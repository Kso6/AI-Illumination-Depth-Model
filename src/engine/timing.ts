/**
 * GPU-side frame timing via timestamp queries.
 *
 * CPU-side `performance.now()` around a submit measures nothing useful: WebGPU
 * submission is asynchronous, so it tells you how long it took to *record*
 * commands, not to run them. Timestamp queries are the only way to substantiate
 * a claim like "inference takes 8 ms", so the profiler is part of the engine
 * rather than a debugging afterthought.
 *
 * Results are read back through a small ring of staging buffers: a frame's
 * timings arrive two or three frames later, which is exactly the right trade —
 * the frame loop never waits on the GPU.
 */

export interface TimingSpan {
  readonly label: string;
  /** Milliseconds of GPU time. */
  readonly ms: number;
}

interface Slot {
  readonly resolve: GPUBuffer;
  readonly staging: GPUBuffer;
  mapped: boolean;
  pending: boolean;
}

const RING = 3;

export class GpuProfiler {
  readonly available: boolean;
  #device: GPUDevice;
  #querySet: GPUQuerySet | undefined;
  #slots: Slot[] = [];
  #labels: string[] = [];
  #frame = 0;
  #latest: TimingSpan[] = [];
  #capacity: number;

  /**
   * @param spans Maximum number of nested/sequential spans per frame. Each span
   *   consumes two timestamps.
   */
  constructor(device: GPUDevice, spans = 4) {
    this.#device = device;
    this.#capacity = spans;
    this.available =
      (device.features as unknown as ReadonlySet<string>).has('timestamp-query') ?? false;

    if (!this.available) return;

    this.#querySet = device.createQuerySet({ type: 'timestamp', count: spans * 2 });
    for (let i = 0; i < RING; i++) {
      this.#slots.push({
        resolve: device.createBuffer({
          size: spans * 2 * 8,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
        staging: device.createBuffer({
          size: spans * 2 * 8,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        mapped: false,
        pending: false,
      });
    }
  }

  /** Call at the start of each frame, before recording. */
  beginFrame(): void {
    this.#labels = [];
  }

  /**
   * Reserves a span and returns the `timestampWrites` descriptor to attach to a
   * pass, or `undefined` when timestamp queries are unsupported.
   */
  span(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.available || !this.#querySet) return undefined;
    const index = this.#labels.length;
    if (index >= this.#capacity) return undefined;
    this.#labels.push(label);
    return {
      querySet: this.#querySet,
      beginningOfPassWriteIndex: index * 2,
      endOfPassWriteIndex: index * 2 + 1,
    };
  }

  /**
   * Records the query resolve into the frame's encoder. Must be called after all
   * passes are ended but before the encoder is finished.
   */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.available || !this.#querySet || this.#labels.length === 0) return;
    const slot = this.#slots[this.#frame % RING]!;
    if (slot.pending) return; // Still being read; skip this frame's timing.
    const count = this.#labels.length * 2;
    encoder.resolveQuerySet(this.#querySet, 0, count, slot.resolve, 0);
    encoder.copyBufferToBuffer(slot.resolve, 0, slot.staging, 0, count * 8);
    slot.pending = true;
  }

  /**
   * Call after submitting. Kicks off the asynchronous read of an older frame's
   * timings; never blocks.
   */
  afterSubmit(): void {
    if (!this.available) return;
    const labels = [...this.#labels];
    const slot = this.#slots[this.#frame % RING]!;
    this.#frame++;
    if (!slot.pending) return;

    void slot.staging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const raw = new BigUint64Array(slot.staging.getMappedRange().slice(0));
        const spans: TimingSpan[] = [];
        for (let i = 0; i < labels.length; i++) {
          const start = raw[i * 2]!;
          const end = raw[i * 2 + 1]!;
          // Timestamps are nanoseconds; a zero or inverted pair means the query
          // did not complete and should be ignored rather than reported as 0 ms.
          if (end > start) {
            spans.push({ label: labels[i]!, ms: Number(end - start) / 1e6 });
          }
        }
        if (spans.length) this.#latest = spans;
        slot.staging.unmap();
      })
      .catch(() => {
        /* device lost or buffer destroyed — timings are best-effort */
      })
      .finally(() => {
        slot.pending = false;
      });
  }

  /** The most recent complete set of timings. */
  get latest(): readonly TimingSpan[] {
    return this.#latest;
  }

  get totalMs(): number {
    return this.#latest.reduce((a, s) => a + s.ms, 0);
  }

  destroy(): void {
    this.#querySet?.destroy();
    for (const s of this.#slots) {
      s.resolve.destroy();
      s.staging.destroy();
    }
    this.#slots = [];
  }
}
