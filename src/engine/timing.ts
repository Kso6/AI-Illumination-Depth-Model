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

/**
 * A staging slot's lifecycle.
 *
 * This must be a single explicit state rather than a pair of booleans. An
 * earlier version tracked only `pending`, which let two failure modes through:
 * a slot that was still recorded-but-not-yet-mapped could be mapped twice, and
 * because a duplicate `mapAsync` on an already-mapping buffer rejects
 * *immediately*, the rejection handler cleared the flag while the first mapping
 * was still outstanding. The next frame to reach that slot then recorded a copy
 * into a buffer WebGPU considered mapped, which is a validation error.
 *
 *   idle     — free; may be recorded into
 *   recorded — a resolve+copy is in this frame's encoder; must not be re-recorded
 *   mapping  — mapAsync is outstanding; must not be recorded into or re-mapped
 */
type SlotState = 'idle' | 'recorded' | 'mapping';

interface Slot {
  readonly resolve: GPUBuffer;
  readonly staging: GPUBuffer;
  state: SlotState;
  /** Labels captured when this slot was recorded, so results cannot desync. */
  labels: string[];
}

const RING = 3;

export class GpuProfiler {
  readonly available: boolean;
  #querySet: GPUQuerySet | undefined;
  #slots: Slot[] = [];
  #labels: string[] = [];
  #frame = 0;
  #latest: TimingSpan[] = [];
  #capacity: number;
  #destroyed = false;

  /**
   * @param spans Maximum number of nested/sequential spans per frame. Each span
   *   consumes two timestamps.
   */
  constructor(device: GPUDevice, spans = 4) {
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
        state: 'idle',
        labels: [],
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
   *
   * Silently skips the frame when every slot is still in flight — dropping a
   * frame's timings is always preferable to stalling the frame loop.
   */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.available || !this.#querySet || this.#labels.length === 0) return;
    const slot = this.#slots[this.#frame % RING]!;
    if (slot.state !== 'idle') return;
    const count = this.#labels.length * 2;
    encoder.resolveQuerySet(this.#querySet, 0, count, slot.resolve, 0);
    encoder.copyBufferToBuffer(slot.resolve, 0, slot.staging, 0, count * 8);
    // Capture the labels with the slot: by the time the map resolves, several
    // more frames will have called `beginFrame` and replaced `this.#labels`.
    slot.labels = [...this.#labels];
    slot.state = 'recorded';
  }

  /**
   * Call after submitting. Kicks off the asynchronous read of this frame's
   * timings; never blocks.
   */
  afterSubmit(): void {
    if (!this.available) return;
    const slot = this.#slots[this.#frame % RING]!;
    this.#frame++;
    if (slot.state !== 'recorded') return;
    slot.state = 'mapping';

    const labels = slot.labels;
    void slot.staging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        if (this.#destroyed) return;
        const raw = new BigUint64Array(slot.staging.getMappedRange().slice(0));
        const spans: TimingSpan[] = [];
        for (let i = 0; i < labels.length; i++) {
          const start = raw[i * 2]!;
          const end = raw[i * 2 + 1]!;
          // Timestamps are nanoseconds. A pair that is still all-zero means the
          // query never completed; an inverted pair means it is unusable. But an
          // *equal* pair is legitimate — a pass can genuinely take less than the
          // timer's resolution — and dropping those would silently shorten the
          // report and misalign it with the labels the caller opened.
          const valid = end >= start && !(start === 0n && end === 0n);
          if (valid) {
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
        // Only the mapping that actually started may return the slot to `idle`.
        if (slot.state === 'mapping') slot.state = 'idle';
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
    // Destroying a buffer with an outstanding `mapAsync` rejects that promise;
    // the flag stops the handler from touching freed objects afterwards.
    this.#destroyed = true;
    this.#querySet?.destroy();
    for (const s of this.#slots) {
      s.resolve.destroy();
      s.staging.destroy();
      s.state = 'idle';
    }
    this.#slots = [];
  }
}
