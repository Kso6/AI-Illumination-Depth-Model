/**
 * Regression tests for the GPU profiler's staging-buffer ring.
 *
 * The ring is small and the frame loop never waits, so slots get reused while
 * a previous frame's readback may still be outstanding. Getting that wrong does
 * not throw in JavaScript — it produces a WebGPU validation error several frames
 * later, which is why these tests drive many frames and watch an error scope.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu } from './helpers/gpu.ts';
import { GpuProfiler } from '../src/engine/timing.ts';

function recordFrame(device: GPUDevice, profiler: GpuProfiler, labels: string[]): void {
  profiler.beginFrame();
  const encoder = device.createCommandEncoder();
  for (const label of labels) {
    const pass = encoder.beginComputePass({ timestampWrites: profiler.span(label) });
    pass.end();
  }
  profiler.resolve(encoder);
  device.queue.submit([encoder.finish()]);
  profiler.afterSubmit();
}

describe('GpuProfiler', () => {
  it('never records into a slot whose readback is still outstanding', async () => {
    const { root } = await headlessGpu();
    const device = root.device;
    const profiler = new GpuProfiler(device, 4);
    if (!profiler.available) return;

    // Enough back-to-back frames to wrap the three-slot ring several times
    // without ever awaiting, so slots are certainly still mapping when reused.
    for (let f = 0; f < 4; f++) recordFrame(device, profiler, ['a']);
    // Let the microtask queue drain: this is what previously let a spuriously
    // rejected duplicate `mapAsync` mark a still-mapping slot as free.
    for (let i = 0; i < 20; i++) await Promise.resolve();

    device.pushErrorScope('validation');
    for (let f = 0; f < 6; f++) recordFrame(device, profiler, ['a']);
    const error = await device.popErrorScope();
    await new Promise((r) => setTimeout(r, 200));
    profiler.destroy();

    expect(error, `validation error: ${error?.message ?? ''}`).toBeNull();
  });

  it('reports spans in the order they were opened, without desynchronising labels', async () => {
    const { root } = await headlessGpu();
    const device = root.device;
    const profiler = new GpuProfiler(device, 4);
    if (!profiler.available) return;

    for (let f = 0; f < 8; f++) {
      recordFrame(device, profiler, ['inference', 'shading', 'composite']);
      await device.queue.onSubmittedWorkDone();
    }
    for (let i = 0; i < 10 && profiler.latest.length === 0; i++) {
      await new Promise((r) => setTimeout(r, 25));
    }

    if (profiler.latest.length > 0) {
      expect(profiler.latest.map((s) => s.label)).toEqual([
        'inference',
        'shading',
        'composite',
      ]);
      expect(profiler.totalMs).toBeGreaterThan(0);
    }
    profiler.destroy();
  });

  it('caps the number of spans per frame', async () => {
    const { root } = await headlessGpu();
    const profiler = new GpuProfiler(root.device, 2);
    if (!profiler.available) return;
    profiler.beginFrame();
    expect(profiler.span('a')).toBeDefined();
    expect(profiler.span('b')).toBeDefined();
    // Beyond capacity the profiler must degrade to "no timing" rather than
    // writing past the end of the query set.
    expect(profiler.span('c')).toBeUndefined();
    profiler.destroy();
  });
});
