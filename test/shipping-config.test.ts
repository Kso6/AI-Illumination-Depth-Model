/**
 * Runs the *shipping* 448x448 configuration on a real device.
 *
 * Everything else in the suite runs the network at 64x64 so a CPU reference can
 * keep up. That leaves one thing unverified: whether the tilings, workgroup
 * sizes and workgroup-memory allocations the choosers pick at full resolution
 * are actually legal and actually run. This test builds all forty dispatches at
 * 448 and executes them under error scopes.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu } from './helpers/gpu.ts';
import { mulberry32 } from './helpers/tensors.ts';
import { uploadImage, readRgba16f } from './helpers/textures.ts';
import { ILLUMINA_DEPTH_448 } from '../src/model/arch.ts';
import { createDepthModel } from '../src/model/runner.ts';
import { synthesizeWeights } from '../src/model/weights.ts';

describe('PROBE shipped 448 configuration', () => {
  it('builds and executes all 40 dispatches without validation errors', async () => {
    const { root, device } = await headlessGpu();
    const arch = ILLUMINA_DEPTH_448;
    const S = arch.inputSize;
    const rand = mulberry32(11);
    const px = new Uint8Array(S * S * 4);
    for (let i = 0; i < px.length; i += 4) {
      px[i] = (rand() * 255) | 0; px[i+1] = (rand()*255)|0; px[i+2] = (rand()*255)|0; px[i+3] = 255;
    }
    const source = uploadImage(root, S, px);

    device.pushErrorScope('validation');
    device.pushErrorScope('internal');
    device.pushErrorScope('out-of-memory');

    const model = createDepthModel(root, { arch, weights: synthesizeWeights(arch, 5), source });
    expect(model.stats.dispatches).toBe(40);
    expect(model.arch.ops.length).toBe(40);

    const enc = root['~unstable'].createCommandEncoder({ label: 'probe-448' });
    const pass = enc.beginComputePass({ label: 'inference' });
    model.record(pass);
    pass.end();
    enc.submit();
    await device.queue.onSubmittedWorkDone();

    const oom = await device.popErrorScope();
    const internal = await device.popErrorScope();
    const validation = await device.popErrorScope();
    expect(validation?.message ?? null).toBe(null);
    expect(internal?.message ?? null).toBe(null);
    expect(oom?.message ?? null).toBe(null);

    const depth = await readRgba16f(root, model.currentDepth(), S);
    let finite = 0, min = Infinity, max = -Infinity;
    for (let p = 0; p < S * S; p++) {
      const v = depth[p * 4]!;
      if (Number.isFinite(v)) { finite++; min = Math.min(min, v); max = Math.max(max, v); }
    }
    console.log('448 depth: finite', finite, '/', S*S, 'range', min, max);
    expect(finite).toBe(S * S);
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeLessThanOrEqual(1);
    model.destroy();
  }, 600000);
});
