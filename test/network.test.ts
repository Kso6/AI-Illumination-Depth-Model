/**
 * End-to-end verification of the whole graph.
 *
 * The network is instantiated at 64×64 rather than 448×448 — geometrically
 * identical (same blocks, channel widths, kernel sizes and five downsamples),
 * just 49× less work — so a complete CPU forward pass finishes in seconds and
 * can be compared against the GPU result op for op.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu } from './helpers/gpu.ts';
import { compare, download, mulberry32 } from './helpers/tensors.ts';
import { readRgba16f, uploadImage } from './helpers/textures.ts';
import { buildArchitecture, costOf, ILLUMINA_DEPTH_448 } from '../src/model/arch.ts';
import { createDepthModel } from '../src/model/runner.ts';
import { synthesizeWeights, serializeWeights, parseWeights } from '../src/model/weights.ts';
import {
  forwardReference,
  refBilateralUpsample,
  refPreprocess,
} from '../src/model/reference.ts';
import { planArena, validateArena } from '../src/engine/arena.ts';

const SIZE = 64;
const RANGE_SIGMA = 0.12;

/** A deterministic test image with edges, gradients and texture. */
function testImage(size: number): Uint8Array {
  const rand = mulberry32(0x1234);
  const px = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const disc = (x - size * 0.35) ** 2 + (y - size * 0.4) ** 2 < (size * 0.22) ** 2;
      const stripe = ((x + y) >> 3) & 1;
      px[i] = disc ? 230 : Math.floor((x / size) * 200 + stripe * 25);
      px[i + 1] = disc ? 60 : Math.floor((y / size) * 180 + rand() * 20);
      px[i + 2] = disc ? 90 : Math.floor(120 + Math.sin(x * 0.4) * 60);
      px[i + 3] = 255;
    }
  }
  return px;
}

describe('IlluminaDepth network', () => {
  it('matches the CPU reference end to end', async () => {
    const { root } = await headlessGpu();
    const arch = buildArchitecture(SIZE);
    const weights = synthesizeWeights(arch, 20240819);

    const pixels = testImage(SIZE);
    const source = uploadImage(root, SIZE, pixels);

    const model = createDepthModel(root, {
      arch,
      weights,
      source,
      decodeSrgb: true,
      exposure: 1,
    });
    model.setTemporal(0.35, 3.0, RANGE_SIGMA);

    // Everything — every convolution, the head, the upsample — in one encoder.
    const encoder = root['~unstable'].createCommandEncoder();
    const pass = encoder.beginComputePass();
    model.record(pass);
    pass.end();
    encoder.submit();

    // --- CPU reference ----------------------------------------------------
    // `uvScale = 1` and a source the same size as the network input put every
    // sample exactly on a texel centre, so bilinear sampling is an exact fetch
    // and the reference needs no resampling of its own.
    const rgbaFloat = new Float32Array(SIZE * SIZE * 4);
    for (let i = 0; i < rgbaFloat.length; i++) rgbaFloat[i] = pixels[i]! / 255;
    const pre = refPreprocess(rgbaFloat, SIZE, arch.mean, arch.std, true, 1);
    const values = forwardReference(arch, weights, pre.input);
    const wantLow = values.get('depth_low')!;
    const wantFull = refBilateralUpsample(wantLow, pre.sceneColor, RANGE_SIGMA);

    // --- half-resolution disparity ----------------------------------------
    // `input` is deliberately not checked here: the arena recycles its buffer
    // immediately after the stem reads it, so a post-`record` readback would
    // return whichever later tensor now owns that memory. The preprocess kernel
    // is verified on its own in `kernels.test.ts`.
    const lowTensor = model.debugTensor('depth_low')!;
    expect(lowTensor.liveAtEnd).toBe(true);
    const gotLow = await download(root, lowTensor.buffer, (SIZE / 2) * (SIZE / 2) * 4);
    // The head broadcasts its scalar across all four channels; compare channel 0.
    const gotLow0 = new Float32Array((SIZE / 2) * (SIZE / 2));
    const wantLow0 = new Float32Array(gotLow0.length);
    for (let i = 0; i < gotLow0.length; i++) {
      gotLow0[i] = gotLow[i * 4]!;
      wantLow0[i] = wantLow.data[i * 4]!;
    }
    const lowCmp = compare(gotLow0, wantLow0, { abs: 1.5e-3, rel: 5e-3 });
    expect(
      lowCmp.worst,
      `depth_low maxAbs=${lowCmp.maxAbs.toExponential(3)} maxRel=${lowCmp.maxRel.toExponential(3)}`,
    ).toEqual([]);

    // --- full-resolution depth texture ------------------------------------
    const gotDepth = await readRgba16f(root, model.currentDepth(), SIZE);
    const gotDepth0 = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < gotDepth0.length; i++) gotDepth0[i] = gotDepth[i * 4]!;
    // Stored as fp16, so ~2^-11 relative quantisation on top of the fp32 drift.
    const fullCmp = compare(gotDepth0, wantFull.data, { abs: 2e-3, rel: 1e-2 });
    expect(
      fullCmp.worst,
      `depth maxAbs=${fullCmp.maxAbs.toExponential(3)} maxRel=${fullCmp.maxRel.toExponential(3)}`,
    ).toEqual([]);

    model.destroy();
  });

  it('produces a finite disparity field in [0, 1]', async () => {
    const { root } = await headlessGpu();
    const arch = buildArchitecture(SIZE);
    const model = createDepthModel(root, {
      arch,
      weights: synthesizeWeights(arch, 7),
      source: uploadImage(root, SIZE, testImage(SIZE)),
    });

    const encoder = root['~unstable'].createCommandEncoder();
    const pass = encoder.beginComputePass();
    model.record(pass);
    pass.end();
    encoder.submit();

    const depth = await readRgba16f(root, model.currentDepth(), SIZE);
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < SIZE * SIZE; i++) {
      const v = depth[i * 4]!;
      expect(Number.isFinite(v)).toBe(true);
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeLessThanOrEqual(1);
    model.destroy();
  });

  it('records exactly one dispatch per op', async () => {
    const { root } = await headlessGpu();
    const arch = buildArchitecture(SIZE);
    const model = createDepthModel(root, {
      arch,
      weights: synthesizeWeights(arch, 1),
      source: uploadImage(root, SIZE, testImage(SIZE)),
    });
    expect(model.stats.dispatches).toBe(arch.ops.length);
    expect(model.stats.dispatches).toBe(40);
    model.destroy();
  });

  it('reuses arena buffers without overlapping live ranges', () => {
    for (const size of [64, 448]) {
      const arch = buildArchitecture(size);
      const exclude = new Set(['depth']);
      const plan = planArena(arch, exclude);
      // Throws if any two co-resident tensors were given the same buffer.
      validateArena(arch, plan, exclude);
      expect(plan.totalBytes).toBeLessThan(plan.naiveBytes);
    }
  });

  it('round-trips weights through the .idm container', () => {
    const arch = buildArchitecture(SIZE);
    const original = synthesizeWeights(arch, 99);
    const restored = parseWeights(serializeWeights(arch, original), arch);
    expect(restored.size).toBe(original.size);
    for (const [name, want] of original) {
      expect(Array.from(restored.get(name)!.subarray(0, 8))).toEqual(
        Array.from(want.subarray(0, 8)),
      );
    }
  });

  it('rejects a weight file built for a different architecture', () => {
    const arch = buildArchitecture(SIZE);
    const bytes = serializeWeights(arch, synthesizeWeights(arch, 5));
    const view = new DataView(bytes);
    const headerLen = view.getUint32(0, true);
    const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(bytes, 4, headerLen)),
    ) as { arch: string };
    expect(header.arch).toBe(arch.name);
    expect(() => parseWeights(bytes, { ...arch, name: 'SomethingElse' })).toThrow(
      /SomethingElse/,
    );
  });

  it('keeps the shipping configuration within its FLOP budget', () => {
    const cost = costOf(ILLUMINA_DEPTH_448);
    // 8 ms on an M1 Max needs the network to stay near 2 GFLOP; guard against
    // an architecture edit silently blowing the budget.
    expect(cost.totalMacs * 2).toBeLessThan(3e9);
    expect(cost.dispatches).toBe(40);
    expect(cost.totalParams).toBeLessThan(2e6);
  });
});
