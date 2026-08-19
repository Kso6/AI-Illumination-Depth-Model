/**
 * Per-kernel numerical parity: every WGSL kernel is executed on a real WebGPU
 * device and checked against the CPU reference in `src/model/reference.ts`.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu } from './helpers/gpu.ts';
import {
  allocVec4,
  compare,
  download,
  mulberry32,
  randomFloats,
  uploadVec4,
  writeFloats,
} from './helpers/tensors.ts';
import { makePointwise, pointwiseLayout } from '../src/model/kernels/pointwise.ts';
import { dwpwLayout, makeDepthwisePointwise } from '../src/model/kernels/dwpw.ts';
import { convLayout, makeConv } from '../src/model/kernels/conv.ts';
import { lateralLayout, makeLateral } from '../src/model/kernels/lateral.ts';
import { PreprocessParams, makePreprocess, preprocessLayout } from '../src/model/kernels/io.ts';
import { uploadImage, readRgba16f } from './helpers/textures.ts';
import { textureStorage2d, vec2f, vec4f } from 'typegpu/data';
import { packBias, packConv, packDepthwise, packPointwise } from '../src/model/layout.ts';
import {
  refConv,
  refDepthwisePointwise,
  refLateral,
  refPointwise,
  refPreprocess,
} from '../src/model/reference.ts';
import type { Activation } from '../src/model/arch.ts';

describe('pointwise convolution', () => {
  const cases: Array<{
    name: string;
    h: number;
    w: number;
    inC: number;
    outC: number;
    act: Activation;
    residual: boolean;
  }> = [
    { name: 'square, exact tiling', h: 16, w: 16, inC: 32, outC: 32, act: 'relu6', residual: false },
    { name: 'ragged pixels', h: 13, w: 17, inC: 12, outC: 8, act: 'hardswish', residual: false },
    { name: 'ragged channels', h: 7, w: 9, inC: 20, outC: 12, act: 'linear', residual: true },
    { name: 'wide expansion', h: 8, w: 8, inC: 24, outC: 96, act: 'hardswish', residual: false },
    { name: 'narrow projection', h: 14, w: 14, inC: 704, outC: 176, act: 'linear', residual: true },
    { name: 'single group out', h: 5, w: 5, inC: 8, outC: 4, act: 'relu', residual: false },
  ];

  for (const c of cases) {
    it(`matches the reference — ${c.name}`, async () => {
      const { root } = await headlessGpu();
      const rand = mulberry32(0xc0ffee ^ (c.inC * 131 + c.outC));

      const pixels = c.h * c.w;
      const srcData = randomFloats(pixels * c.inC, rand);
      const weights = randomFloats(c.outC * c.inC, rand, 0.25);
      const bias = randomFloats(c.outC, rand, 0.5);
      const resData = randomFloats(pixels * c.outC, rand);

      const want = refPointwise(
        { h: c.h, w: c.w, c: c.inC, data: srcData },
        weights,
        bias,
        c.outC,
        c.act,
        c.residual ? { h: c.h, w: c.w, c: c.outC, data: resData } : undefined,
      );

      const kernel = makePointwise({
        inC4: c.inC / 4,
        outC4: c.outC / 4,
        pixels,
        act: c.act,
        residual: c.residual,
      });

      const src = uploadVec4(root, srcData);
      const wgt = uploadVec4(root, packPointwise(weights, c.outC, c.inC));
      const bia = uploadVec4(root, packBias(bias, c.outC));
      const res = uploadVec4(root, resData);
      const dst = allocVec4(root, pixels * c.outC);

      const group = root.createBindGroup(pointwiseLayout, { src, wgt, bias: bia, res, dst });
      root
        .createComputePipeline({ compute: kernel.fn })
        .with(group)
        .dispatchWorkgroups(...kernel.dispatch);

      const got = await download(root, dst, pixels * c.outC);
      const cmp = compare(got, want.data);
      expect(
        cmp.worst,
        `maxAbs=${cmp.maxAbs.toExponential(3)} maxRel=${cmp.maxRel.toExponential(3)} ` +
          `tiling: ${kernel.tiling}`,
      ).toEqual([]);
    });
  }

  it('writes nothing outside the destination tensor', async () => {
    const { root } = await headlessGpu();
    const rand = mulberry32(7);
    // 13×13 pixels with a 12-channel output will not divide evenly into any
    // tile, so out-of-range lanes must be masked off rather than clamped into
    // the last element.
    const h = 13;
    const w = 13;
    const inC = 8;
    const outC = 12;
    const pixels = h * w;

    const srcData = randomFloats(pixels * inC, rand);
    const weights = randomFloats(outC * inC, rand, 0.25);
    const bias = randomFloats(outC, rand, 0.5);

    const kernel = makePointwise({ inC4: inC / 4, outC4: outC / 4, pixels, act: 'linear', residual: false });

    const src = uploadVec4(root, srcData);
    const wgt = uploadVec4(root, packPointwise(weights, outC, inC));
    const bia = uploadVec4(root, packBias(bias, outC));
    const guardFloats = pixels * outC + 64;
    const dst = allocVec4(root, guardFloats);
    const sentinel = new Float32Array(guardFloats).fill(-12345);
    writeFloats(root, root.unwrap(dst), sentinel);

    const group = root.createBindGroup(pointwiseLayout, { src, wgt, bias: bia, res: src, dst });
    root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);

    const got = await download(root, dst, guardFloats);
    const tail = got.subarray(pixels * outC);
    expect(Array.from(tail).every((v) => v === -12345)).toBe(true);
  });
});

describe('fused depthwise + pointwise', () => {
  const cases: Array<{
    name: string;
    inH: number;
    inW: number;
    inC: number;
    outC: number;
    k: number;
    stride: number;
    midAct: Activation;
    act: Activation;
    residual: boolean;
  }> = [
    { name: '3x3 s1 16->24', inH: 12, inW: 12, inC: 16, outC: 24, k: 3, stride: 1, midAct: 'relu6', act: 'linear', residual: false },
    { name: '3x3 s2 24->96', inH: 16, inW: 16, inC: 24, outC: 96, k: 3, stride: 2, midAct: 'linear', act: 'hardswish', residual: false },
    { name: '5x5 s2 32->128', inH: 14, inW: 14, inC: 32, outC: 128, k: 5, stride: 2, midAct: 'linear', act: 'hardswish', residual: false },
    { name: '5x5 s1 224->56 residual', inH: 8, inW: 8, inC: 224, outC: 56, k: 5, stride: 1, midAct: 'hardswish', act: 'linear', residual: true },
    { name: '5x5 s1 704->176 residual', inH: 6, inW: 6, inC: 704, outC: 176, k: 5, stride: 1, midAct: 'hardswish', act: 'linear', residual: true },
    { name: 'ragged 13x11 3x3 s1', inH: 13, inW: 11, inC: 48, outC: 48, k: 3, stride: 1, midAct: 'hardswish', act: 'hardswish', residual: true },
    { name: 'ragged 15x13 5x5 s2', inH: 15, inW: 13, inC: 104, outC: 416, k: 5, stride: 2, midAct: 'linear', act: 'hardswish', residual: false },
  ];

  for (const c of cases) {
    it(`matches the reference — ${c.name}`, async () => {
      const { root } = await headlessGpu();
      const rand = mulberry32(0xbeef ^ (c.inC * 31 + c.outC * 7 + c.k));
      const pad = (c.k - 1) >> 1;
      const outH = Math.floor((c.inH + 2 * pad - c.k) / c.stride) + 1;
      const outW = Math.floor((c.inW + 2 * pad - c.k) / c.stride) + 1;

      const srcData = randomFloats(c.inH * c.inW * c.inC, rand);
      const dwW = randomFloats(c.inC * c.k * c.k, rand, 0.5);
      const dwB = randomFloats(c.inC, rand, 0.3);
      const pwW = randomFloats(c.outC * c.inC, rand, 0.2);
      const pwB = randomFloats(c.outC, rand, 0.3);
      const resData = randomFloats(outH * outW * c.outC, rand);

      const want = refDepthwisePointwise(
        { h: c.inH, w: c.inW, c: c.inC, data: srcData },
        dwW, dwB, c.midAct, pwW, pwB, c.outC, c.k, c.stride, c.act,
        c.residual ? { h: outH, w: outW, c: c.outC, data: resData } : undefined,
      );
      expect([want.h, want.w]).toEqual([outH, outW]);

      const kernel = makeDepthwisePointwise({
        inC4: c.inC / 4, outC4: c.outC / 4,
        inH: c.inH, inW: c.inW, outH, outW,
        k: c.k, stride: c.stride, midAct: c.midAct, act: c.act, residual: c.residual,
      });

      const src = uploadVec4(root, srcData);
      const dwWgt = uploadVec4(root, packDepthwise(dwW, c.inC, c.k));
      const dwBias = uploadVec4(root, packBias(dwB, c.inC));
      const pwWgt = uploadVec4(root, packPointwise(pwW, c.outC, c.inC));
      const pwBias = uploadVec4(root, packBias(pwB, c.outC));
      const res = uploadVec4(root, resData);
      const dst = allocVec4(root, outH * outW * c.outC);

      const group = root.createBindGroup(dwpwLayout, { src, dwWgt, dwBias, pwWgt, pwBias, res, dst });
      root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);

      const got = await download(root, dst, outH * outW * c.outC);
      const cmp = compare(got, want.data);
      expect(
        cmp.worst,
        `maxAbs=${cmp.maxAbs.toExponential(3)} maxRel=${cmp.maxRel.toExponential(3)}\n${kernel.tiling}`,
      ).toEqual([]);
    });
  }
});

describe('dense convolution (stem)', () => {
  const cases = [
    { name: '3x3 s2 4->16', inH: 32, inW: 32, inC: 4, outC: 16, k: 3, stride: 2, act: 'hardswish' as Activation },
    { name: '3x3 s1 8->12', inH: 9, inW: 11, inC: 8, outC: 12, k: 3, stride: 1, act: 'relu6' as Activation },
    { name: '5x5 s2 4->32 ragged', inH: 15, inW: 13, inC: 4, outC: 32, k: 5, stride: 2, act: 'linear' as Activation },
  ];

  for (const c of cases) {
    it(`matches the reference — ${c.name}`, async () => {
      const { root } = await headlessGpu();
      const rand = mulberry32(0x5eed ^ (c.outC * 17 + c.k));
      const pad = (c.k - 1) >> 1;
      const outH = Math.floor((c.inH + 2 * pad - c.k) / c.stride) + 1;
      const outW = Math.floor((c.inW + 2 * pad - c.k) / c.stride) + 1;

      const srcData = randomFloats(c.inH * c.inW * c.inC, rand);
      const w = randomFloats(c.outC * c.inC * c.k * c.k, rand, 0.3);
      const b = randomFloats(c.outC, rand, 0.4);

      const want = refConv({ h: c.inH, w: c.inW, c: c.inC, data: srcData }, w, b, c.outC, c.k, c.stride, c.act);
      expect([want.h, want.w]).toEqual([outH, outW]);

      const kernel = makeConv({
        inC4: c.inC / 4, outC4: c.outC / 4,
        inH: c.inH, inW: c.inW, outH, outW, k: c.k, stride: c.stride, act: c.act,
      });

      const src = uploadVec4(root, srcData);
      const wgt = uploadVec4(root, packConv(w, c.outC, c.inC, c.k));
      const bias = uploadVec4(root, packBias(b, c.outC));
      const dst = allocVec4(root, outH * outW * c.outC);

      const group = root.createBindGroup(convLayout, { src, wgt, bias, dst });
      root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);

      const got = await download(root, dst, outH * outW * c.outC);
      const cmp = compare(got, want.data);
      expect(cmp.worst, `maxAbs=${cmp.maxAbs.toExponential(3)} ${kernel.tiling}`).toEqual([]);
    });
  }
});

describe('FPN lateral fusion', () => {
  const cases = [
    { name: '14->28, 96ch, skip 104', ch: 7, cw: 7, outC: 96, skipC: 104, act: 'linear' as Activation },
    { name: '4->8, 32ch, skip 24', ch: 4, cw: 4, outC: 32, skipC: 24, act: 'hardswish' as Activation },
    { name: 'non-square 5x3', ch: 5, cw: 3, outC: 48, skipC: 32, act: 'linear' as Activation },
  ];

  for (const c of cases) {
    it(`matches the reference — ${c.name}`, async () => {
      const { root } = await headlessGpu();
      const rand = mulberry32(0xfeed ^ (c.outC * 13 + c.skipC));
      const outH = c.ch * 2;
      const outW = c.cw * 2;

      const coarseData = randomFloats(c.ch * c.cw * c.outC, rand);
      const skipData = randomFloats(outH * outW * c.skipC, rand);
      const w = randomFloats(c.outC * c.skipC, rand, 0.25);
      const b = randomFloats(c.outC, rand, 0.4);

      const want = refLateral(
        { h: c.ch, w: c.cw, c: c.outC, data: coarseData },
        { h: outH, w: outW, c: c.skipC, data: skipData },
        w, b, c.outC, c.act,
      );

      const kernel = makeLateral({
        outC4: c.outC / 4, skipC4: c.skipC / 4,
        coarseH: c.ch, coarseW: c.cw, outH, outW, act: c.act,
      });

      const coarse = uploadVec4(root, coarseData);
      const skip = uploadVec4(root, skipData);
      const wgt = uploadVec4(root, packPointwise(w, c.outC, c.skipC));
      const bias = uploadVec4(root, packBias(b, c.outC));
      const dst = allocVec4(root, outH * outW * c.outC);

      const group = root.createBindGroup(lateralLayout, { coarse, skip, wgt, bias, dst });
      root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);

      const got = await download(root, dst, outH * outW * c.outC);
      const cmp = compare(got, want.data);
      expect(cmp.worst, `maxAbs=${cmp.maxAbs.toExponential(3)} ${kernel.tiling}`).toEqual([]);
    });
  }
});

describe('preprocess', () => {
  it('normalises, linearises and mirrors the source into scene colour', async () => {
    const { root } = await headlessGpu();
    const SIZE = 32;
    const rand = mulberry32(0xa11ce);

    const bytes = new Uint8Array(SIZE * SIZE * 4);
    for (let i = 0; i < bytes.length; i += 4) {
      bytes[i] = Math.floor(rand() * 256);
      bytes[i + 1] = Math.floor(rand() * 256);
      bytes[i + 2] = Math.floor(rand() * 256);
      bytes[i + 3] = 255;
    }
    const source = uploadImage(root, SIZE, bytes);

    const mean = [0.485, 0.456, 0.406] as const;
    const std = [0.229, 0.224, 0.225] as const;
    const EXPOSURE = 1.3;

    const kernel = makePreprocess({ size: SIZE, mean, std });
    const dst = allocVec4(root, SIZE * SIZE * 4);
    const sceneColor = root
      .createTexture({ size: [SIZE, SIZE], format: 'rgba16float' })
      .$usage('sampled', 'storage');

    const group = root.createBindGroup(preprocessLayout, {
      source,
      samp: root.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
      params: root.createUniform(PreprocessParams, {
        uvScale: vec2f(1, 1),
        uvOffset: vec2f(0, 0),
        options: vec4f(1, EXPOSURE, 0, 0),
      }),
      dst,
      sceneColor: sceneColor.createView(textureStorage2d('rgba16float', 'write-only')),
    });
    root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);

    const rgbaFloat = new Float32Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) rgbaFloat[i] = bytes[i]! / 255;
    const want = refPreprocess(rgbaFloat, SIZE, mean, std, true, EXPOSURE);

    const gotInput = await download(root, dst, SIZE * SIZE * 4);
    const inputCmp = compare(gotInput, want.input.data, { abs: 1e-5, rel: 1e-5 });
    expect(inputCmp.worst, `input maxAbs=${inputCmp.maxAbs.toExponential(3)}`).toEqual([]);

    // Scene colour is fp16, so only ~3 decimal digits survive.
    const gotScene = await readRgba16f(root, sceneColor, SIZE);
    const sceneCmp = compare(gotScene, want.sceneColor.data, { abs: 2e-3, rel: 2e-3 });
    expect(sceneCmp.worst, `sceneColor maxAbs=${sceneCmp.maxAbs.toExponential(3)}`).toEqual([]);
  });
});
