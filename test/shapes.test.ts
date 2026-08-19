/**
 * Shape coverage for the convolution kernels.
 *
 * The main parity suite runs tidy shapes. These are the awkward ones: single
 * pixels, single rows and columns, channel counts that divide no tile evenly,
 * input channel counts that are not a multiple of the depthwise kernel's
 * workgroup-memory chunk, odd inputs under stride 2, and the exact shapes the
 * shipping 448 network uses. Every case is checked against the CPU reference.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu } from './helpers/gpu.ts';
import { allocVec4, compare, download, mulberry32, randomFloats, uploadVec4, writeFloats } from './helpers/tensors.ts';
import { makePointwise, pointwiseLayout } from '../src/model/kernels/pointwise.ts';
import { dwpwLayout, makeDepthwisePointwise } from '../src/model/kernels/dwpw.ts';
import { convLayout, makeConv } from '../src/model/kernels/conv.ts';
import { lateralLayout, makeLateral } from '../src/model/kernels/lateral.ts';
import { packBias, packConv, packDepthwise, packPointwise } from '../src/model/layout.ts';
import { refConv, refDepthwisePointwise, refLateral, refPointwise } from '../src/model/reference.ts';
import type { Activation } from '../src/model/arch.ts';

const TOL = { abs: 3e-4, rel: 3e-3 };

describe('PROBE dwpw', () => {
  const cases = [
    { name: '1x1 tensor, inC4=5 outC4=3', inH: 1, inW: 1, inC: 20, outC: 12, k: 3, stride: 1 },
    { name: '1x1 tensor k5 s2', inH: 1, inW: 1, inC: 36, outC: 4, k: 5, stride: 2 },
    { name: '1-row 1x17 k5 s1', inH: 1, inW: 17, inC: 36, outC: 20, k: 5, stride: 1 },
    { name: '1-col 17x1 k3 s2', inH: 17, inW: 1, inC: 12, outC: 44, k: 3, stride: 2 },
    { name: 'inC4=1 (chunk 16 waste)', inH: 5, inW: 5, inC: 4, outC: 60, k: 3, stride: 1 },
    { name: 'inC4=17 not mult of chunk', inH: 6, inW: 6, inC: 68, outC: 36, k: 5, stride: 1 },
    { name: 'inC4=29 not mult of chunk', inH: 5, inW: 7, inC: 116, outC: 12, k: 3, stride: 1 },
    { name: 'odd stride2 9x7 k3', inH: 9, inW: 7, inC: 28, outC: 52, k: 3, stride: 2 },
    { name: 'odd stride2 7x9 k5', inH: 7, inW: 9, inC: 52, outC: 28, k: 5, stride: 2 },
    { name: 'odd stride2 11x11 k5', inH: 11, inW: 11, inC: 44, outC: 4, k: 5, stride: 2 },
    { name: 'REAL e9 14x14 704->176 k5 res', inH: 14, inW: 14, inC: 704, outC: 176, k: 5, stride: 1, residual: true },
    { name: 'REAL e8 28x28 104->416 k5 s2', inH: 28, inW: 28, inC: 104, outC: 416, k: 5, stride: 2 },
    { name: 'REAL e2 dw_expand 224 24->96 s2', inH: 224, inW: 224, inC: 24, outC: 96, k: 3, stride: 2 },
    { name: 'outC4=1 with big inC', inH: 4, inW: 4, inC: 704, outC: 4, k: 3, stride: 1 },
  ] as Array<{ name: string; inH: number; inW: number; inC: number; outC: number; k: number; stride: number; residual?: boolean }>;

  for (const c of cases) {
    it(c.name, async () => {
      const { root } = await headlessGpu();
      const rand = mulberry32(0x1234 ^ (c.inC * 977 + c.outC * 13 + c.k * 5 + c.stride + c.inH));
      const pad = (c.k - 1) >> 1;
      const outH = Math.floor((c.inH + 2 * pad - c.k) / c.stride) + 1;
      const outW = Math.floor((c.inW + 2 * pad - c.k) / c.stride) + 1;
      const residual = !!c.residual;
      const midAct: Activation = 'hardswish';
      const act: Activation = 'relu6';

      const srcData = randomFloats(c.inH * c.inW * c.inC, rand);
      const dwW = randomFloats(c.inC * c.k * c.k, rand, 0.4);
      const dwB = randomFloats(c.inC, rand, 0.3);
      const pwW = randomFloats(c.outC * c.inC, rand, 0.1);
      const pwB = randomFloats(c.outC, rand, 0.3);
      const resData = randomFloats(outH * outW * c.outC, rand);

      const want = refDepthwisePointwise(
        { h: c.inH, w: c.inW, c: c.inC, data: srcData },
        dwW, dwB, midAct, pwW, pwB, c.outC, c.k, c.stride, act,
        residual ? { h: outH, w: outW, c: c.outC, data: resData } : undefined,
      );

      const kernel = makeDepthwisePointwise({
        inC4: c.inC / 4, outC4: c.outC / 4,
        inH: c.inH, inW: c.inW, outH, outW,
        k: c.k, stride: c.stride, midAct, act, residual,
      });

      const src = uploadVec4(root, srcData);
      const dwWgt = uploadVec4(root, packDepthwise(dwW, c.inC, c.k));
      const dwBias = uploadVec4(root, packBias(dwB, c.inC));
      const pwWgt = uploadVec4(root, packPointwise(pwW, c.outC, c.inC));
      const pwBias = uploadVec4(root, packBias(pwB, c.outC));
      const res = uploadVec4(root, resData);
      const guard = 64;
      const dst = allocVec4(root, outH * outW * c.outC + guard);
      writeFloats(root, root.unwrap(dst), new Float32Array(outH * outW * c.outC + guard).fill(-98765));

      const group = root.createBindGroup(dwpwLayout, { src, dwWgt, dwBias, pwWgt, pwBias, res, dst });
      root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);

      const got = await download(root, dst, outH * outW * c.outC + guard);
      const cmp = compare(got.subarray(0, outH * outW * c.outC), want.data, TOL);
      expect(cmp.worst, `${kernel.tiling} out=${outH}x${outW} maxAbs=${cmp.maxAbs}`).toEqual([]);
      const tail = got.subarray(outH * outW * c.outC);
      expect(Array.from(tail).every((v) => v === -98765), 'overran dst').toBe(true);
    });
  }
});

describe('PROBE pointwise', () => {
  const cases = [
    { name: '1 pixel, 704->4', h: 1, w: 1, inC: 704, outC: 4 },
    { name: '1 pixel, 4->704', h: 1, w: 1, inC: 4, outC: 704 },
    { name: '1 row 1x37, inC4=17', h: 1, w: 37, inC: 68, outC: 36 },
    { name: 'prime pixels 101, inC4=29', h: 1, w: 101, inC: 116, outC: 12 },
    { name: 'REAL 14x14 176->704', h: 14, w: 14, inC: 176, outC: 704 },
    { name: 'REAL 112x112 32->128', h: 112, w: 112, inC: 32, outC: 128 },
  ];
  for (const c of cases) {
    it(c.name, async () => {
      const { root } = await headlessGpu();
      const rand = mulberry32(0x777 ^ (c.inC * 31 + c.outC + c.w));
      const pixels = c.h * c.w;
      const srcData = randomFloats(pixels * c.inC, rand);
      const weights = randomFloats(c.outC * c.inC, rand, 0.12);
      const bias = randomFloats(c.outC, rand, 0.4);
      const resData = randomFloats(pixels * c.outC, rand);
      const want = refPointwise({ h: c.h, w: c.w, c: c.inC, data: srcData }, weights, bias, c.outC, 'hardswish', { h: c.h, w: c.w, c: c.outC, data: resData });
      const kernel = makePointwise({ inC4: c.inC / 4, outC4: c.outC / 4, pixels, act: 'hardswish', residual: true });
      const src = uploadVec4(root, srcData);
      const wgt = uploadVec4(root, packPointwise(weights, c.outC, c.inC));
      const bia = uploadVec4(root, packBias(bias, c.outC));
      const res = uploadVec4(root, resData);
      const dst = allocVec4(root, pixels * c.outC);
      const group = root.createBindGroup(pointwiseLayout, { src, wgt, bias: bia, res, dst });
      root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);
      const got = await download(root, dst, pixels * c.outC);
      const cmp = compare(got, want.data, TOL);
      expect(cmp.worst, `${kernel.tiling} maxAbs=${cmp.maxAbs}`).toEqual([]);
    });
  }
});

describe('PROBE conv (stem)', () => {
  const cases = [
    { name: '1x1 in, k3 s1', inH: 1, inW: 1, inC: 4, outC: 16, k: 3, stride: 1 },
    { name: 'odd 9x7 k3 s2', inH: 9, inW: 7, inC: 4, outC: 16, k: 3, stride: 2 },
    { name: 'odd 7x9 k5 s2', inH: 7, inW: 9, inC: 8, outC: 12, k: 5, stride: 2 },
    { name: '1-row 1x33 k3 s2', inH: 1, inW: 33, inC: 4, outC: 20, k: 3, stride: 2 },
    { name: 'REAL stem 448 4->16 k3 s2', inH: 448, inW: 448, inC: 4, outC: 16, k: 3, stride: 2 },
  ];
  for (const c of cases) {
    it(c.name, async () => {
      const { root } = await headlessGpu();
      const rand = mulberry32(0x99 ^ (c.inH * 7 + c.inW * 13 + c.k));
      const pad = (c.k - 1) >> 1;
      const outH = Math.floor((c.inH + 2 * pad - c.k) / c.stride) + 1;
      const outW = Math.floor((c.inW + 2 * pad - c.k) / c.stride) + 1;
      const srcData = randomFloats(c.inH * c.inW * c.inC, rand);
      const weights = randomFloats(c.outC * c.inC * c.k * c.k, rand, 0.2);
      const bias = randomFloats(c.outC, rand, 0.4);
      const want = refConv({ h: c.inH, w: c.inW, c: c.inC, data: srcData }, weights, bias, c.outC, c.k, c.stride, 'hardswish');
      expect([want.h, want.w]).toEqual([outH, outW]);
      const kernel = makeConv({ inC4: c.inC / 4, outC4: c.outC / 4, inH: c.inH, inW: c.inW, outH, outW, k: c.k, stride: c.stride, act: 'hardswish' });
      const src = uploadVec4(root, srcData);
      const wgt = uploadVec4(root, packConv(weights, c.outC, c.inC, c.k));
      const bia = uploadVec4(root, packBias(bias, c.outC));
      const dst = allocVec4(root, outH * outW * c.outC);
      const group = root.createBindGroup(convLayout, { src, wgt, bias: bia, dst });
      root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);
      const got = await download(root, dst, outH * outW * c.outC);
      const cmp = compare(got, want.data, TOL);
      expect(cmp.worst, `${kernel.tiling} maxAbs=${cmp.maxAbs}`).toEqual([]);
    });
  }
});

describe('PROBE lateral', () => {
  const cases = [
    { name: 'coarse 1x1 -> 2x2', cw: 1, ch: 1, outC: 8, skipC: 12 },
    { name: 'coarse 7x7 -> 14x14, skipC4=26', cw: 7, ch: 7, outC: 12, skipC: 104 },
    { name: 'coarse 3x5 -> 6x10', cw: 3, ch: 5, outC: 4, skipC: 4 },
    { name: 'REAL d3 14x14 -> 28x28 skip104 out96', cw: 14, ch: 14, outC: 96, skipC: 104 },
    { name: 'REAL d0 112x112 -> 224x224 skip24 out32', cw: 112, ch: 112, outC: 32, skipC: 24 },
  ];
  for (const c of cases) {
    it(c.name, async () => {
      const { root } = await headlessGpu();
      const rand = mulberry32(0xabc ^ (c.cw * 17 + c.ch * 3 + c.outC));
      const outW = c.cw * 2, outH = c.ch * 2;
      const coarseData = randomFloats(c.ch * c.cw * c.outC, rand);
      const skipData = randomFloats(outH * outW * c.skipC, rand);
      const weights = randomFloats(c.outC * c.skipC, rand, 0.2);
      const bias = randomFloats(c.outC, rand, 0.4);
      const want = refLateral(
        { h: c.ch, w: c.cw, c: c.outC, data: coarseData },
        { h: outH, w: outW, c: c.skipC, data: skipData },
        weights, bias, c.outC, 'hardswish',
      );
      const kernel = makeLateral({ outC4: c.outC / 4, skipC4: c.skipC / 4, coarseH: c.ch, coarseW: c.cw, outH, outW, act: 'hardswish' });
      const coarse = uploadVec4(root, coarseData);
      const skip = uploadVec4(root, skipData);
      const wgt = uploadVec4(root, packPointwise(weights, c.outC, c.skipC));
      const bia = uploadVec4(root, packBias(bias, c.outC));
      const dst = allocVec4(root, outH * outW * c.outC);
      const group = root.createBindGroup(lateralLayout, { coarse, skip, wgt, bias: bia, dst });
      root.createComputePipeline({ compute: kernel.fn }).with(group).dispatchWorkgroups(...kernel.dispatch);
      const got = await download(root, dst, outH * outW * c.outC);
      const cmp = compare(got, want.data, TOL);
      expect(cmp.worst, `${kernel.tiling} maxAbs=${cmp.maxAbs}`).toEqual([]);
    });
  }
});
