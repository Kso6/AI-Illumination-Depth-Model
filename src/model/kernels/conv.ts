/**
 * Dense k×k convolution. Used only for the stem, where the input has four
 * channels and a dense convolution is both cheaper and more expressive than a
 * depthwise-separable one.
 *
 * Three-dimensional dispatch, register-blocked over pixels along x, with the
 * spatial taps fully unrolled. Zero padding is applied via biased unsigned
 * comparisons so no signed arithmetic appears in the address computation.
 */
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import type { Activation } from '../arch.ts';
import { activationFn } from './activation.ts';
import { block4, type BuiltKernel } from './pointwise.ts';

/** See the note on `LOAD_COST` in `../layout.ts`. */
const LOAD_COST = 4;

export const convLayout = tgpu
  .bindGroupLayout({
    src: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    wgt: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    bias: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    dst: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'mutable' },
  })
  .$name('conv');

export interface ConvConfig {
  readonly inC4: number;
  readonly outC4: number;
  readonly inH: number;
  readonly inW: number;
  readonly outH: number;
  readonly outW: number;
  readonly k: number;
  readonly stride: number;
  readonly act: Activation;
}

/**
 * Picks a workgroup shape for a spatially-indexed kernel: `ocLanes` threads
 * along the channel axis and `tx × ty` along the image, 64 threads in total.
 *
 * Costed the same way as `chooseTiling`, with an extra halo term: a k×k kernel
 * reads a `(tileX + k - 1) × (ty + k - 1)` input region, so a 2×32 tile pays a
 * far larger halo than an 8×8 tile of the same area. Without that term the
 * search happily returns long, thin tiles that are legal but read the input
 * several times over.
 *
 * @param k Kernel extent, for the halo term. Use 1 for purely pointwise work.
 */
export function spatialTiling(
  outC4: number,
  outW: number,
  outH: number,
  maxAcc: number,
  inC4 = 8,
  k = 3,
): { ocLanes: number; tx: number; ty: number; pptX: number; oc4PerThread: number } {
  let best = { ocLanes: 1, tx: 8, ty: 8, pptX: 1, oc4PerThread: 1 };
  let bestCost = Infinity;
  for (const ocLanes of [1, 2, 4, 8]) {
    const spatial = 64 / ocLanes;
    for (const tx of [1, 2, 4, 8, 16, 32, 64]) {
      if (tx > spatial) continue;
      const ty = spatial / tx;
      if (!Number.isInteger(ty)) continue;
      for (const pptX of [1, 2, 4]) {
        for (const oc4PerThread of [1, 2, 4]) {
          if (pptX * oc4PerThread > maxAcc) continue;
          const chanTile = ocLanes * oc4PerThread;
          const tileX = tx * pptX;
          const pixTile = tileX * ty;

          const workgroups =
            Math.ceil(outC4 / chanTile) * Math.ceil(outW / tileX) * Math.ceil(outH / ty);
          const macs = pixTile * chanTile * 4 * inC4 * 4 * k * k;
          const inputLoads = (tileX + k - 1) * (ty + k - 1) * inC4;
          const weightLoads = chanTile * inC4 * 4 * k * k;

          const cost = workgroups * (macs + LOAD_COST * (inputLoads + weightLoads));
          if (cost < bestCost) {
            bestCost = cost;
            best = { ocLanes, tx, ty, pptX, oc4PerThread };
          }
        }
      }
    }
  }
  return best;
}

export function makeConv(cfg: ConvConfig): BuiltKernel {
  const { inC4, outC4, inH, inW, outH, outW, k, stride } = cfg;
  const pad = (k - 1) >> 1;
  const t = spatialTiling(outC4, outW, outH, 8, inC4, k);
  const { ocLanes, tx, ty, pptX, oc4PerThread } = t;
  const tileX = tx * pptX;

  const act = activationFn(cfg.act);
  const Acc = d.arrayOf(d.vec4f, pptX * oc4PerThread);

  const fn = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [ocLanes, tx, ty],
  })((input) => {
    'use gpu';
    const oc4Base = input.gid.x * oc4PerThread;
    const ox0 = input.gid.y * pptX;
    const oy = input.gid.z;

    const acc = Acc();
    for (const o of tgpu.unroll(std.range(oc4PerThread))) {
      const b = d.vec4f(convLayout.$.bias[oc4Base + o]);
      for (const i of tgpu.unroll(std.range(pptX))) {
        acc[o * pptX + i] = d.vec4f(b);
      }
    }

    const byBase = oy * stride;
    for (const ky of tgpu.unroll(std.range(k))) {
      const by = byBase + ky;
      if (by >= pad && by < inH + pad) {
        const sy = by - pad;
        for (const kx of tgpu.unroll(std.range(k))) {
          for (let ic4 = d.u32(0); ic4 < inC4; ic4++) {
            for (const o of tgpu.unroll(std.range(oc4PerThread))) {
              const wo = (((oc4Base + o) * inC4 + ic4) * k * k + ky * k + kx) * 4;
              const w0 = d.vec4f(convLayout.$.wgt[wo]);
              const w1 = d.vec4f(convLayout.$.wgt[wo + 1]);
              const w2 = d.vec4f(convLayout.$.wgt[wo + 2]);
              const w3 = d.vec4f(convLayout.$.wgt[wo + 3]);
              for (const i of tgpu.unroll(std.range(pptX))) {
                const bx = (ox0 + i) * stride + kx;
                if (bx >= pad && bx < inW + pad) {
                  const sx = bx - pad;
                  const v = d.vec4f(convLayout.$.src[(sy * inW + sx) * inC4 + ic4]);
                  acc[o * pptX + i] = std.add(acc[o * pptX + i], block4(v, w0, w1, w2, w3));
                }
              }
            }
          }
        }
      }
    }

    for (const o of tgpu.unroll(std.range(oc4PerThread))) {
      const oc4 = oc4Base + o;
      for (const i of tgpu.unroll(std.range(pptX))) {
        const ox = ox0 + i;
        if (oc4 < outC4 && ox < outW && oy < outH) {
          convLayout.$.dst[(oy * outW + ox) * outC4 + oc4] = act(acc[o * pptX + i]);
        }
      }
    }
  });

  return {
    fn,
    dispatch: [
      Math.ceil(outC4 / (ocLanes * oc4PerThread)),
      Math.ceil(outW / tileX),
      Math.ceil(outH / ty),
    ],
    tiling: `${ocLanes}×${tx}×${ty} threads, ${pptX}px × ${oc4PerThread}c4 per thread`,
  };
}
