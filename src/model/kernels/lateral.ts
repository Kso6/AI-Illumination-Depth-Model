/**
 * FPN lateral fusion: `act( bias + upsample2(coarse) + W · skip )`.
 *
 * Each thread owns one *coarse* texel, i.e. a 2×2 block of output pixels. That
 * choice has two payoffs:
 *
 *  - No integer division is needed to get from an output pixel back to its
 *    coarse neighbours (TGSL's `/` is floating-point, following JavaScript).
 *  - The 2×2 block's bilinear taps overlap: nine coarse loads serve four
 *    outputs instead of the sixteen a per-pixel kernel would issue.
 *
 * With `scale_factor = 2` and `align_corners = false`, PyTorch's source
 * coordinate is `(i + 0.5)/2 - 0.5`, so the interpolation weights are always
 * ¾ for the nearer texel and ¼ for the farther one — constants, not arithmetic.
 */
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import type { Activation } from '../arch.ts';
import { activationFn } from './activation.ts';
import { block4, type BuiltKernel } from './pointwise.ts';
import { spatialTiling } from './conv.ts';

export const lateralLayout = tgpu
  .bindGroupLayout({
    coarse: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    skip: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    wgt: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    bias: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    dst: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'mutable' },
  })
  .$name('lateral');

export interface LateralConfig {
  /** Channel groups of the coarse tensor **and** of the output — they match. */
  readonly outC4: number;
  readonly skipC4: number;
  /** Coarse resolution; the output is exactly twice this. */
  readonly coarseH: number;
  readonly coarseW: number;
  readonly outH: number;
  readonly outW: number;
  readonly act: Activation;
}

export function makeLateral(cfg: LateralConfig): BuiltKernel {
  const { outC4, skipC4, coarseH, coarseW, outH, outW } = cfg;
  if (outH !== coarseH * 2 || outW !== coarseW * 2) {
    throw new Error(
      `lateral: output ${outH}×${outW} is not exactly twice the coarse ${coarseH}×${coarseW}`,
    );
  }

  // One thread per coarse texel; four output pixels each, so the accumulator
  // budget is spent on channels rather than pixels.
  const t = spatialTiling(outC4, coarseW, coarseH, 2, skipC4, 1);
  const { ocLanes, tx, ty, oc4PerThread } = t;

  const act = activationFn(cfg.act);
  const Acc = d.arrayOf(d.vec4f, 4 * oc4PerThread);

  /** Clamped fetch of one coarse texel. */
  const fetchCoarse = tgpu
    .fn([d.u32, d.u32, d.u32], d.vec4f)((cx, cy, c4) => {
      'use gpu';
      const x = std.min(cx, coarseW - 1);
      const y = std.min(cy, coarseH - 1);
      return d.vec4f(lateralLayout.$.coarse[(y * coarseW + x) * outC4 + c4]);
    })
    .$name('fetchCoarse');

  const fn = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [ocLanes, tx, ty],
  })((input) => {
    'use gpu';
    const oc4Base = input.gid.x * oc4PerThread;
    const m = input.gid.y;
    const n = input.gid.z;

    const acc = Acc();

    for (const o of tgpu.unroll(std.range(oc4PerThread))) {
      const c4 = oc4Base + o;
      const b = d.vec4f(lateralLayout.$.bias[c4]);

      // Nine clamped coarse taps around (m, n). `m` and `n` are unsigned, so
      // `m - 1` underflows at the left edge; adding one first and fetching
      // `mm - 1` keeps every index in range while still clamping.
      const mm = m + 1;
      const nn = n + 1;
      const cL = std.max(m, 1) - 1;
      const cT = std.max(n, 1) - 1;

      const p00 = fetchCoarse(cL, cT, c4);
      const p10 = fetchCoarse(m, cT, c4);
      const p20 = fetchCoarse(mm, cT, c4);
      const p01 = fetchCoarse(cL, n, c4);
      const p11 = fetchCoarse(m, n, c4);
      const p21 = fetchCoarse(mm, n, c4);
      const p02 = fetchCoarse(cL, nn, c4);
      const p12 = fetchCoarse(m, nn, c4);
      const p22 = fetchCoarse(mm, nn, c4);

      // Rows blended along x for the two output rows, then along y.
      // Output (2m, ·) takes ¼ of the left neighbour and ¾ of the centre;
      // output (2m+1, ·) takes ¾ of the centre and ¼ of the right neighbour.
      const topL = std.add(std.mul(0.25, p00), std.mul(0.75, p10));
      const topR = std.add(std.mul(0.75, p10), std.mul(0.25, p20));
      const midL = std.add(std.mul(0.25, p01), std.mul(0.75, p11));
      const midR = std.add(std.mul(0.75, p11), std.mul(0.25, p21));
      const botL = std.add(std.mul(0.25, p02), std.mul(0.75, p12));
      const botR = std.add(std.mul(0.75, p12), std.mul(0.25, p22));

      acc[o * 4 + 0] = std.add(b, std.add(std.mul(0.25, topL), std.mul(0.75, midL)));
      acc[o * 4 + 1] = std.add(b, std.add(std.mul(0.25, topR), std.mul(0.75, midR)));
      acc[o * 4 + 2] = std.add(b, std.add(std.mul(0.75, midL), std.mul(0.25, botL)));
      acc[o * 4 + 3] = std.add(b, std.add(std.mul(0.75, midR), std.mul(0.25, botR)));
    }

    // Project the skip tensor at full resolution and add it in.
    const x0 = m * 2;
    const y0 = n * 2;
    for (const q of tgpu.unroll(std.range(4))) {
      const px = x0 + (q & 1);
      const py = y0 + (q >> 1);
      if (px < outW && py < outH) {
        const skipBase = (py * outW + px) * skipC4;
        for (let ic4 = d.u32(0); ic4 < skipC4; ic4++) {
          const v = d.vec4f(lateralLayout.$.skip[skipBase + ic4]);
          for (const o of tgpu.unroll(std.range(oc4PerThread))) {
            const wo = ((oc4Base + o) * skipC4 + ic4) * 4;
            acc[o * 4 + q] = std.add(
              acc[o * 4 + q],
              block4(
                v,
                d.vec4f(lateralLayout.$.wgt[wo]),
                d.vec4f(lateralLayout.$.wgt[wo + 1]),
                d.vec4f(lateralLayout.$.wgt[wo + 2]),
                d.vec4f(lateralLayout.$.wgt[wo + 3]),
              ),
            );
          }
        }
      }
    }

    for (const o of tgpu.unroll(std.range(oc4PerThread))) {
      const oc4 = oc4Base + o;
      for (const q of tgpu.unroll(std.range(4))) {
        const px = x0 + (q & 1);
        const py = y0 + (q >> 1);
        if (oc4 < outC4 && px < outW && py < outH) {
          lateralLayout.$.dst[(py * outW + px) * outC4 + oc4] = act(acc[o * 4 + q]);
        }
      }
    }
  });

  return {
    fn,
    dispatch: [
      Math.ceil(outC4 / (ocLanes * oc4PerThread)),
      Math.ceil(coarseW / tx),
      Math.ceil(coarseH / ty),
    ],
    tiling: `${ocLanes}×${tx}×${ty} threads, 2×2 px × ${oc4PerThread}c4 per thread`,
  };
}
