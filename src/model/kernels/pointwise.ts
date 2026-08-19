/**
 * Pointwise (1×1) convolution — the workhorse of the network.
 *
 * A 1×1 convolution is a `C_out × C_in` matrix applied independently at every
 * pixel, so the kernel is a register-blocked GEMM: each thread keeps
 * `ppt × oc4PerThread` accumulators in registers and streams the input channel
 * axis, reusing every loaded input `vec4` across all of its output channels and
 * every loaded weight block across all of its pixels.
 *
 * Bias, activation and the residual addition are all folded in, so a whole
 * bottleneck projection is one dispatch and the result is written exactly once.
 */
import tgpu from 'typegpu';
import type { TgpuComputeFn } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import type { Activation } from '../arch.ts';
import { chooseTiling } from '../layout.ts';
import { activationFn } from './activation.ts';

export const pointwiseLayout = tgpu
  .bindGroupLayout({
    src: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    wgt: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    bias: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    res: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    dst: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'mutable' },
  })
  .$name('pw');

/**
 * `w0·v.x + w1·v.y + w2·v.z + w3·v.w` — one 4×4 weight block applied to one
 * packed input `vec4`. Lowers to four `fma`s.
 */
export const block4 = tgpu
  .fn([d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f], d.vec4f)((v, w0, w1, w2, w3) => {
    'use gpu';
    return std.add(
      std.add(std.mul(v.x, w0), std.mul(v.y, w1)),
      std.add(std.mul(v.z, w2), std.mul(v.w, w3)),
    );
  })
  .$name('block4');

export interface PointwiseConfig {
  /** Input channel groups (`C_in / 4`). */
  readonly inC4: number;
  /** Output channel groups (`C_out / 4`). */
  readonly outC4: number;
  /** Total pixels in the (identically shaped) input and output tensors. */
  readonly pixels: number;
  readonly act: Activation;
  readonly residual: boolean;
}

export interface BuiltKernel {
  readonly fn: TgpuComputeFn<any>;
  /** Workgroup counts for `dispatchWorkgroups`. */
  readonly dispatch: readonly [number, number, number];
  /** Human-readable description of the chosen tiling, for the report. */
  readonly tiling: string;
}

export function makePointwise(cfg: PointwiseConfig): BuiltKernel {
  const { inC4, outC4, pixels, residual } = cfg;
  const t = chooseTiling(outC4, pixels);
  const { ocLanes, pixLanes, ppt, oc4PerThread } = t;

  const act = activationFn(cfg.act);
  const Acc = d.arrayOf(d.vec4f, ppt * oc4PerThread);
  const Vals = d.arrayOf(d.vec4f, ppt);

  const fn = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [ocLanes, pixLanes],
  })((input) => {
    'use gpu';
    const oc4Base = input.gid.x * oc4PerThread;
    const pixBase = input.gid.y * ppt;

    const acc = Acc();
    for (const o of tgpu.unroll(std.range(oc4PerThread))) {
      const b = d.vec4f(pointwiseLayout.$.bias[oc4Base + o]);
      for (const p of tgpu.unroll(std.range(ppt))) {
        acc[o * ppt + p] = d.vec4f(b);
      }
    }

    for (let ic4 = d.u32(0); ic4 < inC4; ic4++) {
      // Hoist the input values: every one is reused by all `oc4PerThread`
      // output groups this thread owns.
      const v = Vals();
      for (const p of tgpu.unroll(std.range(ppt))) {
        v[p] = d.vec4f(pointwiseLayout.$.src[(pixBase + p) * inC4 + ic4]);
      }
      for (const o of tgpu.unroll(std.range(oc4PerThread))) {
        const wo = ((oc4Base + o) * inC4 + ic4) * 4;
        const w0 = d.vec4f(pointwiseLayout.$.wgt[wo]);
        const w1 = d.vec4f(pointwiseLayout.$.wgt[wo + 1]);
        const w2 = d.vec4f(pointwiseLayout.$.wgt[wo + 2]);
        const w3 = d.vec4f(pointwiseLayout.$.wgt[wo + 3]);
        for (const p of tgpu.unroll(std.range(ppt))) {
          acc[o * ppt + p] = std.add(acc[o * ppt + p], block4(v[p], w0, w1, w2, w3));
        }
      }
    }

    for (const o of tgpu.unroll(std.range(oc4PerThread))) {
      const oc4 = oc4Base + o;
      for (const p of tgpu.unroll(std.range(ppt))) {
        const pix = pixBase + p;
        if (oc4 < outC4 && pix < pixels) {
          const at = pix * outC4 + oc4;
          let value = d.vec4f(acc[o * ppt + p]);
          if (residual) {
            value = std.add(value, d.vec4f(pointwiseLayout.$.res[at]));
          }
          pointwiseLayout.$.dst[at] = act(value);
        }
      }
    }
  });

  return {
    fn,
    dispatch: [
      Math.ceil(outC4 / (ocLanes * oc4PerThread)),
      Math.ceil(pixels / (pixLanes * ppt)),
      1,
    ],
    tiling: `${ocLanes}×${pixLanes} threads, ${ppt}px × ${oc4PerThread}c4 per thread`,
  };
}
