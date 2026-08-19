/**
 * Fused depthwise → pointwise convolution: an entire separable block in one
 * dispatch, with the depthwise result never touching device memory.
 *
 * ## Why this is not a naive fusion
 *
 * The pointwise half sums over *all* input channels, so every thread needs the
 * depthwise output for every input channel at its pixels. If each thread simply
 * recomputed the depthwise itself, the work would be duplicated once per output
 * channel tile — `ceil(C_out4 / oc4PerThread)` times over. For the deep blocks
 * (704 → 176 channels) that is an 11× duplication of the depthwise, which costs
 * more than the pointwise it was meant to save. Measured across the whole
 * network it inflates the FLOP count by roughly 37 %.
 *
 * Instead the workgroup computes each depthwise value **once**, cooperatively,
 * into workgroup memory, and every thread then reads it back. Duplication drops
 * to `ceil(C_out4 / (ocLanes · oc4PerThread))` — about +1 % over the whole
 * network — while still costing only one dispatch and one round trip to memory.
 *
 * ## Thread mapping
 *
 * The dispatch is three-dimensional so no integer division is ever needed
 * (TGSL follows JavaScript, where `/` is floating-point division):
 *
 *     gid.x → output channel tile      gid.y → output x tile      gid.z → output y
 *
 * Within a workgroup, thread `(lx, ly, lz)` owns the `pptX` pixels starting at
 * `ly · pptX` on row `lz`, and the `oc4PerThread` channel groups starting at
 * `lx · oc4PerThread`. During the cooperative phase the same thread fills the
 * depthwise values for *its* pixels and for every input channel group congruent
 * to `lx` modulo `ocLanes`, so each (pixel, channel) pair is computed exactly
 * once with no divisions and no atomics.
 */
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import type { Activation } from '../arch.ts';
import { activationFn } from './activation.ts';
import { block4, type BuiltKernel } from './pointwise.ts';

export const dwpwLayout = tgpu
  .bindGroupLayout({
    src: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    /** Depthwise weights, `[c4][ky][kx]` packed as `vec4` over channels. */
    dwWgt: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    dwBias: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    /** Pointwise weights, `[oc4][ic4][j]` packed as `vec4` over output channels. */
    pwWgt: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    pwBias: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    res: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    dst: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'mutable' },
  })
  .$name('dwpw');

export interface DwPwConfig {
  readonly inC4: number;
  readonly outC4: number;
  readonly inH: number;
  readonly inW: number;
  readonly outH: number;
  readonly outW: number;
  readonly k: number;
  readonly stride: number;
  readonly midAct: Activation;
  readonly act: Activation;
  readonly residual: boolean;
}

export interface DwPwTiling {
  readonly ocLanes: number;
  readonly tx: number;
  readonly ty: number;
  readonly pptX: number;
  readonly oc4PerThread: number;
  readonly ic4Chunk: number;
  readonly threads: number;
  readonly ldsBytes: number;
  /** How many times the depthwise is recomputed across channel tiles. */
  readonly recompute: number;
}

/** Workgroup memory budget. The WebGPU guaranteed minimum is 16 KiB. */
const MAX_LDS_BYTES = 16384;

/** See the note on `LOAD_COST` in `../layout.ts`. */
const LOAD_COST = 4;

/**
 * Picks a workgroup shape by explicitly costing the two things that matter:
 * duplicated depthwise work, and idle lanes.
 */
export function chooseDwPwTiling(cfg: DwPwConfig): DwPwTiling {
  const { inC4, outC4, k } = cfg;
  let best: DwPwTiling | undefined;
  let bestCost = Infinity;

  for (const threads of [64, 128, 256]) {
    for (const ocLanes of [1, 2, 4, 8, 16]) {
      if (ocLanes > threads) continue;
      const spatial = threads / ocLanes;
      for (const tx of [1, 2, 4, 8, 16, 32]) {
        if (tx > spatial) continue;
        const ty = spatial / tx;
        if (!Number.isInteger(ty)) continue;
        for (const pptX of [1, 2, 4]) {
          for (const oc4PerThread of [1, 2, 4, 8]) {
            if (pptX * oc4PerThread > 8) continue;

            const chanTile = ocLanes * oc4PerThread;
            const tileX = tx * pptX;
            const pixTile = tileX * ty;

            // Input channel groups staged per cooperative round. A multiple of
            // `ocLanes` keeps the strided fill loop balanced across lanes.
            const target = Math.min(inC4, 16);
            let ic4Chunk = Math.ceil(target / ocLanes) * ocLanes;
            while (ic4Chunk > ocLanes && pixTile * ic4Chunk * 16 > MAX_LDS_BYTES) {
              ic4Chunk -= ocLanes;
            }
            const ldsBytes = pixTile * ic4Chunk * 16;
            if (ldsBytes > MAX_LDS_BYTES || ic4Chunk < 1) continue;

            const recompute = Math.ceil(outC4 / chanTile);

            // Whole-op cost: workgroups × (MACs + LOAD_COST × loads). Using
            // `ceil` for the workgroup count prices idle lanes automatically.
            const workgroups =
              recompute * Math.ceil(cfg.outW / tileX) * Math.ceil(cfg.outH / ty);

            // Per workgroup. The depthwise runs once for the whole tile thanks to
            // the workgroup-memory staging, so it is *not* multiplied by chanTile.
            const dwMacs = pixTile * inC4 * 4 * k * k;
            const pwMacs = pixTile * chanTile * 4 * inC4 * 4;

            // Input reads carry a halo of k-1 in each direction; a tall, narrow
            // tile pays far more for that halo than a square one of equal area,
            // which is what makes this term choose squarish tiles.
            const inputLoads = (tileX + k - 1) * (ty + k - 1) * inC4;
            const weightLoads = chanTile * inC4 * 4 + inC4 * k * k;

            const cost = workgroups * (dwMacs + pwMacs + LOAD_COST * (inputLoads + weightLoads));
            if (cost < bestCost) {
              bestCost = cost;
              best = {
                ocLanes,
                tx,
                ty,
                pptX,
                oc4PerThread,
                ic4Chunk,
                threads,
                ldsBytes,
                recompute,
              };
            }
          }
        }
      }
    }
  }
  if (!best) throw new Error(`no dwpw tiling for ${JSON.stringify(cfg)}`);
  return best;
}

export function makeDepthwisePointwise(cfg: DwPwConfig): BuiltKernel & { tilingInfo: DwPwTiling } {
  const t = chooseDwPwTiling(cfg);
  const { ocLanes, tx, ty, pptX, oc4PerThread, ic4Chunk } = t;

  const { inC4, outC4, inH, inW, outH, outW, k, stride, residual } = cfg;
  const pad = (k - 1) >> 1;
  const tileX = tx * pptX;

  const midAct = activationFn(cfg.midAct);
  const act = activationFn(cfg.act);

  const Acc = d.arrayOf(d.vec4f, pptX * oc4PerThread);
  const share = tgpu
    .workgroupVar(d.arrayOf(d.vec4f, tileX * ty * ic4Chunk))
    .$name('dwShare');

  const fn = tgpu.computeFn({
    in: {
      gid: d.builtin.globalInvocationId,
      lid: d.builtin.localInvocationId,
      wid: d.builtin.workgroupId,
    },
    workgroupSize: [ocLanes, tx, ty],
  })((input) => {
    'use gpu';
    const lx = input.lid.x;
    const localY = input.lid.z;
    const localX0 = input.lid.y * pptX;

    const tileX0 = input.wid.y * tileX;
    const oy = input.wid.z * ty + localY;
    const oc4Base = input.gid.x * oc4PerThread;

    const acc = Acc();
    for (const o of tgpu.unroll(std.range(oc4PerThread))) {
      const b = d.vec4f(dwpwLayout.$.pwBias[oc4Base + o]);
      for (const i of tgpu.unroll(std.range(pptX))) {
        acc[o * pptX + i] = d.vec4f(b);
      }
    }

    // Row of the LDS tile this thread's pixels live on.
    const rowBase = (localY * tileX + localX0) * ic4Chunk;

    for (let chunk = d.u32(0); chunk < inC4; chunk += ic4Chunk) {
      // ---- cooperative depthwise ----------------------------------------
      // Wait for every thread to finish reading the previous chunk before
      // overwriting it.
      std.workgroupBarrier();

      for (let j = lx; j < ic4Chunk; j += ocLanes) {
        const ic4 = chunk + j;
        for (const i of tgpu.unroll(std.range(pptX))) {
          const ox = tileX0 + localX0 + i;
          let sum = d.vec4f();
          if (ic4 < inC4 && ox < outW && oy < outH) {
            sum = d.vec4f(dwpwLayout.$.dwBias[ic4]);
            const wBase = ic4 * k * k;
            // Unsigned coordinates biased by `pad`, so the zero-padding test is
            // a single unsigned comparison with no signed arithmetic.
            const byBase = oy * stride;
            const bxBase = ox * stride;
            for (const ky of tgpu.unroll(std.range(k))) {
              const by = byBase + ky;
              if (by >= pad && by < inH + pad) {
                const sy = by - pad;
                for (const kx of tgpu.unroll(std.range(k))) {
                  const bx = bxBase + kx;
                  if (bx >= pad && bx < inW + pad) {
                    const sx = bx - pad;
                    const v = d.vec4f(dwpwLayout.$.src[(sy * inW + sx) * inC4 + ic4]);
                    const wt = d.vec4f(dwpwLayout.$.dwWgt[wBase + ky * k + kx]);
                    sum = std.add(sum, std.mul(v, wt));
                  }
                }
              }
            }
            sum = midAct(sum);
          }
          share.$[(localY * tileX + localX0 + i) * ic4Chunk + j] = d.vec4f(sum);
        }
      }

      std.workgroupBarrier();

      // ---- pointwise accumulation ---------------------------------------
      for (let c = d.u32(0); c < ic4Chunk; c++) {
        const ic4 = chunk + c;
        for (const o of tgpu.unroll(std.range(oc4PerThread))) {
          const wo = ((oc4Base + o) * inC4 + ic4) * 4;
          const w0 = d.vec4f(dwpwLayout.$.pwWgt[wo]);
          const w1 = d.vec4f(dwpwLayout.$.pwWgt[wo + 1]);
          const w2 = d.vec4f(dwpwLayout.$.pwWgt[wo + 2]);
          const w3 = d.vec4f(dwpwLayout.$.pwWgt[wo + 3]);
          for (const i of tgpu.unroll(std.range(pptX))) {
            const v = d.vec4f(share.$[rowBase + i * ic4Chunk + c]);
            acc[o * pptX + i] = std.add(acc[o * pptX + i], block4(v, w0, w1, w2, w3));
          }
        }
      }
    }

    // ---- store -----------------------------------------------------------
    for (const o of tgpu.unroll(std.range(oc4PerThread))) {
      const oc4 = oc4Base + o;
      for (const i of tgpu.unroll(std.range(pptX))) {
        const ox = tileX0 + localX0 + i;
        if (oc4 < outC4 && ox < outW && oy < outH) {
          const at = (oy * outW + ox) * outC4 + oc4;
          let value = d.vec4f(acc[o * pptX + i]);
          if (residual) {
            value = std.add(value, d.vec4f(dwpwLayout.$.res[at]));
          }
          dwpwLayout.$.dst[at] = act(value);
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
    tiling:
      `${ocLanes}×${tx}×${ty} threads, ${pptX}px × ${oc4PerThread}c4 per thread, ` +
      `ic4Chunk=${ic4Chunk}, lds=${t.ldsBytes}B, dw recompute ×${t.recompute}`,
    tilingInfo: t,
  };
}
