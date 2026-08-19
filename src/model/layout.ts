/**
 * Memory layout conventions shared by the GPU kernels, the CPU reference
 * implementation and the weight exporter.
 *
 * Everything is channels-last with channels packed into groups of four, so that
 * a tensor is an `array<vec4f>` and every load is one 16-byte transaction:
 *
 *     tensor[(y * W + x) * C4 + c4]   →   channels [4·c4 … 4·c4+3] at (x, y)
 *
 * Weights are pre-swizzled at export time into the exact order the kernels walk
 * them, so no kernel ever performs a strided or gathered weight read.
 */

import type { Activation, TensorShape } from './arch.ts';

/** Number of `vec4` groups needed for `c` channels. */
export function groups(c: number): number {
  if (c % 4 !== 0) throw new Error(`channel count ${c} is not a multiple of 4`);
  return c / 4;
}

/** Number of `vec4` elements a tensor occupies. */
export function tensorElements(s: TensorShape): number {
  return s.h * s.w * groups(s.c);
}

/** Flat `vec4` index of channel group `c4` at pixel `(x, y)`. */
export function tensorIndex(s: TensorShape, x: number, y: number, c4: number): number {
  return (y * s.w + x) * groups(s.c) + c4;
}

// ---------------------------------------------------------------------------
// Weight packing
// ---------------------------------------------------------------------------

/**
 * Pointwise (1×1) convolution weights.
 *
 * Logical weight `W[oc][ic]` is stored at
 *
 *     packed[((oc4 · IN_C4) + ic4) · 4 + j][oc % 4]
 *     where oc4 = oc >> 2, ic4 = ic >> 2, j = ic % 4
 *
 * i.e. each `vec4` holds four *output* channels for one input channel. The
 * kernel therefore evaluates a 4×4 block as four `fma`s against the components
 * of one input `vec4`, with fully contiguous weight reads.
 */
export function packPointwise(w: Float32Array, outC: number, inC: number): Float32Array {
  const outC4 = groups(outC);
  const inC4 = groups(inC);
  const packed = new Float32Array(outC4 * inC4 * 4 * 4);
  for (let oc = 0; oc < outC; oc++) {
    for (let ic = 0; ic < inC; ic++) {
      const oc4 = oc >> 2;
      const ic4 = ic >> 2;
      const j = ic & 3;
      const dst = (((oc4 * inC4 + ic4) * 4 + j) << 2) + (oc & 3);
      packed[dst] = w[oc * inC + ic];
    }
  }
  return packed;
}

/**
 * Depthwise k×k weights.
 *
 * Logical weight `W[c][ky][kx]` is stored at
 *
 *     packed[(c4 · k · k + ky · k + kx) · 4 + (c % 4)]
 *
 * All k² taps for one channel group are contiguous, which is the order a thread
 * reads them in.
 */
export function packDepthwise(w: Float32Array, c: number, k: number): Float32Array {
  const c4 = groups(c);
  const packed = new Float32Array(c4 * k * k * 4);
  for (let ch = 0; ch < c; ch++) {
    for (let ky = 0; ky < k; ky++) {
      for (let kx = 0; kx < k; kx++) {
        const dst = (((ch >> 2) * k * k + ky * k + kx) << 2) + (ch & 3);
        packed[dst] = w[(ch * k + ky) * k + kx];
      }
    }
  }
  return packed;
}

/**
 * Dense k×k convolution weights (used by the stem only).
 *
 * Logical weight `W[oc][ic][ky][kx]` is stored at
 *
 *     packed[((oc4 · IN_C4 + ic4) · k · k + ky · k + kx) · 4 + j][oc % 4]
 */
export function packConv(
  w: Float32Array,
  outC: number,
  inC: number,
  k: number,
): Float32Array {
  const outC4 = groups(outC);
  const inC4 = groups(inC);
  const packed = new Float32Array(outC4 * inC4 * k * k * 4 * 4);
  for (let oc = 0; oc < outC; oc++) {
    for (let ic = 0; ic < inC; ic++) {
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const oc4 = oc >> 2;
          const ic4 = ic >> 2;
          const j = ic & 3;
          const dst = ((((oc4 * inC4 + ic4) * k * k + ky * k + kx) * 4 + j) << 2) + (oc & 3);
          packed[dst] = w[((oc * inC + ic) * k + ky) * k + kx];
        }
      }
    }
  }
  return packed;
}

/** Bias vector, padded to a whole number of `vec4`s. */
export function packBias(b: Float32Array, c: number): Float32Array {
  const packed = new Float32Array(groups(c) * 4);
  packed.set(b.subarray(0, c));
  return packed;
}

// ---------------------------------------------------------------------------
// Activations (CPU reference — must match `kernels/activation.ts` exactly)
// ---------------------------------------------------------------------------

export function applyActivation(act: Activation, v: number): number {
  switch (act) {
    case 'linear':
      return v;
    case 'relu':
      return Math.max(v, 0);
    case 'relu6':
      return Math.min(Math.max(v, 0), 6);
    case 'hardswish':
      return (v * Math.min(Math.max(v + 3, 0), 6)) / 6;
  }
}

// ---------------------------------------------------------------------------
// Dispatch tiling
// ---------------------------------------------------------------------------

export interface Tiling {
  /** Threads along the output-channel axis within a workgroup. */
  readonly ocLanes: number;
  /** Threads along the pixel axis within a workgroup. */
  readonly pixLanes: number;
  /** Output pixels each thread computes. */
  readonly ppt: number;
  /** Output channel groups each thread computes. */
  readonly oc4PerThread: number;
  /** True when `outC4` is an exact multiple of `ocLanes · oc4PerThread`. */
  readonly exactChannels: boolean;
  /** True when the pixel count is an exact multiple of `pixLanes · ppt`. */
  readonly exactPixels: boolean;
}

const WORKGROUP_THREADS = 64;

/**
 * Chooses a workgroup shape for a pointwise-style kernel.
 *
 * Priorities, in order:
 *  1. `ocLanes · oc4PerThread` must divide the output channel groups where
 *     possible, so no lane is idle.
 *  2. Prefer more pixel lanes than channel lanes: pixels are plentiful and give
 *     coalesced reads, whereas channel lanes re-read the same input.
 *  3. Keep the accumulator count (`ppt · oc4PerThread`) at 8 `vec4`s or fewer so
 *     occupancy stays high.
 */
export function chooseTiling(outC4: number, pixels: number): Tiling {
  let best: Tiling | undefined;
  let bestScore = -Infinity;

  for (const ocLanes of [1, 2, 4, 8, 16]) {
    const pixLanes = WORKGROUP_THREADS / ocLanes;
    for (const oc4PerThread of [1, 2, 4]) {
      for (const ppt of [1, 2, 4, 8]) {
        // Accumulator budget, in `vec4` registers.
        if (ppt * oc4PerThread > 8) continue;

        const chanTile = ocLanes * oc4PerThread;
        const pixTile = pixLanes * ppt;

        // Fraction of launched threads that do useful work.
        const chanWaste = (Math.ceil(outC4 / chanTile) * chanTile) / outC4;
        const pixWaste = (Math.ceil(pixels / pixTile) * pixTile) / pixels;

        const score =
          // Idle lanes are by far the most expensive mistake.
          -Math.log(chanWaste * pixWaste) * 6 +
          // Reward arithmetic intensity: more work per thread amortises loads.
          Math.log(ppt * oc4PerThread) * 0.6 +
          // Mild preference for pixel-major layouts (coalesced loads).
          Math.log(pixLanes) * 0.25;

        if (score > bestScore) {
          bestScore = score;
          best = {
            ocLanes,
            pixLanes,
            ppt,
            oc4PerThread,
            exactChannels: outC4 % chanTile === 0,
            exactPixels: pixels % pixTile === 0,
          };
        }
      }
    }
  }
  if (!best) throw new Error(`no tiling found for outC4=${outC4} pixels=${pixels}`);
  return best;
}
