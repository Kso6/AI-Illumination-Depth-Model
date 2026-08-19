/**
 * CPU reference implementation of every operator in the network.
 *
 * This is the *specification*: the WGSL kernels in `kernels/`, the PyTorch
 * module in `tools/train/illumina/model.py` and this file must agree bit-for-bit
 * modulo floating-point associativity. `test/kernels.test.ts` checks each GPU
 * kernel against the functions here, and `test/network.test.ts` checks the whole
 * graph end to end.
 *
 * Tensors are plain `Float32Array`s in channels-last order:
 *
 *     data[(y · W + x) · C + ch]
 *
 * Because every channel count is a multiple of four, this is byte-identical to
 * the packed `array<vec4f>` layout the GPU uses — only the *weights* are
 * swizzled (see `layout.ts`), so these buffers can be uploaded and compared
 * without conversion.
 */

import type { Activation, Architecture } from './arch.ts';
import { applyActivation } from './layout.ts';

export interface RefTensor {
  readonly h: number;
  readonly w: number;
  readonly c: number;
  readonly data: Float32Array;
}

export function refTensor(h: number, w: number, c: number, fill?: Float32Array): RefTensor {
  return { h, w, c, data: fill ?? new Float32Array(h * w * c) };
}

/** Pointwise (1×1) convolution + bias + optional residual + activation. */
export function refPointwise(
  src: RefTensor,
  /** `[outC][inC]`, row-major. */
  weights: Float32Array,
  bias: Float32Array,
  outC: number,
  act: Activation,
  residual?: RefTensor,
): RefTensor {
  const out = refTensor(src.h, src.w, outC);
  const inC = src.c;
  for (let p = 0; p < src.h * src.w; p++) {
    for (let oc = 0; oc < outC; oc++) {
      let sum = bias[oc];
      for (let ic = 0; ic < inC; ic++) {
        sum += weights[oc * inC + ic] * src.data[p * inC + ic];
      }
      if (residual) sum += residual.data[p * outC + oc];
      out.data[p * outC + oc] = applyActivation(act, sum);
    }
  }
  return out;
}

/** Dense k×k convolution with `same` zero padding and the given stride. */
export function refConv(
  src: RefTensor,
  /** `[outC][inC][k][k]`, row-major. */
  weights: Float32Array,
  bias: Float32Array,
  outC: number,
  k: number,
  stride: number,
  act: Activation,
): RefTensor {
  const pad = (k - 1) >> 1;
  const oh = Math.floor((src.h + 2 * pad - k) / stride) + 1;
  const ow = Math.floor((src.w + 2 * pad - k) / stride) + 1;
  const out = refTensor(oh, ow, outC);
  const inC = src.c;
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      for (let oc = 0; oc < outC; oc++) {
        let sum = bias[oc];
        for (let ky = 0; ky < k; ky++) {
          const sy = y * stride + ky - pad;
          if (sy < 0 || sy >= src.h) continue;
          for (let kx = 0; kx < k; kx++) {
            const sx = x * stride + kx - pad;
            if (sx < 0 || sx >= src.w) continue;
            const base = (sy * src.w + sx) * inC;
            for (let ic = 0; ic < inC; ic++) {
              sum += weights[((oc * inC + ic) * k + ky) * k + kx] * src.data[base + ic];
            }
          }
        }
        out.data[(y * ow + x) * outC + oc] = applyActivation(act, sum);
      }
    }
  }
  return out;
}

/** Depthwise k×k convolution + bias + activation, `same` zero padding. */
export function refDepthwise(
  src: RefTensor,
  /** `[C][k][k]`, row-major. */
  weights: Float32Array,
  bias: Float32Array,
  k: number,
  stride: number,
  act: Activation,
): RefTensor {
  const pad = (k - 1) >> 1;
  const oh = Math.floor((src.h + 2 * pad - k) / stride) + 1;
  const ow = Math.floor((src.w + 2 * pad - k) / stride) + 1;
  const out = refTensor(oh, ow, src.c);
  const C = src.c;
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      for (let c = 0; c < C; c++) {
        let sum = bias[c];
        for (let ky = 0; ky < k; ky++) {
          const sy = y * stride + ky - pad;
          if (sy < 0 || sy >= src.h) continue;
          for (let kx = 0; kx < k; kx++) {
            const sx = x * stride + kx - pad;
            if (sx < 0 || sx >= src.w) continue;
            sum += weights[(c * k + ky) * k + kx] * src.data[(sy * src.w + sx) * C + c];
          }
        }
        out.data[(y * ow + x) * C + c] = applyActivation(act, sum);
      }
    }
  }
  return out;
}

/**
 * Fused depthwise → pointwise. Equivalent to `refDepthwise` followed by
 * `refPointwise`, but expressed as one function because that is what the GPU
 * does in one dispatch.
 */
export function refDepthwisePointwise(
  src: RefTensor,
  dwWeights: Float32Array,
  dwBias: Float32Array,
  midAct: Activation,
  pwWeights: Float32Array,
  pwBias: Float32Array,
  outC: number,
  k: number,
  stride: number,
  act: Activation,
  residual?: RefTensor,
): RefTensor {
  const mid = refDepthwise(src, dwWeights, dwBias, k, stride, midAct);
  return refPointwise(mid, pwWeights, pwBias, outC, act, residual);
}

/**
 * Bilinear 2× upsample matching PyTorch's
 * `F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)`:
 * the source coordinate for destination `i` is `(i + 0.5) / 2 - 0.5`, clamped
 * to the source extent.
 */
export function refUpsample2(src: RefTensor): RefTensor {
  const oh = src.h * 2;
  const ow = src.w * 2;
  const out = refTensor(oh, ow, src.c);
  // Both taps must be derived from the *unclamped* floor and clamped
  // independently. Clamping the first tap and then adding one would silently
  // shift the second tap inward at the borders.
  const clampY = (v: number) => Math.max(0, Math.min(src.h - 1, v));
  const clampX = (v: number) => Math.max(0, Math.min(src.w - 1, v));
  for (let y = 0; y < oh; y++) {
    const sy = (y + 0.5) / 2 - 0.5;
    const fy0 = Math.floor(sy);
    const y0 = clampY(fy0);
    const y1 = clampY(fy0 + 1);
    const fy = sy - fy0;
    for (let x = 0; x < ow; x++) {
      const sx = (x + 0.5) / 2 - 0.5;
      const fx0 = Math.floor(sx);
      const x0 = clampX(fx0);
      const x1 = clampX(fx0 + 1);
      const fx = sx - fx0;
      for (let c = 0; c < src.c; c++) {
        const a = src.data[(y0 * src.w + x0) * src.c + c];
        const b = src.data[(y0 * src.w + x1) * src.c + c];
        const cc = src.data[(y1 * src.w + x0) * src.c + c];
        const dd = src.data[(y1 * src.w + x1) * src.c + c];
        out.data[(y * ow + x) * src.c + c] =
          a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + cc * (1 - fx) * fy + dd * fx * fy;
      }
    }
  }
  return out;
}

/**
 * FPN lateral fusion: `act( up2(coarse) + W · skip + b )`.
 *
 * The output resolution is the skip's resolution, i.e. twice the coarse one, and
 * the coarse tensor already carries the output's channel width (see the note on
 * `LateralOp` in `arch.ts`), so it needs no projection.
 */
export function refLateral(
  coarse: RefTensor,
  skip: RefTensor,
  /** `[outC][skipC]`, row-major. */
  skipWeights: Float32Array,
  bias: Float32Array,
  outC: number,
  act: Activation,
): RefTensor {
  if (coarse.c !== outC) {
    throw new Error(`refLateral: coarse has ${coarse.c} channels, output has ${outC}`);
  }
  const up = refUpsample2(coarse);
  if (up.h !== skip.h || up.w !== skip.w) {
    throw new Error(
      `refLateral: upsampled coarse is ${up.h}×${up.w}, skip is ${skip.h}×${skip.w}`,
    );
  }
  const out = refTensor(skip.h, skip.w, outC);
  for (let p = 0; p < skip.h * skip.w; p++) {
    for (let oc = 0; oc < outC; oc++) {
      let sum = bias[oc] + up.data[p * outC + oc];
      for (let ic = 0; ic < skip.c; ic++) {
        sum += skipWeights[oc * skip.c + ic] * skip.data[p * skip.c + ic];
      }
      out.data[p * outC + oc] = applyActivation(act, sum);
    }
  }
  return out;
}

/**
 * Depth head: 1×1 convolution to a single channel, then a sigmoid, producing
 * normalised inverse depth (disparity) in `[0, 1]` where 1 is nearest.
 *
 * The result is broadcast across the four channels of the output tensor so it
 * can share the packed `vec4` layout with everything else; the lighting pass
 * reads channel 0.
 */
export function refHead(
  src: RefTensor,
  weights: Float32Array,
  bias: number,
): RefTensor {
  const out = refTensor(src.h, src.w, 4);
  for (let p = 0; p < src.h * src.w; p++) {
    let sum = bias;
    for (let ic = 0; ic < src.c; ic++) sum += weights[ic] * src.data[p * src.c + ic];
    const disparity = 1 / (1 + Math.exp(-sum));
    out.data[p * 4] = disparity;
    out.data[p * 4 + 1] = disparity;
    out.data[p * 4 + 2] = disparity;
    out.data[p * 4 + 3] = disparity;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Whole-network reference
// ---------------------------------------------------------------------------

/** sRGB → linear, matching `srgbToLinear` in `kernels/io.ts`. */
export function srgbToLinearScalar(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function luminanceOf(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export interface PreprocessResult {
  /** The network's input tensor. */
  readonly input: RefTensor;
  /** Linear-light colour at network resolution, the lighting pass's input. */
  readonly sceneColor: RefTensor;
}

/**
 * Reference for the preprocess kernel, given a source image already sampled at
 * the network's resolution with components in `[0, 1]`.
 */
export function refPreprocess(
  rgba: Float32Array,
  size: number,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
  decodeSrgb: boolean,
  exposure: number,
): PreprocessResult {
  const input = refTensor(size, size, 4);
  const sceneColor = refTensor(size, size, 4);
  for (let p = 0; p < size * size; p++) {
    const lin = [0, 1, 2].map((i) => {
      const v = rgba[p * 4 + i]!;
      return (decodeSrgb ? srgbToLinearScalar(v) : v) * exposure;
    }) as [number, number, number];
    sceneColor.data[p * 4] = lin[0];
    sceneColor.data[p * 4 + 1] = lin[1];
    sceneColor.data[p * 4 + 2] = lin[2];
    sceneColor.data[p * 4 + 3] = 1;
    for (let i = 0; i < 3; i++) {
      input.data[p * 4 + i] = (lin[i]! - mean[i]!) / std[i]!;
    }
    input.data[p * 4 + 3] = luminanceOf(lin[0], lin[1], lin[2]);
  }
  return { input, sceneColor };
}

/**
 * Reference for the joint-bilateral upsample, with no temporal history (the
 * first frame), matching `makeDepthUpsample` in `kernels/io.ts`.
 */
export function refBilateralUpsample(
  low: RefTensor,
  guide: RefTensor,
  rangeSigma: number,
): RefTensor {
  const full = low.h * 2;
  const out = refTensor(full, full, 1);
  const lumaAt = (x: number, y: number) => {
    const i = (y * guide.w + x) * guide.c;
    return luminanceOf(guide.data[i]!, guide.data[i + 1]!, guide.data[i + 2]!);
  };
  for (let y = 0; y < full; y++) {
    for (let x = 0; x < full; x++) {
      const lumaHere = lumaAt(x, y);
      const cx = (x + 0.5) * 0.5 - 0.5;
      const cy = (y + 0.5) * 0.5 - 0.5;
      const bx = Math.floor(cx + 0.5);
      const by = Math.floor(cy + 0.5);
      let weighted = 0;
      let total = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const qx = Math.max(0, Math.min(low.w - 1, bx + dx));
          const qy = Math.max(0, Math.min(low.h - 1, by + dy));
          const disparity = low.data[(qy * low.w + qx) * low.c]!;
          const lumaThere = lumaAt(qx * 2, qy * 2);
          const dxs = qx - cx;
          const dys = qy - cy;
          const spatial = Math.exp(-(dxs * dxs + dys * dys) * 0.5);
          const dl = (lumaHere - lumaThere) / Math.max(rangeSigma, 1e-4);
          const w = spatial * Math.exp(-dl * dl * 0.5);
          weighted += disparity * w;
          total += w;
        }
      }
      out.data[y * full + x] = weighted / Math.max(total, 1e-6);
    }
  }
  return out;
}

/**
 * Runs the whole graph on the CPU. Returns every intermediate tensor, keyed by
 * the architecture's tensor names, so a failing end-to-end test can be bisected
 * to the first op that diverges.
 */
export function forwardReference(
  arch: Architecture,
  weights: ReadonlyMap<string, Float32Array>,
  input: RefTensor,
): Map<string, RefTensor> {
  const values = new Map<string, RefTensor>();
  values.set('input', input);

  const get = (name: string): RefTensor => {
    const t = values.get(name);
    if (!t) throw new Error(`forwardReference: ${name} has not been produced yet`);
    return t;
  };
  const w = (key: string): Float32Array => {
    const v = weights.get(key);
    if (!v) throw new Error(`forwardReference: missing weight ${key}`);
    return v;
  };
  const channels = (name: string) => arch.tensors[name]!.c;

  for (const op of arch.ops) {
    switch (op.kind) {
      case 'preprocess':
        break;
      case 'conv':
        values.set(
          op.out,
          refConv(
            get(op.in),
            w(`${op.name}.weight`),
            w(`${op.name}.bias`),
            channels(op.out),
            op.k,
            op.stride,
            op.act,
          ),
        );
        break;
      case 'pw':
        values.set(
          op.out,
          refPointwise(
            get(op.in),
            w(`${op.name}.weight`),
            w(`${op.name}.bias`),
            channels(op.out),
            op.act,
            op.residual ? get(op.residual) : undefined,
          ),
        );
        break;
      case 'dwpw':
        values.set(
          op.out,
          refDepthwisePointwise(
            get(op.in),
            w(`${op.name}.dw_weight`),
            w(`${op.name}.dw_bias`),
            op.midAct,
            w(`${op.name}.pw_weight`),
            w(`${op.name}.pw_bias`),
            channels(op.out),
            op.k,
            op.stride,
            op.act,
            op.residual ? get(op.residual) : undefined,
          ),
        );
        break;
      case 'lateral':
        values.set(
          op.out,
          refLateral(
            get(op.coarse),
            get(op.skip),
            w(`${op.name}.weight`),
            w(`${op.name}.bias`),
            channels(op.out),
            op.act,
          ),
        );
        break;
      case 'head':
        values.set(op.out, refHead(get(op.in), w(`${op.name}.weight`), w(`${op.name}.bias`)[0]!));
        break;
      case 'bilateralUp':
        // Needs the guide image; run `refBilateralUpsample` separately.
        break;
    }
  }
  return values;
}
