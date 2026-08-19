/**
 * The kernels at the two ends of the network: turning a source texture into the
 * input tensor, and turning the decoder's output into a depth texture the
 * lighting pass can sample.
 */
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import type { BuiltKernel } from './pointwise.ts';

/** sRGB electro-optical transfer function, exact (not the 2.2 approximation). */
export const srgbToLinear = tgpu
  .fn([d.vec3f], d.vec3f)((c) => {
    'use gpu';
    const lo = std.div(c, 12.92);
    const hi = std.pow(std.div(std.add(c, d.vec3f(0.055)), 1.055), d.vec3f(2.4));
    return std.select(hi, lo, std.le(c, d.vec3f(0.04045)));
  })
  .$name('srgbToLinear');

/** Rec. 709 relative luminance. */
export const luminance = tgpu
  .fn([d.vec3f], d.f32)((c) => {
    'use gpu';
    return std.dot(c, d.vec3f(0.2126, 0.7152, 0.0722));
  })
  .$name('luminance');

// ---------------------------------------------------------------------------
// Preprocess
// ---------------------------------------------------------------------------

export const PreprocessParams = d.struct({
  /** Maps normalised network coordinates to source-texture coordinates. */
  uvScale: d.vec2f,
  uvOffset: d.vec2f,
  /** x: 1 to decode sRGB, y: exposure multiplier, z/w: reserved. */
  options: d.vec4f,
});

export const preprocessLayout = tgpu
  .bindGroupLayout({
    source: { texture: d.texture2d(d.f32) },
    samp: { sampler: 'filtering' },
    params: { uniform: PreprocessParams },
    /** Network input, NHWC4 with the normalised RGB in xyz and luminance in w. */
    dst: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'mutable' },
    /** Linear-light copy of the source at network resolution, for the shading pass. */
    sceneColor: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  })
  .$name('preprocess');

export interface PreprocessConfig {
  readonly size: number;
  readonly mean: readonly [number, number, number];
  readonly std: readonly [number, number, number];
}

/**
 * Samples the source image at the network's resolution, converts it to linear
 * light, writes both the normalised tensor the network consumes and the linear
 * colour buffer the lighting pass consumes. One dispatch, one read of the source.
 */
export function makePreprocess(cfg: PreprocessConfig): BuiltKernel {
  const { size } = cfg;
  const meanVec = d.vec3f(cfg.mean[0], cfg.mean[1], cfg.mean[2]);
  const invStd = d.vec3f(1 / cfg.std[0], 1 / cfg.std[1], 1 / cfg.std[2]);

  const fn = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [8, 8],
  })((input) => {
    'use gpu';
    const x = input.gid.x;
    const y = input.gid.y;
    if (x < size && y < size) {
      const p = preprocessLayout.$.params;
      const uv = std.add(
        std.mul(
          d.vec2f((d.f32(x) + 0.5) / size, (d.f32(y) + 0.5) / size),
          p.uvScale,
        ),
        p.uvOffset,
      );
      const raw = std.textureSampleLevel(
        preprocessLayout.$.source,
        preprocessLayout.$.samp,
        uv,
        0,
      );
      let rgb = d.vec3f(raw.x, raw.y, raw.z);
      if (p.options.x > 0.5) {
        rgb = srgbToLinear(rgb);
      }
      rgb = std.mul(p.options.y, rgb);

      std.textureStore(
        preprocessLayout.$.sceneColor,
        d.vec2u(x, y),
        d.vec4f(rgb.x, rgb.y, rgb.z, 1),
      );

      const norm = std.mul(std.sub(rgb, meanVec), invStd);
      preprocessLayout.$.dst[y * size + x] = d.vec4f(
        norm.x,
        norm.y,
        norm.z,
        luminance(rgb),
      );
    }
  });

  return {
    fn,
    dispatch: [Math.ceil(size / 8), Math.ceil(size / 8), 1],
    tiling: '8×8 threads, 1 px per thread',
  };
}

// ---------------------------------------------------------------------------
// Depth head
// ---------------------------------------------------------------------------

export const headLayout = tgpu
  .bindGroupLayout({
    src: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    /** One `vec4` per input channel group: the 1×N convolution's weights. */
    wgt: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    /** `bias.x` holds the single scalar bias. */
    bias: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    dst: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'mutable' },
  })
  .$name('head');

/**
 * Collapses the decoder output to a single channel and squashes it with a
 * sigmoid, giving normalised inverse depth (disparity) in `[0, 1]` where 1 is
 * nearest. Predicting disparity rather than metric depth is what lets the model
 * be trained scale- and shift-invariantly across mixed datasets.
 */
export function makeHead(inC4: number, pixels: number): BuiltKernel {
  const fn = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [64],
  })((input) => {
    'use gpu';
    const p = input.gid.x;
    if (p < pixels) {
      let sum = headLayout.$.bias[0].x;
      for (let ic4 = d.u32(0); ic4 < inC4; ic4++) {
        sum += std.dot(
          d.vec4f(headLayout.$.src[p * inC4 + ic4]),
          d.vec4f(headLayout.$.wgt[ic4]),
        );
      }
      const disparity = 1 / (1 + std.exp(-sum));
      headLayout.$.dst[p] = d.vec4f(disparity, disparity, disparity, disparity);
    }
  });

  return {
    fn,
    dispatch: [Math.ceil(pixels / 64), 1, 1],
    tiling: '64 threads, 1 px per thread',
  };
}

// ---------------------------------------------------------------------------
// Edge-aware upsample + temporal stabilisation
// ---------------------------------------------------------------------------

export const DepthPostParams = d.struct({
  /** x: range sigma, y: base temporal blend, z: motion sensitivity, w: reserved. */
  options: d.vec4f,
});

export const depthUpsampleLayout = tgpu
  .bindGroupLayout({
    /** Half-resolution disparity from the head. */
    low: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
    /** Full-resolution linear colour, used as the bilateral guide. */
    guide: { texture: d.texture2d(d.f32) },
    /** Previous frame's stabilised depth. */
    history: { texture: d.texture2d(d.f32) },
    params: { uniform: DepthPostParams },
    dst: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  })
  .$name('depthUpsample');

/**
 * Joint-bilateral upsample (Kopf et al., 2007) from the half-resolution
 * disparity map to full resolution, using the full-resolution luminance as the
 * guide, followed by a temporal exponential blend.
 *
 * A plain bilinear upsample would smear depth across object silhouettes, which
 * shows up in the lighting as haloes around every foreground object. Weighting
 * each low-resolution tap by how similar its guide luminance is to the target
 * pixel's keeps the depth edge exactly on the colour edge.
 *
 * The temporal blend rate rises with the magnitude of the change, so slow
 * flicker is smoothed away while genuine motion is not smeared.
 */
export function makeDepthUpsample(lowW: number, lowH: number, fullSize: number): BuiltKernel {
  const RADIUS = 1;

  const fn = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [8, 8],
  })((input) => {
    'use gpu';
    const x = input.gid.x;
    const y = input.gid.y;
    if (x < fullSize && y < fullSize) {
      const opts = depthUpsampleLayout.$.params.options;

      const guideHere = std.textureLoad(
        depthUpsampleLayout.$.guide,
        d.vec2u(x, y),
        0,
      );
      const lumaHere = luminance(d.vec3f(guideHere.x, guideHere.y, guideHere.z));

      // Continuous low-resolution coordinate of this full-resolution pixel.
      const cx = (d.f32(x) + 0.5) * 0.5 - 0.5;
      const cy = (d.f32(y) + 0.5) * 0.5 - 0.5;
      const bx = std.floor(cx + 0.5);
      const by = std.floor(cy + 0.5);

      let weighted = d.f32(0);
      let total = d.f32(0);

      for (const dy of tgpu.unroll(std.range(2 * RADIUS + 1))) {
        for (const dx of tgpu.unroll(std.range(2 * RADIUS + 1))) {
          const qx = std.clamp(bx + (dx - RADIUS), 0, lowW - 1);
          const qy = std.clamp(by + (dy - RADIUS), 0, lowH - 1);
          const qi = d.u32(qy) * lowW + d.u32(qx);
          const disparity = depthUpsampleLayout.$.low[qi].x;

          // Guide luminance at the full-resolution pixel this tap represents.
          const gx = d.u32(qx) * 2;
          const gy = d.u32(qy) * 2;
          const g = std.textureLoad(depthUpsampleLayout.$.guide, d.vec2u(gx, gy), 0);
          const lumaThere = luminance(d.vec3f(g.x, g.y, g.z));

          const dxs = qx - cx;
          const dys = qy - cy;
          const spatial = std.exp(-(dxs * dxs + dys * dys) * 0.5);
          const dl = (lumaHere - lumaThere) / std.max(opts.x, 1e-4);
          const range = std.exp(-dl * dl * 0.5);

          const w = spatial * range;
          weighted += disparity * w;
          total += w;
        }
      }

      const upsampled = weighted / std.max(total, 1e-6);

      const prev = std.textureLoad(depthUpsampleLayout.$.history, d.vec2u(x, y), 0);
      // On the first frame the history texture is cleared to zero; `prev.w` is
      // the validity marker written by this kernel, so a zero means "no history".
      const change = std.abs(upsampled - prev.x);
      const alpha = std.select(
        std.clamp(opts.y + change * opts.z, opts.y, 1),
        1,
        prev.w < 0.5,
      );
      const stabilised = prev.x + (upsampled - prev.x) * alpha;

      std.textureStore(
        depthUpsampleLayout.$.dst,
        d.vec2u(x, y),
        d.vec4f(stabilised, upsampled, change, 1),
      );
    }
  });

  return {
    fn,
    dispatch: [Math.ceil(fullSize / 8), Math.ceil(fullSize / 8), 1],
    tiling: '8×8 threads, 3×3 bilateral taps',
  };
}
