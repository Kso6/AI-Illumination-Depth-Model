/**
 * Presentation pass: a single full-screen triangle that tone-maps the HDR
 * lighting buffer and writes it to the swap chain.
 *
 * This is the only *render* pass in the frame, and it is recorded into the same
 * command encoder as the forty inference dispatches and the shading dispatch
 * that produced its input.
 */
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';

export const CompositeParams = d.struct({
  /** `x` = tone-map operator id, `y` = contrast, `z` = saturation, `w` = vignette. */
  grade: d.vec4f,
});

export const compositeLayout = tgpu
  .bindGroupLayout({
    hdr: { texture: d.texture2d(d.f32) },
    samp: { sampler: 'filtering' },
    params: { uniform: CompositeParams },
  })
  .$name('composite');

// Hoisted out of the shader body: `layout.$` must be resolved as an external at
// module scope, not bound to a local inside a `'use gpu'` function.
const Q = compositeLayout.$;

/**
 * Narkowicz's fitted ACES curve. Compact, and close enough to the reference RRT
 * for real-time work; the highlight roll-off is what keeps bright specular hits
 * from clipping to flat white.
 */
export const acesFitted = tgpu
  .fn([d.vec3f], d.vec3f)((x) => {
    'use gpu';
    const a = 2.51;
    const b = 0.03;
    const c = 2.43;
    const dd = 0.59;
    const e = 0.14;
    const num = std.mul(x, std.add(std.mul(a, x), d.vec3f(b)));
    const den = std.add(std.mul(x, std.add(std.mul(c, x), d.vec3f(dd))), d.vec3f(e));
    return std.clamp(std.div(num, den), d.vec3f(), d.vec3f(1));
  })
  .$name('acesFitted');

/** Reinhard with a white point, for comparison against ACES. */
export const reinhard = tgpu
  .fn([d.vec3f], d.vec3f)((x) => {
    'use gpu';
    const white = 4;
    const num = std.mul(x, std.add(d.vec3f(1), std.div(x, white * white)));
    return std.clamp(std.div(num, std.add(d.vec3f(1), x)), d.vec3f(), d.vec3f(1));
  })
  .$name('reinhard');

/** Linear → sRGB, exact. */
export const linearToSrgb = tgpu
  .fn([d.vec3f], d.vec3f)((c) => {
    'use gpu';
    const lo = std.mul(12.92, c);
    const hi = std.sub(std.mul(1.055, std.pow(c, d.vec3f(1 / 2.4))), d.vec3f(0.055));
    return std.select(hi, lo, std.le(c, d.vec3f(0.0031308)));
  })
  .$name('linearToSrgb');

/**
 * Full-screen triangle. Three vertices covering the viewport beats two
 * triangles: no diagonal seam, and one less quad of helper-lane waste.
 */
export const compositeVertex = tgpu.vertexFn({
  in: { idx: d.builtin.vertexIndex },
  out: { pos: d.builtin.position, uv: d.vec2f },
})((input) => {
  'use gpu';
  const x = d.f32((input.idx << 1) & 2) * 2 - 1;
  const y = d.f32(input.idx & 2) * 2 - 1;
  return {
    pos: d.vec4f(x, y, 0, 1),
    // Flip y: clip space runs up, texture space runs down.
    uv: d.vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5),
  };
});

export const compositeFragment = tgpu.fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  'use gpu';
  const hdr = std.textureSampleLevel(Q.hdr, Q.samp, input.uv, 0);
  let color = d.vec3f(hdr.x, hdr.y, hdr.z);

  const mapped = std.select(reinhard(color), acesFitted(color), Q.params.grade.x < 0.5);

  // Grade in display-referred space, where contrast and saturation behave the
  // way an artist expects.
  const luma = std.dot(mapped, d.vec3f(0.2126, 0.7152, 0.0722));
  color = std.mix(d.vec3f(luma), mapped, Q.params.grade.z);
  color = std.clamp(
    std.add(d.vec3f(0.5), std.mul(Q.params.grade.y, std.sub(color, d.vec3f(0.5)))),
    d.vec3f(),
    d.vec3f(1),
  );

  const centred = std.sub(input.uv, d.vec2f(0.5));
  const vignette = 1 - Q.params.grade.w * std.dot(centred, centred) * 2;
  color = std.mul(std.clamp(vignette, 0, 1), color);

  const srgb = linearToSrgb(color);
  return d.vec4f(srgb.x, srgb.y, srgb.z, 1);
});
