/**
 * Depth-aware illumination.
 *
 * The estimated depth map is the *only* geometry this renderer has: there is no
 * mesh, no G-buffer from a rasteriser, no normals from an artist. Everything the
 * lighting needs — position, normal, occlusion, shadowing, participating media —
 * is derived from a single-channel disparity image that the network produced a
 * few microseconds earlier, in the same command encoder, and which has never
 * left the GPU.
 *
 * The pass computes, per pixel:
 *
 *  1. **View-space position** by unprojecting the linearised depth through the
 *     camera's field of view.
 *  2. **Normals** by the "accurate" reconstruction of Yuwen Wu / Turánszki:
 *     rather than central differences (which corrupt a two-pixel band around
 *     every silhouette), it extrapolates depth from the two neighbours on each
 *     side and takes the derivative from whichever side extrapolates better.
 *     Because inverse depth is affine in screen space for planar surfaces, and
 *     our disparity is itself affine in inverse depth, the test can be applied
 *     to the raw disparity with no conversion.
 *  3. **Direct lighting** — Lambert diffuse plus a GGX specular lobe, with
 *     smooth distance and cone attenuation.
 *  4. **Screen-space contact shadows** by marching the depth buffer toward each
 *     light, with a per-pixel dither so the step pattern reads as grain rather
 *     than banding.
 *  5. **Horizon-based ambient occlusion**, so creases and contacts darken.
 *  6. **Volumetric scattering** by marching the view ray and sampling each
 *     light's contribution to the participating medium.
 *  7. **Depth fog**, applied last, in linear light.
 *
 * The result is written to an HDR storage texture; `composite.ts` tone-maps it.
 */
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { LightBuffer, MAX_LIGHTS } from './lights.ts';
import { srgbToLinear } from '../model/kernels/io.ts';

export const ShadeParams = d.struct({
  /** `x` = tan(fovY/2), `y` = aspect, `z` = near, `w` = far. */
  camera: d.vec4f,
  /** `x` = AO strength, `y` = shadow strength, `z` = volumetric strength, `w` = exposure. */
  toggles: d.vec4f,
  /** `rgb` = ambient/sky colour, `a` = ambient intensity. */
  ambient: d.vec4f,
  /** `x` = frame index (dither), `y` = roughness, `z` = fog density, `w` = normal strength. */
  misc: d.vec4f,
  /** `rgb` = fog colour, `a` = specular intensity. */
  fog: d.vec4f,
  /** `x` = render width, `y` = render height, `z` = depth width, `w` = depth height. */
  size: d.vec4f,
  /** `x` = debug view id, `y` = albedo mix, `z` = contact-shadow length, `w` = AO radius. */
  debug: d.vec4f,
});

export const shadeLayout = tgpu
  .bindGroupLayout({
    /** Stabilised disparity from the network, `x` channel. */
    depth: { texture: d.texture2d(d.f32) },
    /** Full-resolution source image (sRGB encoded). */
    albedo: { texture: d.texture2d(d.f32) },
    samp: { sampler: 'filtering' },
    params: { uniform: ShadeParams },
    lights: { storage: LightBuffer, access: 'readonly' },
    out: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  })
  .$name('shade');

const P = shadeLayout.$;

/**
 * Disparity → view-space distance.
 *
 * The network predicts *relative* inverse depth, so the mapping to metres is a
 * user choice: `near` and `far` set the scene's depth range. The relationship is
 * deliberately the reciprocal one, `1/z = d·(1/near − 1/far) + 1/far`, which
 * makes `1/z` affine in the disparity and therefore linear in screen space over
 * planar surfaces — the property the normal reconstruction relies on.
 */
export const linearDepth = tgpu
  .fn([d.f32], d.f32)((disparity) => {
    'use gpu';
    const near = std.max(P.params.camera.z, 1e-3);
    const far = std.max(P.params.camera.w, near + 1e-3);
    const invZ = disparity * (1 / near - 1 / far) + 1 / far;
    return 1 / std.max(invZ, 1e-6);
  })
  .$name('linearDepth');

/** Raw disparity fetch, clamped to the depth texture's extent. */
export const fetchDisparity = tgpu
  .fn([d.vec2i], d.f32)((coord) => {
    'use gpu';
    const maxX = d.i32(P.params.size.z) - 1;
    const maxY = d.i32(P.params.size.w) - 1;
    const c = d.vec2i(std.clamp(coord.x, 0, maxX), std.clamp(coord.y, 0, maxY));
    return std.textureLoad(P.depth, c, 0).x;
  })
  .$name('fetchDisparity');

/** Unprojects a pixel centre and a linear depth into view space (right-handed, −Z forward). */
export const viewPosition = tgpu
  .fn([d.vec2f, d.f32], d.vec3f)((uv, z) => {
    'use gpu';
    const ndcX = uv.x * 2 - 1;
    const ndcY = 1 - uv.y * 2;
    const t = P.params.camera.x;
    return d.vec3f(ndcX * P.params.camera.y * t * z, ndcY * t * z, -z);
  })
  .$name('viewPosition');

/** View-space position at an integer depth-map coordinate. */
export const viewPositionAt = tgpu
  .fn([d.vec2i], d.vec3f)((coord) => {
    'use gpu';
    const uv = d.vec2f(
      (d.f32(coord.x) + 0.5) / P.params.size.z,
      (d.f32(coord.y) + 0.5) / P.params.size.w,
    );
    return viewPosition(uv, linearDepth(fetchDisparity(coord)));
  })
  .$name('viewPositionAt');

/**
 * Normal reconstruction, "accurate" variant.
 *
 * For each axis, the depth at the centre is predicted by linearly extrapolating
 * from the two samples on the left and, separately, from the two on the right.
 * A planar surface extrapolates exactly; a silhouette does not. Taking the
 * one-sided derivative from whichever side predicts better keeps the normal
 * correct right up to the edge, instead of smearing a foreground normal two
 * pixels into the background.
 */
export const reconstructNormal = tgpu
  .fn([d.vec2i], d.vec3f)((coord) => {
    'use gpu';
    const c = fetchDisparity(coord);

    const l1 = fetchDisparity(d.vec2i(coord.x - 1, coord.y));
    const l2 = fetchDisparity(d.vec2i(coord.x - 2, coord.y));
    const r1 = fetchDisparity(d.vec2i(coord.x + 1, coord.y));
    const r2 = fetchDisparity(d.vec2i(coord.x + 2, coord.y));
    const u1 = fetchDisparity(d.vec2i(coord.x, coord.y - 1));
    const u2 = fetchDisparity(d.vec2i(coord.x, coord.y - 2));
    const b1 = fetchDisparity(d.vec2i(coord.x, coord.y + 1));
    const b2 = fetchDisparity(d.vec2i(coord.x, coord.y + 2));

    const errL = std.abs(2 * l1 - l2 - c);
    const errR = std.abs(2 * r1 - r2 - c);
    const errU = std.abs(2 * u1 - u2 - c);
    const errB = std.abs(2 * b1 - b2 - c);

    const pc = viewPositionAt(coord);
    const dxLeft = std.sub(pc, viewPositionAt(d.vec2i(coord.x - 1, coord.y)));
    const dxRight = std.sub(viewPositionAt(d.vec2i(coord.x + 1, coord.y)), pc);
    const dyUp = std.sub(pc, viewPositionAt(d.vec2i(coord.x, coord.y - 1)));
    const dyDown = std.sub(viewPositionAt(d.vec2i(coord.x, coord.y + 1)), pc);

    const dx = std.select(dxRight, dxLeft, errL < errR);
    const dy = std.select(dyDown, dyUp, errU < errB);

    // Screen y runs downward while view-space y runs up, so the cross product is
    // taken as (dy × dx) to keep the normal facing the camera.
    const n = std.cross(dy, dx);
    const len = std.length(n);
    return std.select(d.vec3f(0, 0, 1), std.div(n, len), len > 1e-12);
  })
  .$name('reconstructNormal');

/** Interleaved gradient noise — cheap, and stable enough to look like film grain. */
export const dither = tgpu
  .fn([d.vec2f, d.f32], d.f32)((pos, frame) => {
    'use gpu';
    const p = std.add(pos, d.vec2f(frame * 5.588238, frame * 5.588238));
    return std.fract(52.9829189 * std.fract(std.dot(p, d.vec2f(0.06711056, 0.00583715))));
  })
  .$name('dither');

/** Projects a view-space point back to depth-map pixel coordinates. */
export const projectToDepthUv = tgpu
  .fn([d.vec3f], d.vec2f)((p) => {
    'use gpu';
    const z = std.max(-p.z, 1e-4);
    const t = P.params.camera.x;
    const ndcX = p.x / (P.params.camera.y * t * z);
    const ndcY = p.y / (t * z);
    return d.vec2f(ndcX * 0.5 + 0.5, 0.5 - ndcY * 0.5);
  })
  .$name('projectToDepthUv');

const CONTACT_STEPS = 12;

/**
 * Screen-space contact shadow: march from the surface toward the light and see
 * whether the depth buffer says something is in the way.
 *
 * Returns 1 when unoccluded. The `thickness` test prevents a thin foreground
 * object from casting an infinitely deep shadow, which is the classic
 * screen-space artefact.
 */
export const contactShadow = tgpu
  .fn([d.vec3f, d.vec3f, d.f32, d.f32], d.f32)((origin, toLight, maxLength, jitter) => {
    'use gpu';
    const step = std.mul(maxLength / CONTACT_STEPS, toLight);
    let occlusion = d.f32(0);
    let p = std.add(origin, std.mul(jitter, step));
    for (const i of tgpu.unroll(std.range(CONTACT_STEPS))) {
      p = std.add(p, step);
      const uv = projectToDepthUv(p);
      if (uv.x > 0 && uv.x < 1 && uv.y > 0 && uv.y < 1) {
        const coord = d.vec2i(
          d.i32(uv.x * P.params.size.z),
          d.i32(uv.y * P.params.size.w),
        );
        const sceneZ = linearDepth(fetchDisparity(coord));
        const rayZ = -p.z;
        const delta = rayZ - sceneZ;
        // Occluded when the ray is behind the surface, but only by a plausible
        // thickness — beyond that the "occluder" is just unrelated background.
        const thickness = std.max(0.35, sceneZ * 0.08);
        if (delta > 0.012 && delta < thickness) {
          occlusion = std.max(occlusion, 1 - d.f32(i) / CONTACT_STEPS);
        }
      }
      void i;
    }
    return 1 - std.clamp(occlusion, 0, 1);
  })
  .$name('contactShadow');

const AO_DIRECTIONS = 6;
const AO_STEPS = 4;

/**
 * Horizon-based ambient occlusion.
 *
 * For a few screen-space directions, march outward and track the steepest
 * horizon angle the neighbourhood subtends above the tangent plane. The
 * occlusion is the fraction of the hemisphere those horizons cut away.
 */
export const horizonOcclusion = tgpu
  .fn([d.vec2i, d.vec3f, d.vec3f, d.f32, d.f32], d.f32)(
    (coord, position, normal, radiusPx, jitter) => {
      'use gpu';
      let total = d.f32(0);
      for (const i of tgpu.unroll(std.range(AO_DIRECTIONS))) {
        const angle = ((d.f32(i) + jitter) / AO_DIRECTIONS) * 6.2831853;
        const dir = d.vec2f(std.cos(angle), std.sin(angle));
        let horizon = d.f32(-1);
        for (const s of tgpu.unroll(std.range(AO_STEPS))) {
          const dist = radiusPx * ((d.f32(s) + 1) / AO_STEPS);
          const sampleCoord = d.vec2i(
            coord.x + d.i32(dir.x * dist),
            coord.y + d.i32(dir.y * dist),
          );
          const diff = std.sub(viewPositionAt(sampleCoord), position);
          const len = std.length(diff);
          if (len > 1e-5) {
            // Cosine of the angle between the tangent plane and the sample,
            // attenuated so distant samples cannot occlude.
            const cosH = std.dot(normal, std.div(diff, len));
            const falloff = std.clamp(1 - (len * len) / 9, 0, 1);
            horizon = std.max(horizon, cosH * falloff);
          }
          void s;
        }
        total += std.max(horizon, 0);
        void i;
      }
      return std.clamp(1 - total / AO_DIRECTIONS, 0, 1);
    },
  )
  .$name('horizonOcclusion');

/** GGX normal distribution term. */
export const distributionGGX = tgpu
  .fn([d.f32, d.f32], d.f32)((nDotH, roughness) => {
    'use gpu';
    const a = roughness * roughness;
    const a2 = a * a;
    const denom = nDotH * nDotH * (a2 - 1) + 1;
    return a2 / std.max(3.14159265 * denom * denom, 1e-6);
  })
  .$name('distributionGGX');

/** Smith height-correlated visibility, Schlick-GGX approximation. */
export const visibilitySmith = tgpu
  .fn([d.f32, d.f32, d.f32], d.f32)((nDotV, nDotL, roughness) => {
    'use gpu';
    const k = (roughness + 1) * (roughness + 1) / 8;
    const gv = nDotV / (nDotV * (1 - k) + k);
    const gl = nDotL / (nDotL * (1 - k) + k);
    return gv * gl;
  })
  .$name('visibilitySmith');

const VOLUMETRIC_STEPS = 10;

export function makeShade() {
  const fn = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [8, 8],
  })((input) => {
    'use gpu';
    const px = input.gid.x;
    const py = input.gid.y;
    const width = d.u32(P.params.size.x);
    const height = d.u32(P.params.size.y);

    if (px < width && py < height) {
      const uv = d.vec2f(
        (d.f32(px) + 0.5) / P.params.size.x,
        (d.f32(py) + 0.5) / P.params.size.y,
      );

      // Depth-map coordinate for this render pixel. The depth map may be a
      // different (lower) resolution than the render target.
      const dcoord = d.vec2i(
        d.i32(uv.x * P.params.size.z),
        d.i32(uv.y * P.params.size.w),
      );

      const disparity = fetchDisparity(dcoord);
      const z = linearDepth(disparity);
      const position = viewPosition(uv, z);
      const rawNormal = reconstructNormal(dcoord);
      // `normalStrength` lets a flat-ish depth estimate still produce shaping.
      const normal = std.normalize(
        d.vec3f(
          rawNormal.x * P.params.misc.w,
          rawNormal.y * P.params.misc.w,
          rawNormal.z,
        ),
      );

      // Albedo comes from the *source* texture at full resolution, so the image
      // stays crisp even though the geometry is only known at network
      // resolution.
      const srgb = std.textureSampleLevel(P.albedo, P.samp, uv, 0);
      const albedoLinear = srgbToLinear(d.vec3f(srgb.x, srgb.y, srgb.z));
      const albedo = std.mix(d.vec3f(0.6, 0.6, 0.6), albedoLinear, P.params.debug.y);

      const jitter = dither(d.vec2f(d.f32(px), d.f32(py)), P.params.misc.x);
      const viewDir = std.normalize(std.neg(position));
      const nDotV = std.max(std.dot(normal, viewDir), 1e-4);
      const roughness = std.clamp(P.params.misc.y, 0.04, 1);

      const ao = std.mix(
        1,
        horizonOcclusion(dcoord, position, normal, P.params.debug.w, jitter),
        P.params.toggles.x,
      );

      let radiance = std.mul(
        std.mul(P.params.ambient.w * ao, albedo),
        d.vec3f(P.params.ambient.x, P.params.ambient.y, P.params.ambient.z),
      );

      let scattering = d.vec3f();

      for (const li of tgpu.unroll(std.range(MAX_LIGHTS))) {
        const light = P.lights[li];
        const intensity = light.color.w;
        if (intensity > 0) {
          const kind = light.params.x;
          const lightColor = d.vec3f(light.color.x, light.color.y, light.color.z);

          // Direction to the light and distance attenuation.
          let toLight = d.vec3f(0, 0, 1);
          let attenuation = d.f32(1);
          let distance = d.f32(1e6);

          if (kind > 1.5) {
            toLight = std.normalize(
              std.neg(d.vec3f(light.position.x, light.position.y, light.position.z)),
            );
          } else {
            const delta = std.sub(
              d.vec3f(light.position.x, light.position.y, light.position.z),
              position,
            );
            distance = std.max(std.length(delta), 1e-4);
            toLight = std.div(delta, distance);
            const range = std.max(light.params.z, 1e-3);
            // Inverse square, windowed so the light reaches exactly zero at its
            // radius instead of leaving a faint halo forever.
            const window = std.clamp(1 - (distance / range) ** 4, 0, 1);
            attenuation = (window * window) / (distance * distance + 1e-4);
          }

          if (kind > 0.5 && kind < 1.5) {
            const spotAxis = d.vec3f(
              light.direction.x,
              light.direction.y,
              light.direction.z,
            );
            const cosAngle = std.dot(std.neg(toLight), spotAxis);
            const outer = light.direction.w;
            const inner = light.params.y;
            attenuation *= std.smoothstep(outer, std.max(inner, outer + 1e-3), cosAngle);
          }

          const nDotL = std.dot(normal, toLight);
          if (nDotL > 0 && attenuation > 0) {
            let shadow = d.f32(1);
            const shadowAmount = light.params.w * P.params.toggles.y;
            if (shadowAmount > 0) {
              const marchLength = std.min(P.params.debug.z, distance);
              shadow = std.mix(
                1,
                contactShadow(position, toLight, marchLength, jitter),
                shadowAmount,
              );
            }

            const half = std.normalize(std.add(toLight, viewDir));
            const nDotH = std.max(std.dot(normal, half), 0);
            const spec =
              distributionGGX(nDotH, roughness) *
              visibilitySmith(nDotV, nDotL, roughness) *
              P.params.fog.w;

            const energy = attenuation * intensity * nDotL * shadow;
            radiance = std.add(
              radiance,
              std.mul(energy, std.mul(lightColor, std.add(albedo, d.vec3f(spec)))),
            );
          }

          // --- volumetric scattering ---------------------------------------
          // The scattering coefficient is the fog density: a denser medium both
          // attenuates more and scatters more, so the two share one control.
          if (P.params.toggles.z > 0 && kind < 1.5) {
            // March the actual view ray, whose length is |position|, not z.
            const rayLength = std.length(position);
            const rayDir = std.div(position, std.max(rayLength, 1e-5));
            const stepLength = rayLength / VOLUMETRIC_STEPS;
            for (const s of tgpu.unroll(std.range(VOLUMETRIC_STEPS))) {
              const t = ((d.f32(s) + jitter) / VOLUMETRIC_STEPS) * rayLength;
              const samplePos = std.mul(t, rayDir);
              const toL = std.sub(
                d.vec3f(light.position.x, light.position.y, light.position.z),
                samplePos,
              );
              const dist = std.max(std.length(toL), 1e-3);
              const range = std.max(light.params.z, 1e-3);
              const win = std.clamp(1 - (dist / range) ** 4, 0, 1);
              let att = (win * win) / (dist * dist + 1e-4);

              if (kind > 0.5) {
                const spotAxis = d.vec3f(
                  light.direction.x,
                  light.direction.y,
                  light.direction.z,
                );
                const cosAngle = std.dot(std.neg(std.div(toL, dist)), spotAxis);
                att *= std.smoothstep(
                  light.direction.w,
                  std.max(light.params.y, light.direction.w + 1e-3),
                  cosAngle,
                );
              }
              scattering = std.add(
                scattering,
                std.mul(att * intensity * stepLength * P.params.misc.z, lightColor),
              );
              void s;
            }
          }
        }
        void li;
      }

      radiance = std.add(radiance, std.mul(P.params.toggles.z, scattering));

      // --- depth fog, in linear light ------------------------------------
      const fogAmount = 1 - std.exp(-z * P.params.misc.z);
      radiance = std.mix(
        radiance,
        d.vec3f(P.params.fog.x, P.params.fog.y, P.params.fog.z),
        std.clamp(fogAmount, 0, 1),
      );

      radiance = std.mul(P.params.toggles.w, radiance);

      // --- debug views ----------------------------------------------------
      const mode = P.params.debug.x;
      // Copy: TGSL forbids binding a `let` to a reference it may later reassign.
      let outColor = d.vec3f(radiance);
      if (mode > 0.5 && mode < 1.5) {
        outColor = d.vec3f(disparity, disparity, disparity);
      } else if (mode > 1.5 && mode < 2.5) {
        outColor = std.mul(0.5, std.add(normal, d.vec3f(1)));
      } else if (mode > 2.5 && mode < 3.5) {
        outColor = d.vec3f(ao, ao, ao);
      } else if (mode > 3.5) {
        outColor = d.vec3f(albedoLinear);
      }

      std.textureStore(
        P.out,
        d.vec2u(px, py),
        d.vec4f(outColor.x, outColor.y, outColor.z, 1),
      );
    }
  });

  return { fn };
}
