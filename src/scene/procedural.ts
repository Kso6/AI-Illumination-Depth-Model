/**
 * A procedural 3D scene, ray-marched on the GPU, used for three things:
 *
 *  1. **A live source image** that always works, with no camera permission and
 *     no asset download.
 *  2. **Ground truth.** Because the scene is analytic, its exact depth is known,
 *     so the estimated depth can be scored against it (see `src/scene/metrics.ts`)
 *     and the lighting can be previewed with perfect geometry for comparison.
 *  3. **Training data.** Rendering RGB/depth pairs is precisely what a
 *     distillation or supervised run needs, and it happens at hundreds of frames
 *     per second on the GPU.
 *
 * The scene is drawn with a signed-distance field so that both the colour and
 * the depth fall out of the same march, and the whole thing is one draw call
 * recorded into the frame's command encoder before the network runs.
 */
import tgpu from 'typegpu';
import type { TgpuRenderPass, TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';

export const SceneParams = d.struct({
  /** `xyz` = camera position, `w` = time in seconds. */
  camera: d.vec4f,
  /** `xyz` = look-at point, `w` = tan(fovY/2). (`target` is a WGSL keyword.) */
  focus: d.vec4f,
  /** `x` = aspect, `y` = far plane, `z` = key-light azimuth, `w` = scene variant. */
  options: d.vec4f,
});

export const sceneLayout = tgpu
  .bindGroupLayout({ params: { uniform: SceneParams } })
  .$name('scene');

const S = sceneLayout.$;

const sdSphere = tgpu
  .fn([d.vec3f, d.f32], d.f32)((p, r) => {
    'use gpu';
    return std.length(p) - r;
  })
  .$name('sdSphere');

const sdBox = tgpu
  .fn([d.vec3f, d.vec3f], d.f32)((p, b) => {
    'use gpu';
    const q = std.sub(std.abs(p), b);
    return (
      std.length(std.max(q, d.vec3f())) +
      std.min(std.max(q.x, std.max(q.y, q.z)), 0)
    );
  })
  .$name('sdBox');

/** Smooth union, so shapes merge into organic contacts rather than hard seams. */
const smoothUnion = tgpu
  .fn([d.f32, d.f32, d.f32], d.f32)((a, b, k) => {
    'use gpu';
    const h = std.clamp(0.5 + (0.5 * (b - a)) / k, 0, 1);
    return std.mix(b, a, h) - k * h * (1 - h);
  })
  .$name('smoothUnion');

/**
 * The scene. Returns `x` = distance, `y` = material id.
 *
 * Composed to give a depth histogram like a real photograph: a dominant near
 * subject, a mid-ground cluster, and a receding floor that runs to the horizon.
 */
const map = tgpu
  .fn([d.vec3f], d.vec2f)((p) => {
    'use gpu';
    const t = S.params.camera.w;

    // Ground plane with a gentle roll so it is not a perfect gradient.
    let dist = p.y + 1 + 0.06 * std.sin(p.x * 0.7) * std.cos(p.z * 0.6);
    let material = d.f32(1);

    // Near subject: two merged spheres, slowly bobbing.
    const bob = 0.12 * std.sin(t * 0.9);
    const a = sdSphere(std.sub(p, d.vec3f(-0.35, -0.25 + bob, -2.1)), 0.62);
    const b = sdSphere(std.sub(p, d.vec3f(0.4, -0.45 + bob * 0.6, -1.75)), 0.42);
    const blob = smoothUnion(a, b, 0.35);
    if (blob < dist) {
      dist = blob;
      material = 2;
    }

    // Mid-ground pillars.
    for (const i of tgpu.unroll(std.range(4))) {
      const x = -2.4 + d.f32(i) * 1.55;
      const z = -4.6 - d.f32((i * 7) % 5) * 0.55;
      const h = 0.9 + 0.35 * d.f32((i * 3) % 4);
      const box = sdBox(std.sub(p, d.vec3f(x, -1 + h, z)), d.vec3f(0.22, h, 0.22));
      if (box < dist) {
        dist = box;
        material = 3;
      }
      void i;
    }

    // Far arch, giving the background real structure to estimate.
    const arch = std.max(
      sdBox(std.sub(p, d.vec3f(0.2, 0.6, -8.5)), d.vec3f(2.6, 1.9, 0.3)),
      -sdSphere(std.sub(p, d.vec3f(0.2, -0.4, -8.5)), 1.7),
    );
    if (arch < dist) {
      dist = arch;
      material = 4;
    }

    // A floating sphere close to camera, to create a strong occlusion edge.
    const near = sdSphere(
      std.sub(p, d.vec3f(1.25 + 0.25 * std.sin(t * 0.5), 0.15, -1.15)),
      0.28,
    );
    if (near < dist) {
      dist = near;
      material = 5;
    }

    return d.vec2f(dist, material);
  })
  .$name('sceneMap');

const calcNormal = tgpu
  .fn([d.vec3f], d.vec3f)((p) => {
    'use gpu';
    const e = 0.0015;
    const dx = map(std.add(p, d.vec3f(e, 0, 0))).x - map(std.sub(p, d.vec3f(e, 0, 0))).x;
    const dy = map(std.add(p, d.vec3f(0, e, 0))).x - map(std.sub(p, d.vec3f(0, e, 0))).x;
    const dz = map(std.add(p, d.vec3f(0, 0, e))).x - map(std.sub(p, d.vec3f(0, 0, e))).x;
    return std.normalize(d.vec3f(dx, dy, dz));
  })
  .$name('calcNormal');

/** Soft shadow by tracking the closest approach of the shadow ray. */
const softShadow = tgpu
  .fn([d.vec3f, d.vec3f], d.f32)((origin, dir) => {
    'use gpu';
    let res = d.f32(1);
    let t = d.f32(0.04);
    // A real loop, not `tgpu.unroll`: unrolling a 28-step march would inline the
    // whole SDF twenty-eight times and cost more in shader size than it saves.
    for (let i = d.u32(0); i < 28; i++) {
      const h = map(std.add(origin, std.mul(t, dir))).x;
      res = std.min(res, (12 * h) / t);
      t += std.clamp(h, 0.02, 0.4);
      if (res < 0.002 || t > 12) {
        break;
      }
    }
    return std.clamp(res, 0, 1);
  })
  .$name('softShadow');

const materialColor = tgpu
  .fn([d.f32, d.vec3f], d.vec3f)((material, p) => {
    'use gpu';
    if (material < 1.5) {
      // Checkered floor, which gives the depth network strong perspective cues.
      const c = std.floor(p.x * 0.8) + std.floor(p.z * 0.8);
      const checker = std.select(0.18, 0.34, std.fract(c * 0.5) < 0.25);
      return d.vec3f(checker * 1.05, checker, checker * 0.92);
    }
    if (material < 2.5) return d.vec3f(0.82, 0.28, 0.22);
    if (material < 3.5) return d.vec3f(0.28, 0.42, 0.6);
    if (material < 4.5) return d.vec3f(0.55, 0.5, 0.44);
    return d.vec3f(0.9, 0.76, 0.32);
  })
  .$name('materialColor');

export const sceneVertex = tgpu.vertexFn({
  in: { idx: d.builtin.vertexIndex },
  out: { pos: d.builtin.position, uv: d.vec2f },
})((input) => {
  'use gpu';
  const x = d.f32((input.idx << 1) & 2) * 2 - 1;
  const y = d.f32(input.idx & 2) * 2 - 1;
  return { pos: d.vec4f(x, y, 0, 1), uv: d.vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5) };
});

/**
 * Fragment shader writing two targets: the sRGB colour image the network will
 * consume, and the exact distance along the view ray.
 */
export const sceneFragment = tgpu.fragmentFn({
  in: { uv: d.vec2f },
  out: { color: d.vec4f, depth: d.vec4f },
})((input) => {
  'use gpu';
  const p = S.params;
  const origin = d.vec3f(p.camera.x, p.camera.y, p.camera.z);
  const lookAt = d.vec3f(p.focus.x, p.focus.y, p.focus.z);

  const forward = std.normalize(std.sub(lookAt, origin));
  const right = std.normalize(std.cross(forward, d.vec3f(0, 1, 0)));
  const up = std.cross(right, forward);

  const ndcX = input.uv.x * 2 - 1;
  const ndcY = 1 - input.uv.y * 2;
  const dir = std.normalize(
    std.add(
      std.add(
        std.mul(ndcX * p.options.x * p.focus.w, right),
        std.mul(ndcY * p.focus.w, up),
      ),
      forward,
    ),
  );

  // Sphere tracing.
  let t = d.f32(0.02);
  let material = d.f32(0);
  let hit = d.f32(0);
  for (let i = d.u32(0); i < 96; i++) {
    const sample = map(std.add(origin, std.mul(t, dir)));
    if (sample.x < 0.0015 * t) {
      hit = 1;
      material = sample.y;
      break;
    }
    t += sample.x * 0.85;
    if (t > p.options.y) {
      t = p.options.y;
      break;
    }
  }

  // Sky: a simple gradient standing in for an overcast dome.
  const horizon = std.clamp(dir.y * 0.5 + 0.5, 0, 1);
  let color = std.mix(d.vec3f(0.42, 0.47, 0.55), d.vec3f(0.12, 0.18, 0.32), horizon);

  if (hit > 0.5) {
    const pos = std.add(origin, std.mul(t, dir));
    const normal = calcNormal(pos);
    const albedo = materialColor(material, pos);

    const sunAngle = p.options.z;
    const sun = std.normalize(d.vec3f(std.cos(sunAngle), 0.72, std.sin(sunAngle)));
    const shadow = softShadow(std.add(pos, std.mul(0.02, normal)), sun);
    const diffuse = std.max(std.dot(normal, sun), 0) * shadow;
    const sky = 0.5 + 0.5 * normal.y;
    const bounce = std.max(-normal.y, 0) * 0.2;

    const lit = std.mul(
      albedo,
      std.add(
        std.add(std.mul(diffuse * 1.5, d.vec3f(1, 0.94, 0.82)), std.mul(sky * 0.35, d.vec3f(0.35, 0.45, 0.65))),
        d.vec3f(bounce * 0.25),
      ),
    );

    const half = std.normalize(std.sub(sun, dir));
    const spec = std.pow(std.max(std.dot(normal, half), 0), 48) * shadow * 0.35;
    color = std.add(lit, d.vec3f(spec));

    // Aerial perspective so distant geometry reads as distant.
    const fog = 1 - std.exp(-t * 0.055);
    color = std.mix(color, d.vec3f(0.3, 0.36, 0.46), std.clamp(fog, 0, 1));
  }

  // Simple filmic curve then sRGB encode: the network expects a
  // display-referred image, like any photograph or video frame.
  const mapped = std.div(color, std.add(color, d.vec3f(0.6)));
  const srgb = std.pow(std.clamp(mapped, d.vec3f(), d.vec3f(1)), d.vec3f(1 / 2.2));

  return {
    color: d.vec4f(srgb.x, srgb.y, srgb.z, 1),
    // Store the view-space distance along -Z, matching what the lighting pass
    // reconstructs, plus a hit mask so the sky can be excluded from metrics.
    depth: d.vec4f(t * std.dot(dir, forward), hit, 0, 1),
  };
});

export interface ProceduralScene {
  /** Records the scene draw. Must precede the network's dispatches. */
  record(pass: TgpuRenderPass): void;
  update(timeSeconds: number, orbit: boolean): void;
  destroy(): void;
}

/**
 * Draws into caller-owned targets rather than allocating its own, so the colour
 * target can be the one stable source texture that every bind group in the
 * application already points at. Switching between the procedural scene and a
 * camera feed then costs nothing: the same texture is either rendered into or
 * copied into.
 *
 * @param colorTarget `rgba8unorm`, usable as a render attachment and sampled.
 * @param truthTarget `rgba16float`; receives exact depth and a hit mask.
 */
export function createProceduralScene(
  root: TgpuRoot,
  colorFormat: GPUTextureFormat,
  truthFormat: GPUTextureFormat,
): ProceduralScene {
  const params = root.createUniform(SceneParams, {
    camera: d.vec4f(0, 0.35, 0.6, 0),
    focus: d.vec4f(0, -0.15, -3, Math.tan((58 * Math.PI) / 180 / 2)),
    options: d.vec4f(1, 30, 0.9, 0),
  });

  const pipeline = root.createRenderPipeline({
    vertex: sceneVertex,
    fragment: sceneFragment,
    targets: { color: { format: colorFormat }, depth: { format: truthFormat } },
    primitive: { topology: 'triangle-list' },
  });

  const group = root.createBindGroup(sceneLayout, { params });

  return {
    record(pass) {
      pipeline.with(group).with(pass).draw(3);
    },
    update(time, orbit) {
      const angle = orbit ? Math.sin(time * 0.18) * 0.5 : 0;
      const height = 0.35 + (orbit ? Math.sin(time * 0.11) * 0.12 : 0);
      params.write({
        camera: d.vec4f(Math.sin(angle) * 1.4, height, 0.6 + Math.cos(angle) * 0.3, time),
        focus: d.vec4f(0, -0.15, -3, Math.tan((58 * Math.PI) / 180 / 2)),
        options: d.vec4f(1, 30, 0.9 + time * 0.03, 0),
      });
    },
    destroy() {
      /* the caller owns the render targets */
    },
  };
}
