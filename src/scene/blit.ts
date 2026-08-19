/**
 * Scaling a camera frame or a still image into the network's square input.
 *
 * `copyExternalImageToTexture` cannot scale — it copies texels one for one. An
 * earlier version therefore copied a 448×448 *window* out of the middle of the
 * frame, which on a 1280×720 webcam is a heavily zoomed crop of roughly a third
 * of the picture rather than the whole view.
 *
 * So the frame is copied at its native size into a staging texture and then
 * blitted through a fragment shader, which both scales it and applies an
 * aspect-preserving centre crop. The blit is a single full-screen triangle
 * recorded into the frame's command encoder, so it costs one draw and keeps the
 * "everything in one encoder" property intact.
 */
import tgpu from 'typegpu';
import type { TgpuRenderPass, TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';

export const BlitParams = d.struct({
  /** `xy` = UV scale, `zw` = UV offset: together an aspect-preserving centre crop. */
  transform: d.vec4f,
  /** `x` = 1 to mirror horizontally (front cameras are conventionally mirrored). */
  options: d.vec4f,
});

export const blitLayout = tgpu
  .bindGroupLayout({
    src: { texture: d.texture2d(d.f32) },
    samp: { sampler: 'filtering' },
    params: { uniform: BlitParams },
  })
  .$name('blit');

const B = blitLayout.$;

export const blitVertex = tgpu.vertexFn({
  in: { idx: d.builtin.vertexIndex },
  out: { pos: d.builtin.position, uv: d.vec2f },
})((input) => {
  'use gpu';
  const x = d.f32((input.idx << 1) & 2) * 2 - 1;
  const y = d.f32(input.idx & 2) * 2 - 1;
  return { pos: d.vec4f(x, y, 0, 1), uv: d.vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5) };
});

/**
 * Writes two targets so this pass can stand in for the procedural scene's pass,
 * which has the same attachments. The second carries no ground truth — a camera
 * frame has none — and is marked invalid so the metrics refuse to score it.
 */
export const blitFragment = tgpu.fragmentFn({
  in: { uv: d.vec2f },
  out: { color: d.vec4f, depth: d.vec4f },
})((input) => {
  'use gpu';
  const t = B.params.transform;
  let u = input.uv.x;
  if (B.params.options.x > 0.5) {
    u = 1 - u;
  }
  const uv = d.vec2f(u * t.x + t.z, input.uv.y * t.y + t.w);
  const c = std.textureSampleLevel(B.src, B.samp, uv, 0);
  return {
    color: d.vec4f(c.x, c.y, c.z, 1),
    // `y = 0` marks "no ground truth here".
    depth: d.vec4f(0, 0, 0, 1),
  };
});

/**
 * Aspect-preserving centre crop: the largest square of the source, mapped onto
 * the whole square destination.
 */
export function cropTransform(width: number, height: number): [number, number, number, number] {
  if (width <= 0 || height <= 0) return [1, 1, 0, 0];
  const scaleX = width > height ? height / width : 1;
  const scaleY = height > width ? width / height : 1;
  return [scaleX, scaleY, (1 - scaleX) / 2, (1 - scaleY) / 2];
}

export interface Blitter {
  /** Uploads a frame at its native size, recreating the staging texture on resize. */
  upload(source: GPUCopyExternalImageSource, width: number, height: number, mirror: boolean): void;
  record(pass: TgpuRenderPass): void;
  /** True once at least one frame has been uploaded. */
  readonly ready: boolean;
  destroy(): void;
}

export function createBlitter(
  root: TgpuRoot,
  colorFormat: GPUTextureFormat,
  truthFormat: GPUTextureFormat,
): Blitter {
  const params = root.createUniform(BlitParams, {
    transform: d.vec4f(1, 1, 0, 0),
    options: d.vec4f(0, 0, 0, 0),
  });
  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const pipeline = root.createRenderPipeline({
    vertex: blitVertex,
    fragment: blitFragment,
    targets: { color: { format: colorFormat }, depth: { format: truthFormat } },
    primitive: { topology: 'triangle-list' },
  });

  let staging: ReturnType<typeof createStaging> | undefined;
  let group: ReturnType<typeof root.createBindGroup> | undefined;
  let stagingW = 0;
  let stagingH = 0;
  let ready = false;
  let lastMirror: boolean | undefined;
  /** Set when a uniform write failed, so the next frame retries it. */
  let transformDirty = true;

  function createStaging(w: number, h: number) {
    return root
      .createTexture({ size: [w, h], format: colorFormat })
      .$usage('sampled', 'render');
  }

  return {
    upload(source, width, height, mirror) {
      if (width <= 0 || height <= 0) return;
      const resized = !staging || width !== stagingW || height !== stagingH;
      if (resized) {
        staging?.destroy();
        const next = createStaging(width, height);
        staging = next;
        stagingW = width;
        stagingH = height;
        group = root.createBindGroup(blitLayout, {
          src: next.createView(d.texture2d(d.f32)),
          samp: sampler,
          params,
        });
      }
      if (!staging) return;
      // Only touch the uniform when the framing actually changes. Rewriting it
      // on every frame allocates a staging buffer per write for a value that is
      // constant for the whole session.
      if (resized || mirror !== lastMirror || transformDirty) {
        const [sx, sy, ox, oy] = cropTransform(width, height);
        try {
          params.write({
            transform: d.vec4f(sx, sy, ox, oy),
            options: d.vec4f(mirror ? 1 : 0, 0, 0, 0),
          });
          lastMirror = mirror;
          transformDirty = false;
        } catch {
          // Leaving this set makes the next frame retry rather than stranding
          // the blit on a stale transform for the rest of the session.
          transformDirty = true;
        }
      }
      root.device.queue.copyExternalImageToTexture(
        { source },
        { texture: root.unwrap(staging) },
        { width, height },
      );
      ready = true;
    },

    record(pass) {
      if (!group) return;
      pipeline.with(group).with(pass).draw(3);
    },

    get ready() {
      return ready;
    },

    destroy() {
      staging?.destroy();
      staging = undefined;
      ready = false;
    },
  };
}
