/**
 * The blit shader that fits a camera frame or still image into the square input.
 *
 * This is the fix for the camera bug, so it is worth testing on the GPU rather
 * than only unit-testing the transform maths. The old path copied a 448×448
 * *window* out of the frame; the new one covers the full short axis and a
 * centred square of the long axis.
 *
 * The staging texture is filled with `writeTexture` rather than
 * `copyExternalImageToTexture`, because not every WebGPU backend implements
 * external-image import — the software backend used in CI does not. Everything
 * downstream of the upload, which is the part that was wrong, is still covered.
 */
import { describe, expect, it } from 'vitest';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { headlessGpu, readBuffer } from './helpers/gpu.ts';
import {
  BlitParams,
  blitFragment,
  blitLayout,
  blitVertex,
  createBlitter,
  cropTransform,
} from '../src/scene/blit.ts';

const OUT = 64;
const SRC_W = 320;
const SRC_H = 120;

/**
 * A wide test image: red in the leftmost 10 %, green in the rightmost 10 %,
 * white through the centred square, dark grey between. A correct centre crop
 * keeps the white and discards both coloured edges.
 */
function wideImage(): Uint8Array {
  const px = new Uint8Array(SRC_W * SRC_H * 4);
  const squareStart = (SRC_W - SRC_H) / 2;
  for (let y = 0; y < SRC_H; y++) {
    for (let x = 0; x < SRC_W; x++) {
      const i = (y * SRC_W + x) * 4;
      let r = 40, g = 40, b = 40;
      if (x < SRC_W * 0.1) { r = 255; g = 0; b = 0; }
      else if (x >= SRC_W * 0.9) { r = 0; g = 255; b = 0; }
      else if (x >= squareStart + 8 && x < squareStart + SRC_H - 8) { r = 255; g = 255; b = 255; }
      px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = 255;
    }
  }
  return px;
}

describe('blit framing', () => {
  it('covers the full short axis and crops the long axis symmetrically', async () => {
    const { root } = await headlessGpu();

    const staging = root
      .createTexture({ size: [SRC_W, SRC_H], format: 'rgba8unorm' })
      .$usage('sampled');
    const pixels = wideImage();
    root.device.queue.writeTexture(
      { texture: root.unwrap(staging) },
      pixels.buffer as ArrayBuffer,
      { offset: pixels.byteOffset, bytesPerRow: SRC_W * 4, rowsPerImage: SRC_H },
      { width: SRC_W, height: SRC_H },
    );

    const [sx, sy, ox, oy] = cropTransform(SRC_W, SRC_H);
    const params = root.createUniform(BlitParams, {
      transform: d.vec4f(sx, sy, ox, oy),
      options: d.vec4f(0, 0, 0, 0),
    });

    const color = root.createTexture({ size: [OUT, OUT], format: 'rgba8unorm' }).$usage('render');
    const truth = root.createTexture({ size: [OUT, OUT], format: 'rgba16float' }).$usage('render');

    const pipeline = root.createRenderPipeline({
      vertex: blitVertex,
      fragment: blitFragment,
      targets: { color: { format: 'rgba8unorm' }, depth: { format: 'rgba16float' } },
      primitive: { topology: 'triangle-list' },
    });
    const group = root.createBindGroup(blitLayout, {
      src: staging.createView(d.texture2d(d.f32)),
      samp: root.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
      params,
    });

    const encoder = root['~unstable'].createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: color.createView('render'), loadOp: 'clear', storeOp: 'store' },
        { view: truth.createView('render'), loadOp: 'clear', storeOp: 'store' },
      ],
    });
    pipeline.with(group).with(pass).draw(3);
    pass.end();
    encoder.submit();

    const bpr = Math.ceil((OUT * 4) / 256) * 256;
    const stagingBuf = root.device.createBuffer({
      size: bpr * OUT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const e = root.device.createCommandEncoder();
    e.copyTextureToBuffer(
      { texture: root.unwrap(color) },
      { buffer: stagingBuf, bytesPerRow: bpr, rowsPerImage: OUT },
      { width: OUT, height: OUT },
    );
    root.device.queue.submit([e.finish()]);
    const raw = new Uint8Array(await readBuffer(root.device, stagingBuf, bpr * OUT));
    stagingBuf.destroy();

    const at = (x: number, y: number) => {
      const i = y * bpr + x * 4;
      return [raw[i]!, raw[i + 1]!, raw[i + 2]!] as const;
    };

    // The coloured edges live outside the centred square and must be gone.
    let redPixels = 0;
    let greenPixels = 0;
    let whitePixels = 0;
    for (let y = 0; y < OUT; y++) {
      for (let x = 0; x < OUT; x++) {
        const [r, g, b] = at(x, y);
        if (r > 180 && g < 80 && b < 80) redPixels++;
        if (g > 180 && r < 80 && b < 80) greenPixels++;
        if (r > 200 && g > 200 && b > 200) whitePixels++;
      }
    }
    expect(redPixels, 'far-left band leaked into the crop').toBe(0);
    expect(greenPixels, 'far-right band leaked into the crop').toBe(0);
    // The centred square dominates the output, which is the whole point: the
    // old 448-window crop showed a much narrower slice.
    expect(whitePixels).toBeGreaterThan(OUT * OUT * 0.6);

    // The full height of the source must be present, so the top and bottom rows
    // are inside the image rather than clamped padding.
    const top = at(OUT >> 1, 0);
    const bottom = at(OUT >> 1, OUT - 1);
    expect(top[0]).toBeGreaterThan(200);
    expect(bottom[0]).toBeGreaterThan(200);
  });

  it('mirrors horizontally when asked', async () => {
    const { root } = await headlessGpu();
    // A square source split left/dark, right/bright: mirroring must swap them.
    const N = 32;
    const px = new Uint8Array(N * N * 4);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = (y * N + x) * 4;
        const v = x < N / 2 ? 20 : 230;
        px[i] = v; px[i + 1] = v; px[i + 2] = v; px[i + 3] = 255;
      }
    }
    const staging = root.createTexture({ size: [N, N], format: 'rgba8unorm' }).$usage('sampled');
    root.device.queue.writeTexture(
      { texture: root.unwrap(staging) },
      px.buffer as ArrayBuffer,
      { offset: px.byteOffset, bytesPerRow: N * 4, rowsPerImage: N },
      { width: N, height: N },
    );

    const read = async (mirror: boolean) => {
      const params = root.createUniform(BlitParams, {
        transform: d.vec4f(1, 1, 0, 0),
        options: d.vec4f(mirror ? 1 : 0, 0, 0, 0),
      });
      const color = root.createTexture({ size: [OUT, OUT], format: 'rgba8unorm' }).$usage('render');
      const truth = root.createTexture({ size: [OUT, OUT], format: 'rgba16float' }).$usage('render');
      const pipeline = root.createRenderPipeline({
        vertex: blitVertex,
        fragment: blitFragment,
        targets: { color: { format: 'rgba8unorm' }, depth: { format: 'rgba16float' } },
        primitive: { topology: 'triangle-list' },
      });
      const group = root.createBindGroup(blitLayout, {
        src: staging.createView(d.texture2d(d.f32)),
        samp: root.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
        params,
      });
      const enc = root['~unstable'].createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [
          { view: color.createView('render'), loadOp: 'clear', storeOp: 'store' },
          { view: truth.createView('render'), loadOp: 'clear', storeOp: 'store' },
        ],
      });
      pipeline.with(group).with(pass).draw(3);
      pass.end();
      enc.submit();

      const bpr = Math.ceil((OUT * 4) / 256) * 256;
      const buf = root.device.createBuffer({
        size: bpr * OUT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const e2 = root.device.createCommandEncoder();
      e2.copyTextureToBuffer(
        { texture: root.unwrap(color) },
        { buffer: buf, bytesPerRow: bpr, rowsPerImage: OUT },
        { width: OUT, height: OUT },
      );
      root.device.queue.submit([e2.finish()]);
      const bytes = new Uint8Array(await readBuffer(root.device, buf, bpr * OUT));
      buf.destroy();
      const row = (OUT >> 1) * bpr;
      return { left: bytes[row + 4 * 4]!, right: bytes[row + (OUT - 5) * 4]! };
    };

    const plain = await read(false);
    const mirrored = await read(true);
    expect(plain.left).toBeLessThan(80);
    expect(plain.right).toBeGreaterThan(180);
    expect(mirrored.left).toBeGreaterThan(180);
    expect(mirrored.right).toBeLessThan(80);
  });
});

/**
 * Not every browser accepts an `HTMLVideoElement` in
 * `copyExternalImageToTexture`. When it is refused the camera must keep
 * working through a 2-D canvas rather than going black, and — since the refusal
 * is permanent for that browser — the failing path must not be retried on every
 * frame.
 */
describe('external image import fallback', () => {
  it('latches onto the canvas path after one refusal and stops retrying', async () => {
    const { root } = await headlessGpu();
    const queue = root.device.queue;
    const original = queue.copyExternalImageToTexture.bind(queue);

    const fakeCtx = { drawImage: () => {} };
    let created = 0;
    const fakeCanvas = { width: 0, height: 0, getContext: () => fakeCtx };
    const previousDocument = (globalThis as Record<string, unknown>)['document'];
    (globalThis as Record<string, unknown>)['document'] = {
      createElement: (tag: string) => {
        if (tag !== 'canvas') throw new Error(`unexpected element ${tag}`);
        created++;
        return fakeCanvas;
      },
    };

    const attempts: string[] = [];
    try {
      Object.defineProperty(queue, 'copyExternalImageToTexture', {
        configurable: true,
        writable: true,
        value: (src: { source: unknown }) => {
          const viaCanvas = src.source === fakeCanvas;
          attempts.push(viaCanvas ? 'canvas' : 'direct');
          // Stands in for a browser that refuses the source type outright.
          if (!viaCanvas) throw new TypeError('unsupported source type');
        },
      });

      const blitter = createBlitter(root, 'rgba8unorm', 'rgba16float');
      expect(blitter.copyPath).toBe('none');

      const video = { nodeName: 'VIDEO' } as unknown as GPUCopyExternalImageSource;
      blitter.upload(video, 640, 480, true);
      expect(blitter.copyPath).toBe('canvas');
      expect(blitter.ready).toBe(true);
      // The scratch canvas is resized to the source, not the destination: the
      // scaling is the blit shader's job.
      expect([fakeCanvas.width, fakeCanvas.height]).toEqual([640, 480]);

      blitter.upload(video, 640, 480, true);
      blitter.upload(video, 640, 480, true);

      // One refused attempt, then only the working path.
      expect(attempts).toEqual(['direct', 'canvas', 'canvas', 'canvas']);
      // And exactly one scratch canvas for the whole session.
      expect(created).toBe(1);
      blitter.destroy();
    } finally {
      Object.defineProperty(queue, 'copyExternalImageToTexture', {
        configurable: true,
        writable: true,
        value: original,
      });
      if (previousDocument === undefined) {
        delete (globalThis as Record<string, unknown>)['document'];
      } else {
        (globalThis as Record<string, unknown>)['document'] = previousDocument;
      }
    }
  });
});
