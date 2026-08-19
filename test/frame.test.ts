/**
 * Whole-frame verification: inference, depth-aware shading and the tone-mapping
 * draw, recorded into a single command encoder and executed on a real device.
 *
 * These tests are about the *pipeline*, not the numerics of the network (which
 * `network.test.ts` covers): that the passes are compatible, that the depth map
 * flows from a compute kernel into a shading kernel without leaving the GPU,
 * that the ping-pong parity is right across frames, and that the lighting
 * actually responds to depth.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu } from './helpers/gpu.ts';
import { mulberry32 } from './helpers/tensors.ts';
import { uploadImage } from './helpers/textures.ts';
import { readBuffer } from './helpers/gpu.ts';
import { buildArchitecture } from '../src/model/arch.ts';
import { createDepthModel } from '../src/model/runner.ts';
import { synthesizeWeights } from '../src/model/weights.ts';
import { createRenderer, DEFAULT_SETTINGS, type ShadingSettings } from '../src/lighting/renderer.ts';
import { renderFrame } from '../src/engine/frame.ts';
import { GpuProfiler } from '../src/engine/timing.ts';
import type { LightDescription } from '../src/lighting/lights.ts';
import { createSceneSource } from '../src/scene/source.ts';

const SIZE = 64;
const OUT = 64;
const FORMAT: GPUTextureFormat = 'rgba8unorm';

function scene(size: number): Uint8Array {
  const rand = mulberry32(4242);
  const px = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      px[i] = Math.floor(60 + (x / size) * 180);
      px[i + 1] = Math.floor(70 + (y / size) * 150);
      px[i + 2] = Math.floor(90 + rand() * 40);
      px[i + 3] = 255;
    }
  }
  return px;
}

interface Harness {
  target: ReturnType<typeof makeTarget>;
  model: ReturnType<typeof createDepthModel>;
  renderer: ReturnType<typeof createRenderer>;
  root: Awaited<ReturnType<typeof headlessGpu>>['root'];
}

function makeTarget(root: Awaited<ReturnType<typeof headlessGpu>>['root']) {
  return root.createTexture({ size: [OUT, OUT], format: FORMAT }).$usage('render');
}

async function harness(lights: readonly LightDescription[]): Promise<Harness> {
  const { root } = await headlessGpu();
  const arch = buildArchitecture(SIZE);
  const source = uploadImage(root, SIZE, scene(SIZE));
  const model = createDepthModel(root, {
    arch,
    weights: synthesizeWeights(arch, 31337),
    source,
  });
  const renderer = createRenderer(root, {
    width: OUT,
    height: OUT,
    depthSize: SIZE,
    albedo: source,
    depthTextures: [model.debugDepth(0), model.debugDepth(1)],
    presentFormat: FORMAT,
  });
  renderer.setLights(lights);
  renderer.update(DEFAULT_SETTINGS, 0);
  return { target: makeTarget(root), model, renderer, root };
}

/** Reads the rendered `rgba8unorm` target back as bytes. */
async function readTarget(
  root: Awaited<ReturnType<typeof headlessGpu>>['root'],
  target: ReturnType<typeof makeTarget>,
): Promise<Uint8Array> {
  const bytesPerRow = OUT * 4;
  const staging = root.device.createBuffer({
    size: bytesPerRow * OUT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const enc = root.device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: root.unwrap(target) },
    { buffer: staging, bytesPerRow, rowsPerImage: OUT },
    { width: OUT, height: OUT },
  );
  root.device.queue.submit([enc.finish()]);
  const raw = await readBuffer(root.device, staging, bytesPerRow * OUT);
  staging.destroy();
  return new Uint8Array(raw);
}

const KEY_LIGHT: LightDescription[] = [
  {
    kind: 'point',
    position: [0.6, 0.4, -1.2],
    color: [1, 0.86, 0.7],
    intensity: 14,
    radius: 6,
    shadow: 1,
  },
];

describe('frame graph', () => {
  it('renders a lit frame through one encoder and one submit', async () => {
    const { root, model, renderer, target } = await harness(KEY_LIGHT);

    const result = renderFrame({
      root,
      model,
      renderer,
      target: target.createView('render'),
    });

    expect(result.encoders).toBe(1);
    expect(result.submits).toBe(1);
    // 40 inference dispatches + 1 shading dispatch.
    expect(result.dispatches).toBe(41);

    const pixels = await readTarget(root, target);
    let nonBlack = 0;
    let sum = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const v = pixels[i]! + pixels[i + 1]! + pixels[i + 2]!;
      if (v > 0) nonBlack++;
      sum += v;
    }
    expect(nonBlack).toBeGreaterThan(OUT * OUT * 0.9);
    // Not a uniform flat colour.
    const mean = sum / (OUT * OUT * 3);
    expect(mean).toBeGreaterThan(2);
    expect(mean).toBeLessThan(253);

    model.destroy();
    renderer.destroy();
  });

  it('alternates the depth ping-pong so history is never read and written at once', async () => {
    const { root, model, renderer, target } = await harness(KEY_LIGHT);
    const view = target.createView('render');

    const seen: number[] = [];
    for (let frame = 0; frame < 4; frame++) {
      seen.push(model.outputParity());
      renderer.update(DEFAULT_SETTINGS, frame);
      renderFrame({ root, model, renderer, target: view });
    }
    expect(seen).toEqual([1, 0, 1, 0]);

    model.destroy();
    renderer.destroy();
  });

  it('produces different images for different light positions', async () => {
    const { root, model, renderer, target } = await harness(KEY_LIGHT);
    const view = target.createView('render');

    renderFrame({ root, model, renderer, target: view });
    const left = await readTarget(root, target);

    renderer.setLights([
      { ...KEY_LIGHT[0]!, position: [-0.9, -0.3, -0.8], color: [0.5, 0.7, 1] },
    ]);
    renderFrame({ root, model, renderer, target: view });
    const right = await readTarget(root, target);

    let differing = 0;
    for (let i = 0; i < left.length; i++) {
      if (Math.abs(left[i]! - right[i]!) > 2) differing++;
    }
    expect(differing).toBeGreaterThan(left.length * 0.2);

    model.destroy();
    renderer.destroy();
  });

  it('debug views select different outputs', async () => {
    const { root, model, renderer, target } = await harness(KEY_LIGHT);
    const view = target.createView('render');

    const images = new Map<string, Uint8Array>();
    for (const debug of ['lit', 'depth', 'normals', 'occlusion'] as const) {
      const settings: ShadingSettings = { ...DEFAULT_SETTINGS, debug };
      renderer.update(settings, 0);
      renderFrame({ root, model, renderer, target: view });
      images.set(debug, await readTarget(root, target));
    }

    const names = [...images.keys()];
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const a = images.get(names[i]!)!;
        const b = images.get(names[j]!)!;
        let differing = 0;
        for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) differing++;
        expect(differing, `${names[i]} vs ${names[j]} are identical`).toBeGreaterThan(0);
      }
    }

    model.destroy();
    renderer.destroy();
  });

  it('splits into timed passes when profiling', async () => {
    const { root, model, renderer, target } = await harness(KEY_LIGHT);
    const profiler = new GpuProfiler(root.device, 4);
    const view = target.createView('render');

    for (let i = 0; i < 6; i++) {
      renderer.update(DEFAULT_SETTINGS, i);
      renderFrame({ root, model, renderer, target: view, profiler, profile: true });
      await root.device.queue.onSubmittedWorkDone();
    }

    if (profiler.available) {
      // Timings arrive a few frames late; allow the ring to drain.
      for (let i = 0; i < 8 && profiler.latest.length === 0; i++) {
        await new Promise((r) => setTimeout(r, 20));
      }
      expect(profiler.latest.map((s) => s.label)).toEqual([
        'inference',
        'shading',
        'composite',
      ]);
      for (const span of profiler.latest) expect(span.ms).toBeGreaterThan(0);
    }

    profiler.destroy();
    model.destroy();
    renderer.destroy();
  });
});

describe('procedural scene through the whole pipeline', () => {
  it('renders scene, depth, lighting and composite in one encoder', async () => {
    const { root } = await headlessGpu();
    const arch = buildArchitecture(SIZE);

    const source = createSceneSource(root, SIZE);
    const model = createDepthModel(root, {
      arch,
      weights: synthesizeWeights(arch, 8080),
      source: source.texture,
    });
    const renderer = createRenderer(root, {
      width: OUT,
      height: OUT,
      depthSize: SIZE,
      albedo: source.texture,
      depthTextures: [model.debugDepth(0), model.debugDepth(1)],
      presentFormat: FORMAT,
    });
    renderer.setLights(KEY_LIGHT);

    const target = root.createTexture({ size: [OUT, OUT], format: FORMAT }).$usage('render');
    const view = target.createView('render');

    source.update(2.0, false);
    renderer.update(DEFAULT_SETTINGS, 0);

    const result = renderFrame({
      root,
      model,
      renderer,
      target: view,
      scene: {
        record: (pass) => source.record(pass),
        colorTarget: source.texture.createView('render'),
        truthTarget: source.truth.createView('render'),
      },
    });

    // The scene draw plus the composite draw.
    expect(result.draws).toBe(2);
    expect(result.encoders).toBe(1);
    expect(result.submits).toBe(1);

    const pixels = await readTarget(root, target);
    let sum = 0;
    let max = 0;
    const distinct = new Set<number>();
    for (let i = 0; i < pixels.length; i += 4) {
      const v = (pixels[i]! + pixels[i + 1]! + pixels[i + 2]!) / 3;
      sum += v;
      max = Math.max(max, v);
      distinct.add(pixels[i]!);
    }
    const mean = sum / (OUT * OUT);
    expect(mean, 'composite is black').toBeGreaterThan(4);
    expect(max, 'composite has no highlights').toBeGreaterThan(30);
    expect(distinct.size, 'composite is a flat colour').toBeGreaterThan(16);

    source.stop();
    model.destroy();
    renderer.destroy();
  });
});
