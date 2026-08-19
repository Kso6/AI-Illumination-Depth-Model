/**
 * Application entry point.
 *
 * Boots WebGPU, builds the depth network and the renderer, and runs the frame
 * loop. The interesting part is how short the loop is: everything a frame does
 * is recorded by `renderFrame`, in one command encoder.
 */
import tgpu from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { ILLUMINA_DEPTH_448, costOf } from './model/arch.ts';
import { createDepthModel, type DepthModel } from './model/runner.ts';
import { parseWeights, synthesizeWeights, type WeightMap } from './model/weights.ts';
import {
  createRenderer,
  DEFAULT_SETTINGS,
  type DebugView,
  type Renderer,
  type ShadingSettings,
  type ToneMap,
} from './lighting/renderer.ts';
import type { LightDescription } from './lighting/lights.ts';
import { createSceneSource, type SceneSource, type SourceMode } from './scene/source.ts';
import { renderFrame } from './engine/frame.ts';
import { GpuProfiler } from './engine/timing.ts';
import { evaluateDepth, type DepthMetrics } from './scene/metrics.ts';
import { buildPanel, type ControlGroup } from './ui/controls.ts';

const NETWORK_SIZE = ILLUMINA_DEPTH_448.inputSize;
const WEIGHTS_URL = 'weights/illumina-depth-448.idm';

function fail(message: string): never {
  const el = document.querySelector('#boot-error');
  if (el instanceof HTMLElement) {
    el.hidden = false;
    el.textContent = message;
  }
  throw new Error(message);
}

async function loadWeights(): Promise<{ weights: WeightMap; trained: boolean; note: string }> {
  try {
    const response = await fetch(WEIGHTS_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    // A dev server answers an unknown path with index.html rather than a 404,
    // so check that the body actually looks like a container before parsing —
    // otherwise the user sees a byte-offset error instead of "file missing".
    const type = response.headers.get('content-type') ?? '';
    if (type.includes('text/html')) {
      throw new Error('the server returned HTML, so the file is not deployed');
    }
    const buffer = await response.arrayBuffer();
    const weights = parseWeights(buffer, ILLUMINA_DEPTH_448);
    return {
      weights,
      trained: true,
      note: `Loaded ${WEIGHTS_URL} (${(buffer.byteLength / 1e6).toFixed(2)} MB)`,
    };
  } catch (err) {
    return {
      weights: synthesizeWeights(ILLUMINA_DEPTH_448, 1),
      trained: false,
      note:
        `No trained weights found at ${WEIGHTS_URL} (${(err as Error).message}). ` +
        'Running with Kaiming-initialised weights: the pipeline is exercised end to ' +
        'end, but the depth is an arbitrary smooth field, not a reconstruction. ' +
        'See tools/README.md to train and export real weights.',
    };
  }
}

interface AppState {
  settings: ShadingSettings;
  lights: LightDescription[];
  orbit: boolean;
  profile: boolean;
  paused: boolean;
  lightMotion: number;
}

async function main(): Promise<void> {
  const canvas = document.querySelector('#viewport');
  if (!(canvas instanceof HTMLCanvasElement)) fail('Missing #viewport canvas');

  if (!navigator.gpu) {
    fail(
      'WebGPU is not available in this browser. Chrome 113+, Edge 113+ or Safari 18+ ' +
        'on a machine with a supported GPU is required.',
    );
  }

  let root: TgpuRoot;
  try {
    root = await tgpu.init();
  } catch (err) {
    fail(`Could not acquire a GPU device: ${(err as Error).message}`);
  }

  const presentFormat = navigator.gpu.getPreferredCanvasFormat();
  const context = root.configureContext({
    canvas,
    format: presentFormat,
    alphaMode: 'opaque',
  });

  const { weights, trained, note } = await loadWeights();

  const source: SceneSource = createSceneSource(root, NETWORK_SIZE);

  const model: DepthModel = createDepthModel(root, {
    weights,
    source: source.texture,
    decodeSrgb: true,
    exposure: 1,
  });

  const state: AppState = {
    settings: { ...DEFAULT_SETTINGS },
    lights: [
      {
        kind: 'point',
        position: [0.9, 0.5, -1.4],
        color: [1, 0.83, 0.62],
        intensity: 9,
        radius: 7,
        shadow: 1,
      },
      {
        kind: 'point',
        position: [-1.2, 0.1, -2.2],
        color: [0.35, 0.6, 1],
        intensity: 6,
        radius: 8,
        shadow: 0.8,
      },
      {
        kind: 'spot',
        position: [0, 1.6, -0.4],
        direction: [0, -1, -0.35],
        color: [1, 0.95, 0.9],
        intensity: 22,
        radius: 9,
        innerAngle: 0.28,
        outerAngle: 0.52,
        shadow: 1,
      },
    ],
    orbit: true,
    profile: false,
    paused: false,
    lightMotion: 0.5,
  };

  const sized = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    return { w, h };
  };
  let { w, h } = sized();
  canvas.width = w;
  canvas.height = h;

  const renderer: Renderer = createRenderer(root, {
    width: w,
    height: h,
    depthSize: NETWORK_SIZE,
    albedo: source.texture,
    depthTextures: [model.debugDepth(0), model.debugDepth(1)],
    presentFormat,
  });
  renderer.setLights(state.lights);

  const profiler = new GpuProfiler(root.device, 4);

  // Created once: `createView` allocates, and the frame loop must not.
  const sceneColorView = source.texture.createView('render');
  const sceneTruthView = source.truth.createView('render');
  const scenePass = {
    record: (pass: Parameters<typeof source.record>[0]) => source.record(pass),
    colorTarget: sceneColorView,
    truthTarget: sceneTruthView,
  };

  // --- UI -----------------------------------------------------------------
  const statusEl = document.querySelector('#status');
  const statsEl = document.querySelector('#stats');
  const metricsEl = document.querySelector('#metrics');
  const panelEl = document.querySelector('#panel');
  if (
    !(statusEl instanceof HTMLElement) ||
    !(statsEl instanceof HTMLElement) ||
    !(metricsEl instanceof HTMLElement) ||
    !(panelEl instanceof HTMLElement)
  ) {
    fail('Missing UI elements');
  }

  statusEl.textContent = note;
  statusEl.classList.toggle('warn', !trained);

  const cost = costOf(ILLUMINA_DEPTH_448);
  let metrics: DepthMetrics | undefined;
  let evaluating = false;

  const s = state.settings;
  const groups: ControlGroup[] = [
    {
      title: 'Source',
      controls: [
        {
          kind: 'select',
          label: 'Input',
          options: ['procedural', 'camera', 'image'] as const,
          get: () => source.mode,
          set: (v) => {
            void source.setMode(v as SourceMode).then(() => {
              statusEl.textContent = source.label;
            });
          },
        },
        {
          kind: 'toggle',
          label: 'Animate scene',
          get: () => state.orbit,
          set: (v) => (state.orbit = v),
        },
        {
          kind: 'button',
          label: 'Load image…',
          onClick: () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.addEventListener('change', async () => {
              const file = input.files?.[0];
              if (!file) return;
              const bitmap = await createImageBitmap(file);
              source.loadImage(bitmap);
              statusEl.textContent = source.label;
              bitmap.close();
            });
            input.click();
          },
        },
      ],
    },
    {
      title: 'View',
      controls: [
        {
          kind: 'select',
          label: 'Show',
          options: ['lit', 'depth', 'normals', 'occlusion', 'albedo'] as const,
          get: () => s.debug,
          set: (v) => (s.debug = v as DebugView),
        },
        {
          kind: 'select',
          label: 'Tone map',
          options: ['aces', 'reinhard'] as const,
          get: () => s.toneMap,
          set: (v) => (s.toneMap = v as ToneMap),
        },
        { kind: 'slider', label: 'Exposure', min: 0.1, max: 4, step: 0.01, get: () => s.exposure, set: (v) => (s.exposure = v) },
        { kind: 'slider', label: 'Contrast', min: 0.6, max: 1.6, step: 0.01, get: () => s.contrast, set: (v) => (s.contrast = v) },
        { kind: 'slider', label: 'Saturation', min: 0, max: 2, step: 0.01, get: () => s.saturation, set: (v) => (s.saturation = v) },
        { kind: 'slider', label: 'Vignette', min: 0, max: 1, step: 0.01, get: () => s.vignette, set: (v) => (s.vignette = v) },
      ],
    },
    {
      title: 'Depth interpretation',
      controls: [
        { kind: 'slider', label: 'Near (m)', min: 0.05, max: 3, step: 0.01, get: () => s.near, set: (v) => (s.near = v) },
        { kind: 'slider', label: 'Far (m)', min: 2, max: 60, step: 0.1, get: () => s.far, set: (v) => (s.far = v) },
        { kind: 'slider', label: 'Field of view', min: 20, max: 110, step: 1, get: () => (s.fovY * 180) / Math.PI, set: (v) => (s.fovY = (v * Math.PI) / 180), format: (v) => `${v.toFixed(0)}°` },
        { kind: 'slider', label: 'Normal strength', min: 0.1, max: 4, step: 0.01, get: () => s.normalStrength, set: (v) => (s.normalStrength = v) },
      ],
    },
    {
      title: 'Lighting',
      controls: [
        { kind: 'slider', label: 'Light motion', min: 0, max: 2, step: 0.01, get: () => state.lightMotion, set: (v) => (state.lightMotion = v) },
        { kind: 'slider', label: 'Ambient', min: 0, max: 2, step: 0.01, get: () => s.ambientIntensity, set: (v) => (s.ambientIntensity = v) },
        { kind: 'slider', label: 'Contact shadows', min: 0, max: 1, step: 0.01, get: () => s.shadowStrength, set: (v) => (s.shadowStrength = v) },
        { kind: 'slider', label: 'Shadow reach (m)', min: 0.05, max: 4, step: 0.01, get: () => s.contactShadowLength, set: (v) => (s.contactShadowLength = v) },
        { kind: 'slider', label: 'Ambient occlusion', min: 0, max: 1, step: 0.01, get: () => s.aoStrength, set: (v) => (s.aoStrength = v) },
        { kind: 'slider', label: 'AO radius (px)', min: 2, max: 48, step: 1, get: () => s.aoRadiusPixels, set: (v) => (s.aoRadiusPixels = v) },
        { kind: 'slider', label: 'Volumetric', min: 0, max: 2, step: 0.01, get: () => s.volumetric, set: (v) => (s.volumetric = v) },
        { kind: 'slider', label: 'Fog density', min: 0, max: 0.25, step: 0.001, get: () => s.fogDensity, set: (v) => (s.fogDensity = v), format: (v) => v.toFixed(3) },
        { kind: 'slider', label: 'Roughness', min: 0.04, max: 1, step: 0.01, get: () => s.roughness, set: (v) => (s.roughness = v) },
        { kind: 'slider', label: 'Specular', min: 0, max: 2, step: 0.01, get: () => s.specular, set: (v) => (s.specular = v) },
        { kind: 'slider', label: 'Albedo mix', min: 0, max: 1, step: 0.01, get: () => s.albedoMix, set: (v) => (s.albedoMix = v) },
      ],
    },
    {
      title: 'Diagnostics',
      collapsed: true,
      controls: [
        {
          kind: 'toggle',
          label: 'Per-stage GPU timing',
          get: () => state.profile,
          set: (v) => (state.profile = v),
        },
        {
          kind: 'toggle',
          label: 'Pause',
          get: () => state.paused,
          set: (v) => (state.paused = v),
        },
        {
          kind: 'button',
          label: 'Score against ground truth',
          onClick: () => {
            if (!source.hasGroundTruth) {
              metricsEl.textContent =
                'Ground truth is only available for the procedural scene.';
              return;
            }
            if (evaluating) return;
            evaluating = true;
            metricsEl.textContent = 'Evaluating…';
            void evaluateDepth(root, model.currentDepth(), source.truth, NETWORK_SIZE)
              .then((m) => {
                metrics = m;
              })
              .catch((err: Error) => {
                metricsEl.textContent = `Evaluation failed: ${err.message}`;
              })
              .finally(() => {
                evaluating = false;
              });
          },
        },
      ],
    },
  ];
  buildPanel(panelEl, groups);

  // --- frame loop ---------------------------------------------------------
  let frame = 0;
  let last = performance.now();
  let smoothedFrameMs = 16;
  const start = performance.now();

  const loop = () => {
    requestAnimationFrame(loop);

    const now = performance.now();
    const dt = now - last;
    last = now;
    smoothedFrameMs += (dt - smoothedFrameMs) * 0.08;

    const next = sized();
    if (next.w !== w || next.h !== h) {
      w = next.w;
      h = next.h;
      canvas.width = w;
      canvas.height = h;
      renderer.resize(w, h);
    }

    if (state.paused) return;

    const time = (now - start) / 1000;
    source.update(time, state.orbit);

    // Orbit the two point lights so the depth-driven shading is obvious.
    if (state.lightMotion > 0) {
      const a = time * 0.6 * state.lightMotion;
      state.lights[0] = {
        ...state.lights[0]!,
        position: [Math.cos(a) * 1.5, 0.45 + Math.sin(a * 0.7) * 0.3, -1.6 + Math.sin(a) * 0.7],
      };
      state.lights[1] = {
        ...state.lights[1]!,
        position: [
          Math.cos(a + Math.PI) * 1.7,
          0.1 + Math.cos(a * 0.9) * 0.4,
          -2.2 + Math.sin(a + Math.PI) * 0.8,
        ],
      };
      renderer.setLights(state.lights);
    }

    renderer.update(state.settings, frame);

    const result = renderFrame({
      root,
      model,
      renderer,
      target: context,
      scene: source.needsScenePass ? scenePass : undefined,
      profiler,
      profile: state.profile,
    });

    frame++;

    if ((frame & 7) === 0) {
      const gpu = profiler.latest;
      const gpuLines = gpu.length
        ? gpu.map((t) => `  ${t.label.padEnd(18)} ${t.ms.toFixed(2)} ms`).join('\n')
        : '  (enable per-stage GPU timing)';
      statsEl.textContent = [
        `${(1000 / smoothedFrameMs).toFixed(0)} fps   ${smoothedFrameMs.toFixed(1)} ms/frame`,
        '',
        `network      ${ILLUMINA_DEPTH_448.name}`,
        `resolution   ${NETWORK_SIZE}×${NETWORK_SIZE} → ${w}×${h}`,
        `parameters   ${(cost.totalParams / 1e6).toFixed(2)} M`,
        `work         ${(cost.totalMacs * 2e-9).toFixed(2)} GFLOP / frame`,
        `dispatches   ${result.dispatches}  (${cost.dispatches} inference + 1 shading)`,
        `draws        ${result.draws}`,
        `encoders     ${result.encoders}   submits ${result.submits}`,
        `activations  ${(model.stats.arenaBytes / 1e6).toFixed(1)} MB ` +
          `(${(model.stats.naiveArenaBytes / 1e6).toFixed(1)} MB unaliased)`,
        `weights      ${(model.stats.weightBytes / 1e6).toFixed(2)} MB`,
        '',
        'GPU time',
        gpuLines,
      ].join('\n');

      if (metrics) {
        metricsEl.textContent = [
          `samples  ${metrics.samples}`,
          `AbsRel   ${metrics.absRel.toFixed(4)}   (lower is better)`,
          `RMSE     ${metrics.rmse.toFixed(3)} m`,
          `δ<1.25   ${(metrics.delta1 * 100).toFixed(1)} %`,
          `δ<1.25²  ${(metrics.delta2 * 100).toFixed(1)} %`,
          `δ<1.25³  ${(metrics.delta3 * 100).toFixed(1)} %`,
          `fit      scale ${metrics.scale.toFixed(3)}  shift ${metrics.shift.toFixed(3)}`,
        ].join('\n');
      }
    }
  };

  requestAnimationFrame(loop);

  // A small introspection hook. Headless browsers cannot screenshot a WebGPU
  // canvas, so this is how automated checks confirm the pipeline is producing
  // pixels rather than merely running without errors.
  (window as unknown as Record<string, unknown>).illumina = {
    root,
    model,
    renderer,
    source,
    settings: state.settings,
    stats: () => model.stats,
    /** Reads the HDR lighting buffer back and returns simple statistics. */
    async sampleHdr(): Promise<{ mean: number; min: number; max: number; nonZero: number }> {
      const size = 64;
      const bytesPerRow = Math.ceil((size * 8) / 256) * 256;
      const staging = root.device.createBuffer({
        size: bytesPerRow * size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = root.device.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: root.unwrap(renderer.hdr) },
        { buffer: staging, bytesPerRow, rowsPerImage: size },
        { width: size, height: size },
      );
      root.device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const half = new Uint16Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();

      const decode = (h: number) => {
        const sign = (h & 0x8000) >> 15;
        const exp = (h & 0x7c00) >> 10;
        const frac = h & 0x03ff;
        const v =
          exp === 0 ? frac * 2 ** -24 : exp === 0x1f ? NaN : (frac + 1024) * 2 ** (exp - 25);
        return sign ? -v : v;
      };
      let sum = 0;
      let min = Infinity;
      let max = -Infinity;
      let nonZero = 0;
      let n = 0;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const i = (y * (bytesPerRow / 2)) + x * 4;
          const lum = (decode(half[i]!) + decode(half[i + 1]!) + decode(half[i + 2]!)) / 3;
          sum += lum;
          min = Math.min(min, lum);
          max = Math.max(max, lum);
          if (lum > 1e-4) nonZero++;
          n++;
        }
      }
      return { mean: sum / n, min, max, nonZero: nonZero / n };
    },
  };

  window.addEventListener('beforeunload', () => {
    source.stop();
    renderer.destroy();
    model.destroy();
    profiler.destroy();
  });
}

void main();
