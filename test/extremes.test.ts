/**
 * Every UI slider at both extremes, checked for NaN and infinity.
 *
 * The shading kernel divides by depths, lengths and cone widths that the user
 * can drive to their limits. A NaN in an HDR buffer propagates silently through
 * tone mapping and shows up as a black or white hole, so each slider is swept to
 * both ends (plus a few pathological combinations) and the HDR output is scanned
 * for non-finite values.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu, readBuffer } from './helpers/gpu.ts';
import { mulberry32 } from './helpers/tensors.ts';
import { uploadImage } from './helpers/textures.ts';
import { buildArchitecture } from '../src/model/arch.ts';
import { createDepthModel } from '../src/model/runner.ts';
import { synthesizeWeights } from '../src/model/weights.ts';
import {
  createRenderer,
  DEFAULT_SETTINGS,
  type ShadingSettings,
} from '../src/lighting/renderer.ts';
import { renderFrame } from '../src/engine/frame.ts';
import type { LightDescription } from '../src/lighting/lights.ts';
import { halfToFloat } from '../src/model/weights.ts';

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

// The three lights main.ts actually ships with.
const APP_LIGHTS: LightDescription[] = [
  { kind: 'point', position: [0.9, 0.5, -1.4], color: [1, 0.83, 0.62], intensity: 9, radius: 7, shadow: 1 },
  { kind: 'point', position: [-1.2, 0.1, -2.2], color: [0.35, 0.6, 1], intensity: 6, radius: 8, shadow: 0.8 },
  {
    kind: 'spot', position: [0, 1.6, -0.4], direction: [0, -1, -0.35],
    color: [1, 0.95, 0.9], intensity: 22, radius: 9,
    innerAngle: 0.28, outerAngle: 0.52, shadow: 1,
  },
];

async function readHdr(root: any, tex: any): Promise<Float32Array> {
  const bytesPerRow = OUT * 8;
  const staging = root.device.createBuffer({
    size: bytesPerRow * OUT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const enc = root.device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: root.unwrap(tex) },
    { buffer: staging, bytesPerRow, rowsPerImage: OUT },
    { width: OUT, height: OUT },
  );
  root.device.queue.submit([enc.finish()]);
  const raw = await readBuffer(root.device, staging, bytesPerRow * OUT);
  const u16 = new Uint16Array(raw);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]!);
  staging.destroy();
  return out;
}

async function readTargetBytes(root: any, target: any): Promise<Uint8Array> {
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

/** Every slider in main.ts, at both ends. */
const RANGES: Record<string, [number, number]> = {
  exposure: [0.1, 4],
  contrast: [0.6, 1.6],
  saturation: [0, 2],
  vignette: [0, 1],
  near: [0.05, 3],
  far: [2, 60],
  fovY: [(20 * Math.PI) / 180, (110 * Math.PI) / 180],
  normalStrength: [0.1, 4],
  ambientIntensity: [0, 2],
  shadowStrength: [0, 1],
  contactShadowLength: [0.05, 4],
  aoStrength: [0, 1],
  aoRadiusPixels: [2, 48],
  volumetric: [0, 2],
  fogDensity: [0, 0.25],
  roughness: [0.04, 1],
  specular: [0, 2],
  albedoMix: [0, 1],
};

describe('extreme UI settings', () => {
  it('sweeps every slider extreme and reports NaN/Inf', async () => {
    const { root } = await headlessGpu();
    const arch = buildArchitecture(SIZE);
    const source = uploadImage(root, SIZE, scene(SIZE));
    const model = createDepthModel(root, {
      arch,
      weights: synthesizeWeights(arch, 31337),
      source,
    });
    const renderer = createRenderer(root, {
      width: OUT, height: OUT, depthSize: SIZE,
      albedo: source,
      depthTextures: [model.debugDepth(0), model.debugDepth(1)],
      presentFormat: FORMAT,
    });
    renderer.setLights(APP_LIGHTS);
    const target = root.createTexture({ size: [OUT, OUT], format: FORMAT }).$usage('render');
    const view = target.createView('render');

    const cases: Array<{ name: string; s: ShadingSettings }> = [];
    // one-at-a-time
    for (const [key, [lo, hi]] of Object.entries(RANGES)) {
      for (const v of [lo, hi]) {
        cases.push({
          name: `${key}=${v}`,
          s: { ...DEFAULT_SETTINGS, [key]: v } as ShadingSettings,
        });
      }
    }
    // Adversarial combinations.
    cases.push({
      name: 'MIRROR: roughness=0.04 specular=2 exposure=4',
      s: { ...DEFAULT_SETTINGS, roughness: 0.04, specular: 2, exposure: 4 },
    });
    cases.push({
      name: 'MIRROR+near: roughness=0.04 specular=2 exposure=4 near=0.05 far=2',
      s: { ...DEFAULT_SETTINGS, roughness: 0.04, specular: 2, exposure: 4, near: 0.05, far: 2 },
    });
    cases.push({
      name: 'near>far: near=3 far=2',
      s: { ...DEFAULT_SETTINGS, near: 3, far: 2 },
    });
    cases.push({
      name: 'everything max',
      s: {
        ...DEFAULT_SETTINGS, exposure: 4, contrast: 1.6, saturation: 2, vignette: 1,
        near: 3, far: 60, fovY: (110 * Math.PI) / 180, normalStrength: 4,
        ambientIntensity: 2, shadowStrength: 1, contactShadowLength: 4,
        aoStrength: 1, aoRadiusPixels: 48, volumetric: 2, fogDensity: 0.25,
        roughness: 0.04, specular: 2, albedoMix: 1,
      },
    });
    cases.push({
      name: 'everything min',
      s: {
        ...DEFAULT_SETTINGS, exposure: 0.1, contrast: 0.6, saturation: 0, vignette: 0,
        near: 0.05, far: 2, fovY: (20 * Math.PI) / 180, normalStrength: 0.1,
        ambientIntensity: 0, shadowStrength: 0, contactShadowLength: 0.05,
        aoStrength: 0, aoRadiusPixels: 2, volumetric: 0, fogDensity: 0,
        roughness: 1, specular: 0, albedoMix: 0,
      },
    });

    const bad: string[] = [];
    for (const c of cases) {
      renderer.update(c.s, 0);
      renderFrame({ root, model, renderer, target: view });
      const hdr = await readHdr(root, renderer.hdr);
      let nan = 0, inf = 0, maxv = 0;
      for (let i = 0; i < hdr.length; i += 4) {
        for (let k = 0; k < 3; k++) {
          const v = hdr[i + k]!;
          if (Number.isNaN(v)) nan++;
          else if (!Number.isFinite(v)) inf++;
          else if (v > maxv) maxv = v;
        }
      }
      const px = await readTargetBytes(root, target);
      let black = 0;
      for (let i = 0; i < px.length; i += 4) {
        if (px[i] === 0 && px[i + 1] === 0 && px[i + 2] === 0) black++;
      }
      const line = `${c.name.padEnd(58)} hdrNaN=${nan} hdrInf=${inf} hdrMax=${maxv.toFixed(1)} blackPx=${black}/${OUT * OUT}`;
      console.log(line);
      if (nan > 0 || inf > 0) bad.push(line);
    }

    console.log('\n=== SUMMARY: cases producing NaN/Inf ===');
    for (const b of bad) console.log(b);
    expect(bad).toEqual([]);
  }, 600_000);
});
