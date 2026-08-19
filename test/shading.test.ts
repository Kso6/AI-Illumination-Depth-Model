/**
 * The shading kernel driven by a hand-authored depth map.
 *
 * Using the network's output to test the lighting confounds two things at once.
 * Here the depth map is written directly, so the geometry is known exactly and
 * the reconstructed normals, the light falloff and the specular response can be
 * checked against what that geometry should produce.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu, readBuffer } from './helpers/gpu.ts';
import { uploadImage } from './helpers/textures.ts';
import {
  createRenderer,
  DEFAULT_SETTINGS,
  type ShadingSettings,
} from '../src/lighting/renderer.ts';
import type { LightDescription } from '../src/lighting/lights.ts';
import { halfToFloat } from '../src/model/weights.ts';

const N = 64;
const FORMAT: GPUTextureFormat = 'rgba8unorm';

function f32ToF16(v: number): number {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  f[0] = v;
  const x = u[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let frac = x & 0x7fffff;
  if (exp <= 0) return sign;
  if (exp >= 0x1f) return sign | 0x7c00;
  return sign | (exp << 10) | (frac >>> 13);
}

describe('shade kernel, hand-authored depth', () => {
  it('flat plane facing the camera + head-on light: HDR sanity at UI extremes', async () => {
    const { root } = await headlessGpu();

    // Albedo: mid grey.
    const rgba = new Uint8Array(N * N * 4).fill(180);
    for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;
    const albedo = uploadImage(root, N, rgba);

    // Two depth textures, both a *constant* disparity => a plane parallel to the
    // image plane, normal exactly (0,0,1).
    const DISP = 0.5;
    const mk = () => {
      const t = root
        .createTexture({ size: [N, N], format: 'rgba16float' })
        .$usage('sampled', 'storage');
      const data = new Uint16Array(N * N * 4);
      for (let i = 0; i < N * N; i++) {
        data[i * 4] = f32ToF16(DISP);
        data[i * 4 + 1] = f32ToF16(DISP);
        data[i * 4 + 2] = 0;
        data[i * 4 + 3] = f32ToF16(1);
      }
      root.device.queue.writeTexture(
        { texture: root.unwrap(t) },
        data.buffer as ArrayBuffer,
        { bytesPerRow: N * 8, rowsPerImage: N },
        { width: N, height: N },
      );
      return t;
    };
    const dA = mk();
    const dB = mk();

    const renderer = createRenderer(root, {
      width: N, height: N, depthSize: N,
      albedo,
      depthTextures: [dA, dB],
      presentFormat: FORMAT,
    });

    // near=0.35, far=14 (defaults) -> invZ = 0.5*(1/0.35-1/14)+1/14 = 1.42857-0.0714... compute:
    const near = DEFAULT_SETTINGS.near, far = DEFAULT_SETTINGS.far;
    const invZ = DISP * (1 / near - 1 / far) + 1 / far;
    const z = 1 / invZ;
    console.log(`plane at view z = ${z.toFixed(4)} m (position = (0,0,${(-z).toFixed(4)}))`);

    // A point light exactly 1 m in front of the plane centre, on the view axis.
    // At the centre pixel: normal = (0,0,1), viewDir = (0,0,1), toLight = (0,0,1)
    // => nDotH = 1 exactly: the worst case for the GGX denominator.
    const lights: LightDescription[] = [
      {
        kind: 'point',
        position: [0, 0, -z + 1],
        color: [1, 1, 1],
        intensity: 9,
        radius: 7,
        shadow: 0,
      },
    ];
    renderer.setLights(lights);

    const target = root.createTexture({ size: [N, N], format: FORMAT }).$usage('render');
    const view = target.createView('render');

    const readHdr = async () => {
      const bytesPerRow = N * 8;
      const staging = root.device.createBuffer({
        size: bytesPerRow * N,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const enc = root.device.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: root.unwrap(renderer.hdr) },
        { buffer: staging, bytesPerRow, rowsPerImage: N },
        { width: N, height: N },
      );
      root.device.queue.submit([enc.finish()]);
      const raw = await readBuffer(root.device, staging, bytesPerRow * N);
      const u16 = new Uint16Array(raw);
      const out = new Float32Array(u16.length);
      for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]!);
      staging.destroy();
      return out;
    };
    const readTgt = async () => {
      const bytesPerRow = N * 4;
      const staging = root.device.createBuffer({
        size: bytesPerRow * N,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const enc = root.device.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: root.unwrap(target) },
        { buffer: staging, bytesPerRow, rowsPerImage: N },
        { width: N, height: N },
      );
      root.device.queue.submit([enc.finish()]);
      const raw = await readBuffer(root.device, staging, bytesPerRow * N);
      staging.destroy();
      return new Uint8Array(raw);
    };

    const run = async (name: string, s: ShadingSettings) => {
      renderer.update(s, 0);
      const enc = root['~unstable'].createCommandEncoder({ label: 'probe' });
      const cp = enc.beginComputePass({});
      renderer.recordShade(cp, 0);
      cp.end();
      const rp = enc.beginRenderPass({
        colorAttachments: [
          { view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        ],
      });
      renderer.recordComposite(rp);
      rp.end();
      enc.submit();
      const hdr = await readHdr();
      const px = await readTgt();
      let nan = 0, inf = 0, maxv = 0;
      for (let i = 0; i < hdr.length; i += 4) {
        for (let k = 0; k < 3; k++) {
          const v = hdr[i + k]!;
          if (Number.isNaN(v)) nan++;
          else if (!Number.isFinite(v)) inf++;
          else if (v > maxv) maxv = v;
        }
      }
      // centre pixel of the LDR image
      const ci = ((N / 2) * N + N / 2) * 4;
      let black = 0;
      for (let i = 0; i < px.length; i += 4) {
        if (px[i] === 0 && px[i + 1] === 0 && px[i + 2] === 0) black++;
      }
      const cHdr = ((N / 2) * N + N / 2) * 4;
      console.log(
        `${name.padEnd(52)} hdrNaN=${nan} hdrInf=${inf} hdrMax=${maxv.toFixed(1)} ` +
          `centreHdr=${hdr[cHdr]!.toFixed(1)} centreLdr=${px[ci]},${px[ci + 1]},${px[ci + 2]} blackPx=${black}`,
      );
      return { nan, inf, maxv, black, centreLdr: px[ci]! };
    };

    const results: Record<string, any> = {};
    results.default = await run('DEFAULT settings', DEFAULT_SETTINGS);
    results.mirror = await run('roughness=0.04 specular=2 exposure=4 (UI max)', {
      ...DEFAULT_SETTINGS, roughness: 0.04, specular: 2, exposure: 4,
    });
    results.mirrorDefaultExp = await run('roughness=0.04 specular=0.6 exposure=1.15', {
      ...DEFAULT_SETTINGS, roughness: 0.04,
    });
    results.rough = await run('roughness=1 specular=2', {
      ...DEFAULT_SETTINGS, roughness: 1, specular: 2,
    });

    // Also: what does a *flat* plane give for the normals debug view?
    const nrm = await run('normals view', { ...DEFAULT_SETTINGS, debug: 'normals' });
    console.log('normals-view centre LDR (expect 128,128,255):', nrm.centreLdr);

    console.log(JSON.stringify(results, null, 2));
    expect(results.mirror.nan + results.mirror.inf).toBe(0);
  }, 600_000);
});
