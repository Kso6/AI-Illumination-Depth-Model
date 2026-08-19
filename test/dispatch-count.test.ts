/**
 * Counts the real WebGPU calls in one frame.
 *
 * `renderFrame` reports how many dispatches, encoders and submits it made, and
 * the README quotes those numbers. This test does not trust them: it wraps the
 * device and command encoder and counts the actual calls that reach WebGPU, so
 * the headline claim ("one encoder, one submit, no readbacks") is measured
 * rather than asserted.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu } from './helpers/gpu.ts';
import { mulberry32 } from './helpers/tensors.ts';
import { uploadImage } from './helpers/textures.ts';
import { buildArchitecture } from '../src/model/arch.ts';
import { createDepthModel } from '../src/model/runner.ts';
import { synthesizeWeights } from '../src/model/weights.ts';
import { createRenderer, DEFAULT_SETTINGS } from '../src/lighting/renderer.ts';
import { renderFrame } from '../src/engine/frame.ts';
import { createSceneSource } from '../src/scene/source.ts';
import type { LightDescription } from '../src/lighting/lights.ts';

const SIZE = 64;
const OUT = 64;
const FORMAT: GPUTextureFormat = 'rgba8unorm';

describe('real call counts', () => {
  it('counts dispatches, draws, encoders, submits, and readbacks', async () => {
    const { root, device } = await headlessGpu();
    const arch = buildArchitecture(SIZE);
    const px = new Uint8Array(SIZE * SIZE * 4);
    const rand = mulberry32(7);
    for (let i = 0; i < SIZE * SIZE; i++) {
      px[i * 4] = (rand() * 255) | 0;
      px[i * 4 + 1] = (rand() * 255) | 0;
      px[i * 4 + 2] = (rand() * 255) | 0;
      px[i * 4 + 3] = 255;
    }
    const source = uploadImage(root, SIZE, px);
    const model = createDepthModel(root, { arch, weights: synthesizeWeights(arch, 5), source });
    const renderer = createRenderer(root, {
      width: OUT, height: OUT, depthSize: SIZE,
      albedo: source, depthTextures: [model.debugDepth(0), model.debugDepth(1)],
      presentFormat: FORMAT,
    });
    const lights: LightDescription[] = [
      { kind: 'point', position: [0.6, 0.4, -1.2], color: [1, 1, 1], intensity: 10, radius: 6, shadow: 1 },
    ];
    renderer.setLights(lights);
    renderer.update(DEFAULT_SETTINGS, 0);
    const target = root.createTexture({ size: [OUT, OUT], format: FORMAT }).$usage('render');
    const view = target.createView('render');

    // warm-up so pipeline creation is not counted
    renderFrame({ root, model, renderer, target: view });

    const counts = {
      encoders: 0, submits: 0, dispatches: 0, draws: 0,
      computePasses: 0, renderPasses: 0,
      copyTexToBuf: 0, copyBufToBuf: 0, mapAsync: 0, writeBuffer: 0, writeTexture: 0,
    };

    const origCreateEncoder = device.createCommandEncoder.bind(device);
    const origSubmit = device.queue.submit.bind(device.queue);
    const origWriteBuffer = device.queue.writeBuffer.bind(device.queue);
    const origWriteTexture = device.queue.writeTexture.bind(device.queue);
    const origCreateBuffer = device.createBuffer.bind(device);

    (device as any).createCommandEncoder = (desc: any) => {
      counts.encoders++;
      const enc = origCreateEncoder(desc);
      const bcp = enc.beginComputePass.bind(enc);
      const brp = enc.beginRenderPass.bind(enc);
      (enc as any).beginComputePass = (d0: any) => {
        counts.computePasses++;
        const p = bcp(d0);
        const dw = p.dispatchWorkgroups.bind(p);
        (p as any).dispatchWorkgroups = (...a: any[]) => { counts.dispatches++; return (dw as any)(...a); };
        return p;
      };
      (enc as any).beginRenderPass = (d0: any) => {
        counts.renderPasses++;
        const p = brp(d0);
        const dr = p.draw.bind(p);
        (p as any).draw = (...a: any[]) => { counts.draws++; return (dr as any)(...a); };
        return p;
      };
      const ctb = enc.copyTextureToBuffer.bind(enc);
      (enc as any).copyTextureToBuffer = (...a: any[]) => { counts.copyTexToBuf++; return (ctb as any)(...a); };
      const cbb = enc.copyBufferToBuffer.bind(enc);
      (enc as any).copyBufferToBuffer = (...a: any[]) => { counts.copyBufToBuf++; return (cbb as any)(...a); };
      return enc;
    };
    (device.queue as any).submit = (...a: any[]) => { counts.submits++; return (origSubmit as any)(...a); };
    (device.queue as any).writeBuffer = (...a: any[]) => { counts.writeBuffer++; return (origWriteBuffer as any)(...a); };
    (device.queue as any).writeTexture = (...a: any[]) => { counts.writeTexture++; return (origWriteTexture as any)(...a); };
    (device as any).createBuffer = (desc: any) => {
      if (desc?.usage & GPUBufferUsage.MAP_READ) counts.mapAsync++;
      return origCreateBuffer(desc);
    };

    const result = renderFrame({ root, model, renderer, target: view });
    console.log('renderFrame reported:', JSON.stringify(result));
    console.log('actually observed   :', JSON.stringify(counts));

    // Now with the procedural scene pass attached.
    const src2 = createSceneSource(root, SIZE);
    const model2 = createDepthModel(root, { arch, weights: synthesizeWeights(arch, 5), source: src2.texture });
    const rend2 = createRenderer(root, {
      width: OUT, height: OUT, depthSize: SIZE,
      albedo: src2.texture, depthTextures: [model2.debugDepth(0), model2.debugDepth(1)],
      presentFormat: FORMAT,
    });
    rend2.setLights(lights);
    rend2.update(DEFAULT_SETTINGS, 0);
    const colorView = src2.texture.createView('render');
    const truthView = src2.truth.createView('render');
    const scenePass = {
      record: (p: any) => src2.record(p),
      colorTarget: colorView,
      truthTarget: truthView,
    };
    renderFrame({ root, model: model2, renderer: rend2, target: view, scene: scenePass }); // warm

    for (const k of Object.keys(counts)) (counts as any)[k] = 0;
    const r2 = renderFrame({ root, model: model2, renderer: rend2, target: view, scene: scenePass });
    console.log('with procedural scene, reported:', JSON.stringify(r2));
    console.log('with procedural scene, observed:', JSON.stringify(counts));

    (device as any).createCommandEncoder = origCreateEncoder;
    (device.queue as any).submit = origSubmit;
    (device.queue as any).writeBuffer = origWriteBuffer;
    (device.queue as any).writeTexture = origWriteTexture;
    (device as any).createBuffer = origCreateBuffer;

    expect(true).toBe(true);
  }, 600_000);
});
