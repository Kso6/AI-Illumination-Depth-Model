/**
 * The procedural scene must actually draw. It is the default source, so if it
 * renders black the whole application renders black.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu, readBuffer } from './helpers/gpu.ts';
import { createProceduralScene } from '../src/scene/procedural.ts';
import { halfToFloat } from '../src/model/weights.ts';

const SIZE = 64;

describe('procedural scene', () => {
  it('renders a non-trivial image and matching ground-truth depth', async () => {
    const { root } = await headlessGpu();
    const color = root
      .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
      .$usage('sampled', 'render');
    const truth = root
      .createTexture({ size: [SIZE, SIZE], format: 'rgba16float' })
      .$usage('sampled', 'render');

    const scene = createProceduralScene(root, 'rgba8unorm', 'rgba16float');
    scene.update(1.5, false);

    const encoder = root['~unstable'].createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: color.createView('render'), loadOp: 'clear', storeOp: 'store' },
        { view: truth.createView('render'), loadOp: 'clear', storeOp: 'store' },
      ],
    });
    scene.record(pass);
    pass.end();
    encoder.submit();

    // --- colour ---
    const colorBytes = await (async () => {
      const bpr = Math.ceil((SIZE * 4) / 256) * 256;
      const staging = root.device.createBuffer({
        size: bpr * SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const e = root.device.createCommandEncoder();
      e.copyTextureToBuffer(
        { texture: root.unwrap(color) },
        { buffer: staging, bytesPerRow: bpr, rowsPerImage: SIZE },
        { width: SIZE, height: SIZE },
      );
      root.device.queue.submit([e.finish()]);
      const raw = await readBuffer(root.device, staging, bpr * SIZE);
      staging.destroy();
      return { data: new Uint8Array(raw), stride: bpr };
    })();

    let sum = 0;
    let min = 255;
    let max = 0;
    let distinct = new Set<number>();
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const i = y * colorBytes.stride + x * 4;
        const v = (colorBytes.data[i]! + colorBytes.data[i + 1]! + colorBytes.data[i + 2]!) / 3;
        sum += v;
        min = Math.min(min, v);
        max = Math.max(max, v);
        distinct.add(colorBytes.data[i]!);
      }
    }
    const mean = sum / (SIZE * SIZE);
    expect(mean, 'scene rendered black').toBeGreaterThan(8);
    expect(max, 'scene has no bright pixels').toBeGreaterThan(40);
    expect(distinct.size, 'scene is a flat colour').toBeGreaterThan(8);

    // --- ground-truth depth ---
    const depthData = await (async () => {
      const bpr = Math.ceil((SIZE * 8) / 256) * 256;
      const staging = root.device.createBuffer({
        size: bpr * SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const e = root.device.createCommandEncoder();
      e.copyTextureToBuffer(
        { texture: root.unwrap(truth) },
        { buffer: staging, bytesPerRow: bpr, rowsPerImage: SIZE },
        { width: SIZE, height: SIZE },
      );
      root.device.queue.submit([e.finish()]);
      const raw = await readBuffer(root.device, staging, bpr * SIZE);
      staging.destroy();
      return { data: new Uint16Array(raw), stride: bpr / 2 };
    })();

    let hits = 0;
    let minZ = Infinity;
    let maxZ = 0;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const i = y * depthData.stride + x * 4;
        const z = halfToFloat(depthData.data[i]!);
        const hit = halfToFloat(depthData.data[i + 1]!);
        if (hit > 0.5) {
          hits++;
          minZ = Math.min(minZ, z);
          maxZ = Math.max(maxZ, z);
        }
      }
    }
    expect(hits, 'no geometry was hit by the ray march').toBeGreaterThan(SIZE * SIZE * 0.2);
    expect(minZ).toBeGreaterThan(0.1);
    expect(maxZ).toBeGreaterThan(minZ * 1.5);

    scene.destroy();
    color.destroy();
    truth.destroy();
  });
});
