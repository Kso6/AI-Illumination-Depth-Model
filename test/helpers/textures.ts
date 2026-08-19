import type { TgpuRoot } from 'typegpu';
import { halfToFloat } from '../../src/model/weights.ts';
import type { WorkTexture } from '../../src/model/runner.ts';
import { readBuffer } from './gpu.ts';

/** Creates an `rgba8unorm` texture and fills it with the given bytes. */
export function uploadImage(root: TgpuRoot, size: number, rgba: Uint8Array) {
  const tex = root
    .createTexture({ size: [size, size], format: 'rgba8unorm' })
    .$usage('sampled');
  root.device.queue.writeTexture(
    { texture: root.unwrap(tex) },
    rgba.buffer as ArrayBuffer,
    { offset: rgba.byteOffset, bytesPerRow: size * 4, rowsPerImage: size },
    { width: size, height: size },
  );
  return tex;
}

/**
 * Reads back an `rgba16float` texture as floats.
 * `size * 8` must be a multiple of 256 for the copy to be legal.
 */
export async function readRgba16f(
  root: TgpuRoot,
  texture: WorkTexture,
  size: number,
): Promise<Float32Array> {
  const bytesPerRow = size * 8;
  if (bytesPerRow % 256 !== 0) {
    throw new Error(`readRgba16f: bytesPerRow ${bytesPerRow} is not a multiple of 256`);
  }
  const staging = root.device.createBuffer({
    size: bytesPerRow * size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const enc = root.device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: root.unwrap(texture) },
    { buffer: staging, bytesPerRow, rowsPerImage: size },
    { width: size, height: size },
  );
  root.device.queue.submit([enc.finish()]);

  const raw = await readBuffer(root.device, staging, bytesPerRow * size);
  const u16 = new Uint16Array(raw);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]!);
  staging.destroy();
  return out;
}
