/**
 * Scoring the estimated depth against the procedural scene's exact depth.
 *
 * The network predicts *relative* inverse depth: it is trained scale- and
 * shift-invariantly, so its output is only defined up to an affine transform of
 * disparity. Comparing it to metric ground truth therefore requires first
 * solving for the best `(scale, shift)` in disparity space — the same
 * least-squares alignment MiDaS uses for evaluation — and only then measuring
 * error. Skipping that step would report a large error for a perfect prediction.
 *
 * This runs on demand (a button), not per frame: it reads two textures back to
 * the CPU, which is exactly the kind of GPU→CPU round trip the rest of the
 * renderer exists to avoid.
 */
import type { TgpuRoot } from 'typegpu';
import { halfToFloat } from '../model/weights.ts';

export interface DepthMetrics {
  /** Pixels that had valid ground truth. */
  readonly samples: number;
  /** Absolute relative error, after affine alignment. Lower is better. */
  readonly absRel: number;
  /** Root-mean-square error in metres, after alignment. */
  readonly rmse: number;
  /** Fraction of pixels with `max(zHat/z, z/zHat) < 1.25`. Higher is better. */
  readonly delta1: number;
  readonly delta2: number;
  readonly delta3: number;
  /** The fitted disparity scale and shift, for diagnostics. */
  readonly scale: number;
  readonly shift: number;
}

async function readTexture(
  root: TgpuRoot,
  texture: Parameters<TgpuRoot['unwrap']>[0],
  size: number,
  bytesPerTexel: number,
): Promise<ArrayBuffer> {
  const bytesPerRow = Math.ceil((size * bytesPerTexel) / 256) * 256;
  const staging = root.device.createBuffer({
    size: bytesPerRow * size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = root.device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: root.unwrap(texture) as GPUTexture },
    { buffer: staging, bytesPerRow, rowsPerImage: size },
    { width: size, height: size },
  );
  root.device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
}

/**
 * Compares the predicted disparity texture against the ground-truth depth
 * texture produced by the procedural scene.
 *
 * @param nearFar The depth range the application maps disparity onto; only used
 *   to report the alignment in interpretable units.
 */
export async function evaluateDepth(
  root: TgpuRoot,
  predicted: Parameters<TgpuRoot['unwrap']>[0],
  truth: Parameters<TgpuRoot['unwrap']>[0],
  size: number,
): Promise<DepthMetrics> {
  const bytesPerRow = Math.ceil((size * 8) / 256) * 256;
  const texelsPerRow = bytesPerRow / 8;

  const [predRaw, truthRaw] = await Promise.all([
    readTexture(root, predicted, size, 8),
    readTexture(root, truth, size, 8),
  ]);
  const pred = new Uint16Array(predRaw);
  const gt = new Uint16Array(truthRaw);

  // --- collect valid samples ---------------------------------------------
  // Ground truth is a distance in metres; convert to disparity so the affine
  // fit happens in the space the network actually predicts in.
  const x: number[] = [];
  const y: number[] = [];
  const gtDepth: number[] = [];
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const i = (row * texelsPerRow + col) * 4;
      const hit = halfToFloat(gt[i + 1]!);
      const depth = halfToFloat(gt[i]!);
      if (hit < 0.5 || !(depth > 0.05) || depth > 25) continue;
      x.push(halfToFloat(pred[i]!));
      y.push(1 / depth);
      gtDepth.push(depth);
    }
  }
  const n = x.length;
  if (n < 64) {
    return {
      samples: n,
      absRel: NaN,
      rmse: NaN,
      delta1: 0,
      delta2: 0,
      delta3: 0,
      scale: NaN,
      shift: NaN,
    };
  }

  // --- least-squares affine alignment in disparity space ------------------
  // Solve [[Σx², Σx], [Σx, n]] · [s, t]ᵀ = [Σxy, Σy]ᵀ.
  let sxx = 0;
  let sx = 0;
  let sxy = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sxx += x[i]! * x[i]!;
    sx += x[i]!;
    sxy += x[i]! * y[i]!;
    sy += y[i]!;
  }
  const det = sxx * n - sx * sx;
  const scale = Math.abs(det) > 1e-12 ? (n * sxy - sx * sy) / det : 1;
  const shift = Math.abs(det) > 1e-12 ? (sxx * sy - sx * sxy) / det : 0;

  // --- error metrics on aligned depth -------------------------------------
  let absRel = 0;
  let sqErr = 0;
  let d1 = 0;
  let d2 = 0;
  let d3 = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const disparity = scale * x[i]! + shift;
    if (!(disparity > 1e-4)) continue;
    const zHat = 1 / disparity;
    const z = gtDepth[i]!;
    absRel += Math.abs(zHat - z) / z;
    sqErr += (zHat - z) ** 2;
    const ratio = Math.max(zHat / z, z / zHat);
    if (ratio < 1.25) d1++;
    if (ratio < 1.25 ** 2) d2++;
    if (ratio < 1.25 ** 3) d3++;
    counted++;
  }

  const denom = Math.max(counted, 1);
  return {
    samples: counted,
    absRel: absRel / denom,
    rmse: Math.sqrt(sqErr / denom),
    delta1: d1 / denom,
    delta2: d2 / denom,
    delta3: d3 / denom,
    scale,
    shift,
  };
}
