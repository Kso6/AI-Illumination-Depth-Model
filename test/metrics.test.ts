/**
 * Depth metrics against closed-form answers.
 *
 * Feeds `evaluateDepth` synthetic prediction/ground-truth pairs whose
 * least-squares scale-and-shift fit, AbsRel, RMSE and delta thresholds are known
 * analytically, so the alignment maths is checked rather than merely exercised.
 */
import { describe, expect, it } from 'vitest';
import { headlessGpu } from './helpers/gpu.ts';
import { evaluateDepth } from '../src/scene/metrics.ts';

const N = 64;

function f32ToF16(v: number): number {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  f[0] = v;
  const x = u[0]!;
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const frac = x & 0x7fffff;
  if (exp <= 0) return sign;
  if (exp >= 0x1f) return sign | 0x7c00;
  return sign | (exp << 10) | (frac >>> 13);
}

function mk(root: any, fill: (x: number, y: number) => [number, number, number, number]) {
  const t = root.createTexture({ size: [N, N], format: 'rgba16float' }).$usage('sampled');
  const data = new Uint16Array(N * N * 4);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const v = fill(x, y);
      const i = (y * N + x) * 4;
      for (let k = 0; k < 4; k++) data[i + k] = f32ToF16(v[k]!);
    }
  }
  root.device.queue.writeTexture(
    { texture: root.unwrap(t) },
    data.buffer as ArrayBuffer,
    { bytesPerRow: N * 8, rowsPerImage: N },
    { width: N, height: N },
  );
  return t;
}

describe('metrics', () => {
  it('closed-form scale/shift + metrics', async () => {
    const { root } = await headlessGpu();

    // Two distinct predicted disparities => the LS line passes exactly through
    // the two group means, so scale and shift are analytic.
    //   rows  0-15 : x=0.25, z=4
    //   rows 16-31 : x=0.25, z=2   -> mean y_A = 0.375
    //   rows 32-47 : x=0.75, z=1
    //   rows 48-63 : x=0.75, z=2   -> mean y_B = 0.75
    // s = (0.75-0.375)/(0.75-0.25) = 0.75 ; t = 0.375 - 0.75*0.25 = 0.1875
    const predX = (_x: number, y: number) => (y < 32 ? 0.25 : 0.75);
    const truthZ = (_x: number, y: number) => (y < 16 ? 4 : y < 32 ? 2 : y < 48 ? 1 : 2);

    const pred = mk(root, (x, y) => [predX(x, y), 0, 0, 1]);
    const truth = mk(root, (x, y) => [truthZ(x, y), 1, 0, 1]);

    const m = await evaluateDepth(root, pred, truth, N);
    console.log('measured:', JSON.stringify(m, null, 2));
    console.log('expected: scale 0.75 shift 0.1875 absRel 0.333333 rmse 0.833333 d1 0 d2 1 d3 1 samples 4096');

    expect(m.samples).toBe(4096);
    expect(m.scale).toBeCloseTo(0.75, 5);
    expect(m.shift).toBeCloseTo(0.1875, 5);
    expect(m.absRel).toBeCloseTo(1 / 3, 5);
    expect(m.rmse).toBeCloseTo(5 / 6, 5);
    expect(m.delta1).toBeCloseTo(0, 6);
    expect(m.delta2).toBeCloseTo(1, 6);
    expect(m.delta3).toBeCloseTo(1, 6);
  }, 600_000);

  it('perfect up-to-affine prediction scores zero error', async () => {
    const { root } = await headlessGpu();
    // true disparity y = 1/z ; predicted x = (y - 0.2)/0.5  => s=0.5, t=0.2
    const zOf = (x: number, y: number) => 1 + ((x + y * N) % 97) * 0.1; // 1..10.6
    const pred = mk(root, (x, y) => [(1 / zOf(x, y) - 0.2) / 0.5, 0, 0, 1]);
    const truth = mk(root, (x, y) => [zOf(x, y), 1, 0, 1]);
    const m = await evaluateDepth(root, pred, truth, N);
    console.log('affine-perfect:', JSON.stringify(m));
    expect(m.scale).toBeCloseTo(0.5, 2);
    expect(m.shift).toBeCloseTo(0.2, 2);
    expect(m.absRel).toBeLessThan(2e-3);
    expect(m.delta1).toBeCloseTo(1, 6);
  }, 600_000);

  it('honours the hit mask and the depth window', async () => {
    const { root } = await headlessGpu();
    const pred = mk(root, () => [0.5, 0, 0, 1]);
    // half the pixels marked as sky (hit = 0)
    const truth = mk(root, (_x, y) => [2, y < 32 ? 1 : 0, 0, 1]);
    const m = await evaluateDepth(root, pred, truth, N);
    console.log('mask:', JSON.stringify(m));
    expect(m.samples).toBe(2048);
  }, 600_000);
});
