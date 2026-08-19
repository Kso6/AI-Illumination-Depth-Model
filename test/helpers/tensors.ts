import * as d from 'typegpu/data';
import type { TgpuRoot } from 'typegpu';
import { readBuffer } from './gpu.ts';

/** Deterministic PRNG so failures are reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniform noise in `[-scale, scale)`. */
export function randomFloats(n: number, rand: () => number, scale = 1): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rand() * 2 - 1) * scale;
  return out;
}

export type StorageBuffer = ReturnType<TgpuRoot['createBuffer']>;

/**
 * `queue.writeBuffer` types its source as `ArrayBufferView<ArrayBuffer>`, but a
 * `Float32Array` is typed over `ArrayBufferLike` (which admits
 * `SharedArrayBuffer`). Passing the underlying buffer with an explicit range
 * keeps it exact without an unchecked cast of the view itself.
 */
export function writeFloats(root: TgpuRoot, target: GPUBuffer, data: Float32Array): void {
  root.device.queue.writeBuffer(
    target,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
}

/**
 * Allocates a `vec4f` storage buffer and fills it with raw floats.
 * `data.length` must be a multiple of four.
 */
export function uploadVec4(root: TgpuRoot, data: Float32Array) {
  if (data.length % 4 !== 0) throw new Error(`length ${data.length} is not a multiple of 4`);
  const buf = root.createBuffer(d.arrayOf(d.vec4f, data.length / 4)).$usage('storage');
  writeFloats(root, root.unwrap(buf), data);
  return buf;
}

/** Allocates an empty `vec4f` storage buffer holding `floats` floats. */
export function allocVec4(root: TgpuRoot, floats: number) {
  if (floats % 4 !== 0) throw new Error(`length ${floats} is not a multiple of 4`);
  return root.createBuffer(d.arrayOf(d.vec4f, floats / 4)).$usage('storage');
}

export async function download(
  root: TgpuRoot,
  buf: ReturnType<typeof allocVec4>,
  floats: number,
): Promise<Float32Array> {
  const raw = await readBuffer(root.device, root.unwrap(buf), floats * 4);
  return new Float32Array(raw, 0, floats);
}

export interface Mismatch {
  index: number;
  got: number;
  want: number;
  absErr: number;
  relErr: number;
}

/**
 * Compares two float arrays with a combined absolute/relative tolerance and
 * returns the worst mismatches, so a failure message points at real numbers
 * rather than just "arrays differ".
 */
export function compare(
  got: Float32Array,
  want: Float32Array,
  tol = { abs: 2e-4, rel: 2e-3 },
): { maxAbs: number; maxRel: number; worst: Mismatch[] } {
  if (got.length !== want.length) {
    throw new Error(`length mismatch: got ${got.length}, want ${want.length}`);
  }
  let maxAbs = 0;
  let maxRel = 0;
  const bad: Mismatch[] = [];
  for (let i = 0; i < want.length; i++) {
    const a = got[i];
    const b = want[i];
    const absErr = Math.abs(a - b);
    const relErr = absErr / Math.max(1e-6, Math.abs(b));
    maxAbs = Math.max(maxAbs, absErr);
    maxRel = Math.max(maxRel, relErr);
    if (absErr > tol.abs && relErr > tol.rel) {
      bad.push({ index: i, got: a, want: b, absErr, relErr });
    }
  }
  bad.sort((p, q) => q.absErr - p.absErr);
  return { maxAbs, maxRel, worst: bad.slice(0, 8) };
}
