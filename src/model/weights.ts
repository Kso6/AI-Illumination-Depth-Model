/**
 * Weight container and loader.
 *
 * The on-disk format (`.idm`) is deliberately the same shape as safetensors: a
 * little-endian `u32` header length, a UTF-8 JSON header, then a tightly packed
 * blob. Weights are stored in *logical* PyTorch order — `[out, in, kh, kw]` and
 * friends — never in the GPU's swizzled order, so that
 *
 *   - the exporter in `tools/export` is a direct dump of the state dict,
 *   - the file is inspectable and reusable outside this project, and
 *   - the swizzling logic lives in exactly one tested place (`layout.ts`).
 *
 * Packing 1.4 M parameters at load time costs a few milliseconds, once.
 */
import type { Architecture, Op } from './arch.ts';

export interface WeightSpec {
  readonly name: string;
  readonly shape: readonly number[];
}

export type WeightMap = Map<string, Float32Array>;

function elems(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Every weight tensor the architecture needs, with its logical shape. */
export function weightSpecs(arch: Architecture): WeightSpec[] {
  const out: WeightSpec[] = [];
  const c = (n: string) => arch.tensors[n]!.c;

  for (const op of arch.ops) {
    switch (op.kind) {
      case 'conv':
        out.push({ name: `${op.name}.weight`, shape: [c(op.out), c(op.in), op.k, op.k] });
        out.push({ name: `${op.name}.bias`, shape: [c(op.out)] });
        break;
      case 'pw':
        out.push({ name: `${op.name}.weight`, shape: [c(op.out), c(op.in)] });
        out.push({ name: `${op.name}.bias`, shape: [c(op.out)] });
        break;
      case 'dwpw':
        out.push({ name: `${op.name}.dw_weight`, shape: [c(op.in), op.k, op.k] });
        out.push({ name: `${op.name}.dw_bias`, shape: [c(op.in)] });
        out.push({ name: `${op.name}.pw_weight`, shape: [c(op.out), c(op.in)] });
        out.push({ name: `${op.name}.pw_bias`, shape: [c(op.out)] });
        break;
      case 'lateral':
        out.push({ name: `${op.name}.weight`, shape: [c(op.out), c(op.skip)] });
        out.push({ name: `${op.name}.bias`, shape: [c(op.out)] });
        break;
      case 'head':
        out.push({ name: `${op.name}.weight`, shape: [1, c(op.in)] });
        out.push({ name: `${op.name}.bias`, shape: [1] });
        break;
      case 'preprocess':
      case 'bilateralUp':
        break;
    }
  }
  return out;
}

export function totalParameters(arch: Architecture): number {
  return weightSpecs(arch).reduce((a, s) => a + elems(s.shape), 0);
}

// ---------------------------------------------------------------------------
// Container
// ---------------------------------------------------------------------------

interface IdmHeader {
  format: string;
  arch: string;
  tensors: Record<string, { dtype: 'f32' | 'f16'; shape: number[]; offset: number; length: number }>;
}

const MAGIC = 'idm/1';

/** Decodes an fp16 bit pattern to a JavaScript number. */
function halfToFloat(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  let value: number;
  if (exp === 0) {
    value = frac * 2 ** -24;
  } else if (exp === 0x1f) {
    value = frac === 0 ? Infinity : NaN;
  } else {
    value = (frac + 1024) * 2 ** (exp - 25);
  }
  return sign ? -value : value;
}

export function parseWeights(buffer: ArrayBuffer, arch: Architecture): WeightMap {
  const view = new DataView(buffer);
  if (buffer.byteLength < 4) throw new Error('weights: file is truncated');
  const headerLen = view.getUint32(0, true);
  if (headerLen === 0 || 4 + headerLen > buffer.byteLength) {
    throw new Error(`weights: header length ${headerLen} does not fit the file`);
  }
  const headerText = new TextDecoder().decode(new Uint8Array(buffer, 4, headerLen));
  let header: IdmHeader;
  try {
    header = JSON.parse(headerText) as IdmHeader;
  } catch (e) {
    throw new Error(`weights: header is not valid JSON (${(e as Error).message})`);
  }
  if (header.format !== MAGIC) {
    throw new Error(`weights: unsupported format ${header.format}, expected ${MAGIC}`);
  }
  if (header.arch !== arch.name) {
    throw new Error(`weights: file is for ${header.arch}, this build is ${arch.name}`);
  }

  const base = 4 + headerLen;
  const map: WeightMap = new Map();
  for (const [name, info] of Object.entries(header.tensors)) {
    const count = elems(info.shape);
    const bytes = count * (info.dtype === 'f16' ? 2 : 4);
    if (info.length !== bytes) {
      throw new Error(`weights: ${name} claims ${info.length} bytes but its shape needs ${bytes}`);
    }
    if (base + info.offset + bytes > buffer.byteLength) {
      throw new Error(`weights: ${name} extends past the end of the file`);
    }
    if (info.dtype === 'f32') {
      // The blob is not guaranteed to be 4-byte aligned relative to the file
      // start, so copy rather than aliasing.
      const src = new Uint8Array(buffer, base + info.offset, bytes);
      const dst = new Float32Array(count);
      new Uint8Array(dst.buffer).set(src);
      map.set(name, dst);
    } else {
      const src = new Uint8Array(buffer, base + info.offset, bytes);
      const u16 = new Uint16Array(count);
      new Uint8Array(u16.buffer).set(src);
      const dst = new Float32Array(count);
      for (let i = 0; i < count; i++) dst[i] = halfToFloat(u16[i]!);
      map.set(name, dst);
    }
  }

  // Fail loudly and specifically rather than producing silent garbage.
  const missing: string[] = [];
  for (const spec of weightSpecs(arch)) {
    const got = map.get(spec.name);
    if (!got) {
      missing.push(spec.name);
    } else if (got.length !== elems(spec.shape)) {
      throw new Error(
        `weights: ${spec.name} has ${got.length} values, expected ${elems(spec.shape)} ` +
          `for shape [${spec.shape.join(', ')}]`,
      );
    }
  }
  if (missing.length) {
    throw new Error(
      `weights: ${missing.length} tensor(s) missing, first few: ${missing.slice(0, 5).join(', ')}`,
    );
  }
  return map;
}

/** Writes a weight map in the `.idm` container format. */
export function serializeWeights(arch: Architecture, weights: WeightMap): ArrayBuffer {
  const specs = weightSpecs(arch);
  const tensors: IdmHeader['tensors'] = {};
  let offset = 0;
  for (const spec of specs) {
    const data = weights.get(spec.name);
    if (!data) throw new Error(`serializeWeights: missing ${spec.name}`);
    const length = data.length * 4;
    tensors[spec.name] = { dtype: 'f32', shape: [...spec.shape], offset, length };
    offset += length;
  }
  const header: IdmHeader = { format: MAGIC, arch: arch.name, tensors };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const buffer = new ArrayBuffer(4 + headerBytes.length + offset);
  const view = new DataView(buffer);
  view.setUint32(0, headerBytes.length, true);
  new Uint8Array(buffer, 4).set(headerBytes);
  const base = 4 + headerBytes.length;
  for (const spec of specs) {
    const data = weights.get(spec.name)!;
    const info = tensors[spec.name]!;
    new Uint8Array(buffer, base + info.offset, info.length).set(
      new Uint8Array(data.buffer, data.byteOffset, info.length),
    );
  }
  return buffer;
}

// ---------------------------------------------------------------------------
// Deterministic initialisation
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller, so the initialisation is genuinely Gaussian. */
function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Kaiming-normal initialisation, matching `torch.nn.init.kaiming_normal_` with
 * `mode='fan_in'`, `nonlinearity='relu'`.
 *
 * These weights are **not trained**: they let the whole pipeline run, be
 * benchmarked and be tested end to end, but the depth they produce is a smooth
 * arbitrary field, not a scene reconstruction. Train real weights with
 * `tools/train` and load the resulting `.idm` file.
 */
export function synthesizeWeights(arch: Architecture, seed = 1): WeightMap {
  const rand = mulberry32(seed);
  const map: WeightMap = new Map();
  for (const spec of weightSpecs(arch)) {
    const n = elems(spec.shape);
    const data = new Float32Array(n);
    if (spec.name.endsWith('bias')) {
      // Leave biases at zero, as PyTorch's conv default does.
      map.set(spec.name, data);
      continue;
    }
    // fan_in is everything but the leading (output) dimension. For a depthwise
    // kernel of shape [C, k, k] the groups make fan_in just k·k.
    const fanIn = spec.name.includes('dw_weight')
      ? spec.shape[1]! * spec.shape[2]!
      : elems(spec.shape.slice(1));
    const std = Math.sqrt(2 / Math.max(fanIn, 1));
    // The head is scaled down so an untrained network outputs a mid-grey
    // disparity rather than saturated noise.
    const scale = spec.name.startsWith('head') ? std * 0.05 : std;
    for (let i = 0; i < n; i++) data[i] = gaussian(rand) * scale;
    map.set(spec.name, data);
  }
  return map;
}

/** Splits a weight map into the per-op tensors the GPU runner uploads. */
export function opWeights(op: Op, weights: WeightMap): Record<string, Float32Array> {
  const need = (suffix: string): Float32Array => {
    const key = `${op.name}.${suffix}`;
    const value = weights.get(key);
    if (!value) throw new Error(`weights: ${key} is missing`);
    return value;
  };
  switch (op.kind) {
    case 'conv':
    case 'pw':
    case 'lateral':
    case 'head':
      return { weight: need('weight'), bias: need('bias') };
    case 'dwpw':
      return {
        dwWeight: need('dw_weight'),
        dwBias: need('dw_bias'),
        pwWeight: need('pw_weight'),
        pwBias: need('pw_bias'),
      };
    default:
      return {};
  }
}
