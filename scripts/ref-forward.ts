/**
 * Run the CPU reference network from the command line.
 *
 * `tools/export/verify_parity.py` needs to evaluate *this* implementation --- the
 * one in `src/model/reference.ts`, which is the normative specification the WGSL
 * kernels are tested against --- on exactly the weights and exactly the input
 * that the PyTorch model just saw. Python cannot import TypeScript, so the two
 * halves talk through two files and this script.
 *
 * Interchange format
 * ------------------
 * Both files use the same trivially parseable container the `.idm` weight file
 * uses, so there is one layout to document and one to get right:
 *
 *     [u32 little-endian headerLength]
 *     [UTF-8 JSON header, `headerLength` bytes]
 *     [tightly packed tensor blob]
 *
 * For weights the header is the `.idm` header (`{"format":"idm/1","arch":...}`)
 * and `src/model/weights.ts` parses it. For activations the header is
 *
 *     {"format":"ntb/1",
 *      "tensors":{ name: {dtype:"f32"|"f16", shape:[...], offset, length} }}
 *
 * ("named tensor bundle"). `offset` is relative to the start of the blob,
 * `length` is in bytes, and the JSON key order is the blob order.
 *
 * **Activation tensors are stored channels-last, shaped `[h, w, c]`**, which is
 * precisely the memory order of a `RefTensor.data`, so no transposition happens
 * on this side. `verify_parity.py` permutes its NCHW PyTorch tensors once, on
 * the Python side, where the cost is irrelevant.
 *
 * Usage
 * -----
 *     npx vite-node scripts/ref-forward.ts -- \
 *       --weights out/parity.idm --input out/in.ntb --out out/ref.ntb
 *
 *     # no PyTorch anywhere: deterministic synthetic weights, dumped as .idm
 *     npx vite-node scripts/ref-forward.ts -- \
 *       --weights synthetic:7 --dump-weights out/synth.idm \
 *       --input out/in.ntb --out out/ref.ntb --size 64
 *
 * The input bundle must contain either `rgba` (`[S, S, 4]`, sRGB-encoded
 * components in `[0, 1]`), in which case the preprocess step runs here too and
 * is itself compared, or a ready-made `input` tensor (`[S, S, 4]`, already
 * normalised). The output bundle contains every intermediate activation keyed by
 * the architecture's tensor names, so a divergence can be bisected to the first
 * op that disagrees. A one-line JSON summary is printed on stdout.
 *
 * `--size` defaults to the height of the input tensor. Parity runs use a small
 * geometrically identical network (64 is the smallest legal size) because this
 * reference is scalar JavaScript: 2.2 GFLOP at 448 would take minutes, whereas
 * 45 MFLOP at 64 takes a moment and exercises every op the same way.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { buildArchitecture } from '../src/model/arch.ts';
import type { Architecture } from '../src/model/arch.ts';
import { halfToFloat, parseWeights, serializeWeights, synthesizeWeights } from '../src/model/weights.ts';
import type { WeightMap } from '../src/model/weights.ts';
import { forwardReference, refPreprocess, refTensor } from '../src/model/reference.ts';
import type { RefTensor } from '../src/model/reference.ts';

// ---------------------------------------------------------------------------
// Named tensor bundles
// ---------------------------------------------------------------------------

const BUNDLE_FORMAT = 'ntb/1';

interface BundleTensorInfo {
  dtype: 'f32' | 'f16';
  shape: number[];
  offset: number;
  length: number;
}

interface BundleHeader {
  format: string;
  tensors: Record<string, BundleTensorInfo>;
}

interface NamedTensor {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

function elementCount(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Reads a file into a standalone `ArrayBuffer` (`slice` copies, so it is not shared). */
function readArrayBuffer(path: string): ArrayBuffer {
  const bytes = new Uint8Array(readFileSync(path));
  return bytes.slice().buffer;
}

function readBundle(path: string): Map<string, NamedTensor> {
  const buffer = readArrayBuffer(path);
  if (buffer.byteLength < 4) throw new Error(`${path}: file is truncated`);
  const view = new DataView(buffer);
  const headerLen = view.getUint32(0, true);
  if (headerLen === 0 || 4 + headerLen > buffer.byteLength) {
    throw new Error(`${path}: header length ${headerLen} does not fit the file`);
  }
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, headerLen)),
  ) as BundleHeader;
  if (header.format !== BUNDLE_FORMAT) {
    throw new Error(`${path}: format ${header.format}, expected ${BUNDLE_FORMAT}`);
  }

  const base = 4 + headerLen;
  const out = new Map<string, NamedTensor>();
  for (const [name, info] of Object.entries(header.tensors)) {
    const count = elementCount(info.shape);
    const bytes = count * (info.dtype === 'f16' ? 2 : 4);
    if (info.length !== bytes) {
      throw new Error(`${path}: ${name} claims ${info.length} bytes, shape needs ${bytes}`);
    }
    if (base + info.offset + bytes > buffer.byteLength) {
      throw new Error(`${path}: ${name} extends past the end of the file`);
    }
    // The blob is not guaranteed to be 4-byte aligned relative to the file
    // start, so copy into a fresh typed array rather than aliasing.
    const src = new Uint8Array(buffer, base + info.offset, bytes);
    const data = new Float32Array(count);
    if (info.dtype === 'f32') {
      new Uint8Array(data.buffer).set(src);
    } else {
      const half = new Uint16Array(count);
      new Uint8Array(half.buffer).set(src);
      for (let i = 0; i < count; i++) data[i] = halfToFloat(half[i]);
    }
    out.set(name, { shape: [...info.shape], data });
  }
  return out;
}

function writeBundle(path: string, tensors: ReadonlyMap<string, NamedTensor>): void {
  const header: BundleHeader = { format: BUNDLE_FORMAT, tensors: {} };
  let offset = 0;
  for (const [name, tensor] of tensors) {
    const count = elementCount(tensor.shape);
    if (count !== tensor.data.length) {
      throw new Error(
        `${name}: shape [${tensor.shape.join(', ')}] needs ${count} values, got ${tensor.data.length}`,
      );
    }
    const length = count * 4;
    header.tensors[name] = { dtype: 'f32', shape: [...tensor.shape], offset, length };
    offset += length;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + headerBytes.length + offset);
  new DataView(out.buffer).setUint32(0, headerBytes.length, true);
  out.set(headerBytes, 4);
  const base = 4 + headerBytes.length;
  for (const [name, tensor] of tensors) {
    const info = header.tensors[name];
    out.set(
      new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, info.length),
      base + info.offset,
    );
  }
  writeFileSync(path, out);
}

function fromRefTensor(t: RefTensor): NamedTensor {
  return { shape: [t.h, t.w, t.c], data: t.data };
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

interface Options {
  weights: string;
  input: string | null;
  out: string | null;
  dumpWeights: string | null;
  size: number | null;
  exposure: number;
  decodeSrgb: boolean;
}

const USAGE = `usage: npx vite-node scripts/ref-forward.ts -- [options]

  --weights <path|synthetic[:seed]>  .idm weight container, or deterministic
                                     synthetic weights (default synthetic:1)
  --input <path.ntb>                 bundle holding "rgba" or "input"
  --out <path.ntb>                   where to write every activation
  --dump-weights <path.idm>          also write the weights actually used
  --size <n>                         network size; default: input height
  --exposure <f>                     preprocess exposure multiplier (default 1)
  --no-srgb                          treat "rgba" as already linear
  --help
`;

function parseArgs(argv: readonly string[]): Options {
  const opts: Options = {
    weights: 'synthetic:1',
    input: null,
    out: null,
    dumpWeights: null,
    size: null,
    exposure: 1,
    decodeSrgb: true,
  };
  const need = (i: number, flag: string): string => {
    const value = argv[i + 1];
    if (value === undefined) throw new Error(`${flag} needs a value`);
    return value;
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '--weights':
        opts.weights = need(i, arg);
        i++;
        break;
      case '--input':
        opts.input = need(i, arg);
        i++;
        break;
      case '--out':
        opts.out = need(i, arg);
        i++;
        break;
      case '--dump-weights':
        opts.dumpWeights = need(i, arg);
        i++;
        break;
      case '--size':
        opts.size = Number.parseInt(need(i, arg), 10);
        i++;
        break;
      case '--exposure':
        opts.exposure = Number.parseFloat(need(i, arg));
        i++;
        break;
      case '--no-srgb':
        opts.decodeSrgb = false;
        break;
      case '--help':
      case '-h':
        process.stdout.write(USAGE);
        process.exit(0);
        break;
      default:
        throw new Error(`unknown argument ${arg}\n\n${USAGE}`);
    }
  }
  return opts;
}

function loadWeights(spec: string, arch: Architecture): WeightMap {
  if (spec === 'synthetic' || spec.startsWith('synthetic:')) {
    const seedText = spec.slice('synthetic'.length + 1);
    const seed = seedText.length > 0 ? Number.parseInt(seedText, 10) : 1;
    if (!Number.isFinite(seed)) throw new Error(`bad synthetic seed in ${spec}`);
    return synthesizeWeights(arch, seed);
  }
  return parseWeights(readArrayBuffer(spec), arch);
}

/**
 * Resolves the network input, running `refPreprocess` here when the caller
 * supplied a raw image so that the preprocessing itself is covered by the parity
 * check rather than assumed identical.
 */
function resolveInput(
  bundle: Map<string, NamedTensor>,
  arch: Architecture,
  opts: Options,
): { input: RefTensor; extra: Map<string, NamedTensor> } {
  const extra = new Map<string, NamedTensor>();
  const size = arch.inputSize;

  const rgba = bundle.get('rgba');
  if (rgba) {
    const want = [size, size, 4];
    if (rgba.shape.join('x') !== want.join('x')) {
      throw new Error(`rgba has shape [${rgba.shape.join(', ')}], expected [${want.join(', ')}]`);
    }
    const pre = refPreprocess(
      rgba.data,
      size,
      arch.mean,
      arch.std,
      opts.decodeSrgb,
      opts.exposure,
    );
    extra.set('scene_color', fromRefTensor(pre.sceneColor));
    return { input: pre.input, extra };
  }

  const given = bundle.get('input');
  if (!given) throw new Error('input bundle contains neither "rgba" nor "input"');
  if (given.shape.join('x') !== [size, size, 4].join('x')) {
    throw new Error(`input has shape [${given.shape.join(', ')}], expected [${size}, ${size}, 4]`);
  }
  return { input: refTensor(size, size, 4, given.data), extra };
}

function main(): void {
  const opts = parseArgs(process.argv.slice(2));

  const bundle = opts.input ? readBundle(opts.input) : new Map<string, NamedTensor>();

  let size = opts.size;
  if (size === null) {
    const probe = bundle.get('rgba') ?? bundle.get('input');
    if (!probe) throw new Error('--size is required when no --input is given');
    size = probe.shape[0];
  }
  const arch = buildArchitecture(size);
  const weights = loadWeights(opts.weights, arch);

  if (opts.dumpWeights) {
    writeFileSync(opts.dumpWeights, new Uint8Array(serializeWeights(arch, weights)));
  }

  if (!opts.input) {
    process.stdout.write(
      `${JSON.stringify({ arch: arch.name, size, tensors: 0, dumpedWeights: opts.dumpWeights })}\n`,
    );
    return;
  }

  const { input, extra } = resolveInput(bundle, arch, opts);
  const values = forwardReference(arch, weights, input);

  const out = new Map<string, NamedTensor>();
  for (const [name, tensor] of values) out.set(name, fromRefTensor(tensor));
  for (const [name, tensor] of extra) out.set(name, tensor);

  if (opts.out) writeBundle(opts.out, out);

  process.stdout.write(
    `${JSON.stringify({
      arch: arch.name,
      size,
      tensors: out.size,
      names: [...out.keys()],
      out: opts.out,
      dumpedWeights: opts.dumpWeights,
    })}\n`,
  );
}

main();
