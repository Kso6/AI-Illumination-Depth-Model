/**
 * IlluminaDepth-448 — architecture definition.
 *
 * This module is the single source of truth for the network. The GPU runner, the
 * CPU reference implementation, the weight container, the FLOP budget and the
 * PyTorch exporter are all generated from the op list below, so the four can
 * never drift apart.
 *
 * Design constraints that shaped it:
 *
 *  - **8 ms frame budget on an M1 Max.** The GPU peaks at ~10.4 TFLOP/s fp32.
 *    Hand-written WGSL convolutions on small tensors realistically land at
 *    5–20 % of peak (they are bandwidth- and latency-bound, not ALU-bound), so
 *    the usable budget is ~0.5–2 TFLOP/s. Targeting ~2 GFLOP of work leaves
 *    comfortable headroom for the lighting and draw passes in the same frame.
 *
 *  - **Every op is exactly one compute dispatch.** Depthwise convolutions are
 *    fused into the pointwise convolution that follows them, and bias,
 *    activation and residual addition are always folded into the producing
 *    kernel. Nothing is ever written to memory just to be read straight back.
 *
 *  - **Channels are multiples of four** so every tensor packs into `vec4`s, and
 *    every load from a storage buffer is a full 16-byte (fp32) or 8-byte (fp16)
 *    transaction.
 *
 *  - **No runtime normalisation layers.** BatchNorm is folded into the preceding
 *    convolution's weights and bias at export time (see `tools/export`), so the
 *    inference graph is pure conv/bias/activation.
 */

/** Activation functions available to fused kernels. */
export type Activation = 'linear' | 'relu' | 'relu6' | 'hardswish';

/** Names of the tensors that flow between ops. */
export type TensorName = string;

/** A tensor's shape, channels-last. */
export interface TensorShape {
  readonly h: number;
  readonly w: number;
  /** Logical channel count. Always a multiple of 4. */
  readonly c: number;
}

/**
 * Convert the source image texture into the network's input tensor.
 * Fuses: sample → linearise → resize to 448×448 → mean/std normalise → pack NHWC4.
 */
export interface PreprocessOp {
  readonly kind: 'preprocess';
  readonly name: string;
  readonly out: TensorName;
}

/**
 * Strided k×k convolution reading the input tensor directly. Used only for the
 * stem, where the input has 4 channels and a dense convolution is cheaper than a
 * depthwise-separable one.
 */
export interface ConvOp {
  readonly kind: 'conv';
  readonly name: string;
  readonly in: TensorName;
  readonly out: TensorName;
  readonly k: number;
  readonly stride: number;
  readonly act: Activation;
}

/**
 * Pointwise (1×1) convolution + bias + activation, with an optional residual
 * addend. This is the workhorse: it is a batched `C_out × C_in` matrix applied
 * to every pixel, which maps onto a register-blocked GEMM.
 */
export interface PointwiseOp {
  readonly kind: 'pw';
  readonly name: string;
  readonly in: TensorName;
  readonly out: TensorName;
  readonly act: Activation;
  /** If set, this tensor is added to the result before the activation. */
  readonly residual?: TensorName;
}

/**
 * Fused depthwise k×k (+bias +activation) immediately followed by a pointwise
 * convolution (+bias +activation +optional residual).
 *
 * The depthwise result is never written to memory: each thread computes the
 * depthwise output for one input channel group, activates it, and accumulates it
 * straight into the pointwise accumulators. That halves both the dispatch count
 * and the memory traffic of a classic depthwise-separable block.
 */
export interface DepthwisePointwiseOp {
  readonly kind: 'dwpw';
  readonly name: string;
  readonly in: TensorName;
  readonly out: TensorName;
  readonly k: number;
  readonly stride: number;
  /** Activation applied to the depthwise result, before the pointwise. */
  readonly midAct: Activation;
  /** Activation applied to the pointwise result. */
  readonly act: Activation;
  readonly residual?: TensorName;
}

/**
 * Decoder fusion, the top-down half of an FPN. In one dispatch it
 *
 *   1. bilinearly upsamples the coarse tensor by 2× (`align_corners=false`),
 *   2. projects the encoder skip through a 1×1 convolution,
 *   3. sums them with a bias and applies the activation.
 *
 * The coarse tensor needs **no** projection of its own: each level's refine step
 * already emits the channel width the next level up expects. Doing the width
 * change on the coarse side of the previous (half-resolution) level rather than
 * here saves ~284 MFLOP per frame versus projecting both inputs at full
 * resolution, and costs no extra dispatch.
 *
 * Consequently `coarse` must have exactly the same channel count as `out`.
 */
export interface LateralOp {
  readonly kind: 'lateral';
  readonly name: string;
  /** Coarse tensor from the level below; upsampled 2×. */
  readonly coarse: TensorName;
  /** Encoder skip connection at the target resolution. */
  readonly skip: TensorName;
  readonly out: TensorName;
  readonly act: Activation;
}

/**
 * Final head: 1×1 convolution to a single-channel inverse-depth map at the
 * decoder's resolution, written to a storage texture.
 */
export interface HeadOp {
  readonly kind: 'head';
  readonly name: string;
  readonly in: TensorName;
  /** Resolution of the produced (single channel) depth texture. */
  readonly out: TensorName;
}

/**
 * Edge-aware joint-bilateral upsample of the head's output to full resolution,
 * guided by the luminance of the source image. Recovers crisp depth
 * discontinuities that a bilinear upsample would smear.
 */
export interface BilateralUpOp {
  readonly kind: 'bilateralUp';
  readonly name: string;
  readonly in: TensorName;
  readonly out: TensorName;
}

export type Op =
  | PreprocessOp
  | ConvOp
  | PointwiseOp
  | DepthwisePointwiseOp
  | LateralOp
  | HeadOp
  | BilateralUpOp;

export interface Architecture {
  readonly name: string;
  /** Network input resolution (square). */
  readonly inputSize: number;
  /** Per-channel mean/std used to normalise the input, in linear RGB. */
  readonly mean: readonly [number, number, number];
  readonly std: readonly [number, number, number];
  readonly tensors: Readonly<Record<TensorName, TensorShape>>;
  readonly ops: readonly Op[];
}

// ---------------------------------------------------------------------------
// Builder helpers — keep the definition below readable.
// ---------------------------------------------------------------------------

const tensors: Record<TensorName, TensorShape> = {};
const ops: Op[] = [];

function T(name: TensorName, h: number, w: number, c: number): TensorName {
  if (c % 4 !== 0) {
    throw new Error(`tensor ${name}: channel count ${c} is not a multiple of 4`);
  }
  tensors[name] = { h, w, c };
  return name;
}

function push<O extends Op>(op: O): O {
  ops.push(op);
  return op;
}

// ---------------------------------------------------------------------------
// The network.
// ---------------------------------------------------------------------------

const S = 448;

/** Normalised RGB input, 448×448×4 (alpha carries luminance for later reuse). */
const x0 = T('input', S, S, 4);
push({ kind: 'preprocess', name: 'preprocess', out: x0 });

// --- Stem: 448 → 224 -------------------------------------------------------
const stem = T('stem', 224, 224, 16);
push({ kind: 'conv', name: 'stem', in: x0, out: stem, k: 3, stride: 2, act: 'hardswish' });

// A depthwise-separable block with no expansion, as in MobileNetV2's first
// bottleneck. Produces the highest-resolution skip.
const s0 = T('s0', 224, 224, 24);
push({
  kind: 'dwpw',
  name: 'e1',
  in: stem,
  out: s0,
  k: 3,
  stride: 1,
  midAct: 'relu6',
  act: 'linear',
});

/**
 * A downsampling block. The depthwise runs *first*, at stride 2, so the
 * expensive channel expansion happens at the lower resolution — a 4× saving
 * versus MobileNetV2's ordering, which expands before downsampling.
 *
 * Emits two dispatches: `dwpw` (depthwise s2 → expand) and `pw` (project).
 */
function downBlock(
  prefix: string,
  input: TensorName,
  h: number,
  w: number,
  expand: number,
  out: number,
  k: number,
): TensorName {
  const mid = T(`${prefix}.mid`, h, w, expand);
  push({
    kind: 'dwpw',
    name: `${prefix}.dw_expand`,
    in: input,
    out: mid,
    k,
    stride: 2,
    midAct: 'linear',
    act: 'hardswish',
  });
  const o = T(`${prefix}.out`, h, w, out);
  push({ kind: 'pw', name: `${prefix}.project`, in: mid, out: o, act: 'linear' });
  return o;
}

/**
 * A residual inverted-bottleneck block: expand 1×1 → depthwise k×k → project
 * 1×1, with the input added back. Two dispatches; the depthwise is fused into
 * the projection.
 */
function irBlock(
  prefix: string,
  input: TensorName,
  h: number,
  w: number,
  channels: number,
  expand: number,
  k: number,
): TensorName {
  const mid = T(`${prefix}.mid`, h, w, expand);
  push({ kind: 'pw', name: `${prefix}.expand`, in: input, out: mid, act: 'hardswish' });
  const o = T(`${prefix}.out`, h, w, channels);
  push({
    kind: 'dwpw',
    name: `${prefix}.dw_project`,
    in: mid,
    out: o,
    k,
    stride: 1,
    midAct: 'hardswish',
    act: 'linear',
    residual: input,
  });
  return o;
}

// --- Stage 1: 224 → 112, 32 channels ---------------------------------------
let t = downBlock('e2', s0, 112, 112, 96, 32, 3);
t = irBlock('e3', t, 112, 112, 32, 128, 3);
const s1 = t;

// --- Stage 2: 112 → 56, 56 channels ----------------------------------------
t = downBlock('e4', s1, 56, 56, 128, 56, 5);
t = irBlock('e5a', t, 56, 56, 56, 224, 5);
t = irBlock('e5b', t, 56, 56, 56, 224, 5);
const s2 = t;

// --- Stage 3: 56 → 28, 104 channels ----------------------------------------
t = downBlock('e6', s2, 28, 28, 224, 104, 5);
t = irBlock('e7a', t, 28, 28, 104, 416, 5);
t = irBlock('e7b', t, 28, 28, 104, 416, 5);
t = irBlock('e7c', t, 28, 28, 104, 416, 5);
const s3 = t;

// --- Stage 4: 28 → 14, 176 channels (bottleneck) ---------------------------
t = downBlock('e8', s3, 14, 14, 416, 176, 5);
t = irBlock('e9a', t, 14, 14, 176, 704, 5);
t = irBlock('e9b', t, 14, 14, 176, 704, 5);
t = irBlock('e9c', t, 14, 14, 176, 704, 5);
const bottleneck = t;

// --- Decoder ---------------------------------------------------------------
// A lightweight FPN. Each level upsamples the coarser tensor, adds a projected
// encoder skip, then refines with one fused depthwise-separable convolution.

const d4 = T('d4', 14, 14, 96);
push({ kind: 'pw', name: 'd4.lateral', in: bottleneck, out: d4, act: 'hardswish' });

/**
 * One decoder level. `channels` is the width at this level (which must match the
 * coarse tensor coming in); `nextChannels` is the width the level above expects,
 * produced by the refine step so the next lateral needs no projection.
 */
function decoderLevel(
  prefix: string,
  coarse: TensorName,
  skip: TensorName,
  h: number,
  w: number,
  channels: number,
  nextChannels: number,
): TensorName {
  const fused = T(`${prefix}.fused`, h, w, channels);
  push({ kind: 'lateral', name: `${prefix}.lateral`, coarse, skip, out: fused, act: 'linear' });
  const o = T(`${prefix}.out`, h, w, nextChannels);
  push({
    kind: 'dwpw',
    name: `${prefix}.refine`,
    in: fused,
    out: o,
    k: 3,
    stride: 1,
    midAct: 'hardswish',
    act: 'hardswish',
    // A residual is only meaningful when the width is unchanged.
    ...(channels === nextChannels ? { residual: fused } : {}),
  });
  return o;
}

const d3 = decoderLevel('d3', d4, s3, 28, 28, 96, 64);
const d2 = decoderLevel('d2', d3, s2, 56, 56, 64, 48);
const d1 = decoderLevel('d1', d2, s1, 112, 112, 48, 32);
const d0 = decoderLevel('d0', d1, s0, 224, 224, 32, 32);

// --- Head ------------------------------------------------------------------
// A single-channel inverse-depth (disparity) map at 224², then an edge-aware
// upsample to full 448² guided by the source image.
const depthLow = T('depth_low', 224, 224, 4);
push({ kind: 'head', name: 'head', in: d0, out: depthLow });

const depthFull = T('depth', S, S, 4);
push({ kind: 'bilateralUp', name: 'upsample', in: depthLow, out: depthFull });

export const ILLUMINA_DEPTH_448: Architecture = {
  name: 'IlluminaDepth-448',
  inputSize: S,
  // ImageNet statistics, applied in linear-light space.
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  tensors,
  ops,
};

// ---------------------------------------------------------------------------
// Derived facts: parameter counts, FLOPs, dispatch count, memory.
// ---------------------------------------------------------------------------

export interface OpCost {
  readonly name: string;
  readonly kind: Op['kind'];
  /** Multiply–accumulate operations. FLOPs are 2× this. */
  readonly macs: number;
  /** Learned parameters (weights + biases). */
  readonly params: number;
  readonly outShape: TensorShape;
}

export interface ArchCost {
  readonly ops: readonly OpCost[];
  readonly totalMacs: number;
  readonly totalParams: number;
  /** One dispatch per op. */
  readonly dispatches: number;
  /** Bytes of tensor storage if every tensor were live simultaneously. */
  readonly naiveTensorBytes: (bytesPerElement: number) => number;
}

/** Computes exact MAC and parameter counts for an architecture. */
export function costOf(arch: Architecture): ArchCost {
  const shape = (n: TensorName): TensorShape => {
    const s = arch.tensors[n];
    if (!s) throw new Error(`unknown tensor ${n}`);
    return s;
  };

  const costs: OpCost[] = arch.ops.map((op): OpCost => {
    switch (op.kind) {
      case 'preprocess':
        return { name: op.name, kind: op.kind, macs: 0, params: 0, outShape: shape(op.out) };

      case 'conv': {
        const o = shape(op.out);
        const i = shape(op.in);
        const macs = o.h * o.w * o.c * i.c * op.k * op.k;
        return { name: op.name, kind: op.kind, macs, params: o.c * i.c * op.k * op.k + o.c, outShape: o };
      }

      case 'pw': {
        const o = shape(op.out);
        const i = shape(op.in);
        const macs = o.h * o.w * o.c * i.c;
        return { name: op.name, kind: op.kind, macs, params: o.c * i.c + o.c, outShape: o };
      }

      case 'dwpw': {
        const o = shape(op.out);
        const i = shape(op.in);
        // Depthwise runs at the *output* resolution, over the input channels.
        const dwMacs = o.h * o.w * i.c * op.k * op.k;
        const pwMacs = o.h * o.w * o.c * i.c;
        const params = i.c * op.k * op.k + i.c + o.c * i.c + o.c;
        return { name: op.name, kind: op.kind, macs: dwMacs + pwMacs, params, outShape: o };
      }

      case 'lateral': {
        const o = shape(op.out);
        const s = shape(op.skip);
        const c = shape(op.coarse);
        if (c.c !== o.c) {
          throw new Error(
            `${op.name}: coarse tensor has ${c.c} channels but the output has ${o.c}; ` +
              'the previous level\'s refine step must emit the matching width',
          );
        }
        // One 1×1 projection of the skip; the bilinear upsample is lerps, not MACs.
        const macs = o.h * o.w * o.c * s.c;
        return { name: op.name, kind: op.kind, macs, params: o.c * s.c + o.c, outShape: o };
      }

      case 'head': {
        const o = shape(op.out);
        const i = shape(op.in);
        const macs = o.h * o.w * i.c;
        return { name: op.name, kind: op.kind, macs, params: i.c + 1, outShape: o };
      }

      case 'bilateralUp':
        return { name: op.name, kind: op.kind, macs: 0, params: 0, outShape: shape(op.out) };
    }
  });

  const totalMacs = costs.reduce((a, c) => a + c.macs, 0);
  const totalParams = costs.reduce((a, c) => a + c.params, 0);

  return {
    ops: costs,
    totalMacs,
    totalParams,
    dispatches: arch.ops.length,
    naiveTensorBytes: (bpe: number) =>
      Object.values(arch.tensors).reduce((a, s) => a + s.h * s.w * s.c * bpe, 0),
  };
}

/** Human-readable cost report, used by `scripts/arch-report.ts` and the tests. */
export function formatCost(arch: Architecture): string {
  const cost = costOf(arch);
  const lines: string[] = [];
  const pad = (s: string | number, n: number) => String(s).padStart(n);
  const padr = (s: string | number, n: number) => String(s).padEnd(n);

  lines.push(`${arch.name} — ${arch.inputSize}×${arch.inputSize}`);
  lines.push('');
  lines.push(
    `${padr('op', 20)}${padr('kind', 13)}${pad('out h×w×c', 16)}${pad('MFLOP', 10)}${pad('params', 10)}`,
  );
  lines.push('─'.repeat(69));
  for (const c of cost.ops) {
    const s = `${c.outShape.h}×${c.outShape.w}×${c.outShape.c}`;
    lines.push(
      `${padr(c.name, 20)}${padr(c.kind, 13)}${pad(s, 16)}${pad((c.macs * 2e-6).toFixed(1), 10)}${pad(c.params, 10)}`,
    );
  }
  lines.push('─'.repeat(69));
  lines.push(
    `${padr('TOTAL', 33)}${pad('', 16)}${pad((cost.totalMacs * 2e-6).toFixed(1), 10)}${pad(cost.totalParams, 10)}`,
  );
  lines.push('');
  lines.push(`dispatches            : ${cost.dispatches}`);
  lines.push(`GFLOP / frame         : ${(cost.totalMacs * 2e-9).toFixed(3)}`);
  lines.push(`parameters            : ${(cost.totalParams / 1e6).toFixed(2)} M`);
  lines.push(`weights @ fp16        : ${(cost.totalParams * 2) / 1e6} MB`);
  lines.push(
    `all tensors live @fp16: ${(cost.naiveTensorBytes(2) / 1e6).toFixed(2)} MB (before arena aliasing)`,
  );
  lines.push('');
  lines.push('Projected M1 Max inference time at various achieved efficiencies');
  lines.push('(peak fp32 throughput 10.4 TFLOP/s):');
  for (const eff of [0.05, 0.1, 0.15, 0.2, 0.3]) {
    const ms = (cost.totalMacs * 2) / (10.4e12 * eff) * 1e3;
    lines.push(`  ${pad((eff * 100).toFixed(0) + '%', 4)} of peak → ${ms.toFixed(2)} ms`);
  }
  return lines.join('\n');
}
