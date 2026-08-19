/**
 * Builds every pipeline and bind group the network needs, once, and then
 * records the whole forward pass into a caller-supplied compute pass.
 *
 * Nothing here allocates GPU resources per frame and nothing reads back to the
 * CPU: `record()` only issues dispatches. That is what allows inference,
 * lighting and drawing to share a single `GPUCommandEncoder` and a single
 * `queue.submit`, with no interop, no staging buffers and no fences between the
 * depth model and the renderer that consumes its output.
 */
import type {
  SampledFlag,
  TgpuBindGroup,
  TgpuComputePass,
  TgpuComputePipeline,
  TgpuRoot,
  TgpuTexture,
} from 'typegpu';
import * as d from 'typegpu/data';
import type { Architecture, Op } from './arch.ts';
import { ILLUMINA_DEPTH_448 } from './arch.ts';
import {
  groups,
  packBias,
  packConv,
  packDepthwise,
  packPointwise,
} from './layout.ts';
import { planArena, validateArena, type ArenaPlan } from '../engine/arena.ts';
import { opWeights, type WeightMap } from './weights.ts';
import { makePointwise, pointwiseLayout } from './kernels/pointwise.ts';
import { makeDepthwisePointwise, dwpwLayout } from './kernels/dwpw.ts';
import { makeConv, convLayout } from './kernels/conv.ts';
import { makeLateral, lateralLayout } from './kernels/lateral.ts';
import {
  DepthPostParams,
  PreprocessParams,
  depthUpsampleLayout,
  headLayout,
  makeDepthUpsample,
  makeHead,
  makePreprocess,
  preprocessLayout,
} from './kernels/io.ts';

/**
 * Allocating through these two helpers (rather than calling `root.createTexture`
 * / `createBuffer` inline) keeps the usage flags in the inferred type, which is
 * what lets `createBindGroup` check statically that a resource is bindable where
 * it is used.
 */
function createWorkTexture(root: TgpuRoot, size: number) {
  return root
    .createTexture({ size: [size, size], format: 'rgba16float' })
    .$usage('sampled', 'storage');
}

function createStorage(root: TgpuRoot, vec4Count: number) {
  return root.createBuffer(d.arrayOf(d.vec4f, vec4Count)).$usage('storage');
}

type Buf = ReturnType<typeof createStorage>;
export type WorkTexture = ReturnType<typeof createWorkTexture>;

/** Any 2D texture the preprocessing kernel can sample from. */
export type SourceTexture = SampledFlag &
  TgpuTexture<{ size: readonly number[]; format: GPUTextureFormat }>;

/** A single recorded dispatch. */
interface Step {
  readonly name: string;
  readonly kind: Op['kind'];
  readonly pipeline: TgpuComputePipeline;
  readonly group: TgpuBindGroup;
  readonly dispatch: readonly [number, number, number];
  readonly tiling: string;
}

export interface DepthModelOptions {
  readonly arch?: Architecture;
  readonly weights: WeightMap;
  /**
   * Source image, at least as large as the network input. The caller keeps it
   * up to date (from a video frame, a still, or a scene rendered on the GPU).
   */
  readonly source: SourceTexture;
  /** How the source maps onto the network's square input. */
  readonly uvScale?: readonly [number, number];
  readonly uvOffset?: readonly [number, number];
  /** Set false when the source texture is already linear-light. */
  readonly decodeSrgb?: boolean;
  readonly exposure?: number;
}

export interface DepthModelStats {
  readonly dispatches: number;
  readonly arenaBytes: number;
  readonly naiveArenaBytes: number;
  readonly weightBytes: number;
  readonly steps: ReadonlyArray<{ name: string; kind: string; dispatch: readonly number[]; tiling: string }>;
}

export interface DepthModel {
  readonly arch: Architecture;
  /**
   * Stabilised depth for the current frame, `rgba16float`:
   * `x` = disparity in `[0, 1]` (1 = nearest), `y` = this frame's raw
   * (untemporalised) disparity, `z` = temporal change magnitude, `w` = 1.
   */
  currentDepth(): WorkTexture;
  /** Linear-light copy of the source at network resolution. */
  readonly sceneColor: WorkTexture;
  /** Records the full forward pass. Call once per frame, inside one encoder. */
  record(pass: TgpuComputePass): void;
  /** Advances the temporal history ping-pong. Call once per frame after record. */
  swapHistory(): void;
  /**
   * Index of the depth texture the next `record()` will write, for callers that
   * hold per-parity bind groups over the same ping-pong pair.
   */
  outputParity(): number;
  /**
   * Index of the texture holding the most recent result. This is *not* always
   * `outputParity()`: after `swapHistory()` the two differ, and a caller that
   * skipped inference this frame must shade from the last thing actually
   * written, not from the slot that is about to be overwritten.
   */
  freshParity(): number;
  /** Both depth ping-pong textures, so a consumer can build per-parity bind groups. */
  debugDepth(index: number): WorkTexture;
  setExposure(exposure: number): void;
  setTemporal(baseAlpha: number, motionSensitivity: number, rangeSigma: number): void;
  readonly stats: DepthModelStats;
  /**
   * Looks up the arena buffer backing a named intermediate tensor, for tests and
   * the debug visualiser.
   *
   * `liveAtEnd` is false when the arena has handed that buffer to a later
   * tensor, in which case reading it after a full `record()` yields the *other*
   * tensor's data, not this one's.
   */
  debugTensor(
    name: string,
  ): { buffer: Buf; shape: { h: number; w: number; c: number }; liveAtEnd: boolean } | undefined;
  destroy(): void;
}

function upload(root: TgpuRoot, data: Float32Array): Buf {
  const buf = createStorage(root, data.length / 4);
  root.device.queue.writeBuffer(
    root.unwrap(buf),
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
  return buf;
}

export function createDepthModel(root: TgpuRoot, options: DepthModelOptions): DepthModel {
  const arch = options.arch ?? ILLUMINA_DEPTH_448;
  const size = arch.inputSize;

  // --- textures -----------------------------------------------------------
  const sceneColor = createWorkTexture(root, size);

  // Ping-pong so the upsample kernel can read last frame's result while writing
  // this frame's; WebGPU forbids reading and writing one texture in a dispatch.
  const depth: [WorkTexture, WorkTexture] = [
    createWorkTexture(root, size),
    createWorkTexture(root, size),
  ];
  let historyIndex = 0;
  /** Slot most recently written by `record()`. */
  let lastWritten = 1;

  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  // --- uniforms -----------------------------------------------------------
  const preprocessParams = root.createUniform(PreprocessParams, {
    uvScale: d.vec2f(options.uvScale?.[0] ?? 1, options.uvScale?.[1] ?? 1),
    uvOffset: d.vec2f(options.uvOffset?.[0] ?? 0, options.uvOffset?.[1] ?? 0),
    options: d.vec4f(options.decodeSrgb === false ? 0 : 1, options.exposure ?? 1, 0, 0),
  });
  const depthPostParams = root.createUniform(DepthPostParams, {
    options: d.vec4f(0.12, 0.35, 3.0, 0),
  });

  // --- arena --------------------------------------------------------------
  const textureBacked = new Set(['depth']);
  const plan: ArenaPlan = planArena(arch, textureBacked);
  validateArena(arch, plan, textureBacked);
  const arena: Buf[] = plan.bufferSizes.map((n) => createStorage(root, n));
  const tensorBuf = (name: string): Buf => {
    const idx = plan.assignment.get(name);
    if (idx === undefined) throw new Error(`runner: ${name} has no arena buffer`);
    return arena[idx]!;
  };

  // --- per-op resources ---------------------------------------------------
  const steps: Step[] = [];
  const ownedBuffers: Buf[] = [];
  let weightBytes = 0;

  const uploadWeights = (data: Float32Array): Buf => {
    weightBytes += data.byteLength;
    const buf = upload(root, data);
    ownedBuffers.push(buf);
    return buf;
  };

  const shape = (n: string) => arch.tensors[n]!;

  let upsampleStep!: {
    name: string;
    pipeline: TgpuComputePipeline;
    groups: TgpuBindGroup[];
    dispatch: readonly [number, number, number];
    tiling: string;
  };


  for (const op of arch.ops) {
    switch (op.kind) {
      case 'preprocess': {
        const k = makePreprocess({ size, mean: arch.mean, std: arch.std });
        const group = root.createBindGroup(preprocessLayout, {
          source: options.source,
          samp: sampler,
          params: preprocessParams,
          dst: tensorBuf(op.out),
          sceneColor: sceneColor.createView(
            d.textureStorage2d('rgba16float', 'write-only'),
          ),
        });
        steps.push({
          name: op.name,
          kind: op.kind,
          pipeline: root.createComputePipeline({ compute: k.fn }),
          group,
          dispatch: k.dispatch,
          tiling: k.tiling,
        });
        break;
      }

      case 'conv': {
        const si = shape(op.in);
        const so = shape(op.out);
        const w = opWeights(op, options.weights);
        const k = makeConv({
          inC4: groups(si.c),
          outC4: groups(so.c),
          inH: si.h,
          inW: si.w,
          outH: so.h,
          outW: so.w,
          k: op.k,
          stride: op.stride,
          act: op.act,
        });
        const group = root.createBindGroup(convLayout, {
          src: tensorBuf(op.in),
          wgt: uploadWeights(packConv(w.weight!, so.c, si.c, op.k)),
          bias: uploadWeights(packBias(w.bias!, so.c)),
          dst: tensorBuf(op.out),
        });
        steps.push({
          name: op.name,
          kind: op.kind,
          pipeline: root.createComputePipeline({ compute: k.fn }),
          group,
          dispatch: k.dispatch,
          tiling: k.tiling,
        });
        break;
      }

      case 'pw': {
        const si = shape(op.in);
        const so = shape(op.out);
        const w = opWeights(op, options.weights);
        const k = makePointwise({
          inC4: groups(si.c),
          outC4: groups(so.c),
          pixels: so.h * so.w,
          act: op.act,
          residual: op.residual !== undefined,
        });
        const group = root.createBindGroup(pointwiseLayout, {
          src: tensorBuf(op.in),
          wgt: uploadWeights(packPointwise(w.weight!, so.c, si.c)),
          bias: uploadWeights(packBias(w.bias!, so.c)),
          // Unused residual slots are bound to the source buffer: a valid,
          // already-resident binding the shader provably never reads.
          res: tensorBuf(op.residual ?? op.in),
          dst: tensorBuf(op.out),
        });
        steps.push({
          name: op.name,
          kind: op.kind,
          pipeline: root.createComputePipeline({ compute: k.fn }),
          group,
          dispatch: k.dispatch,
          tiling: k.tiling,
        });
        break;
      }

      case 'dwpw': {
        const si = shape(op.in);
        const so = shape(op.out);
        const w = opWeights(op, options.weights);
        const k = makeDepthwisePointwise({
          inC4: groups(si.c),
          outC4: groups(so.c),
          inH: si.h,
          inW: si.w,
          outH: so.h,
          outW: so.w,
          k: op.k,
          stride: op.stride,
          midAct: op.midAct,
          act: op.act,
          residual: op.residual !== undefined,
        });
        const group = root.createBindGroup(dwpwLayout, {
          src: tensorBuf(op.in),
          dwWgt: uploadWeights(packDepthwise(w.dwWeight!, si.c, op.k)),
          dwBias: uploadWeights(packBias(w.dwBias!, si.c)),
          pwWgt: uploadWeights(packPointwise(w.pwWeight!, so.c, si.c)),
          pwBias: uploadWeights(packBias(w.pwBias!, so.c)),
          res: tensorBuf(op.residual ?? op.in),
          dst: tensorBuf(op.out),
        });
        steps.push({
          name: op.name,
          kind: op.kind,
          pipeline: root.createComputePipeline({ compute: k.fn }),
          group,
          dispatch: k.dispatch,
          tiling: k.tiling,
        });
        break;
      }

      case 'lateral': {
        const sc = shape(op.coarse);
        const ss = shape(op.skip);
        const so = shape(op.out);
        const w = opWeights(op, options.weights);
        const k = makeLateral({
          outC4: groups(so.c),
          skipC4: groups(ss.c),
          coarseH: sc.h,
          coarseW: sc.w,
          outH: so.h,
          outW: so.w,
          act: op.act,
        });
        const group = root.createBindGroup(lateralLayout, {
          coarse: tensorBuf(op.coarse),
          skip: tensorBuf(op.skip),
          wgt: uploadWeights(packPointwise(w.weight!, so.c, ss.c)),
          bias: uploadWeights(packBias(w.bias!, so.c)),
          dst: tensorBuf(op.out),
        });
        steps.push({
          name: op.name,
          kind: op.kind,
          pipeline: root.createComputePipeline({ compute: k.fn }),
          group,
          dispatch: k.dispatch,
          tiling: k.tiling,
        });
        break;
      }

      case 'head': {
        const si = shape(op.in);
        const so = shape(op.out);
        const w = opWeights(op, options.weights);
        const k = makeHead(groups(si.c), so.h * so.w);
        // The head is a 1×C convolution; pad both operands out to whole `vec4`s.
        const biasVec = new Float32Array(4);
        biasVec[0] = w.bias![0]!;
        const group = root.createBindGroup(headLayout, {
          src: tensorBuf(op.in),
          wgt: uploadWeights(Float32Array.from(w.weight!)),
          bias: uploadWeights(biasVec),
          dst: tensorBuf(op.out),
        });
        steps.push({
          name: op.name,
          kind: op.kind,
          pipeline: root.createComputePipeline({ compute: k.fn }),
          group,
          dispatch: k.dispatch,
          tiling: k.tiling,
        });
        break;
      }

      case 'bilateralUp': {
        const si = shape(op.in);
        const k = makeDepthUpsample(si.w, si.h, size);
        // Two bind groups, one per ping-pong parity, both built up front.
        const built = [0, 1].map((i) =>
          root.createBindGroup(depthUpsampleLayout, {
            low: tensorBuf(op.in),
            guide: sceneColor.createView(d.texture2d(d.f32)),
            history: depth[i]!.createView(d.texture2d(d.f32)),
            params: depthPostParams,
            dst: depth[1 - i]!.createView(d.textureStorage2d('rgba16float', 'write-only')),
          }),
        );
        const pipeline = root.createComputePipeline({ compute: k.fn });
        // Recorded specially below so it can pick the right parity each frame.
        upsampleStep = { name: op.name, pipeline, groups: built, dispatch: k.dispatch, tiling: k.tiling };
        break;
      }
    }
  }

  const stats: DepthModelStats = {
    dispatches: steps.length + 1,
    arenaBytes: plan.totalBytes,
    naiveArenaBytes: plan.naiveBytes,
    weightBytes,
    steps: [
      ...steps.map((s) => ({ name: s.name, kind: s.kind, dispatch: s.dispatch, tiling: s.tiling })),
      { name: 'upsample', kind: 'bilateralUp', dispatch: [0, 0, 0], tiling: '' },
    ],
  };

  return {
    arch,
    sceneColor,
    currentDepth: () => depth[lastWritten]!,
    record(pass: TgpuComputePass) {
      lastWritten = 1 - historyIndex;
      for (const s of steps) {
        s.pipeline.with(s.group).with(pass).dispatchWorkgroups(...s.dispatch);
      }
      upsampleStep.pipeline
        .with(upsampleStep.groups[historyIndex]!)
        .with(pass)
        .dispatchWorkgroups(...upsampleStep.dispatch);
    },
    swapHistory() {
      historyIndex = 1 - historyIndex;
    },
    outputParity: () => 1 - historyIndex,
    freshParity: () => lastWritten,
    debugDepth: (index: number) => depth[index === 0 ? 0 : 1],
    setExposure(exposure: number) {
      preprocessParams.writePartial({
        options: d.vec4f(options.decodeSrgb === false ? 0 : 1, exposure, 0, 0),
      });
    },
    setTemporal(baseAlpha: number, motionSensitivity: number, rangeSigma: number) {
      depthPostParams.writePartial({
        options: d.vec4f(rangeSigma, baseAlpha, motionSensitivity, 0),
      });
    },
    debugTensor(name: string) {
      const shape = arch.tensors[name];
      const idx = plan.assignment.get(name);
      if (!shape || idx === undefined) return undefined;
      // The tensor still owns its buffer only if nothing written later shares it.
      const producedAt = arch.ops.findIndex((op) => op.out === name);
      const stolen = arch.ops.some(
        (op, i) => i > producedAt && plan.assignment.get(op.out) === idx,
      );
      return { buffer: arena[idx]!, shape, liveAtEnd: !stolen };
    },
    stats,
    destroy() {
      for (const b of [...arena, ...ownedBuffers]) b.destroy();
      sceneColor.destroy();
      depth[0].destroy();
      depth[1].destroy();
    },
  };
}
