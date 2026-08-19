/**
 * The frame graph.
 *
 * This is the file the whole project exists to make possible. A frame is:
 *
 *     encoder ─┬─ compute pass ─┬─ 40 inference dispatches
 *              │                └─  1 shading dispatch
 *              └─ render pass  ──   1 tone-mapping draw
 *     submit
 *
 * One `GPUCommandEncoder`. One `queue.submit`. No readbacks, no staging buffers,
 * no fences, no interop between an ML runtime and a renderer — because there is
 * no separate ML runtime. The depth map is written by a WGSL kernel and read by
 * another WGSL kernel forty microseconds later, and WebGPU's own intra-pass
 * ordering guarantees are the only synchronisation involved.
 *
 * `profile: true` splits the compute pass into one pass per stage so timestamp
 * queries can attribute GPU time to inference versus lighting. That costs extra
 * pass boundaries, which is exactly why it is not the default.
 */
import type { ColorAttachment, TgpuRenderPass, TgpuRoot } from 'typegpu';
import type { DepthModel } from '../model/runner.ts';
import type { Renderer } from '../lighting/renderer.ts';
import type { GpuProfiler } from './timing.ts';

/**
 * An optional scene rendered into the source texture at the top of the frame.
 * When present, the image the network analyses is itself produced on the GPU in
 * this same encoder, so a frame contains no CPU-side image handling at all.
 */
export interface ScenePass {
  record(pass: TgpuRenderPass): void;
  readonly colorTarget: ColorAttachment['view'];
  readonly truthTarget: ColorAttachment['view'];
}

export interface FrameOptions {
  readonly root: TgpuRoot;
  readonly model: DepthModel;
  readonly renderer: Renderer;
  /**
   * Where the composite pass draws. Normally a configured `GPUCanvasContext`;
   * tests pass a render-view of an offscreen texture instead, which is why this
   * is not narrowed to a canvas.
   */
  readonly target: ColorAttachment['view'];
  readonly scene?: ScenePass | undefined;
  readonly profiler?: GpuProfiler | undefined;
  /** Split into per-stage passes so timestamps can attribute time. */
  readonly profile?: boolean;
  readonly clearColor?: GPUColor;
}

export interface FrameResult {
  /** Compute dispatches recorded, including the shading pass. */
  readonly dispatches: number;
  /** Draw calls recorded: the composite, plus the scene when one is drawn. */
  readonly draws: number;
  /** Command encoders used. Always one. */
  readonly encoders: number;
  /** Queue submissions. Always one. */
  readonly submits: number;
}

/**
 * Records and submits one frame.
 *
 * The caller is responsible for having updated the source texture and the
 * renderer's uniforms beforehand; nothing in here touches the queue except the
 * single submit at the end.
 */
export function renderFrame(options: FrameOptions): FrameResult {
  const { root, model, renderer, target, scene, profiler, profile = false } = options;

  profiler?.beginFrame();

  const encoder = root['~unstable'].createCommandEncoder({ label: 'illumina-frame' });

  if (scene) {
    const scenePass = encoder.beginRenderPass({
      label: 'procedural-scene',
      colorAttachments: [
        { view: scene.colorTarget, loadOp: 'clear', storeOp: 'store' },
        { view: scene.truthTarget, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    scene.record(scenePass);
    scenePass.end();
  }

  const parity = model.outputParity();

  const dispatches = model.stats.dispatches + 1;

  if (profile) {
    // Two passes so the boundary between them can be timestamped. The extra
    // pass boundary is the price of attribution.
    const inferencePass = encoder.beginComputePass({
      label: 'inference',
      timestampWrites: profiler?.span('inference'),
    });
    model.record(inferencePass);
    inferencePass.end();

    const shadePass = encoder.beginComputePass({
      label: 'shading',
      timestampWrites: profiler?.span('shading'),
    });
    renderer.recordShade(shadePass, parity);
    shadePass.end();
  } else {
    const pass = encoder.beginComputePass({
      label: 'inference+shading',
      timestampWrites: profiler?.span('inference+shading'),
    });
    model.record(pass);
    renderer.recordShade(pass, parity);
    pass.end();
  }

  const renderPass = encoder.beginRenderPass({
    label: 'composite',
    colorAttachments: [
      {
        view: target,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: options.clearColor ?? { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
    timestampWrites: profiler?.span('composite'),
  });
  renderer.recordComposite(renderPass);
  renderPass.end();

  // Timestamp resolution has to be recorded into the same encoder, after the
  // passes it measures and before the encoder is finished.
  if (profiler?.available) {
    profiler.resolve(root.unwrap(encoder));
  }

  encoder.submit();
  profiler?.afterSubmit();
  model.swapHistory();

  return { dispatches, draws: scene ? 2 : 1, encoders: 1, submits: 1 };
}
