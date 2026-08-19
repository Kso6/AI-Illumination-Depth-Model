/**
 * Where the image comes from.
 *
 * Whatever the mode, the rest of the application sees exactly one source
 * texture, created once and never replaced. That matters more than it looks:
 * every bind group in the network and the renderer references it, so a mode
 * switch that allocated a new texture would force rebuilding forty-odd bind
 * groups mid-frame. Instead the texture is either *rendered into* (procedural
 * scene) or *copied into* (camera, video, still image).
 *
 * Note on the single-encoder claim: the procedural scene's draw is recorded into
 * the frame's encoder like everything else. Camera and video frames arrive
 * through `copyExternalImageToTexture`, which is a queue operation by
 * construction — WebGPU offers no way to import an external image inside a
 * command encoder. That copy is frame *acquisition*, and stays outside the
 * inference → lighting → draw chain, which remains a single encoder and a
 * single submit.
 */
import type { TgpuRenderPass, TgpuRoot } from 'typegpu';
import { createProceduralScene, type ProceduralScene } from './procedural.ts';

export type SourceMode = 'procedural' | 'camera' | 'image';

const COLOR_FORMAT: GPUTextureFormat = 'rgba8unorm';
const TRUTH_FORMAT: GPUTextureFormat = 'rgba16float';

function createColor(root: TgpuRoot, size: number) {
  return root
    .createTexture({ size: [size, size], format: COLOR_FORMAT })
    .$usage('sampled', 'render');
}

function createTruth(root: TgpuRoot, size: number) {
  return root
    .createTexture({ size: [size, size], format: TRUTH_FORMAT })
    .$usage('sampled', 'render');
}

export interface SceneSource {
  /** The one stable source texture. */
  readonly texture: ReturnType<typeof createColor>;
  /**
   * Ground-truth depth for the procedural scene: `x` = view-space metres,
   * `y` = 1 where geometry was hit. Meaningless in camera and image modes.
   */
  readonly truth: ReturnType<typeof createTruth>;
  readonly mode: SourceMode;
  readonly hasGroundTruth: boolean;
  /** Human-readable description of what is currently feeding the network. */
  readonly label: string;

  setMode(mode: SourceMode): Promise<void>;
  loadImage(source: ImageBitmap | HTMLImageElement): void;
  /** Per-frame update; uploads a video frame when in camera mode. */
  update(timeSeconds: number, orbit: boolean): void;
  /** Records the procedural draw. A no-op in the other modes. */
  record(pass: TgpuRenderPass): void;
  /** True when this frame needs a render pass for the scene. */
  readonly needsScenePass: boolean;
  stop(): void;
}

export function createSceneSource(root: TgpuRoot, size: number): SceneSource {
  const texture = createColor(root, size);
  const truth = createTruth(root, size);
  const scene: ProceduralScene = createProceduralScene(root, COLOR_FORMAT, TRUTH_FORMAT);

  let mode: SourceMode = 'procedural';
  let label = 'Procedural scene (ray-marched, ground truth available)';
  let video: HTMLVideoElement | undefined;
  let stream: MediaStream | undefined;

  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    stream = undefined;
    if (video) {
      video.srcObject = null;
      video = undefined;
    }
  };

  /** Copies a source of any aspect ratio into the square texture, centre-cropped. */
  const copyCropped = (
    src: ImageBitmap | HTMLVideoElement | HTMLImageElement,
    w: number,
    h: number,
  ) => {
    if (w === 0 || h === 0) return;
    const side = Math.min(w, h);
    root.device.queue.copyExternalImageToTexture(
      {
        source: src as GPUCopyExternalImageSource,
        origin: { x: Math.floor((w - side) / 2), y: Math.floor((h - side) / 2) },
      },
      { texture: root.unwrap(texture) },
      // The copy cannot scale, so take the largest centred square that fits and
      // let the preprocess kernel's sampler do the resampling to 448².
      { width: Math.min(side, size), height: Math.min(side, size) },
    );
  };

  return {
    texture,
    truth,
    get mode() {
      return mode;
    },
    get hasGroundTruth() {
      return mode === 'procedural';
    },
    get label() {
      return label;
    },
    get needsScenePass() {
      return mode === 'procedural';
    },

    async setMode(next) {
      if (next === mode) return;
      if (mode === 'camera') stopCamera();
      mode = next;

      if (next === 'procedural') {
        label = 'Procedural scene (ray-marched, ground truth available)';
        return;
      }
      if (next === 'image') {
        label = 'Still image';
        return;
      }

      label = 'Camera — requesting permission…';
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        });
        video = document.createElement('video');
        video.srcObject = stream;
        video.playsInline = true;
        video.muted = true;
        await video.play();
        label = `Camera — ${video.videoWidth}×${video.videoHeight}`;
      } catch (err) {
        mode = 'procedural';
        label = `Camera unavailable (${(err as Error).name}); using the procedural scene`;
      }
    },

    loadImage(src) {
      mode = 'image';
      const w = 'width' in src ? src.width : 0;
      const h = 'height' in src ? src.height : 0;
      copyCropped(src, w, h);
      label = `Still image — ${w}×${h}`;
    },

    update(time, orbit) {
      if (mode === 'procedural') {
        scene.update(time, orbit);
      } else if (mode === 'camera' && video && video.readyState >= 2) {
        copyCropped(video, video.videoWidth, video.videoHeight);
      }
    },

    record(pass) {
      if (mode === 'procedural') scene.record(pass);
    },

    stop() {
      stopCamera();
      scene.destroy();
      texture.destroy();
      truth.destroy();
    },
  };
}
