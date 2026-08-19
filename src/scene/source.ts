/**
 * Where the image comes from.
 *
 * Whatever the mode, the rest of the application sees exactly one source
 * texture, created once and never replaced. That matters more than it looks:
 * every bind group in the network and the renderer references it, so a mode
 * switch that allocated a new texture would force rebuilding forty-odd bind
 * groups mid-frame.
 *
 * The texture is always *rendered into*, by one of two draws recorded in the
 * frame's own command encoder: the procedural scene's ray-march, or a blit that
 * scales a camera frame or still image down to the network's square input.
 * Camera and image frames reach the GPU through `copyExternalImageToTexture`,
 * which is a queue operation by construction — WebGPU offers no way to import an
 * external image inside a command encoder. That copy is frame *acquisition* and
 * stays outside the inference → lighting → draw chain, which remains one encoder
 * and one submit.
 */
import type { TgpuRenderPass, TgpuRoot } from 'typegpu';
import { createProceduralScene, type ProceduralScene } from './procedural.ts';
import { createBlitter, type Blitter } from './blit.ts';

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
   * `y` = 1 where geometry was hit. The blit writes `y = 0`, so camera and
   * image frames are correctly reported as having no ground truth.
   */
  readonly truth: ReturnType<typeof createTruth>;
  readonly mode: SourceMode;
  readonly hasGroundTruth: boolean;
  /** Human-readable description of what is currently feeding the network. */
  readonly label: string;
  /**
   * True when the source changed this frame. A still image does not change, so
   * the frame loop can skip re-running forty dispatches on identical input.
   */
  readonly changed: boolean;

  setMode(mode: SourceMode): Promise<void>;
  loadImage(source: ImageBitmap): void;
  /** Per-frame update; uploads a video frame when in camera mode. */
  update(timeSeconds: number, orbit: boolean): void;
  /** Records the draw that fills the source texture. */
  record(pass: TgpuRenderPass): void;
  stop(): void;
}

export function createSceneSource(root: TgpuRoot, size: number): SceneSource {
  const texture = createColor(root, size);
  const truth = createTruth(root, size);
  const scene: ProceduralScene = createProceduralScene(root, COLOR_FORMAT, TRUTH_FORMAT);
  const blitter: Blitter = createBlitter(root, COLOR_FORMAT, TRUTH_FORMAT);

  let mode: SourceMode = 'procedural';
  let label = 'Procedural scene (ray-marched, ground truth available)';
  let video: HTMLVideoElement | undefined;
  let stream: MediaStream | undefined;
  let cameraReady = false;
  let changed = true;
  /** Redraws to run after a one-shot source change, so both ping-pong slots see it. */
  let dirtyFrames = 0;

  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    stream = undefined;
    if (video) {
      video.srcObject = null;
      video.remove();
      video = undefined;
    }
    cameraReady = false;
  };

  /**
   * Which source is actually driving the texture right now. While the camera is
   * still negotiating permission or waiting for its first frame, the procedural
   * scene keeps drawing — otherwise the texture would freeze on a stale frame
   * and the display would look broken for as long as the user takes to click
   * "Allow".
   */
  const activeMode = (): SourceMode =>
    mode === 'camera' && !cameraReady ? 'procedural' : mode;

  return {
    texture,
    truth,
    get mode() {
      return mode;
    },
    get hasGroundTruth() {
      return activeMode() === 'procedural';
    },
    get label() {
      return label;
    },
    get changed() {
      return changed;
    },

    async setMode(next) {
      if (next === mode) return;
      if (mode === 'camera') stopCamera();
      mode = next;
      changed = true;

      if (next === 'procedural') {
        label = 'Procedural scene (ray-marched, ground truth available)';
        return;
      }
      if (next === 'image') {
        label = blitter.ready ? 'Still image' : 'Still image — none loaded yet';
        dirtyFrames = 3;
        return;
      }

      label = 'Camera — requesting permission…';
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          // The network consumes 448x448, so a 720p feed is wasted work at every
          // stage: more to decode, more to copy across the bus, and a staging
          // texture three times the size. 640x480 still exceeds the short axis
          // the centre crop needs, so nothing is lost visually.
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        });
        const el = document.createElement('video');
        el.srcObject = stream;
        el.playsInline = true;
        el.muted = true;
        el.autoplay = true;
        // A detached media element is not guaranteed to decode: Safari will not
        // produce frames for one that is outside the document, and Chrome may
        // throttle it. Keeping it in the DOM but invisible is what makes the
        // camera path work across browsers.
        el.style.cssText =
          'position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;left:-10px;top:-10px';
        document.body.append(el);
        await el.play();
        video = el;
        label = 'Camera — starting…';
      } catch (err) {
        stopCamera();
        mode = 'procedural';
        label = `Camera unavailable (${(err as Error).name}); using the procedural scene`;
      }
    },

    loadImage(source) {
      mode = 'image';
      stopCamera();
      try {
        blitter.upload(source, source.width, source.height, false);
        label = `Still image — ${source.width}×${source.height}`;
      } catch (err) {
        // Not all WebGPU backends implement external-image import; a user
        // action must surface that, not throw out of the event handler.
        label = `Could not load image: ${(err as Error).message}`;
      }
      // The image is static, so inference would otherwise run forever on
      // identical input. Redraw a few frames to fill both depth history slots,
      // then let the frame loop go idle.
      dirtyFrames = 3;
      changed = true;
    },

    update(time, orbit) {
      // Pull a camera frame *before* deciding what to draw. Gating the upload on
      // `activeMode()` deadlocks: that reports 'procedural' until `cameraReady`
      // is set, and `cameraReady` is only ever set here — so the camera would
      // stay one step from working forever.
      if (mode === 'camera' && video) {
        // `videoWidth` is only meaningful once metadata has arrived, and
        // `readyState >= 2` guarantees a decoded frame exists to copy.
        if (video.readyState >= 2 && video.videoWidth > 0) {
          // Mark the camera live *before* uploading. If the upload throws, the
          // frame is merely stale — but leaving `cameraReady` false would strand
          // the app on the procedural fallback with a misleading label, which is
          // far harder to diagnose than a dropped frame.
          if (!cameraReady) {
            cameraReady = true;
            label = `Camera — ${video.videoWidth}×${video.videoHeight}`;
          }
          try {
            blitter.upload(video, video.videoWidth, video.videoHeight, true);
          } catch (err) {
            label = `Camera frame dropped: ${(err as Error).message}`;
          }
        }
      }

      const active = activeMode();
      if (active === 'procedural') {
        scene.update(time, orbit);
        changed = true;
        return;
      }
      if (active === 'camera') {
        changed = true;
        return;
      }

      // Still image: nothing moves, so report unchanged once the redraws are done.
      changed = dirtyFrames > 0;
      if (dirtyFrames > 0) dirtyFrames--;
    },

    record(pass) {
      if (activeMode() === 'procedural') {
        scene.record(pass);
      } else {
        blitter.record(pass);
      }
    },

    stop() {
      stopCamera();
      scene.destroy();
      blitter.destroy();
      texture.destroy();
      truth.destroy();
    },
  };
}
