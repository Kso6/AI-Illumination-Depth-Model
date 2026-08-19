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
import { createBlitter, type Blitter, type CopyPath } from './blit.ts';

export type SourceMode = 'procedural' | 'camera' | 'image';

/**
 * Builds the hidden video element that a camera stream decodes into.
 *
 * Exported so the browser probe can exercise the real element rather than a
 * copy of it: every property here exists because some engine refuses to decode
 * without it, and a probe against a re-typed element would prove nothing.
 */
export function attachCameraElement(stream: MediaStream): HTMLVideoElement {
  const el = document.createElement('video');
  el.playsInline = true;
  // `muted` must be set before the stream is attached for autoplay to be
  // allowed, and the attribute has to be there too — the property alone is not
  // enough in every engine.
  el.muted = true;
  el.defaultMuted = true;
  el.setAttribute('muted', '');
  el.setAttribute('playsinline', '');
  el.autoplay = true;
  // Rendered, not merely present. A media element that is `display:none`,
  // zero-sized, or entirely outside the viewport is not guaranteed to decode —
  // WebKit in particular will hold a MediaStream element at `readyState` 0
  // forever, which is exactly the "Camera — starting…" that never advanced. Two
  // visible pixels in the corner at 1 % opacity keep it a rendered element.
  el.style.cssText =
    'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0.01;' +
    'pointer-events:none;z-index:-1';
  document.body.append(el);
  el.srcObject = stream;
  return el;
}

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
  /** When the camera was requested, so a stall can be reported rather than hung on. */
  let cameraRequestedAt = 0;
  let frameTick = 0;
  let stallReported = false;
  let lastCopyPath: CopyPath = 'none';

  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    stream = undefined;
    if (video) {
      video.srcObject = null;
      video.remove();
      video = undefined;
    }
    cameraReady = false;
    stallReported = false;
    lastCopyPath = 'none';
  };

  /**
   * Everything that decides whether a frame can be pulled, in one string. A
   * camera that produces no frames is otherwise indistinguishable from one that
   * was never granted, and the two have completely different fixes.
   */
  const cameraDiagnosis = (): string => {
    if (!video) return 'no video element';
    const track = stream?.getVideoTracks()[0];
    return [
      `readyState ${video.readyState}`,
      `${video.videoWidth}x${video.videoHeight}`,
      video.paused ? 'paused' : 'playing',
      track ? `track ${track.readyState}${track.muted ? ' (muted)' : ''}` : 'no track',
    ].join(', ');
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
      cameraRequestedAt = performance.now();
      stallReported = false;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          // The network consumes 448x448, so a 720p feed is wasted work at every
          // stage: more to decode, more to copy across the bus, and a staging
          // texture three times the size. 640x480 still exceeds the short axis
          // the centre crop needs, so nothing is lost visually.
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        });
        const el = attachCameraElement(stream);
        // Assigned before `play()` resolves. The promise can stay pending for a
        // long time, and `update()` polling a live element is what actually
        // detects the first frame; waiting here only delays that.
        video = el;
        label = 'Camera — starting…';
        el.play().catch(() => {
          // Reported by the stall diagnosis below rather than thrown away: a
          // rejected play() with frames still arriving is not fatal.
        });
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
        frameTick++;
        // `videoWidth` is only meaningful once metadata has arrived, and
        // `readyState >= 2` guarantees a decoded frame exists to copy.
        if (video.readyState >= 2 && video.videoWidth > 0) {
          // Mark the camera live *before* uploading. If the upload throws, the
          // frame is merely stale — but leaving `cameraReady` false would strand
          // the app on the procedural fallback with a misleading label, which is
          // far harder to diagnose than a dropped frame.
          if (!cameraReady) {
            cameraReady = true;
            stallReported = false;
            label = `Camera — ${video.videoWidth}×${video.videoHeight}`;
          }
          try {
            blitter.upload(video, video.videoWidth, video.videoHeight, true);
            // Which import path won is worth showing: it is the difference
            // between a zero-copy import and a per-frame trip through a 2-D
            // canvas, and it is invisible from anywhere else.
            if (blitter.copyPath !== lastCopyPath) {
              lastCopyPath = blitter.copyPath;
              label = `Camera — ${video.videoWidth}×${video.videoHeight} · ${lastCopyPath} copy`;
            }
          } catch (err) {
            label = `Camera frame dropped: ${(err as Error).message}`;
          }
        } else {
          // Autoplay can be refused, and an element that was throttled while
          // hidden can come back paused. Retrying costs nothing every half
          // second or so and recovers both cases without a user gesture.
          if (video.paused && frameTick % 30 === 0) {
            video.play().catch(() => {});
          }
          // A camera that is granted but silent looks identical to one that was
          // never granted, so say which it is instead of sitting on "starting…".
          if (!stallReported && performance.now() - cameraRequestedAt > 3000) {
            stallReported = true;
            label = `Camera — permission granted but no frames yet (${cameraDiagnosis()})`;
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
