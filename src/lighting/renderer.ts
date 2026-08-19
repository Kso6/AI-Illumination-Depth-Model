/**
 * The lighting and presentation half of the frame: one compute dispatch that
 * shades the image from the estimated depth, and one draw that tone-maps it.
 *
 * Both are recorded into the caller's command encoder, alongside the network's
 * dispatches, so the depth map goes from "written by the last inference kernel"
 * to "sampled by the shading kernel" without a submit, a fence, or a copy.
 */
import type {
  TgpuBindGroup,
  TgpuComputePass,
  TgpuComputePipeline,
  TgpuRenderPass,
  TgpuRenderPipeline,
  TgpuRoot,
} from 'typegpu';
import * as d from 'typegpu/data';
import { LightBuffer, packLights, type LightDescription } from './lights.ts';
import { ShadeParams, makeShade, shadeLayout } from './shade.ts';
import {
  CompositeParams,
  compositeFragment,
  compositeLayout,
  compositeVertex,
} from './composite.ts';
import type { SourceTexture, WorkTexture } from '../model/runner.ts';

export type ToneMap = 'aces' | 'reinhard';

export type DebugView = 'lit' | 'depth' | 'normals' | 'occlusion' | 'albedo';

const DEBUG_IDS: Record<DebugView, number> = {
  lit: 0,
  depth: 1,
  normals: 2,
  occlusion: 3,
  albedo: 4,
};

export interface RendererOptions {
  /** Output resolution. Usually the canvas backing-store size. */
  readonly width: number;
  readonly height: number;
  /** Resolution of the depth map produced by the network. */
  readonly depthSize: number;
  /** Full-resolution source image, used as albedo. */
  readonly albedo: SourceTexture;
  /** The two ping-ponged depth textures the model writes. */
  readonly depthTextures: readonly [WorkTexture, WorkTexture];
  readonly presentFormat: GPUTextureFormat;
}

export interface ShadingSettings {
  /** Vertical field of view in radians. */
  fovY: number;
  /** Depth range the network's relative disparity is mapped onto, in metres. */
  near: number;
  far: number;
  aoStrength: number;
  shadowStrength: number;
  volumetric: number;
  exposure: number;
  ambient: readonly [number, number, number];
  ambientIntensity: number;
  roughness: number;
  fogDensity: number;
  fogColor: readonly [number, number, number];
  specular: number;
  normalStrength: number;
  albedoMix: number;
  contactShadowLength: number;
  aoRadiusPixels: number;
  toneMap: ToneMap;
  contrast: number;
  saturation: number;
  vignette: number;
  debug: DebugView;
}

export const DEFAULT_SETTINGS: ShadingSettings = {
  fovY: (58 * Math.PI) / 180,
  near: 0.35,
  far: 14,
  aoStrength: 0.85,
  shadowStrength: 0.8,
  volumetric: 0.35,
  exposure: 1.15,
  ambient: [0.16, 0.2, 0.3],
  ambientIntensity: 0.5,
  roughness: 0.55,
  fogDensity: 0.028,
  fogColor: [0.05, 0.07, 0.12],
  specular: 0.6,
  normalStrength: 1.0,
  albedoMix: 1.0,
  contactShadowLength: 0.8,
  aoRadiusPixels: 14,
  toneMap: 'aces',
  contrast: 1.04,
  saturation: 1.06,
  vignette: 0.35,
  debug: 'lit',
};

export interface Renderer {
  /** Records the shading dispatch. `parity` selects the depth ping-pong slot. */
  recordShade(pass: TgpuComputePass, parity: number): void;
  /** Records the tone-mapping draw. */
  recordComposite(pass: TgpuRenderPass): void;
  setLights(lights: readonly LightDescription[]): void;
  update(settings: ShadingSettings, frame: number): void;
  resize(width: number, height: number): void;
  readonly hdr: WorkTexture;
  destroy(): void;
}

function createHdr(root: TgpuRoot, width: number, height: number) {
  return root
    .createTexture({ size: [width, height], format: 'rgba16float' })
    .$usage('sampled', 'storage');
}

export function createRenderer(root: TgpuRoot, options: RendererOptions): Renderer {
  let { width, height } = options;
  let hdr = createHdr(root, width, height);

  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const shadeParams = root.createUniform(ShadeParams, {
    camera: d.vec4f(1, 1, 0.35, 14),
    toggles: d.vec4f(1, 1, 0.35, 1),
    ambient: d.vec4f(0.16, 0.2, 0.3, 0.5),
    misc: d.vec4f(0, 0.55, 0.028, 1),
    fog: d.vec4f(0.05, 0.07, 0.12, 0.6),
    size: d.vec4f(width, height, options.depthSize, options.depthSize),
    debug: d.vec4f(0, 1, 0.8, 14),
  });

  const compositeParams = root.createUniform(CompositeParams, {
    grade: d.vec4f(0, 1.04, 1.06, 0.35),
  });

  const lightBuffer = root.createBuffer(LightBuffer, packLights([])).$usage('storage');

  const shadePipeline: TgpuComputePipeline = root.createComputePipeline({
    compute: makeShade().fn,
  });

  const compositePipeline: TgpuRenderPipeline = root.createRenderPipeline({
    vertex: compositeVertex,
    fragment: compositeFragment,
    targets: { format: options.presentFormat },
    primitive: { topology: 'triangle-list' },
  });

  /** One shading bind group per depth ping-pong parity, plus one per HDR target. */
  let shadeGroups: TgpuBindGroup[] = [];
  let compositeGroup: TgpuBindGroup;

  const buildGroups = () => {
    shadeGroups = [0, 1].map((i) =>
      root.createBindGroup(shadeLayout, {
        depth: options.depthTextures[i]!.createView(d.texture2d(d.f32)),
        albedo: options.albedo,
        samp: sampler,
        params: shadeParams,
        lights: lightBuffer,
        out: hdr.createView(d.textureStorage2d('rgba16float', 'write-only')),
      }),
    );
    compositeGroup = root.createBindGroup(compositeLayout, {
      hdr: hdr.createView(d.texture2d(d.f32)),
      samp: sampler,
      params: compositeParams,
    });
  };
  buildGroups();

  const dispatch = (): [number, number, number] => [
    Math.ceil(width / 8),
    Math.ceil(height / 8),
    1,
  ];

  return {
    recordShade(pass, parity) {
      shadePipeline
        .with(shadeGroups[parity]!)
        .with(pass)
        .dispatchWorkgroups(...dispatch());
    },

    recordComposite(pass) {
      compositePipeline.with(compositeGroup).with(pass).draw(3);
    },

    setLights(lights) {
      lightBuffer.write(packLights(lights));
    },

    update(s, frame) {
      shadeParams.write({
        camera: d.vec4f(Math.tan(s.fovY / 2), width / height, s.near, s.far),
        toggles: d.vec4f(s.aoStrength, s.shadowStrength, s.volumetric, s.exposure),
        ambient: d.vec4f(s.ambient[0], s.ambient[1], s.ambient[2], s.ambientIntensity),
        // The frame index only feeds the dither; wrapping keeps it in the range
        // where f32 still resolves consecutive integers.
        misc: d.vec4f(frame % 4096, s.roughness, s.fogDensity, s.normalStrength),
        fog: d.vec4f(s.fogColor[0], s.fogColor[1], s.fogColor[2], s.specular),
        size: d.vec4f(width, height, options.depthSize, options.depthSize),
        debug: d.vec4f(
          DEBUG_IDS[s.debug],
          s.albedoMix,
          s.contactShadowLength,
          s.aoRadiusPixels,
        ),
      });
      compositeParams.write({
        grade: d.vec4f(
          s.toneMap === 'aces' ? 0 : 1,
          s.contrast,
          s.saturation,
          s.vignette,
        ),
      });
    },

    resize(w, h) {
      if (w === width && h === height) return;
      width = Math.max(1, w);
      height = Math.max(1, h);
      hdr.destroy();
      hdr = createHdr(root, width, height);
      buildGroups();
    },

    get hdr() {
      return hdr;
    },

    destroy() {
      hdr.destroy();
      lightBuffer.destroy();
    },
  };
}
