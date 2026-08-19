/**
 * Headless WebGPU harness.
 *
 * Boots Google Dawn (via `@kmamal/gpu`) against whatever Vulkan ICD is present —
 * a real GPU when one exists, Mesa's `lavapipe` software rasteriser otherwise.
 * That gives the test-suite a genuine Tint-validated WebGPU implementation, so
 * every kernel in this repo is compiled *and executed* in CI rather than merely
 * type-checked.
 */
import tgpu from 'typegpu';
import type { TgpuRoot } from 'typegpu';

type DawnModule = {
  create(flags: string[]): GPU;
  destroy(instance: GPU): void;
} & Record<string, unknown>;

const WEBGPU_GLOBALS = [
  'GPUBufferUsage',
  'GPUMapMode',
  'GPUTextureUsage',
  'GPUShaderStage',
  'GPUColorWrite',
] as const;

export type HeadlessGpu = {
  root: TgpuRoot;
  device: GPUDevice;
  /** Features the adapter actually granted (e.g. `shader-f16`). */
  features: ReadonlySet<string>;
  dispose(): void;
};

let cached: HeadlessGpu | undefined;

/** Boots (once per process) a headless TypeGPU root. */
export async function headlessGpu(): Promise<HeadlessGpu> {
  if (cached) return cached;

  const mod = (await import('@kmamal/gpu')).default as unknown as DawnModule;
  // TypeGPU reads these as ambient globals when building descriptors.
  for (const key of WEBGPU_GLOBALS) {
    if (!(key in globalThis)) {
      (globalThis as Record<string, unknown>)[key] = mod[key];
    }
  }

  const instance = mod.create([]);
  const adapter = await instance.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('headlessGpu: no WebGPU adapter available');

  const wanted: GPUFeatureName[] = ['shader-f16', 'timestamp-query'];
  const requiredFeatures = wanted.filter((f) => adapter.features.has(f));

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.addEventListener?.('uncapturederror', (ev) => {
    // Surface validation errors as test failures rather than console noise.
    throw new Error(`WebGPU uncaptured error: ${(ev as GPUUncapturedErrorEvent).error.message}`);
  });

  const root = tgpu.initFromDevice({ device });
  cached = {
    root,
    device,
    features: new Set(device.features as unknown as Iterable<string>),
    dispose() {
      root.destroy();
      mod.destroy(instance);
      cached = undefined;
    },
  };
  return cached;
}

/** Reads a storage buffer back to the host as a typed array. */
export async function readBuffer(
  device: GPUDevice,
  src: GPUBuffer,
  byteLength: number,
  srcOffset = 0,
): Promise<ArrayBuffer> {
  const size = Math.ceil(byteLength / 4) * 4;
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, srcOffset, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
}
