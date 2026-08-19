import { describe, expect, it } from 'vitest';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { headlessGpu, readBuffer } from './helpers/gpu.ts';

describe('toolchain', () => {
  it('transpiles a TGSL kernel to WGSL and runs it on a real device', async () => {
    const { root, device } = await headlessGpu();

    const layout = tgpu.bindGroupLayout({
      src: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
      dst: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'mutable' },
    });

    const N = 64;
    const kernel = tgpu.computeFn({
      in: { gid: d.builtin.globalInvocationId },
      workgroupSize: [64],
    })((input) => {
      'use gpu';
      const i = input.gid.x;
      const v = layout.$.src[i] as d.v4f;
      layout.$.dst[i] = d.vec4f(v.x * 2, v.y + 1, v.z * v.z, -v.w);
    });

    const pipeline = root.createComputePipeline({ compute: kernel });

    const srcData = Array.from({ length: N }, (_, i) => d.vec4f(i, i, i, i));
    const src = root.createBuffer(d.arrayOf(d.vec4f, N), srcData).$usage('storage');
    const dst = root.createBuffer(d.arrayOf(d.vec4f, N)).$usage('storage');

    const group = root.createBindGroup(layout, { src, dst });
    pipeline.with(group).dispatchWorkgroups(1);

    const raw = await readBuffer(device, root.unwrap(dst), N * 16);
    const out = new Float32Array(raw);
    expect(Array.from(out.slice(0, 4))).toEqual([0, 1, 0, -0]);
    expect(Array.from(out.slice(4, 8))).toEqual([2, 2, 1, -1]);
    expect(Array.from(out.slice(12, 16))).toEqual([6, 4, 9, -3]);
  });
});
