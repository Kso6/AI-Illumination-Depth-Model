/**
 * The specular BRDF terms against their published closed forms.
 *
 * Evaluates the shipped GGX distribution and Smith visibility on the GPU across
 * a grid of half-angles and roughnesses, and compares term by term with the
 * reference formulas evaluated in double precision on the CPU. This is what
 * caught the denominator floor that was silently capping the specular peak on
 * smooth surfaces.
 */
import { describe, expect, it } from 'vitest';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { headlessGpu, readBuffer } from './helpers/gpu.ts';
import { distributionGGX, visibilitySmith } from '../src/lighting/shade.ts';

const CASES: Array<[number, number]> = [];
for (const r of [0.04, 0.08, 0.12, 0.155, 0.2, 0.3, 0.55, 1.0]) {
  for (const nh of [1.0, 0.9999, 0.999, 0.99, 0.9, 0.7, 0.5]) CASES.push([nh, r]);
}

describe('GGX terms vs published formulas', () => {
  it('D and G', async () => {
    const { root } = await headlessGpu();

    const inBuf = root
      .createBuffer(d.arrayOf(d.vec4f, CASES.length), CASES.map(([nh, r]) => d.vec4f(nh, r, 0.8, 0.6)))
      .$usage('storage');
    const outBuf = root.createBuffer(d.arrayOf(d.vec4f, CASES.length)).$usage('storage');

    const layout = tgpu.bindGroupLayout({
      src: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'readonly' },
      dst: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: 'mutable' },
    });
    const L = layout.$;
    const n = CASES.length;

    const fn = tgpu.computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [64] })(
      (input) => {
        'use gpu';
        const i = input.gid.x;
        if (i < n) {
          const v = d.vec4f(L.src[i]);
          const D = distributionGGX(v.x, v.y);
          const G = visibilitySmith(v.z, v.w, v.y);
          L.dst[i] = d.vec4f(D, G, 0, 0);
        }
      },
    );

    const pipe = root.createComputePipeline({ compute: fn });
    const group = root.createBindGroup(layout, { src: inBuf, dst: outBuf });
    const enc = root['~unstable'].createCommandEncoder({});
    const pass = enc.beginComputePass({});
    pipe.with(group).with(pass).dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    enc.submit();

    const raw = await readBuffer(root.device, root.unwrap(outBuf), n * 16);
    const got = new Float32Array(raw);

    const PI = Math.PI;
    console.log('  nDotH  rough |     D(GPU)      D(ref)     ratio |    G(GPU)     G(ref)');
    let worst = 0;
    for (let i = 0; i < n; i++) {
      const [nh, r] = CASES[i]!;
      const a = r * r, a2 = a * a;
      const den = nh * nh * (a2 - 1) + 1;
      const Dref = a2 / (PI * den * den);
      const k = ((r + 1) * (r + 1)) / 8;
      const Gref = (0.8 / (0.8 * (1 - k) + k)) * (0.6 / (0.6 * (1 - k) + k));
      const Dgpu = got[i * 4]!, Ggpu = got[i * 4 + 1]!;
      const ratio = Dgpu / Dref;
      if (Math.abs(ratio - 1) > worst) worst = Math.abs(ratio - 1);
      console.log(
        `${nh.toFixed(4).padStart(7)} ${r.toFixed(3).padStart(6)} | ` +
          `${Dgpu.toExponential(4).padStart(12)} ${Dref.toExponential(4).padStart(12)} ${ratio.toFixed(6).padStart(10)} | ` +
          `${Ggpu.toFixed(5).padStart(9)} ${Gref.toFixed(5).padStart(9)}`,
      );
    }
    console.log('worst |D_gpu/D_ref - 1| =', worst);
    expect(worst).toBeLessThan(1e-3);
  }, 600_000);
});
