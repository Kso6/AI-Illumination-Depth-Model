/**
 * Activation functions, as TypeGPU functions shared by every kernel.
 *
 * These must stay numerically identical to `applyActivation` in `../layout.ts`
 * (the CPU reference) and to `illumina/model.py` (the training reference); the
 * parity tests in `test/kernels.test.ts` enforce it.
 */
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import type { Activation } from '../arch.ts';

export type ActFn = ReturnType<typeof identity>;

function identity() {
  return tgpu.fn([d.vec4f], d.vec4f)((v) => {
    'use gpu';
    // Must construct a copy: WGSL cannot return a reference to a parameter.
    return d.vec4f(v);
  });
}

const actLinear = identity().$name('act_linear');

const actRelu = tgpu
  .fn([d.vec4f], d.vec4f)((v) => {
    'use gpu';
    return std.max(v, d.vec4f());
  })
  .$name('act_relu');

const actRelu6 = tgpu
  .fn([d.vec4f], d.vec4f)((v) => {
    'use gpu';
    return std.clamp(v, d.vec4f(), d.vec4f(6));
  })
  .$name('act_relu6');

/** `x · relu6(x + 3) / 6` — MobileNetV3's hard-swish. */
const actHardswish = tgpu
  .fn([d.vec4f], d.vec4f)((v) => {
    'use gpu';
    return std.mul(v, std.mul(1 / 6, std.clamp(std.add(v, d.vec4f(3)), d.vec4f(), d.vec4f(6))));
  })
  .$name('act_hardswish');

const TABLE: Record<Activation, ActFn> = {
  linear: actLinear,
  relu: actRelu,
  relu6: actRelu6,
  hardswish: actHardswish,
};

export function activationFn(act: Activation): ActFn {
  return TABLE[act];
}
