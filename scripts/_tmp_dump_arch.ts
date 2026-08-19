import { buildArchitecture } from '../src/model/arch.ts';
import { weightSpecs } from '../src/model/weights.ts';
const size = Number(process.env.IDM_SIZE ?? 448);
const arch = buildArchitecture(size);
const ops = arch.ops.map((o: any) => ({
  kind: o.kind, name: o.name, in: o.in ?? null, out: o.out ?? null,
  coarse: o.coarse ?? null, skip: o.skip ?? null,
  k: o.k ?? null, stride: o.stride ?? null,
  midAct: o.midAct ?? null, act: o.act ?? null, residual: o.residual ?? null,
}));
const tensors = Object.fromEntries(Object.entries(arch.tensors).map(([k, v]: any) => [k, [v.h, v.w, v.c]]));
console.log(JSON.stringify({ name: arch.name, inputSize: arch.inputSize, mean: arch.mean, std: arch.std, ops, tensors, specs: weightSpecs(arch) }));
