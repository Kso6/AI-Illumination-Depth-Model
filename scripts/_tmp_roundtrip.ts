import { readFileSync } from 'node:fs';
import { buildArchitecture } from '../src/model/arch.ts';
import { parseWeights, weightSpecs } from '../src/model/weights.ts';

const path = process.argv[2] ?? process.env.IDM_PATH!;
const sumsPath = process.argv[3] ?? process.env.IDM_SUMS!;
const bytes = new Uint8Array(readFileSync(path));
const arch = buildArchitecture(64);
const map = parseWeights(bytes.slice().buffer, arch);
const sums = JSON.parse(readFileSync(sumsPath, 'utf8')) as Record<string, number>;
let worst = 0;
let worstName = '';
let count = 0;
for (const spec of weightSpecs(arch)) {
  const got = map.get(spec.name)!;
  let s = 0;
  for (let i = 0; i < got.length; i++) s += got[i];
  const d = Math.abs(s - sums[spec.name]);
  const scale = Math.max(1, Math.abs(sums[spec.name]));
  if (d / scale > worst) { worst = d / scale; worstName = spec.name; }
  count++;
}
console.log(JSON.stringify({ ok: true, path, tensors: count, worstRelSumDiff: worst, worstName }));
