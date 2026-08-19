import { readFileSync, writeFileSync } from 'node:fs';
import { buildArchitecture } from '../src/model/arch.ts';
import { parseWeights, weightSpecs } from '../src/model/weights.ts';
const path = process.argv[2]!;
const out = process.argv[3]!;
const arch = buildArchitecture(64);
const map = parseWeights(new Uint8Array(readFileSync(path)).slice().buffer, arch);
// Concatenate every tensor, in weightSpecs order, as raw f32 little-endian.
const specs = weightSpecs(arch);
let total = 0;
for (const s of specs) total += map.get(s.name)!.length;
const all = new Float32Array(total);
let o = 0;
for (const s of specs) { const v = map.get(s.name)!; all.set(v, o); o += v.length; }
writeFileSync(out, Buffer.from(all.buffer, all.byteOffset, all.byteLength));
console.log(JSON.stringify({ values: total }));
