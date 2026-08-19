import { readFileSync } from 'node:fs';
import { ILLUMINA_DEPTH_448 } from '../src/model/arch.ts';
import { parseWeights, totalParameters } from '../src/model/weights.ts';
const m = parseWeights(new Uint8Array(readFileSync(process.argv[2]!)).slice().buffer, ILLUMINA_DEPTH_448);
let n = 0; let finite = true;
for (const [, v] of m) { n += v.length; for (const x of v) if (!Number.isFinite(x)) finite = false; }
console.log(JSON.stringify({ tensors: m.size, values: n, expected: totalParameters(ILLUMINA_DEPTH_448), allFinite: finite }));
