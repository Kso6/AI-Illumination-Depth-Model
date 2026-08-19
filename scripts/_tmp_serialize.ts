import { readFileSync, writeFileSync } from 'node:fs';
import { buildArchitecture } from '../src/model/arch.ts';
import { parseWeights, serializeWeights } from '../src/model/weights.ts';
const inPath = process.argv[2]!, outPath = process.argv[3]!;
const arch = buildArchitecture(Number(process.argv[4] ?? 64));
const map = parseWeights(new Uint8Array(readFileSync(inPath)).slice().buffer, arch);
writeFileSync(outPath, new Uint8Array(serializeWeights(arch, map)));
console.log('reserialized');
