import { buildArchitecture } from '../src/model/arch.ts';
import { weightSpecs, totalParameters } from '../src/model/weights.ts';
const size = Number(process.env.IDM_SIZE ?? 448);
const arch = buildArchitecture(size);
const specs = weightSpecs(arch);
console.log(JSON.stringify({ name: arch.name, count: specs.length, params: totalParameters(arch), specs }, null, 1));
