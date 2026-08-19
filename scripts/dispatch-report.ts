/**
 * Prints the dispatch geometry every kernel will actually use at the shipping
 * 448×448 resolution, and checks each against the WebGPU limits that are
 * *guaranteed* rather than the ones a particular device happens to report.
 *
 * The test-suite runs the network at 64×64 so a CPU reference can keep up; this
 * script is what confirms the 448 configuration is legal, since a tiling chooser
 * that picks a 512-invocation workgroup or a 20 KiB workgroup array would only
 * fail on the real thing.
 *
 *   npm run dispatch
 */
import { ILLUMINA_DEPTH_448, type Op } from '../src/model/arch.ts';
import { chooseTiling, groups } from '../src/model/layout.ts';
import { chooseDwPwTiling } from '../src/model/kernels/dwpw.ts';
import { spatialTiling } from '../src/model/kernels/conv.ts';

/** WebGPU guaranteed minimums — not what any one adapter reports. */
const LIMITS = {
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeWorkgroupsPerDimension: 65535,
  maxStorageBuffersPerShaderStage: 8,
};

interface Row {
  name: string;
  kind: Op['kind'];
  workgroup: [number, number, number];
  dispatch: [number, number, number];
  lds: number;
  threads: number;
  notes: string[];
}

const arch = ILLUMINA_DEPTH_448;
const shape = (n: string) => arch.tensors[n]!;
const rows: Row[] = [];

for (const op of arch.ops) {
  switch (op.kind) {
    case 'preprocess':
    case 'bilateralUp': {
      const o = shape(op.out);
      rows.push({
        name: op.name,
        kind: op.kind,
        workgroup: [8, 8, 1],
        dispatch: [Math.ceil(o.w / 8), Math.ceil(o.h / 8), 1],
        lds: 0,
        threads: 64,
        notes: [],
      });
      break;
    }
    case 'conv': {
      const i = shape(op.in);
      const o = shape(op.out);
      const t = spatialTiling(groups(o.c), o.w, o.h, 8);
      rows.push({
        name: op.name,
        kind: op.kind,
        workgroup: [t.ocLanes, t.tx, t.ty],
        dispatch: [
          Math.ceil(groups(o.c) / (t.ocLanes * t.oc4PerThread)),
          Math.ceil(o.w / (t.tx * t.pptX)),
          Math.ceil(o.h / t.ty),
        ],
        lds: 0,
        threads: t.ocLanes * t.tx * t.ty,
        notes: [`in ${i.c}→${o.c}`, `${t.pptX}px×${t.oc4PerThread}c4/thread`],
      });
      break;
    }
    case 'pw':
    case 'lateral':
    case 'head': {
      const o = shape(op.out);
      const outC4 = groups(o.c);
      if (op.kind === 'head') {
        rows.push({
          name: op.name,
          kind: op.kind,
          workgroup: [64, 1, 1],
          dispatch: [Math.ceil((o.h * o.w) / 64), 1, 1],
          lds: 0,
          threads: 64,
          notes: [],
        });
        break;
      }
      if (op.kind === 'lateral') {
        const c = shape(op.coarse);
        const t = spatialTiling(outC4, c.w, c.h, 2);
        rows.push({
          name: op.name,
          kind: op.kind,
          workgroup: [t.ocLanes, t.tx, t.ty],
          dispatch: [
            Math.ceil(outC4 / (t.ocLanes * t.oc4PerThread)),
            Math.ceil(c.w / t.tx),
            Math.ceil(c.h / t.ty),
          ],
          lds: 0,
          threads: t.ocLanes * t.tx * t.ty,
          notes: [`2×2 px × ${t.oc4PerThread}c4/thread`],
        });
        break;
      }
      const t = chooseTiling(outC4, o.h * o.w);
      rows.push({
        name: op.name,
        kind: op.kind,
        workgroup: [t.ocLanes, t.pixLanes, 1],
        dispatch: [
          Math.ceil(outC4 / (t.ocLanes * t.oc4PerThread)),
          Math.ceil((o.h * o.w) / (t.pixLanes * t.ppt)),
          1,
        ],
        lds: 0,
        threads: t.ocLanes * t.pixLanes,
        notes: [`${t.ppt}px×${t.oc4PerThread}c4/thread`],
      });
      break;
    }
    case 'dwpw': {
      const i = shape(op.in);
      const o = shape(op.out);
      const cfg = {
        inC4: groups(i.c),
        outC4: groups(o.c),
        inH: i.h,
        inW: i.w,
        outH: o.h,
        outW: o.w,
        k: op.k,
        stride: op.stride,
        midAct: op.midAct,
        act: op.act,
        residual: op.residual !== undefined,
      };
      const t = chooseDwPwTiling(cfg);
      rows.push({
        name: op.name,
        kind: op.kind,
        workgroup: [t.ocLanes, t.tx, t.ty],
        dispatch: [
          Math.ceil(groups(o.c) / (t.ocLanes * t.oc4PerThread)),
          Math.ceil(o.w / (t.tx * t.pptX)),
          Math.ceil(o.h / t.ty),
        ],
        lds: t.ldsBytes,
        threads: t.ocLanes * t.tx * t.ty,
        notes: [`k${op.k}s${op.stride}`, `dw×${t.recompute}`, `chunk ${t.ic4Chunk}`],
      });
      break;
    }
  }
}

const problems: string[] = [];
for (const r of rows) {
  const [wx, wy, wz] = r.workgroup;
  if (r.threads > LIMITS.maxComputeInvocationsPerWorkgroup) {
    problems.push(`${r.name}: ${r.threads} invocations exceeds ${LIMITS.maxComputeInvocationsPerWorkgroup}`);
  }
  if (wx > LIMITS.maxComputeWorkgroupSizeX) problems.push(`${r.name}: workgroup x ${wx} too large`);
  if (wy > LIMITS.maxComputeWorkgroupSizeY) problems.push(`${r.name}: workgroup y ${wy} too large`);
  if (wz > LIMITS.maxComputeWorkgroupSizeZ) problems.push(`${r.name}: workgroup z ${wz} too large`);
  if (r.lds > LIMITS.maxComputeWorkgroupStorageSize) {
    problems.push(`${r.name}: ${r.lds} B of workgroup storage exceeds ${LIMITS.maxComputeWorkgroupStorageSize}`);
  }
  for (const [axis, n] of r.dispatch.entries()) {
    if (n < 1) problems.push(`${r.name}: dispatch axis ${axis} is ${n}`);
    if (n > LIMITS.maxComputeWorkgroupsPerDimension) {
      problems.push(`${r.name}: dispatch axis ${axis} is ${n}, over ${LIMITS.maxComputeWorkgroupsPerDimension}`);
    }
  }
}

const pad = (s: string | number, n: number) => String(s).padEnd(n);
const padStart = (s: string | number, n: number) => String(s).padStart(n);

console.log(`${arch.name} — dispatch geometry at ${arch.inputSize}×${arch.inputSize}\n`);
console.log(
  `${pad('op', 18)}${pad('kind', 12)}${pad('workgroup', 14)}${pad('dispatch', 18)}` +
    `${padStart('thr', 5)}${padStart('lds', 8)}  notes`,
);
console.log('─'.repeat(104));
let totalWorkgroups = 0;
for (const r of rows) {
  const wg = r.workgroup.join('×');
  const ds = r.dispatch.join('×');
  totalWorkgroups += r.dispatch[0] * r.dispatch[1] * r.dispatch[2];
  console.log(
    `${pad(r.name, 18)}${pad(r.kind, 12)}${pad(wg, 14)}${pad(ds, 18)}` +
      `${padStart(r.threads, 5)}${padStart(r.lds ? `${r.lds}B` : '-', 8)}  ${r.notes.join(' ')}`,
  );
}
console.log('─'.repeat(104));
console.log(`dispatches       : ${rows.length}`);
console.log(`workgroups total : ${totalWorkgroups.toLocaleString('en-US')}`);
console.log(
  `threads total    : ${rows
    .reduce((a, r) => a + r.dispatch[0] * r.dispatch[1] * r.dispatch[2] * r.threads, 0)
    .toLocaleString('en-US')}`,
);
console.log(`peak workgroup storage: ${Math.max(...rows.map((r) => r.lds))} B`);

if (problems.length) {
  console.error(`\n${problems.length} LIMIT VIOLATION(S):`);
  for (const p of problems) console.error(`  ✗ ${p}`);
  process.exitCode = 1;
} else {
  console.log('\nAll dispatches are within the WebGPU guaranteed minimum limits.');
}
