/**
 * The contract between the Python exporter and the TypeScript loader.
 *
 * These two live in different languages and are edited by different people at
 * different times, and nothing about a weight file is self-describing enough to
 * fail loudly when they drift: a renamed tensor or a reordered blob produces a
 * model that loads cleanly and predicts nonsense. So the contract is tested
 * directly, by having Python write a real container and TypeScript read it.
 *
 * The test skips itself when Python is unavailable rather than failing, since
 * the TypeScript half of the project must stay buildable without it.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { ILLUMINA_DEPTH_448 } from '../src/model/arch.ts';
import { parseWeights, weightSpecs } from '../src/model/weights.ts';

function pythonAvailable(): boolean {
  try {
    execFileSync('python3', ['--version'], { stdio: 'ignore' });
    return existsSync('tools/export/export_idm.py');
  } catch {
    return false;
  }
}

/** Same deterministic value the Python side writes, so both can be compared. */
function expectedValue(tensorIndex: number, element: number): number {
  return ((tensorIndex * 7919 + element * 31) % 1000) / 1000 - 0.5;
}

describe('Python exporter ↔ TypeScript loader', () => {
  const run = pythonAvailable() ? it : it.skip;

  run('round-trips a container written by tools/export/export_idm.py', () => {
    const specs = weightSpecs(ILLUMINA_DEPTH_448);
    const dir = mkdtempSync(join(tmpdir(), 'idm-'));
    try {
      const idm = join(dir, 'roundtrip.idm');
      const script = join(dir, 'write.py');
      // Drives the exporter's own container primitives — which are deliberately
      // torch-free — so this exercises the real writer, not a reimplementation.
      writeFileSync(
        script,
        [
          'import json, math, sys',
          'sys.path.insert(0, "tools/export")',
          'from export_idm import TensorEntry, write_container',
          'specs = json.load(open(sys.argv[1]))',
          'entries = []',
          'for i, s in enumerate(specs):',
          '    n = math.prod(s["shape"])',
          '    vals = [((i * 7919 + j * 31) % 1000) / 1000.0 - 0.5 for j in range(n)]',
          '    entries.append(TensorEntry(name=s["name"], shape=tuple(s["shape"]), values=vals))',
          'write_container(sys.argv[2], entries)',
        ].join('\n'),
      );
      const specFile = join(dir, 'specs.json');
      writeFileSync(
        specFile,
        JSON.stringify(specs.map((s) => ({ name: s.name, shape: [...s.shape] }))),
      );

      execFileSync('python3', [script, specFile, idm], { stdio: 'pipe' });

      const buf = readFileSync(idm);
      const bytes = buf.buffer.slice(
        buf.byteOffset,
        buf.byteOffset + buf.byteLength,
      ) as ArrayBuffer;

      // Throws if a name is missing, a shape disagrees, or the arch name differs.
      const weights = parseWeights(bytes, ILLUMINA_DEPTH_448);
      expect(weights.size).toBe(specs.length);

      let worst = 0;
      let checked = 0;
      specs.forEach((spec, i) => {
        const n = spec.shape.reduce((a, b) => a * b, 1);
        const got = weights.get(spec.name);
        expect(got, `${spec.name} missing`).toBeDefined();
        expect(got!.length, `${spec.name} length`).toBe(n);
        // Sample rather than compare all 1.4 M values; a blob-order or stride
        // error shows up in the first few elements of any tensor.
        for (let j = 0; j < Math.min(n, 64); j++) {
          worst = Math.max(worst, Math.abs(got![j]! - expectedValue(i, j)));
          checked++;
        }
      });
      expect(checked).toBeGreaterThan(1000);
      // Only fp32 rounding of the decimal literals should separate them.
      expect(worst).toBeLessThan(1e-6);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  run('derives exactly the same weight names, shapes and order as the model', () => {
    const out = execFileSync(
      'python3',
      ['-c', PYTHON_NAME_DUMP],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] },
    );
    const fromPython = out.trim().split('\n');
    const fromTypeScript = weightSpecs(ILLUMINA_DEPTH_448).map(
      (s) => `${s.name}\t${s.shape.join(',')}`,
    );
    // Order matters as much as content: it is the blob order.
    expect(fromPython).toEqual(fromTypeScript);
  });
});

/**
 * Imports the PyTorch model module with a stub `torch`, which works because the
 * architecture table and `weight_specs` are plain dataclasses — only the
 * `nn.Module` subclasses need the real library.
 */
const PYTHON_NAME_DUMP = `
import sys, types
class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Any()
    def __getattr__(self, n): return _Any()
class _Mod(types.ModuleType):
    def __getattr__(self, n): return _Any()
torch = _Mod('torch'); torch.Tensor = type('Tensor', (), {})
nn = _Mod('torch.nn')
nn.Module = type('Module', (object,), {'__init__': lambda self, *a, **k: None})
F = _Mod('torch.nn.functional')
nn.functional = F; torch.nn = nn
sys.modules['torch'] = torch
sys.modules['torch.nn'] = nn
sys.modules['torch.nn.functional'] = F
sys.path.insert(0, 'tools/train')
from illumina import model as M
arch = M.build_architecture(448)
for s in M.weight_specs(arch):
    print(s.name + '\\t' + ','.join(str(d) for d in s.shape))
`;
