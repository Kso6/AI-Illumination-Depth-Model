/**
 * Tensor memory arena.
 *
 * Activations are short-lived: most tensors are read by exactly one op and are
 * dead immediately afterwards. Only the four encoder skip connections survive
 * across the whole decoder. Allocating one buffer per tensor would reserve
 * ~66 MB (fp32) of device memory for a network whose true high-water mark is a
 * fraction of that, and would spread the working set across enough distinct
 * allocations to lose cache locality.
 *
 * So the planner computes each tensor's live range from the op list and reuses
 * buffers greedily, smallest-fit-first, once their previous occupant is dead.
 * The plan is deterministic and is asserted in `test/arena.test.ts`.
 */
import type { Architecture, Op, TensorName } from '../model/arch.ts';
import { tensorElements } from '../model/layout.ts';

/** Tensors an op reads. */
export function opInputs(op: Op): TensorName[] {
  switch (op.kind) {
    case 'preprocess':
      return [];
    case 'conv':
    case 'pw':
    case 'dwpw':
      return op.kind === 'conv' ? [op.in] : [op.in, ...(op.residual ? [op.residual] : [])];
    case 'lateral':
      return [op.coarse, op.skip];
    case 'head':
    case 'bilateralUp':
      return [op.in];
  }
}

/** The tensor an op writes. */
export function opOutput(op: Op): TensorName {
  return op.out;
}

export interface ArenaPlan {
  /** Size of each pooled buffer, in `vec4` elements. */
  readonly bufferSizes: readonly number[];
  /** Which pooled buffer each tensor lives in. */
  readonly assignment: ReadonlyMap<TensorName, number>;
  /** Total bytes reserved at fp32. */
  readonly totalBytes: number;
  /** Bytes that one-buffer-per-tensor would have needed. */
  readonly naiveBytes: number;
}

/**
 * Plans buffer reuse.
 *
 * @param exclude Tensors that are not backed by an arena buffer (e.g. the final
 *   depth map, which lives in a texture).
 */
export function planArena(arch: Architecture, exclude: ReadonlySet<TensorName>): ArenaPlan {
  // Last op index that reads each tensor.
  const lastRead = new Map<TensorName, number>();
  arch.ops.forEach((op, i) => {
    for (const t of opInputs(op)) lastRead.set(t, i);
  });

  const bufferSizes: number[] = [];
  const assignment = new Map<TensorName, number>();
  /** Buffer index → true when available for reuse. */
  const free: boolean[] = [];
  /** Which tensors currently occupy which buffer, to release them. */
  const occupant = new Map<number, TensorName>();

  arch.ops.forEach((op, i) => {
    const out = opOutput(op);
    if (!exclude.has(out) && !assignment.has(out)) {
      const need = tensorElements(arch.tensors[out]!);

      // Smallest free buffer that fits; grow the smallest free one if none fits.
      let chosen = -1;
      let chosenSize = Infinity;
      for (let b = 0; b < bufferSizes.length; b++) {
        if (free[b] && bufferSizes[b]! >= need && bufferSizes[b]! < chosenSize) {
          chosen = b;
          chosenSize = bufferSizes[b]!;
        }
      }
      if (chosen === -1) {
        // Prefer enlarging an existing free buffer over adding another one:
        // fewer, larger allocations keep the working set contiguous.
        let grow = -1;
        let growSize = -1;
        for (let b = 0; b < bufferSizes.length; b++) {
          if (free[b] && bufferSizes[b]! > growSize) {
            grow = b;
            growSize = bufferSizes[b]!;
          }
        }
        if (grow === -1) {
          chosen = bufferSizes.length;
          bufferSizes.push(need);
          free.push(false);
        } else {
          chosen = grow;
          bufferSizes[grow] = need;
        }
      }
      free[chosen] = false;
      assignment.set(out, chosen);
      occupant.set(chosen, out);
    }

    // Release inputs whose last read is this op — *after* the output has been
    // placed, so an op can never alias its own source and destination.
    for (const t of opInputs(op)) {
      if (lastRead.get(t) === i) {
        const b = assignment.get(t);
        if (b !== undefined && occupant.get(b) === t) {
          free[b] = true;
        }
      }
    }
  });

  const totalBytes = bufferSizes.reduce((a, n) => a + n * 16, 0);
  const naiveBytes = Object.entries(arch.tensors)
    .filter(([n]) => !exclude.has(n))
    .reduce((a, [, s]) => a + tensorElements(s) * 16, 0);

  return { bufferSizes, assignment, totalBytes, naiveBytes };
}

/**
 * Verifies that no two tensors with overlapping live ranges were placed in the
 * same buffer. Cheap enough to run at startup in development, and run as a test.
 */
export function validateArena(
  arch: Architecture,
  plan: ArenaPlan,
  exclude: ReadonlySet<TensorName>,
): void {
  const firstWrite = new Map<TensorName, number>();
  const lastRead = new Map<TensorName, number>();
  arch.ops.forEach((op, i) => {
    if (!firstWrite.has(op.out)) firstWrite.set(op.out, i);
    for (const t of opInputs(op)) lastRead.set(t, i);
  });

  const names = Object.keys(arch.tensors).filter((n) => !exclude.has(n));
  for (const a of names) {
    for (const b of names) {
      if (a >= b) continue;
      if (plan.assignment.get(a) !== plan.assignment.get(b)) continue;
      const a0 = firstWrite.get(a) ?? 0;
      const a1 = lastRead.get(a) ?? a0;
      const b0 = firstWrite.get(b) ?? 0;
      const b1 = lastRead.get(b) ?? b0;
      if (a0 <= b1 && b0 <= a1) {
        throw new Error(
          `arena: ${a} [${a0}..${a1}] and ${b} [${b0}..${b1}] overlap in buffer ` +
            `${plan.assignment.get(a)}`,
        );
      }
    }
  }

  for (const n of names) {
    const b = plan.assignment.get(n);
    if (b === undefined) throw new Error(`arena: ${n} was never assigned a buffer`);
    const need = tensorElements(arch.tensors[n]!);
    if (plan.bufferSizes[b]! < need) {
      throw new Error(`arena: buffer ${b} holds ${plan.bufferSizes[b]} but ${n} needs ${need}`);
    }
  }
}
