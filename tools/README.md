# IlluminaDepth-448 — training and export tools

The Python half of the project. The browser half needs none of it: it loads a
single `.idm` file. Everything here exists to produce that file, and to prove
that what it contains is the same function the browser will run.

```
tools/
  requirements.txt
  train/
    train.py                 CLI: distillation, AdamW, cosine + warmup, AMP, EMA
    illumina/
      model.py               the network, from the same declarative op table
      losses.py              MiDaS scale-and-shift-invariant loss
      data.py                image folder dataset + frozen teacher
  export/
    export_idm.py            fold BatchNorm, verify, write the .idm container
    verify_parity.py         PyTorch vs. the TypeScript CPU reference, op by op
```

The specification lives on the TypeScript side and is not duplicated by hand:

| what | authority | Python mirror |
| --- | --- | --- |
| layer table, tensor shapes | `src/model/arch.ts` | `illumina.model.build_architecture` |
| op semantics | `src/model/reference.ts` | `illumina.model` layers |
| weight names, shapes, order | `src/model/weights.ts` (`weightSpecs`) | `illumina.model.weight_specs` |
| activation formulas | `src/model/layout.ts` | `illumina.model.apply_activation` |

`verify_parity.py` is what keeps the mirror honest. Run it after any change to
either side.

---

## 1. Install

Python 3.11 or newer.

```bash
# GPU users: install the torch build that matches your CUDA runtime first,
# otherwise pip may give you a CPU-only wheel.
#   https://pytorch.org/get-started/locally/
python -m pip install -r tools/requirements.txt
```

The TypeScript side needs its own install once, for `verify_parity.py`:

```bash
npm install
```

---

## 2. Check the pipeline before training anything

Nothing below needs a dataset, a GPU or a teacher.

```bash
# the layer table, FLOP budget and parameter count
npx vite-node scripts/arch-report.ts

# the same table, from the Python port -- the two must agree
python tools/train/illumina/model.py

# a Kaiming-initialised container, so the loader, the kernels and the frame
# graph can be exercised end to end. The depth it produces is a smooth
# arbitrary field, not a reconstruction.
python tools/export/export_idm.py --random --seed 1 \
    --out public/weights/random-448.idm

# PyTorch vs. the TypeScript CPU reference, on identical weights and input
python tools/export/verify_parity.py
```

`verify_parity.py` builds a geometrically identical 64x64 network (the smallest
legal size — the encoder downsamples five times), exports it, runs
`scripts/ref-forward.ts`, and prints a max-absolute-difference table with one
row per op. It takes a few seconds. Expect `1e-7` near the stem growing to a few
`1e-6` at the head; that is floating-point associativity, not disagreement. A
row at `1e-2`, or one bad row surrounded by clean ones, is a real bug in that
op.

`--size 448` runs the same check on the shipping network. The reference is
scalar JavaScript, so it takes minutes rather than seconds; worth doing once
before a release.

---

## 3. Data

**What you need.** A directory of ordinary photographs — any depth of
subdirectory nesting, `.jpg` / `.png` / `.webp`. No depth labels: the targets
come from the frozen teacher.

**How many.** Be honest with yourself here:

| distinct source images | what you get |
| --- | --- |
| < 5 k | a smoke test. The student memorises the teacher's output on those images. |
| 50 k | a model that works on scenes resembling the training set and visibly fails elsewhere. This is the practical floor. |
| 300 k – 1 M | a model that generalises. Public options: a COCO/Places/OpenImages mix, or an LAION/CC12M subset. |

Diversity matters more than count. 200 k images from one webcam is worse than
50 k from everywhere: indoor and outdoor, close and far, day and night, people
and empty rooms, wide and telephoto.

**Precompute the teacher.** Strongly recommended — it is the difference between
a training run that is GPU-bound on a 90 GFLOP teacher and one that is bound on
a 6.6 GFLOP student.

```bash
python -m illumina.data --root /data/images --cache-dir /data/cache \
    --teacher depth-anything-v2-small --device cuda --batch-size 8
# run from tools/train/, or set PYTHONPATH=tools/train
```

The cache is written once and reused by every epoch and every run.

---

## 4. Train

```bash
python tools/train/train.py \
    --data-root /data/images \
    --teacher-cache /data/cache \
    --out runs/v1 \
    --epochs 30 --batch-size 32 --lr 3e-4 --amp bf16
```

Resume exactly where it stopped:

```bash
python tools/train/train.py --data-root /data/images --teacher-cache /data/cache \
    --out runs/v1 --resume runs/v1/last.pt
```

A 20-step smoke test with no teacher download and no dataset requirements:

```bash
python tools/train/train.py --data-root /data/sample --teacher synthetic \
    --out /tmp/smoke --max-steps 20 --batch-size 4 --num-workers 0
```

`--help` lists every option with its default. The script prints a cost estimate
derived from your actual dataset size and settings before it does any work.

Validation reports AbsRel and delta1 after a least-squares affine alignment of
the prediction to the target — the standard protocol for relative-depth models,
which are only defined up to scale and shift. Checkpoints go to
`runs/v1/last.pt` and `runs/v1/best.pt`; both hold the raw and the EMA weights
in training (BatchNorm) shape, so a checkpoint is always resumable.

### Honest cost

Measured against the architecture's real FLOP counts, at 448x448:

| item | cost |
| --- | --- |
| student forward | 2.198 GFLOP / image |
| student training step (fwd + bwd) | ~6.6 GFLOP / image |
| teacher forward (Depth Anything V2 Small) | ~90 GFLOP / image |
| building the teacher cache, 300 k images | 3–5 GPU-hours, once, on an A100 |
| training to a usable model, 1–3 M samples seen | 15–30 A100-hours |
| the same on one RTX 4090 | roughly 25–50 hours |

So: **one to two days on a single good GPU**, plus a few hours of cache
building, plus however long it takes to assemble the images. The student is so
cheap that a run is input-pipeline bound long before it is GPU bound — if
throughput disappoints, the fix is almost always more `--num-workers`, faster
storage, or a pre-resized copy of the dataset, not a bigger GPU.

Not honest to claim: that fewer than ~50 k distinct images produces something
worth shipping, or that any of this beats the teacher. Distillation buys a 40x
FLOP reduction and a browser-sized download; it does not buy accuracy.

---

## 5. Export

```bash
# float32, 5,572,036 bytes
python tools/export/export_idm.py --checkpoint runs/v1/best.pt \
    --out public/weights/illumina-448.idm

# float16, 2,790,663 bytes -- same file, half the download
python tools/export/export_idm.py --checkpoint runs/v1/best.pt \
    --out public/weights/illumina-448.f16.idm --dtype f16
```

What it does, in order:

1. loads the checkpoint and picks the EMA weights when the checkpoint has them
   (`--weights model` / `--weights ema` to override);
2. folds every BatchNorm into the convolution in front of it —
   `w' = w * gamma / sqrt(var + eps)`, `b' = (b - mean) * gamma / sqrt(var + eps) + beta`
   — so the inference graph is pure conv / bias / activation, exactly what
   `arch.ts` promises;
3. checks all 112 tensors against `weightSpecs()` for presence, shape, order and
   finiteness, and reports *every* problem at once rather than the first;
4. writes the container, reads it back and compares. The float32 round-trip must
   be exact; the float16 one prints its worst quantisation error (about `5e-4`
   on Kaiming-initialised weights).

Then point the app at the file and reload.

---

## 6. Verify

```bash
python tools/export/verify_parity.py                 # size 64, seed 1
python tools/export/verify_parity.py --size 128 --keep
python tools/export/verify_parity.py --dtype f16     # exercise the fp16 blob
python tools/export/verify_parity.py --size 448      # slow, pre-release
```

`--dtype f16` re-quantises the PyTorch weights from the file before running, so
the table still measures implementation disagreement rather than quantisation
error.

Exits non-zero if any op falls outside `--atol + --rtol * max|reference|`
(defaults `1e-5` and `1e-4`). The first failing row in architecture order is the
one to fix; every row after it inherits the error.

The comparison covers preprocessing (sRGB decode, exposure, and the fact that
the ImageNet mean/std are applied in **linear light**, not in sRGB), every
convolution, every residual, every lateral fusion and the sigmoid head. It does
not cover the joint-bilateral upsample, which needs the guide image and is an
inference-only shader pass with no trainable parameters.

---

## 7. File formats

### `.idm` — the weight container

```
[u32 little-endian headerLength]
[UTF-8 JSON header, exactly headerLength bytes]
[tightly packed blob]

header = {"format": "idm/1",
          "arch":   "IlluminaDepth-448",
          "tensors": {name: {"dtype": "f32"|"f16", "shape": [...],
                             "offset": <bytes into the blob>,
                             "length": <bytes>}}}
```

Deliberately the same shape as safetensors, and simple enough that both readers
(`parseWeights` in `src/model/weights.ts`, `read_container` in
`tools/export/export_idm.py`) are about twenty lines.

Guarantees the loader relies on: tensors appear in the blob, and as JSON keys,
in `weightSpecs()` order; every tensor is contiguous, row-major, in *logical*
PyTorch shape (`[out, in, kh, kw]`, `[out, in]`, `[C, k, k]`) — the swizzle into
the GPU's `vec4` groups happens at load time in `src/model/layout.ts`, in one
tested place; `length` is always `prod(shape) * itemsize` and offsets are tight.

Writing it is pure Python, so the format can be exercised on a machine with no
PyTorch. As a check on both writers, the Python one produces a **byte-identical**
file to the TypeScript `serializeWeights` given the same tensors.

### `.ntb` — the activation bundle

The same container with `"format": "ntb/1"` and no `arch` field, used only by
`verify_parity.py` and `scripts/ref-forward.ts` to exchange a random image and
every intermediate activation. Activations are stored **channels-last,
`[h, w, c]`** — the exact memory order of a `RefTensor.data`, so nothing is
transposed on the TypeScript side; the PyTorch NCHW tensors are permuted once on
the Python side.

### `scripts/ref-forward.ts`

The TypeScript entry point `verify_parity.py` shells out to. Usable on its own:

```bash
# every activation, from deterministic synthetic weights, no PyTorch involved
npx vite-node scripts/ref-forward.ts -- \
    --weights synthetic:7 --size 64 --input in.ntb --out ref.ntb

# dump those synthetic weights as a real .idm
npx vite-node scripts/ref-forward.ts -- \
    --weights synthetic:7 --size 448 --dump-weights synth.idm
```

---

## 8. Troubleshooting

**`cannot import illumina.model`** — the training requirements are not
installed. The container code itself does not need PyTorch; only reading a
checkpoint or building a network does.

**`weights: 37 tensor(s) missing`, from the browser** — the `.idm` was not
written by `export_idm.py`, or was written for a different architecture.
Re-export; the exporter's own check reports the same problem with names before
the file is written.

**`the TypeScript reference failed`** — run `npm install`, then try the command
`verify_parity.py` printed on its own. `--node-cmd` overrides how a `.ts` file
is run.

**Parity fails at exactly one op** — read that op in `src/model/reference.ts`
and its layer in `illumina/model.py` side by side. In practice the divergences
worth suspecting are: a residual added after the activation instead of before;
`align_corners=True` in the lateral upsample; projecting the coarse tensor in a
lateral (it is deliberately *not* projected — the previous level's refine step
already emitted the right width); and normalising in sRGB rather than in linear
light.

**Parity fails everywhere, including `preprocess`** — the inputs differ, not the
network. Check `--exposure` and `--linear-input`.
