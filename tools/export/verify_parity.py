"""Prove that the PyTorch network and the TypeScript reference are the same function.

The project has two independent implementations of IlluminaDepth-448: the
PyTorch one in ``tools/train/illumina/model.py``, which is trained, and the
scalar CPU reference in ``src/model/reference.ts``, which is the normative
specification the WGSL kernels are tested against. Two implementations of the
same layer table drift in exactly the places that are easy to get wrong and hard
to notice --- the order of a residual add and its activation, which side of a
lateral connection gets projected, whether ``align_corners`` is false, whether
the ImageNet normalisation happens in sRGB or in linear light. A model that is
wrong in one of those ways still trains, still produces plausible-looking depth,
and is simply worse in the browser than it was in training.

So: run both, on the same weights and the same input, and print the maximum
absolute difference per op.

How
---
1. Build the PyTorch model (BatchNorm-free) at a small but geometrically
   identical size and initialise it from a seeded generator.
2. Fold and export those exact weights with ``export_idm.write_container``,
   producing a real ``.idm`` file --- so the container is on the critical path
   of the test rather than tested separately.
3. Write a random sRGB image into an activation bundle (see below).
4. Shell out to ``npx vite-node scripts/ref-forward.ts``, which loads the
   ``.idm`` through ``parseWeights``, runs ``refPreprocess`` and
   ``forwardReference``, and writes every intermediate activation back out.
5. Compare, op by op, in architecture order.

Interchange format
------------------
Both directions use the container documented in ``tools/export/export_idm.py``
--- ``[u32 headerLength][JSON header][packed blob]`` --- with format tag
``"ntb/1"`` and no ``arch`` field::

    {"format": "ntb/1",
     "tensors": {name: {"dtype": "f32", "shape": [h, w, c],
                        "offset": ..., "length": ...}}}

Activations are stored **channels-last, ``[h, w, c]``**, which is the exact
memory order of a ``RefTensor.data`` on the TypeScript side. PyTorch's NCHW
tensors are permuted here, once, where the cost does not matter.

The input bundle holds ``rgba`` (``[S, S, 4]``, sRGB in ``[0, 1]``, alpha
unused) so that preprocessing runs on both sides and is compared too. The output
bundle holds every tensor named in the architecture, plus ``input`` and
``scene_color``.

Why a small network by default
------------------------------
``forwardReference`` is scalar JavaScript. At 448 it is 2.2 GFLOP and takes
minutes; at 64 --- the smallest legal size, since the encoder downsamples five
times --- it is 45 MFLOP, runs in seconds, and exercises every op, every
activation, every residual and every lateral in exactly the same way. Use
``--size 448`` for a final check if you are patient.

Interpreting the numbers
------------------------
The two implementations sum in different orders, so agreement is up to
floating-point associativity, not bitwise. Differences grow with depth as
rounding accumulates: a few 1e-7 at the stem and a few 1e-6 by the head is
healthy. A difference of 1e-2, or one that appears suddenly at a single op while
its neighbours are clean, is a semantic bug in that op.

Usage
-----
::

    python tools/export/verify_parity.py                  # size 64, seed 1
    python tools/export/verify_parity.py --size 128 --keep
    python tools/export/verify_parity.py --dtype f16      # exercise the fp16 blob

Exits non-zero if any op is outside ``--atol + --rtol * max|reference|``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_idm import (  # noqa: E402
    BUNDLE_FORMAT,
    TensorEntry,
    entries_from_state_dict,
    flatten_tensor,
    read_container,
    verify_entries,
    write_container,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REF_SCRIPT = "scripts/ref-forward.ts"

# Ops that produce a tensor both implementations compute. `preprocess` is
# compared separately (it has no weights) and `bilateralUp` is an inference-only
# shader pass that needs the guide image, so it is not part of the trained graph.
_COMPARED_KINDS = {"conv", "pw", "dwpw", "lateral", "head"}


# ---------------------------------------------------------------------------
# Comparison bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """One tensor, computed twice."""

    label: str
    kind: str
    shape: tuple[int, int, int]
    reference: Sequence[float]
    candidate: Sequence[float]

    def stats(self) -> tuple[float, float, float]:
        """Returns ``(max_abs_diff, mean_abs_diff, max_abs_reference)``."""
        if len(self.reference) != len(self.candidate):
            raise ValueError(
                f"{self.label}: reference has {len(self.reference)} values, "
                f"PyTorch has {len(self.candidate)}"
            )
        worst = 0.0
        total = 0.0
        scale = 0.0
        for a, b in zip(self.reference, self.candidate):
            delta = abs(a - b)
            total += delta
            if delta > worst:
                worst = delta
            if abs(a) > scale:
                scale = abs(a)
        mean = total / max(len(self.reference), 1)
        return worst, mean, scale


def select_channels(
    flat: Sequence[float], shape: tuple[int, int, int], keep: int
) -> list[float]:
    """Keeps the first ``keep`` channels of an ``[h, w, c]`` flat tensor."""
    h, w, c = shape
    if keep == c:
        return list(flat)
    out: list[float] = []
    for pixel in range(h * w):
        base = pixel * c
        out.extend(flat[base : base + keep])
    return out


def channel_spread(flat: Sequence[float], shape: tuple[int, int, int]) -> float:
    """Largest difference between channels of the same pixel.

    The head broadcasts its single disparity value across all four channels so
    the result shares the packed ``vec4`` layout; this checks that it really did.
    """
    h, w, c = shape
    worst = 0.0
    for pixel in range(h * w):
        base = pixel * c
        block = flat[base : base + c]
        worst = max(worst, max(block) - min(block))
    return worst


# ---------------------------------------------------------------------------
# Driving the TypeScript side
# ---------------------------------------------------------------------------


def run_reference(
    weights_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    size: int,
    exposure: float,
    decode_srgb: bool,
    node_cmd: str,
    repo_root: Path = _REPO_ROOT,
    verbose: bool = True,
) -> dict[str, Any]:
    """Runs ``scripts/ref-forward.ts`` and returns its JSON summary.

    Kept free of torch so the plumbing can be exercised on a machine that has
    only the TypeScript half installed.
    """
    command = [
        *shlex.split(node_cmd),
        _REF_SCRIPT,
        "--",
        "--weights",
        str(weights_path),
        "--input",
        str(input_path),
        "--out",
        str(output_path),
        "--size",
        str(size),
        "--exposure",
        repr(float(exposure)),
    ]
    if not decode_srgb:
        command.append("--no-srgb")

    if verbose:
        print("running", " ".join(shlex.quote(part) for part in command))
    result = subprocess.run(
        command, cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            "the TypeScript reference failed "
            f"(exit {result.returncode}).\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    lines = [
        line for line in result.stdout.splitlines() if line.strip().startswith("{")
    ]
    if not lines:
        raise SystemExit(f"no JSON summary from ref-forward.ts:\n{result.stdout}")
    return json.loads(lines[-1])


def write_input_bundle(path: Path, rgba_hwc: Sequence[float], size: int) -> None:
    """Writes the ``rgba`` input bundle the reference script reads."""
    write_container(
        path,
        [TensorEntry("rgba", (size, size, 4), rgba_hwc)],
        dtype="f32",
        arch=None,
        fmt=BUNDLE_FORMAT,
    )


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------


def _to_hwc(tensor: Any) -> list[float]:
    """``[1, C, H, W]`` torch tensor -> flat ``[H, W, C]`` list of floats."""
    return flatten_tensor(tensor[0].permute(1, 2, 0))


def build_comparisons(
    arch: Any,
    torch_values: dict[str, Any],
    torch_input: Any,
    torch_scene: Any,
    reference: dict[str, TensorEntry],
) -> list[Comparison]:
    """Pairs every tensor the two sides both computed, in architecture order."""
    comparisons: list[Comparison] = []

    def ref_entry(name: str) -> TensorEntry:
        entry = reference.get(name)
        if entry is None:
            raise SystemExit(
                f"the reference produced no tensor named {name!r}; it emitted "
                f"{sorted(reference)}"
            )
        if len(entry.shape) != 3:
            raise SystemExit(f"{name}: reference shape {entry.shape} is not [h, w, c]")
        return entry

    # Preprocessing, which has no weights but three easy ways to be wrong.
    entry = ref_entry("input")
    comparisons.append(
        Comparison(
            "preprocess -> input",
            "preprocess",
            entry.shape,  # type: ignore[arg-type]
            entry.values,
            _to_hwc(torch_input),
        )
    )
    if "scene_color" in reference:
        scene = ref_entry("scene_color")
        h, w, c = scene.shape  # type: ignore[misc]
        comparisons.append(
            Comparison(
                "preprocess -> scene_color",
                "preprocess",
                (h, w, 3),
                select_channels(scene.values, (h, w, c), 3),
                _to_hwc(torch_scene),
            )
        )

    for op in arch.ops:
        if op.kind not in _COMPARED_KINDS:
            continue
        name = op.out
        entry = ref_entry(name)
        h, w, c = entry.shape  # type: ignore[misc]
        got = torch_values.get(name)
        if got is None:
            raise SystemExit(f"the PyTorch model produced no tensor named {name!r}")
        torch_channels = int(got.shape[1])
        if torch_channels > c:
            raise SystemExit(
                f"{name}: PyTorch has {torch_channels} channels, reference has {c}"
            )
        comparisons.append(
            Comparison(
                f"{op.name} -> {name}",
                op.kind,
                (h, w, torch_channels),
                select_channels(entry.values, (h, w, c), torch_channels),
                _to_hwc(got),
            )
        )
    return comparisons


def report(
    comparisons: Sequence[Comparison], atol: float, rtol: float
) -> tuple[int, float]:
    """Prints the per-op table. Returns ``(failures, worst_absolute_difference)``."""
    header = (
        f"{'op -> tensor':<34}{'kind':<12}{'shape':>16}"
        f"{'max |diff|':>13}{'mean |diff|':>13}{'rel':>11}  status"
    )
    print()
    print(header)
    print("-" * len(header))

    failures = 0
    overall = 0.0
    for comparison in comparisons:
        worst, mean, scale = comparison.stats()
        overall = max(overall, worst)
        limit = atol + rtol * scale
        ok = worst <= limit
        failures += 0 if ok else 1
        relative = worst / scale if scale > 0 else 0.0
        shape = "x".join(str(dim) for dim in comparison.shape)
        print(
            f"{comparison.label:<34}{comparison.kind:<12}{shape:>16}"
            f"{worst:>13.3e}{mean:>13.3e}{relative:>11.2e}  "
            f"{'ok' if ok else 'FAIL'}"
        )
    print("-" * len(header))
    return failures, overall


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    sys.path.insert(0, str(_REPO_ROOT / "tools" / "train"))
    try:
        import torch  # noqa: PLC0415

        from illumina.model import (  # noqa: E402,PLC0415
            build_model,
            preprocess,
            weight_specs,
        )
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            f"cannot import the PyTorch model ({exc}).\n"
            "    python -m pip install -r tools/requirements.txt"
        ) from exc

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    model = build_model(input_size=args.size, with_bn=False)
    generator = torch.Generator().manual_seed(args.seed)
    model.reset_parameters(generator=generator)
    model.eval()
    arch = model.arch

    specs = weight_specs(arch)
    entries = entries_from_state_dict(model.fold_batchnorm(), specs)
    verify_entries(entries, specs)

    workdir = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="idm-parity-"))
    )
    workdir.mkdir(parents=True, exist_ok=True)
    weights_path = workdir / "parity.idm"
    input_path = workdir / "input.ntb"
    output_path = workdir / "reference.ntb"

    write_container(weights_path, entries, dtype=args.dtype, arch=arch.name)
    print(
        f"{arch.name} at {args.size}x{args.size}: "
        f"{len(entries)} tensors, {sum(e.numel for e in entries)} parameters, "
        f"blob dtype {args.dtype}"
    )
    print(f"workdir {workdir}")

    # An fp16 blob is re-read so that PyTorch runs on the *quantised* weights the
    # reference will see; otherwise the comparison would measure quantisation
    # error rather than implementation disagreement.
    if args.dtype == "f16":
        _, quantised = read_container(weights_path)
        state = model.state_dict()
        for spec in specs:
            state[spec.name] = torch.tensor(
                quantised[spec.name].values, dtype=torch.float32
            ).reshape(spec.shape)
        model.load_state_dict(state, strict=True)

    rgb = torch.rand(1, 3, args.size, args.size, generator=generator)
    alpha = torch.ones(1, 1, args.size, args.size)
    write_input_bundle(input_path, _to_hwc(torch.cat([rgb, alpha], dim=1)), args.size)

    net_input, scene_color = preprocess(
        rgb,
        mean=arch.mean,
        std=arch.std,
        decode_srgb=not args.linear_input,
        exposure=args.exposure,
    )
    _, torch_values = model(net_input, return_intermediates=True)

    summary = run_reference(
        weights_path,
        input_path,
        output_path,
        size=args.size,
        exposure=args.exposure,
        decode_srgb=not args.linear_input,
        node_cmd=args.node_cmd,
    )
    print(f"reference emitted {summary['tensors']} tensors")

    _, reference = read_container(output_path)
    comparisons = build_comparisons(
        arch, torch_values, net_input, scene_color, reference
    )
    failures, overall = report(comparisons, args.atol, args.rtol)

    depth = reference.get("depth_low")
    if depth is not None:
        spread = channel_spread(depth.values, depth.shape)  # type: ignore[arg-type]
        print(f"head broadcast across the vec4: max channel spread {spread:.3e}")

    print(f"worst absolute difference anywhere: {overall:.3e}")
    if failures:
        print(
            f"{failures} op(s) outside atol {args.atol:g} + rtol {args.rtol:g} * "
            "max|reference|"
        )
        print(
            "The first failing op in architecture order is the one to fix; "
            "everything after it inherits the error."
        )
    else:
        print(
            f"all {len(comparisons)} tensors agree within "
            f"atol {args.atol:g} + rtol {args.rtol:g} * max|reference|"
        )

    if not args.keep and not args.workdir:
        for leftover in (weights_path, input_path, output_path):
            leftover.unlink(missing_ok=True)
        workdir.rmdir()

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_parity.py",
        description=(
            "Run the PyTorch model and the TypeScript CPU reference on identical "
            "random weights and identical input; report the maximum absolute "
            "difference per op."
        ),
        epilog=(
            "examples:\n"
            "  python tools/export/verify_parity.py\n"
            "  python tools/export/verify_parity.py --size 128 --keep\n"
            "  python tools/export/verify_parity.py --dtype f16\n"
            "\n"
            "note: the reference is scalar JavaScript, so runtime grows with the\n"
            "square of --size. 64 takes seconds, 448 takes minutes.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="network input size, a multiple of 32 (default: 64)",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1)")
    parser.add_argument(
        "--dtype",
        choices=("f32", "f16"),
        default="f32",
        help="dtype of the exchanged .idm blob; f16 re-quantises the PyTorch "
        "weights to match, so the comparison stays honest (default: f32)",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-5, help="absolute tolerance (default: 1e-5)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="tolerance relative to max|reference| for that tensor (default: 1e-4)",
    )
    parser.add_argument(
        "--exposure", type=float, default=1.0, help="preprocess exposure (default: 1.0)"
    )
    parser.add_argument(
        "--linear-input",
        action="store_true",
        help="treat the random image as linear light, skipping the sRGB decode",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="directory for the intermediate .idm and .ntb files "
        "(default: a temporary directory, deleted on success)",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the temporary files for inspection"
    )
    parser.add_argument(
        "--node-cmd",
        default="npx vite-node",
        help="how to run a TypeScript file (default: 'npx vite-node')",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
