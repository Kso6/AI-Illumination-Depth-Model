"""Write the ``.idm`` weight container the browser runtime loads.

This is the last step of the training pipeline and the only place where the
PyTorch side and the TypeScript side exchange numbers, so it is deliberately
paranoid: it folds every BatchNorm away, checks each tensor against the
authoritative list in ``src/model/weights.ts`` (mirrored by
``illumina.model.weight_specs``) for presence, shape and finiteness, and then
reads the file back and compares it with what went in.

The container
-------------
Byte-for-byte what ``parseWeights`` in ``src/model/weights.ts`` expects, and
deliberately the same shape as safetensors::

    [u32 little-endian headerLength]
    [UTF-8 JSON header, exactly headerLength bytes]
    [tightly packed blob]

    header = {"format": "idm/1",
              "arch":   "IlluminaDepth-448",
              "tensors": {name: {"dtype": "f32"|"f16",
                                 "shape": [...],
                                 "offset": <bytes into the blob>,
                                 "length": <bytes>}}}

Rules that the loader relies on and that this writer guarantees:

* tensors appear in the blob, and as JSON keys, in ``weightSpecs()`` order;
* every tensor is contiguous, row-major, in *logical* PyTorch shape --- ``[out,
  in, kh, kw]`` for the stem, ``[out, in]`` for 1x1 convolutions, ``[C, k, k]``
  for depthwise kernels. The swizzle into the GPU's ``vec4`` groups happens at
  load time in ``src/model/layout.ts``, in one tested place;
* ``length`` always equals ``prod(shape) * itemsize``, and offsets are tight
  (no padding), so ``offset`` of tensor *i* is the sum of the lengths before it;
* ``f32`` and ``f16`` are little-endian IEEE-754. ``f16`` halves the download
  (2.78 MB -> 1.39 MB) and is what the fp16 GPU path wants; the rounding error
  it introduces is reported before the file is written.

Why the container primitives below do not import torch
------------------------------------------------------
Reading and writing the container is pure Python (``array`` and ``struct``), so
the format can be exercised, and cross-checked against the TypeScript reader,
on a machine that has no PyTorch installed. Only the parts that must open a
checkpoint or build a network import torch, and they do it lazily.

Usage
-----
::

    # export a trained checkpoint (EMA weights when the checkpoint has them)
    python tools/export/export_idm.py --checkpoint runs/v1/best.pt \\
        --out public/weights/illumina-448.idm

    # half precision
    python tools/export/export_idm.py --checkpoint runs/v1/best.pt \\
        --out public/weights/illumina-448.f16.idm --dtype f16

    # a Kaiming-initialised file, so the container and the runtime can be
    # tested end to end before any training has happened
    python tools/export/export_idm.py --random --seed 1 \\
        --out public/weights/random-448.idm

Run with ``--help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Literal, Mapping, Sequence

__all__ = [
    "Dtype",
    "TensorEntry",
    "IDM_FORMAT",
    "BUNDLE_FORMAT",
    "ARCH_NAME",
    "write_container",
    "read_container",
    "pack_values",
    "unpack_values",
    "entries_from_state_dict",
    "verify_entries",
    "export_checkpoint",
    "export_random",
]

Dtype = Literal["f32", "f16"]

IDM_FORMAT = "idm/1"
ARCH_NAME = "IlluminaDepth-448"
#: Format tag of the activation bundles exchanged with scripts/ref-forward.ts.
BUNDLE_FORMAT = "ntb/1"

ITEMSIZE: dict[str, int] = {"f32": 4, "f16": 2}

#: Largest finite value representable in IEEE-754 binary16.
HALF_MAX = 65504.0

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _requirements_path() -> Path:
    return _REPO_ROOT / "tools" / "requirements.txt"


# ---------------------------------------------------------------------------
# Container primitives (no torch, no numpy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorEntry:
    """One tensor on its way into or out of a container.

    ``values`` is the flat, row-major, C-contiguous content; ``shape`` is the
    logical shape the header advertises.
    """

    name: str
    shape: tuple[int, ...]
    values: Sequence[float]

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    def validate(self) -> None:
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"{self.name}: non-positive dimension in {self.shape}")
        if len(self.values) != self.numel:
            raise ValueError(
                f"{self.name}: shape {self.shape} needs {self.numel} values, "
                f"got {len(self.values)}"
            )


def pack_values(values: Sequence[float], dtype: Dtype) -> tuple[bytes, int]:
    """Packs floats little-endian. Returns ``(bytes, saturated_count)``.

    ``saturated_count`` is the number of finite inputs whose magnitude exceeded
    the binary16 range and were therefore clamped to +-65504 (never silently
    turned into infinity, which would poison the whole feature map downstream).
    It is always zero for ``f32``.
    """
    if dtype == "f32":
        buf = array("f", values)
        if sys.byteorder != "little":
            buf.byteswap()
        return buf.tobytes(), 0
    if dtype != "f16":
        raise ValueError(f"unsupported dtype {dtype!r}")

    # struct's 'e' code is IEEE-754 binary16 with round-half-to-even, which is
    # exactly what torch.Tensor.half() does. It raises OverflowError instead of
    # returning inf, so out-of-range values are clamped explicitly. Bulk-packing
    # in chunks keeps the common (no-overflow) path fast without building a
    # million-argument call.
    out = bytearray()
    saturated = 0
    chunk = 4096
    for start in range(0, len(values), chunk):
        block = values[start : start + chunk]
        try:
            out += struct.pack(f"<{len(block)}e", *block)
        except (OverflowError, ValueError):
            for value in block:
                v = float(value)
                if math.isfinite(v) and abs(v) > HALF_MAX:
                    v = math.copysign(HALF_MAX, v)
                    saturated += 1
                out += struct.pack("<e", v)
    return bytes(out), saturated


def unpack_values(raw: bytes, dtype: Dtype) -> list[float]:
    """Inverse of :func:`pack_values`."""
    if dtype == "f32":
        buf = array("f")
        buf.frombytes(raw)
        if sys.byteorder != "little":
            buf.byteswap()
        return list(buf)
    if dtype != "f16":
        raise ValueError(f"unsupported dtype {dtype!r}")
    count = len(raw) // 2
    return list(struct.unpack(f"<{count}e", raw))


def write_container(
    path: str | Path,
    entries: Sequence[TensorEntry],
    *,
    dtype: Dtype = "f32",
    arch: str | None = ARCH_NAME,
    fmt: str = IDM_FORMAT,
) -> dict[str, Any]:
    """Writes ``entries`` to ``path`` in blob order and returns the header.

    ``arch=None`` omits the ``arch`` field, which is how the activation bundles
    exchanged with ``scripts/ref-forward.ts`` (``fmt="ntb/1"``) are written.
    """
    seen: set[str] = set()
    tensors: dict[str, Any] = {}
    blob = bytearray()
    saturated_total = 0
    for entry in entries:
        entry.validate()
        if entry.name in seen:
            raise ValueError(f"duplicate tensor {entry.name}")
        seen.add(entry.name)
        raw, saturated = pack_values(entry.values, dtype)
        saturated_total += saturated
        expected = entry.numel * ITEMSIZE[dtype]
        if len(raw) != expected:
            raise AssertionError(
                f"{entry.name}: packed {len(raw)} bytes, expected {expected}"
            )
        tensors[entry.name] = {
            "dtype": dtype,
            "shape": list(entry.shape),
            "offset": len(blob),
            "length": len(raw),
        }
        blob += raw

    header: dict[str, Any] = {"format": fmt}
    if arch is not None:
        header["arch"] = arch
    header["tensors"] = tensors

    # separators=(",",":") keeps the header compact; json preserves insertion
    # order, so the JSON key order is the blob order the loader can rely on.
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(struct.pack("<I", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob)
    tmp.replace(path)

    if saturated_total:
        print(
            f"warning: {saturated_total} value(s) exceeded the f16 range and were "
            f"clamped to +-{HALF_MAX:.0f}",
            file=sys.stderr,
        )
    return header


def read_container(path: str | Path) -> tuple[dict[str, Any], "dict[str, TensorEntry]"]:
    """Reads a container back. Returns ``(header, {name: TensorEntry})``.

    Used by ``--check`` below and by ``verify_parity.py``; it is also the
    simplest available description of the format, being about twenty lines.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 4:
        raise ValueError(f"{path}: file is truncated")
    (header_len,) = struct.unpack_from("<I", raw, 0)
    if header_len == 0 or 4 + header_len > len(raw):
        raise ValueError(f"{path}: header length {header_len} does not fit the file")
    header = json.loads(raw[4 : 4 + header_len].decode("utf-8"))
    base = 4 + header_len

    entries: dict[str, TensorEntry] = {}
    for name, info in header["tensors"].items():
        dtype: Dtype = info["dtype"]
        if dtype not in ITEMSIZE:
            raise ValueError(f"{path}: {name} has unsupported dtype {dtype!r}")
        shape = tuple(int(dim) for dim in info["shape"])
        count = 1
        for dim in shape:
            count *= dim
        expected = count * ITEMSIZE[dtype]
        if info["length"] != expected:
            raise ValueError(
                f"{path}: {name} claims {info['length']} bytes but shape {shape} "
                f"needs {expected}"
            )
        start = base + info["offset"]
        if start + expected > len(raw):
            raise ValueError(f"{path}: {name} extends past the end of the file")
        values = unpack_values(raw[start : start + expected], dtype)
        entries[name] = TensorEntry(name=name, shape=shape, values=values)
    return header, entries


# ---------------------------------------------------------------------------
# Bridge to the PyTorch side
# ---------------------------------------------------------------------------


def _import_model() -> ModuleType:
    """Imports ``illumina.model`` lazily, with an actionable error message."""
    train_root = _REPO_ROOT / "tools" / "train"
    if str(train_root) not in sys.path:
        sys.path.insert(0, str(train_root))
    try:
        from illumina import model  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            f"cannot import illumina.model ({exc}).\n"
            "Install the training requirements first:\n"
            f"    python -m pip install -r {_requirements_path()}\n"
            "PyTorch is only needed to read a checkpoint or build a network; the "
            ".idm container itself is pure Python."
        ) from exc
    return model


def flatten_tensor(tensor: Any) -> list[float]:
    """Flattens a torch (or numpy) tensor to a list of Python floats.

    Deliberately duck-typed so this module does not import torch. ``tolist()``
    on 1.39 M float32 values costs a fraction of a second, which is nothing next
    to a training run, and it removes every dtype and endianness question: a
    float32 promoted to a Python float and written back with ``array('f')`` is
    exact.
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()
    if hasattr(tensor, "reshape"):
        tensor = tensor.reshape(-1)
    values = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)
    if values and isinstance(values[0], list):  # pragma: no cover - defensive
        raise ValueError("flatten_tensor received a tensor it could not flatten")
    return [float(v) for v in values]


def entries_from_state_dict(
    state: Mapping[str, Any], specs: Iterable[Any]
) -> list[TensorEntry]:
    """Converts an inference state dict to container entries, in ``specs`` order."""
    return [
        TensorEntry(
            name=spec.name,
            shape=tuple(int(dim) for dim in spec.shape),
            values=flatten_tensor(state[spec.name]),
        )
        for spec in specs
    ]


def verify_entries(entries: Sequence[TensorEntry], specs: Sequence[Any]) -> None:
    """Checks the export against ``weightSpecs()``; raises with every problem at once.

    Catching one problem per run turns exporting into a guessing game, and the
    browser-side error ("weights: 37 tensor(s) missing") arrives far too late.
    """
    by_name = {entry.name: entry for entry in entries}
    problems: list[str] = []

    for spec in specs:
        want = tuple(int(dim) for dim in spec.shape)
        entry = by_name.get(spec.name)
        if entry is None:
            problems.append(f"missing tensor {spec.name} (shape {list(want)})")
            continue
        if entry.shape != want:
            problems.append(
                f"{spec.name}: shape {list(entry.shape)}, expected {list(want)}"
            )
            continue
        bad = sum(1 for v in entry.values if not math.isfinite(v))
        if bad:
            problems.append(f"{spec.name}: {bad} non-finite value(s)")

    expected_names = {spec.name for spec in specs}
    for name in by_name:
        if name not in expected_names:
            problems.append(f"unexpected tensor {name}, not in weightSpecs()")

    order = [spec.name for spec in specs]
    if [entry.name for entry in entries] != order:
        problems.append(
            "tensor order does not match weightSpecs(); the loader tolerates it "
            "but the blob layout is then not the documented one"
        )

    if problems:
        raise ValueError(
            f"{len(problems)} problem(s) in the export:\n  " + "\n  ".join(problems)
        )


def _checkpoint_state(path: Path, prefer: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loads a training checkpoint and picks the raw or EMA weights.

    ``prefer`` is ``"auto"`` (EMA when present, else the raw weights), ``"ema"``
    or ``"model"``.
    """
    import torch  # noqa: PLC0415

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise SystemExit(
            f"{path} does not look like an illumina checkpoint (no 'model' key). "
            "Checkpoints are written by tools/train/train.py."
        )
    meta = {
        key: value
        for key, value in payload.items()
        if not isinstance(value, dict) or key == "metrics"
    }

    ema = payload.get("ema")
    if prefer == "ema":
        if ema is None:
            raise SystemExit(f"{path} has no EMA weights; use --weights model")
        chosen, source = ema, "ema"
    elif prefer == "model":
        chosen, source = payload["model"], "model"
    else:
        chosen, source = (
            (ema, "ema") if ema is not None else (payload["model"], "model")
        )

    meta["weights_source"] = source
    meta["input_size"] = int(payload.get("input_size", 448))
    meta["with_bn"] = bool(payload.get("with_bn", True))
    return dict(chosen), meta


def export_checkpoint(
    checkpoint: Path,
    out: Path,
    *,
    dtype: Dtype = "f32",
    prefer: str = "auto",
    input_size: int | None = None,
    check: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Folds a checkpoint's BatchNorms away and writes the ``.idm`` file."""
    model_module = _import_model()
    state, meta = _checkpoint_state(checkpoint, prefer)
    size = input_size if input_size is not None else meta["input_size"]

    model = model_module.build_model(input_size=size, with_bn=meta["with_bn"])
    model.load_state_dict(state, strict=True)
    model.eval()
    folded = model.fold_batchnorm()

    specs = model_module.weight_specs(model.arch)
    entries = entries_from_state_dict(folded, specs)
    verify_entries(entries, specs)

    if verbose:
        print(
            f"checkpoint {checkpoint}\n"
            f"  step {meta.get('step', '?')}  epoch {meta.get('epoch', '?')}  "
            f"weights: {meta['weights_source']}  input_size: {size}\n"
            f"  teacher: {meta.get('teacher', 'unknown')}"
        )
    return _finish(entries, out, model.arch.name, dtype, check, verbose)


def export_random(
    out: Path,
    *,
    input_size: int = 448,
    seed: int = 1,
    dtype: Dtype = "f32",
    check: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Writes a Kaiming-initialised container, no training required.

    The point is to be able to test the container, the loader, the GPU kernels
    and the whole frame graph before a single gradient step exists. The depth
    such a file produces is a smooth arbitrary field, not a reconstruction ---
    the same caveat ``synthesizeWeights`` in ``src/model/weights.ts`` carries.
    """
    # _import_model first: it turns a missing PyTorch into an actionable
    # message instead of a bare ModuleNotFoundError traceback.
    model_module = _import_model()
    import torch  # noqa: PLC0415

    model = model_module.build_model(input_size=input_size, with_bn=False)
    generator = torch.Generator().manual_seed(seed)
    model.reset_parameters(generator=generator)
    model.eval()

    specs = model_module.weight_specs(model.arch)
    entries = entries_from_state_dict(model.fold_batchnorm(), specs)
    verify_entries(entries, specs)

    if verbose:
        print(f"random weights: seed {seed}, input_size {input_size} (NOT trained)")
    return _finish(entries, out, model.arch.name, dtype, check, verbose)


def _finish(
    entries: Sequence[TensorEntry],
    out: Path,
    arch_name: str,
    dtype: Dtype,
    check: bool,
    verbose: bool,
) -> dict[str, Any]:
    header = write_container(out, entries, dtype=dtype, arch=arch_name)
    total = sum(entry.numel for entry in entries)
    size_bytes = out.stat().st_size

    if check:
        _, back = read_container(out)
        if set(back) != {entry.name for entry in entries}:
            raise AssertionError("round-trip lost or invented tensors")
        worst = 0.0
        worst_name = ""
        for entry in entries:
            got = back[entry.name].values
            if back[entry.name].shape != entry.shape:
                raise AssertionError(f"{entry.name}: shape changed on round-trip")
            for a, b in zip(entry.values, got):
                delta = abs(a - b)
                if delta > worst:
                    worst, worst_name = delta, entry.name
        if dtype == "f32" and worst != 0.0:
            # float32 -> Python float -> array('f') is lossless in both
            # directions, so anything but zero here is a real bug.
            raise AssertionError(
                f"f32 round-trip is not exact: {worst_name} differs by {worst:g}"
            )
        if verbose:
            note = (
                "exact"
                if dtype == "f32"
                else f"max quantisation error {worst:.3g} ({worst_name})"
            )
            print(f"  round-trip: {note}")

    if verbose:
        print(
            f"wrote {out}\n"
            f"  arch {arch_name}  dtype {dtype}  tensors {len(entries)}  "
            f"parameters {total} ({total / 1e6:.2f} M)\n"
            f"  file size {size_bytes} bytes ({size_bytes / 1e6:.2f} MB)"
        )
    return header


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_idm.py",
        description=(
            "Fold BatchNorm and write the .idm weight container that "
            "src/model/weights.ts loads."
        ),
        epilog=(
            "examples:\n"
            "  python tools/export/export_idm.py --checkpoint runs/v1/best.pt \\\n"
            "      --out public/weights/illumina-448.idm\n"
            "  python tools/export/export_idm.py --checkpoint runs/v1/best.pt \\\n"
            "      --out public/weights/illumina-448.f16.idm --dtype f16\n"
            "  python tools/export/export_idm.py --random --seed 1 \\\n"
            "      --out public/weights/random-448.idm\n"
            "\n"
            "verify the result against the TypeScript reference with\n"
            "  python tools/export/verify_parity.py --size 64\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint", type=Path, help="checkpoint written by tools/train/train.py"
    )
    source.add_argument(
        "--random",
        action="store_true",
        help="emit Kaiming-initialised weights instead, so the container and the "
        "runtime can be tested without training",
    )
    parser.add_argument("--out", type=Path, required=True, help="destination .idm file")
    parser.add_argument(
        "--dtype",
        choices=("f32", "f16"),
        default="f32",
        help="blob element type (default: f32; f16 halves the download to ~1.39 MB)",
    )
    parser.add_argument(
        "--weights",
        choices=("auto", "ema", "model"),
        default="auto",
        help="which copy of the weights to export from a checkpoint "
        "(default: auto, meaning EMA when the checkpoint has them)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="override the network size; must be a multiple of 32 "
        "(default: 448, or whatever the checkpoint recorded)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="seed for --random (default: 1)"
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="skip reading the file back and comparing it with what was written",
    )
    parser.add_argument("--quiet", action="store_true", help="print nothing on success")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet
    try:
        if args.random:
            export_random(
                args.out,
                input_size=args.input_size if args.input_size is not None else 448,
                seed=args.seed,
                dtype=args.dtype,
                check=args.check,
                verbose=verbose,
            )
        else:
            export_checkpoint(
                args.checkpoint,
                args.out,
                dtype=args.dtype,
                prefer=args.weights,
                input_size=args.input_size,
                check=args.check,
                verbose=verbose,
            )
    except (ValueError, FileNotFoundError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
