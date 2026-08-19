"""IlluminaDepth-448 training package.

Three modules, each a direct counterpart of a piece of the TypeScript runtime:

* :mod:`illumina.model`  -- the network, built from a Python port of the
  declarative op table in ``src/model/arch.ts``, with a BatchNorm training
  variant and the folding step that produces the inference weights.
* :mod:`illumina.losses` -- the MiDaS scale- and shift-invariant objective the
  student is distilled with.
* :mod:`illumina.data`   -- image folder dataset and frozen teacher.

Nothing here imports the browser code; the two halves are kept in agreement by
``tools/export/verify_parity.py``, which runs both on identical weights.
"""
