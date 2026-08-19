# Weights

Put a trained checkpoint here as `illumina-depth-448.idm`.

The application fetches `weights/illumina-depth-448.idm` at startup. If the file
is absent it falls back to a deterministic Kaiming initialisation and says so in
the sidebar — the pipeline runs end to end, but the depth is an arbitrary smooth
field rather than a reconstruction.

`.idm` files are produced by `tools/export/export_idm.py`. See
[`../../tools/README.md`](../../tools/README.md) for training and export.

The format is a `u32` little-endian header length, a UTF-8 JSON header, then a
tightly packed blob — the same shape as safetensors. Weights are stored in
logical PyTorch order; the GPU-side swizzling happens at load time in
`src/model/layout.ts`.
