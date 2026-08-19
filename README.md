# IlluminaDepth

Real-time monocular depth estimation and depth-aware relighting, written
end-to-end in [TypeGPU](https://typegpu.com). A 448×448 convolutional depth
network, a screen-space lighting pass and the presentation draw are recorded
into **one `GPUCommandEncoder` and submitted once**. The depth map is produced by
a WGSL kernel and consumed by another WGSL kernel microseconds later; it never
touches the CPU, never round-trips through a buffer readback, and never crosses
an interop boundary between an ML runtime and a renderer — because there is no
separate ML runtime.

```
encoder ─┬─ render pass  ──   1 draw    procedural scene (optional source)
         ├─ compute pass ─┬─ 40 dispatches   IlluminaDepth-448 inference
         │                └─  1 dispatch     depth-aware shading
         └─ render pass  ──   1 draw         tone-map and present
submit ×1
```

---

## What is actually here

| | |
|---|---|
| Network | `IlluminaDepth-448`, 1.39 M parameters, **2.20 GFLOP** per frame |
| Input / output | 448×448 RGB → 448×448 relative inverse depth (disparity) |
| Dispatches | **40** for inference (one per op), +1 for lighting |
| Command encoders / submits per frame | **1 / 1** |
| Activation memory | **19.0 MB** with arena aliasing (63.1 MB without) |
| Weight memory | 5.56 MB at fp32 |
| Authoring | TypeScript kernels (`'use gpu'`) transpiled to WGSL by TypeGPU 0.12 |
| Verification | 79 tests executing real WGSL on a real WebGPU device |

Everything in that table is measured, not estimated: run `npm run arch` for the
layer table and open the app for the live counters.

---

## Performance, honestly

The design target was **8 ms per frame on an M1 Max**. Here is the reasoning and
exactly what has and has not been verified.

The M1 Max GPU peaks near 10.4 TFLOP/s fp32. Hand-written WGSL convolutions on
tensors this small are latency- and bandwidth-bound rather than ALU-bound, and
realistically achieve 5–20 % of peak. Against a 2.20 GFLOP workload:

| achieved fraction of peak | projected inference time |
|---|---|
| 5 % | 4.23 ms |
| 10 % | 2.11 ms |
| 15 % | 1.41 ms |
| 20 % | 1.06 ms |

The architecture was sized so that even the pessimistic end of that range leaves
room for the lighting pass and the draw inside 8 ms.

**What has been verified:** every kernel and the whole 40-dispatch graph execute
correctly and match a CPU reference numerically, on a real WebGPU
implementation (Google Dawn). The app runs at 60 fps (vsync-capped) in Chrome.

**What has not been verified:** the actual millisecond figure on an M1 Max. This
project was developed in a Linux container with no Apple GPU, so the table above
is a projection from a FLOP budget, not a measurement. To measure it, open the
app on the target machine and enable *Diagnostics → Per-stage GPU timing*: that
splits the frame into separately timestamped passes and reports true GPU
milliseconds for inference, shading and compositing via `timestamp-query`. CPU
timers around `submit()` would measure only command recording and are not used
anywhere in this project.

---

## Weights

**The weights shipped with this repository are untrained.** There is no
checkpoint in the box. On startup the app looks for
`public/weights/illumina-depth-448.idm`; if it is absent it falls back to a
deterministic Kaiming initialisation and says so in the sidebar, in orange.

With untrained weights the whole pipeline runs, the counters are real and the
lighting is real, but the depth is a smooth arbitrary field rather than a
reconstruction of the scene. Nothing is faked to hide this.

To get real weights, `tools/` contains the complete pipeline: a PyTorch model
built from the same layer table, a distillation trainer, and an exporter that
writes the `.idm` container this app loads. See [`tools/README.md`](tools/README.md).

Because the architecture is defined once in `src/model/arch.ts` and consumed by
the GPU runner, the CPU reference, the weight container *and* the PyTorch model,
the four cannot drift apart.

---

## Running it

```bash
npm install
npm run dev          # http://localhost:5173
```

Requires Node 20+ and a browser with WebGPU: Chrome/Edge 113+, or Safari 18+.

The test-suite's headless WebGPU implementation (`@kmamal/gpu`) is a native
addon and is listed as an *optional* dependency, so `npm install` still succeeds
where no prebuilt binary exists. The application never uses it; only `npm test`
does.

```bash
npm run build        # typecheck + production bundle
npm test             # 79 tests on a real GPU (see "Testing" below)
npm run arch         # print the layer table, FLOP and parameter budget
```

### Sources

* **Procedural** (default) — a ray-marched SDF scene rendered on the GPU in the
  same encoder. Because it is analytic, its exact depth is known, so
  *Diagnostics → Score against ground truth* can report AbsRel, RMSE and δ<1.25
  after the least-squares affine alignment that relative-depth models require.
* **Camera** — webcam frames.
* **Image** — any still you drop in.

### Views

`lit`, `depth`, `normals`, `occlusion` and `albedo` let you inspect each stage of
the shading independently.

---

## How the inference works

### Everything is one dispatch per op

The layer table in `src/model/arch.ts` is a list of operations, and each maps to
exactly one compute dispatch. Bias, activation and residual addition are always
folded into the kernel that produces the value, so no tensor is ever written to
memory purely to be read straight back.

The one that matters most is `dwpw`: a depthwise convolution fused into the
pointwise convolution that follows it. The naive version of this fusion is a
trap — the pointwise sums over *all* input channels, so each thread needs the
depthwise result for every channel at its pixels, and recomputing it per output
channel tile inflates the whole network's FLOP count by about **37 %**. Instead
the workgroup computes each depthwise value once, cooperatively, into workgroup
memory. Duplication drops to about **1 %** while still costing one dispatch.
(`src/model/kernels/dwpw.ts` derives this in full.)

### Kernels are specialised, not parameterised

Each kernel is produced by a factory that closes over its shapes, so `C_in`,
strides, tile sizes and activation choice become literals in the generated WGSL
rather than uniform reads. Loop bounds are compile-time constants and the hot
loops are unrolled with `tgpu.unroll`.

The dispatch geometry is chosen per op by an explicit cost model that prices
idle lanes, duplicated work and weight traffic (`chooseTiling`,
`chooseDwPwTiling`), rather than by a fixed workgroup size.

### Layout

Tensors are channels-last with channels packed in groups of four, so a tensor is
an `array<vec4f>` and every load is a full 16-byte transaction. Weights are
pre-swizzled at load time into exactly the order the kernels walk them, so no
kernel performs a strided or gathered weight read. The packed float layout is
byte-identical to plain NHWC, which is what lets the tests compare GPU and CPU
buffers without conversion.

Activations live in a pooled arena: live ranges are computed from the op list and
buffers are reused once their occupant is dead, which is where 63.1 MB becomes
19.0 MB.

### Architecture

A MobileNet-style encoder and a lightweight FPN decoder, sized to the FLOP
budget:

* Stem: 3×3 stride-2 convolution, 448 → 224.
* Five stages of inverted-residual blocks (16 → 24 → 32 → 56 → 104 → 176
  channels), 5×5 depthwise kernels at low resolution where they are cheap.
* Downsampling blocks run the depthwise **first**, at stride 2, so the expensive
  channel expansion happens at the lower resolution — a 4× saving over
  MobileNetV2's ordering.
* An FPN decoder in which each level's refine step already emits the width the
  level above expects, so the lateral fusion needs no projection of the coarse
  tensor. That alone saves ~284 MFLOP per frame.
* A sigmoid head producing disparity at 224², then an edge-aware joint-bilateral
  upsample to 448² guided by image luminance, with temporal stabilisation folded
  into the same dispatch.

---

## How the lighting works

The estimated depth map is the *only* geometry the renderer has. From a single
disparity image the shading pass derives, per pixel:

1. **View-space position**, by unprojecting linearised depth through the camera's
   field of view. The disparity → depth mapping is deliberately reciprocal, which
   makes `1/z` affine in disparity and therefore linear in screen space over
   planar surfaces — a property the next step depends on.
2. **Normals**, using the "accurate" reconstruction of Yuwen Wu and Turánszki:
   depth is extrapolated from the two neighbours on each side and the derivative
   is taken from whichever side extrapolates better. Central differences would
   corrupt a two-pixel band around every silhouette, which reads as a halo around
   every object.
3. **Direct lighting** — Lambert plus a GGX specular lobe, with windowed
   inverse-square falloff so lights reach exactly zero at their radius.
4. **Screen-space contact shadows**, ray-marching the depth buffer toward each
   light with a thickness test and per-pixel dither.
5. **Horizon-based ambient occlusion**.
6. **Volumetric scattering** along the view ray, and depth fog, both in linear
   light.

Tone mapping (fitted ACES or Reinhard), grading and sRGB encoding happen in the
composite draw.

---

## Testing

The test suite is not a typecheck. `@kmamal/gpu` boots Google Dawn against
whatever Vulkan driver is present — a real GPU when there is one, Mesa's
`lavapipe` software rasteriser otherwise — so every kernel is compiled by Tint
and **executed**, then compared against the CPU reference in
`src/model/reference.ts`.

```
npm test
```

**The network**
* `kernels.test.ts` — every kernel against the reference: ragged shapes,
  stride 2, 5×5 kernels, residuals, and a check that out-of-range lanes write
  nothing.
* `shapes.test.ts` — the awkward shapes: single pixels, single rows and
  columns, channel counts that divide no tile evenly, input channel counts that
  are not a multiple of the depthwise workgroup-memory chunk, and the exact
  shapes the shipping network uses.
* `network.test.ts` — the whole graph at 64×64 (geometrically identical, 49×
  less work) compared op by op against a complete CPU forward pass. Also covers
  arena liveness and weight-container round-tripping.
* `shipping-config.test.ts` — builds and executes all forty dispatches at the
  real 448×448, which is the only way to know the tilings chosen at full
  resolution are legal.

**The frame**
* `frame.test.ts` — one encoder, one submit, correct ping-pong parity, lighting
  that responds to light position, and real timestamp queries.
* `dispatch-count.test.ts` — wraps the device and counts the WebGPU calls that
  actually happen, so the "one encoder, one submit, no readbacks" claim is
  measured rather than asserted.
* `timing.test.ts` — the profiler's staging ring under back-to-back frames.

**The lighting**
* `brdf.test.ts` — the GGX and Smith terms against their published closed forms.
* `extremes.test.ts` — every UI slider at both ends, scanned for NaN and infinity.
* `shading.test.ts` — the shading kernel driven by a hand-authored depth map, so
  the geometry is known exactly.
* `metrics.test.ts` — depth metrics against closed-form answers.
* `scene.test.ts` — the procedural scene renders geometry and consistent depth.

**The Python contract**
* `interop.test.ts` — the exporter and the loader agree on weight names, shapes
  and blob order, and a container written by Python is read back by TypeScript.

Having two independent implementations is what found the real bugs: a
border-clamping error in the CPU bilinear upsample, a reference-vs-value
aliasing bug in a kernel, two lifetime defects in the profiler's staging ring,
and a denominator floor that was silently capping the specular highlight on
smooth surfaces by five orders of magnitude. None would have been visible from
a screenshot.

### Note on headless browsers

Headless Chrome cannot screenshot a WebGPU canvas in this configuration — a
canvas cleared to solid red also captures as black. Visual verification is
therefore done by reading rendered textures back and asserting on pixel
statistics, not by image capture.

---

## Layout

```
src/
  model/
    arch.ts            layer table — the single source of truth
    layout.ts          packing, indexing, tiling cost models
    reference.ts       CPU reference implementation (the spec)
    weights.ts         .idm container, loader, initialisation
    runner.ts          pipelines, bind groups, recording
    kernels/           pointwise, dwpw, conv, lateral, io, activation
  lighting/
    shade.ts           depth-aware shading kernel
    composite.ts       tone mapping and presentation
    lights.ts          light structs
    renderer.ts        lighting resources
  engine/
    frame.ts           the single-encoder frame graph
    arena.ts           activation memory planning
    timing.ts          timestamp-query profiler
  scene/
    procedural.ts      ray-marched SDF scene + ground truth
    source.ts          procedural / camera / image
    metrics.ts         scale-and-shift-invariant depth scoring
tools/                 PyTorch model, trainer, exporter
test/                  executed-on-GPU test suite
```

---

## Licence

MIT.
