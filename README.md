# High Frequency Mask

**Stops an upscaler wrecking the flat parts of your image.**

Upscale and refine passes invent detail in surfaces that should stay smooth — skies,
walls, panels, skin — and drift the colour while doing it. **Flux Klein is especially
prone to both.** This node builds a mask from the image's own texture (white where
there is real detail, black where it is flat) and hands it to the sampler as a noise
mask. The flat regions are then never touched, so they keep their original tone and
stay clean.

Everything runs in torch on the GPU and handles batches.

## Quick start for upscaling

Connect `image` from your source, `samples` from your latent, and the `latent` output
into the sampler. Then **change the defaults** — they ship tuned too coarse for this job:

| setting | default | use this | why |
|---|---|---|---|
| `radius_override` | 0 (auto ≈ 20 @1024) | **5–7 @1024, 10–14 @2048** | auto is ~3× too coarse; this is where protection peaks |
| `entrauschen` | 1.00 | **0.00** | on clean renders it smooths away real texture before measuring |
| `weichheit` | 1.00 | **0.5–1.0** | soft edge, so the mask leaves no seam |
| `groesse` | 1.00 | **0.5–1.0** | real detail keeps a safety margin |

Measured effect on the test render — flat-area leak is what an upscaler gets to
hallucinate into:

| | flat leak | texture kept | flat pixels fully protected |
|---|---|---|---|
| defaults | 0.116 | 0.297 | 81.5% |
| **tuned as above** | **0.038** | **0.417** | **90.4%** |

Three times less leak *and* more real texture retained.

## Inputs

> These are the current widget names, which are German. English names
> (`sensitivity`, `grow`, `feather`, `grain_suppress`, `highpass_radius`) are the
> next change, together with sliders.

| input | what it does |
|---|---|
| `staerke` | how much counts as detail. Nearly all the useful travel is below 1.2 |
| `groesse` | expand the white areas; negative shrinks |
| `weichheit` | edge softness between white and black |
| `entrauschen` | pre-smoothing against grain and JPEG. **Set 0 for clean sources** |
| `invert` | swap: protect the detail, sample the flat areas |
| `samples` *(optional)* | if connected, the mask is attached as the latent's noise mask |
| `radius_override` | high-pass radius in px. `0` = automatic from image size |
| `black_override` / `white_override` | manual levels — see the caveat below |

Outputs: `mask`, `mask_preview` (as IMAGE), `latent`, `info`.

## Finding it, and help in the node

Double-click the canvas and search **upscale mask**, **protect flat areas**, **colour
shift**, **too much detail**, **flux klein**, **detail mask**, **high pass**, or the
German **Detailmaske** / **Flaechen schuetzen** / **Farbverschiebung**. 30 aliases, so
you needn't remember the node's name. It lives under **mask**.

Every input has an English tooltip on hover, all four outputs are documented, and the
node's help panel carries the settings guidance above.

## Known issues

Each pinned by a test — see [CLAUDE.md](CLAUDE.md).

- Images at or above ~4096×4096 raise `quantile() input tensor is too large`.
  **This is reachable on a normal upscale.**
- Large `radius_override` with high `weichheit` overflows the blur's padding
- `noise_mask` can reach the sampler with the wrong batch size
- `black_override` / `white_override` are **high-pass contrast**, not brightness —
  useful values are near 0–30 — and setting either silently disables `staerke`

## Tests

`pytest` is not installed in the ComfyUI portable python, so the suite runs standalone:

```bash
python_embeded/python.exe ComfyUI/custom_nodes/high-frequency-mask/tests/run_tests.py
```

Known bugs are `XFAIL`, so the suite is green today and turns red when one is fixed
without being deregistered. `tests/test_radius.py` run directly prints the radius
tables. Full measurements: **[docs/upscale-settings.md](docs/upscale-settings.md)**.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. No dependencies beyond ComfyUI.
