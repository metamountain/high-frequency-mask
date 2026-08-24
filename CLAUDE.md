# High Frequency Mask — working notes

## What this is actually for

**Protecting flat areas during ComfyUI upscale and refine passes.** Upscalers invent
detail in surfaces that should stay smooth and shift their colour while doing it;
**Flux Klein is especially prone to both**. The node masks those regions out so the
sampler never touches them.

This framing matters for every decision here. The metric is not "does the mask find
texture" — it is **how much leaks through in flat regions**, because every bit of leak
is an upscaler free to hallucinate. Optimise for protection, not for coverage.

## The detector: guided filter, not a plain high pass

The original `g - gaussian_blur(g)` had two problems, both measured:

1. **It rectified.** `.clamp(min=0)` kept only the bright side of every edge,
   throwing away half the signal. Using `.abs()` alone lifts texture retention
   from .350 to .537 and halves the halo.
2. **A Gaussian low-pass cannot tell an edge from texture.** It smears straight
   across a strong edge, so the difference lights up flat ground on BOTH sides --
   a halo. That is the pale glow in sky beside a cliff, and across the glass
   inside a window frame. It is the single largest source of leak.

Replacing the low-pass with a self-guided edge-preserving filter fixes both.
Measured at `grow 0`, same autolevel:

| detector | halo leak | texture kept |
|---|---|---|
| rectified high pass (original) | .371 | .350 |
| both edge sides (`.abs()`) | .184 | .537 |
| **guided filter** | **.009** | **.515** |

41x less halo on the cliff, 7x on the concrete, *while finding more texture*.
It is not a protection/detail trade -- the original was losing on both counts.

**The advantage decays as `grow` rises**, because dilation floods the halo zone
by itself:

| grow | halo: plain -> guided | white: plain -> guided |
|---|---|---|
| 0.15 (4px) | .592 -> .287 (2.1x) | 20.3% -> **41.8%** |
| 0.5 (15px) | .927 -> .739 (1.3x) | 62.8% -> 65.3% |
| 1.0 (30px) | .999 -> .986 (1.0x) | 72.9% -> 72.7% |

So the detector matters most at low `grow`. At every setting it yields *more*
white, never less -- it blocks smarter, not harder.

Things that did NOT work: combining guided with multi-scale is worse than guided
alone (.026 vs .009 -- over-suppression), and subtracting a gradient term from
local std is worse than doing nothing (.654).

## The defaults are wrong for this job

Measured on `tests/assets/spraycrete.png` at 1024px:

| | flat leak | texture kept | flat fully protected |
|---|---|---|---|
| shipped defaults | .116 | .297 | 81.5% |
| radius 0.35×, `grain_filter` 0 | **.038** | **.417** | **90.4%** |

Two independent causes, both worth fixing in code rather than documentation:

- **The auto radius is ~3× too coarse.** Leak falls monotonically with radius and
  plateaus below 0.3×; texture retention peaks at 0.25–0.35×. So `min(W,H)/52` should
  be closer to `min(W,H)/150`. In px: 5–7 at 1024, 10–14 at 2048. **Still open** —
  changing it moves every saved mask, so it stays a documented `radius_override`
  recommendation rather than a new default for now.
- ~~**`grain_filter` defaults to 1.0**~~ **Fixed.** It pre-smooths the image before
  anything is measured, which on clean renders cost 60% more leak *and* removed real
  texture. Default is now 0; raise it only for grainy or JPEG sources.

Full tables: `docs/upscale-settings.md`.

## `base` drives four things at once

`base = radius_override or max(4, round(min(W, H) / 52))` (`__init__.py:109`).
These are not independently tunable:

| consumer | value | line |
|---|---|---|
| grain pre-blur sigma | `base / 12 * grain_filter` | `:113` |
| high-pass sigma | `base / 2` | `:116` |
| grow/shrink px | `base * 1.5 * abs(grow)` | `:135` |
| feather sigma | `base * 0.5 * feather` | `:139` |

This coupling is why turning the radius down also turns grain suppression down — the
two "fixes" above are partly the same fix. Decoupling `grain_filter` from `base` is the
single highest-value change to the maths.

## Units trap: levels are high-pass contrast, not brightness

`black_override` / `white_override` are 0–255 **of the high-pass image**. Auto values
land near `black 5..10 / white 12..76`, i.e. in the bottom 10–30% of the nominal range.

They are also a hidden mode switch: if *either* is > 0 the quantile autolevel is
bypassed entirely and `strength` becomes a no-op. Black-only input silently falls back
to `white = 255` and yields a near-empty mask.

## Fixed breakage (was reproduced in `tests/test_regressions.py`, now XFAIL-free)

- **`torch.quantile` caps at 2^24 = 16,777,216 elements.** Images at or above
  ~4096×4096 raised `RuntimeError: quantile() input tensor is too large` — reachable
  on a normal upscale. Fixed by `_quantile()`: strided subsample down to under the
  cap before calling `torch.quantile`, keeping millions of points either way.
- **Reflect-pad overflow.** `_blur` needed `1.5 * base * feather < min(W,H)`. Fixed
  by clamping the kernel radius to `min(H,W) - 1` (and renormalising the truncated
  kernel) inside `_blur` itself, so no caller has to reason about it.
- **Latent batch mismatch fell through.** The guard only handled `mm.shape[0] == 1`;
  for `1 < mask_batch < latent_batch` the `mm[:sh[0]]` slice was a no-op and a
  wrong-sized `noise_mask` reached the sampler. Fixed by tiling `mm` up to
  `latent_batch` with `repeat` before slicing, which also subsumes the old
  batch-of-1 special case.

## Weakening the mask: opacity

`strength` looked "stuck" for users because `grow` re-dilates white back out as fast
as `strength` shrinks it — the two sliders fight, and past a point `strength` has
no visible effect. `opacity` (new, appended last in `optional` so it cannot shift any
saved graph's widget positions) multiplies the finished mask *after* grow, feather and
invert have all run: `m = m * opacity` when `opacity < 1.0`. It is the one control
that reliably attenuates the result regardless of what the other sliders are doing.

## Live preview

`/hfmask/preview` mirrors `/hfmask/auto`'s round-trip pattern: the browser posts the
current widget values, the backend re-runs `build()` on the cached image
(`_LAST_IMAGE`, now `{"img": tensor, "orig_edge": int}`) and returns a mask PNG, an
overlay PNG (source tinted red where the mask would sample) and white/black/mean
stats, all base64. `web/hfmask.js` hooks every relevant widget's `callback`, debounces
120ms and aborts the in-flight request when a new one starts, so a fast drag does not
queue stale frames behind the latest one.

One thing this had to get right: `radius_override` is an **absolute px value tuned
for the full-resolution image**, but the cache is capped at 512px on the short edge.
Passing it through unscaled would make the preview look far coarser than the real
run. The handler rescales it by `cached_edge / orig_edge` before calling `build()`.

## Corrected: `strength` is not dead above 1.4

An earlier reading — that the slider stops working above ~1.4 — came from a synthetic
test chart whose high-pass histogram was far more zero-inflated than real images. On
real input the slider works across its whole range (coverage .078 → .308 from 0.4 to
2.0). It *is* badly compressed: 0.4 → 1.0 moves texture by .168, 1.4 → 2.0 by .018.
A usability problem worth remapping, not a bug. Enforced by
`test_sensitivity_is_compressed_at_the_top`.

**Lesson:** synthetic fixtures misled on the one question that mattered. The suite now
uses only the real render.

## Conventions that are correct as written

- `_erode = -_dilate(-x)` — `max_pool2d` pads with `-inf`, negating to `+inf`, so
  erosion correctly does not eat inward from the frame.
- `samples.copy()` — shallow copy is the ComfyUI idiom; the caller's dict is not mutated.
- **Scaling the radius with resolution is right here**, even though WAS, KJNodes and
  Masquerade all use a fixed ~10px. A mask that must select the same regions at every
  step of an upscale chain cannot use an absolute radius. Verified stable to IoU
  0.87–0.98 from 768px to 2048px.
- Per-image quantiles are deliberate, but they make masks flicker across video frames.
  That flicker — not manual level control — is the real reason the overrides exist.

## Environment

Portable install. Use `../../../python_embeded/python.exe`, not a system Python.
ComfyUI 0.33.1, frontend 1.48.7, torch 2.12.1+cu130. `pytest` is **not** installed —
`tests/run_tests.py` runs standalone.

`custom_nodes/` is gitignored by ComfyUI (`.gitignore:8`), so this directory is
free-standing and safe as its own repo.
