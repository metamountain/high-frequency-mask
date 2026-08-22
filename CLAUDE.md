# High Frequency Mask — working notes

## What this is actually for

**Protecting flat areas during ComfyUI upscale and refine passes.** Upscalers invent
detail in surfaces that should stay smooth and shift their colour while doing it;
**Flux Klein is especially prone to both**. The node masks those regions out so the
sampler never touches them.

This framing matters for every decision here. The metric is not "does the mask find
texture" — it is **how much leaks through in flat regions**, because every bit of leak
is an upscaler free to hallucinate. Optimise for protection, not for coverage.

## The defaults are wrong for this job

Measured on `tests/assets/spraycrete.png` at 1024px:

| | flat leak | texture kept | flat fully protected |
|---|---|---|---|
| shipped defaults | .116 | .297 | 81.5% |
| radius 0.35×, `entrauschen` 0 | **.038** | **.417** | **90.4%** |

Two independent causes, both worth fixing in code rather than documentation:

- **The auto radius is ~3× too coarse.** Leak falls monotonically with radius and
  plateaus below 0.3×; texture retention peaks at 0.25–0.35×. So `min(W,H)/52` should
  be closer to `min(W,H)/150`. In px: 5–7 at 1024, 10–14 at 2048.
- **`entrauschen` defaults to 1.0**, which pre-smooths the image before anything is
  measured. On clean renders that costs 60% more leak *and* removes real texture. It
  should default to 0 and only be raised for grainy or JPEG sources.

Full tables: `docs/upscale-settings.md`.

## `base` drives four things at once

`base = radius_override or max(4, round(min(W, H) / 52))` (`__init__.py:109`).
These are not independently tunable:

| consumer | value | line |
|---|---|---|
| grain pre-blur sigma | `base / 12 * entrauschen` | `:113` |
| high-pass sigma | `base / 2` | `:116` |
| grow/shrink px | `base * 1.5 * abs(groesse)` | `:135` |
| feather sigma | `base * 0.5 * weichheit` | `:139` |

This coupling is why turning the radius down also turns grain suppression down — the
two "fixes" above are partly the same fix. Decoupling `entrauschen` from `base` is the
single highest-value change to the maths.

## Units trap: levels are high-pass contrast, not brightness

`black_override` / `white_override` are 0–255 **of the high-pass image**. Auto values
land near `black 5..10 / white 12..76`, i.e. in the bottom 10–30% of the nominal range.

They are also a hidden mode switch: if *either* is > 0 the quantile autolevel is
bypassed entirely and `staerke` becomes a no-op. Black-only input silently falls back
to `white = 255` and yields a near-empty mask.

## Known breakage (reproduced in `tests/test_regressions.py`)

- **`torch.quantile` caps at 2^24 = 16,777,216 elements** (`:130`). Images at or above
  ~4096×4096 raise `RuntimeError: quantile() input tensor is too large`. **Reachable
  on a normal upscale** — two files in this install's `input/` already exceed it.
- **Reflect-pad overflow** (`:141`). `_blur` needs `1.5 * base * weichheit < min(W,H)`.
  Clamp `r` to `min(W,H) - 1`.
- **Latent batch mismatch falls through** (`:167`). The guard only handles
  `mm.shape[0] == 1`; for `1 < mask_batch < latent_batch` the `mm[:sh[0]]` slice is a
  no-op and a wrong-sized `noise_mask` reaches the sampler.

## Corrected: `staerke` is not dead above 1.4

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
