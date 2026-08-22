# High Frequency Mask — working notes

ComfyUI custom node. Builds a mask that is white where an image has texture and
black where it is flat, so a sampler only touches the detailed regions.
Single file: `__init__.py`. All torch, GPU, batch-aware.

## Pipeline

```
grayscale -> pre-blur (grain_suppress) -> highpass (g - blur(g)) -> clamp>=0, x3
          -> per-image quantile autolevel -> grow/shrink -> feather -> invert
```

## The one thing to know: `base` drives four things at once

`base = radius_override or max(4, round(min(W, H) / 52))` (`__init__.py:109`).
Changing it moves **all** of these together — they are not independently tunable:

| consumer | value | line |
|---|---|---|
| grain pre-blur sigma | `base / 12 * grain_suppress` | `:113` |
| highpass sigma       | `base / 2`                  | `:116` |
| grow/shrink px       | `base * 1.5 * abs(grow)`    | `:135` |
| feather sigma        | `base * 0.5 * feather`      | `:139` |

This coupling is measurable, not theoretical: raising the radius also raises grain
suppression, which over-smooths the image before analysis and makes high radius
values look worse than they are. See `docs/radius-study.md` — decoupling grain from
`base` recovers roughly half the lost discrimination at the top of the range.

## Units trap: levels are high-pass contrast, not brightness

`black_override` / `white_override` are 0–255 **of the high-pass image**, not of the
source. Auto-derived values land near `black 0 / white 8..26`, i.e. in the bottom 10%
of the nominal range. Any UI that exposes these must say so or they are unusable.

Also: they are a hidden mode switch. If *either* is > 0 the entire quantile autolevel
is bypassed and `staerke` becomes a no-op (verified: identical output at 0.4/1.0/2.0).
Black-only input silently falls back to `white = 255` and yields a near-empty mask.

## Known breakage (all reproduced — see `tests/test_regressions.py`)

- **`torch.quantile` caps at 2^24 = 16,777,216 elements** (`:130`). Any image at or
  above ~4096x4096 raises `RuntimeError: quantile() input tensor is too large`.
  Two files in this install's `input/` already exceed it (6720x4480, 4096x5120).
- **Reflect-pad overflow** (`:141`). `_blur` needs `1.5 * base * feather < min(W, H)`.
  Reachable via `radius_override=120, weichheit=4` at 512px. Switching the radius
  control to a relative multiplier mostly defuses this, but the `max(4, ...)` floor
  keeps it alive on very small inputs — still clamp `r` to `min(W, H) - 1`.
- **Latent batch mismatch falls through** (`:167`). The guard only handles
  `mm.shape[0] == 1`; for `1 < mask_batch < latent_batch` the `mm[:sh[0]]` slice is a
  no-op and a wrong-sized `noise_mask` reaches the sampler.
- **`staerke` is dead above ~1.4.** `pw` and `pb` converge as it rises (62/60 at 2.0).
  On a real, zero-inflated high-pass histogram both land on 0, so the
  `max(blk + 0.03, ...)` floor sets the white point instead of the quantile. Measured
  sky/detail separation is identical at 1.4, 1.8 and 2.0.

## Conventions that are correct as written

- `_erode = -_dilate(-x)` — `max_pool2d` pads with `-inf`, which negates to `+inf`,
  so erosion correctly does not eat inward from the frame.
- `samples.copy()` — shallow copy is the ComfyUI idiom, the caller's dict is not mutated.
- Per-image quantiles are deliberate, but they are also the reason masks flicker
  across video frames. That flicker — not manual level control — is the real reason
  the override inputs exist.

## Environment

Portable install. Use `../../../python_embeded/python.exe`, not a system Python.
ComfyUI 0.33.1, frontend 1.48.7, torch 2.12.1+cu130. `pytest` is **not** installed —
`tests/run_tests.py` runs standalone.

`custom_nodes/` is gitignored by ComfyUI (`.gitignore:8`), so this directory is
free-standing and safe to make its own repo.
