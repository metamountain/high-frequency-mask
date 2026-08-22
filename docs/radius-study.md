# Radius: what the measurements say

Everything here is reproducible from the repo alone:

```
python_embeded/python.exe tests/test_radius.py
```

The fixture is `tests/assets/testchart.png` — a generated 1024x1024 chart with a
smooth gradient background and six 256px patches, each carrying band-limited noise
at one known texture scale (4, 8, 16, 32, 64, 128 px). A photo would work too, but
you never know exactly where its detail is; the chart has exact ground truth, so
"did the mask find the 8px texture and ignore the flat field" is an exact question.

Score = `mean(mask) inside a patch − mean(mask) on the flat background`.

## The radius optimum tracks the texture scale

`grain_suppress = 0`, so this isolates the high-pass radius:

| texture | 0.25 | 0.35 | 0.5 | 0.7 | 1.0 | 1.4 | 2.0 | 2.8 | 4.0 | peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 4px   | .413 | .421 | **.421** | .416 | .410 | .399 | .384 | .374 | .367 | 0.5 |
| 8px   | .329 | .370 | .392 | **.397** | .397 | .390 | .379 | .370 | .363 | 0.7 |
| 16px  | .139 | .227 | .315 | .364 | .390 | **.393** | .384 | .379 | .378 | 1.4 |
| 32px  | .018 | .040 | .108 | .203 | .278 | .315 | **.328** | .325 | .322 | 2.0 |
| 64px  | .029 | .035 | .055 | .096 | .169 | .260 | .335 | .361 | **.373** | 4.0 |
| 128px | .003 | -.002 | -.003 | -.001 | .005 | .020 | .056 | .119 | **.219** | 4.0+ |

**There is no single correct radius.** Every multiplier from 0.25 to 4.0 is the best
value for *some* structure size, and the peak moves monotonically with it. That is
the argument for exposing it as a slider rather than picking a constant.

Two range conclusions:

- **0.25–4.00 is the right span.** It covers texture from 4px to 64px, which is the
  useful scope for a detail mask. 128px still hasn't peaked at 4.0, but structures
  that large are composition, not detail.
- **Default 1.0 targets ~16px structures.** For skin, fabric and foliage the sweet
  spot is 0.5–0.7 — the default is roughly 2x too coarse for fine texture.

## Grain suppression steals the top of the range

`base` drives the pre-analysis blur as well as the high-pass, so turning the radius
up also turns up smoothing that runs *before* anything is measured. Same sweep with
`grain_suppress = 1` (as shipped):

| texture | 0.25 | 0.35 | 0.5 | 0.7 | 1.0 | 1.4 | 2.0 | 2.8 | 4.0 |
|---|---|---|---|---|---|---|---|---|---|
| 4px  | .413 | .424 | .421 | .413 | .400 | .369 | .324 | .266 | **.173** |
| 8px  | .329 | .374 | .396 | .400 | .398 | .384 | .365 | .343 | **.299** |
| 16px | .139 | .231 | .322 | .371 | .396 | .392 | .378 | .369 | **.352** |

At 4px texture the far end of the slider loses more than half its discrimination
(.367 → .173) — and none of that loss is the high-pass, it is the coupled pre-blur.
Decoupling `grain_suppress` from `base` makes the whole slider usable.

## Resolution stability of `min(W, H) / 52`

Same chart at several resolutions, masks compared at 256px against the 1024 run,
`grain_suppress = 0` so downscale-induced grain removal isn't measured as radius error.
Corroborated on three real photographs (3024x4032, 2880x2880, 5256x3504):

| | 384 | 512 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|
| IoU vs reference | .71–.83 | .80–.93 | .87–.95 | ref | .89–.98 | .87–.97 |

**The scaling law holds from ~768px upward** and degrades below it, where the
`max(4, ...)` floor and pixel quantisation take over rather than the constant. Any
resolution test must stay above 256px or it measures the floor, not the law.

## How this compares to other tools

| tool | control | default | scales with resolution? |
|---|---|---|---|
| this node | `min(W,H)/52`, sigma = base/2 | sigma 10 @1024px | yes |
| WAS Image High Pass Filter | `radius` int 1–500 | 10 | no |
| KJNodes Sharpen (high_pass) | `radius` float 0.5–5.0 | 1.0 | no |
| Masquerade blur | `radius` int 0–48 | 10 | no |
| Photoshop frequency separation | Gaussian radius | 3–15, ~10 typical | by hand |

The ecosystem converges on **~10px absolute**. This node lands on sigma 10 at 1024px
— matching convention there — then diverges upward with resolution, reaching sigma
~38 on a 4032px image, about 4x the top of photographic practice.

That divergence is **not a bug**. Sharpening wants a fixed small radius to control
halos; a detail *mask* wants the same perceptual regions selected regardless of
resolution, which is exactly what the stability table above confirms the scaling law
delivers. An absolute radius would make the mask differ wildly between 512 and 2048.
The law is right for this use case even though it is unusual for the ecosystem — the
*constant* is simply on the coarse side, which is what the default-1.0 note is about.

Sources: [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui),
[Masquerade](https://github.com/BadCafeCode/masquerade-nodes-comfyui),
[KJNodes](https://github.com/kijai/ComfyUI-KJNodes),
[Fstoppers frequency separation guide](https://fstoppers.com/post-production/ultimate-guide-frequency-separation-technique-8699),
[PhotoshopCAFE](https://photoshopcafe.com/tutorial/frequency-separation.htm)
