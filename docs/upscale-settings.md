# Settings for upscaling: what the measurements say

Reproduce from the repo:

```
python_embeded/python.exe tests/test_radius.py
```

## The fixtures

Two **ComfyUI generations** — the kind of image this actually runs on, not
photographs. Both pair large genuinely flat regions with genuine texture, which is
where an upscaler goes wrong: it invents detail in the flat parts and shifts colour.

| fixture | flat | textured |
|---|---|---|
| `spraycrete.jpg` | smooth concrete panels, overcast sky | rough sprayed aggregate |
| `degenerator.jpg` | large flat sky, snow field, panels | cliff rock, weathered stone |

1024px, JPEG q95. The compression is high-frequency and could in principle
contaminate the measurement, so it was checked: PNG vs JPEG q95 shifts the leak from
.0378 to .0388 and protection from 90.4% to 90.3% — negligible, at a quarter the size.

Flat and textured regions are separated by local standard deviation per 8px block
(bottom and top quartile). **The number that matters is the flat-area leak** — every
bit the mask lets through there is an upscaler free to hallucinate detail and drift
colour.

## Detector: guided beats the plain high pass at everything

Same autolevel, `grow 0`, `feather 0` -- only the detector differs. "Halo leak"
is the mask level over flat blocks within 24px of a strong edge, which is the
sky-beside-the-cliff and inside-a-window case.

| detector | halo (cliff) | halo (concrete) | texture kept |
|---|---|---|---|
| rectified high pass (original) | .371 | .185 | .350 / .377 |
| both edge sides (`.abs()`) | .184 | .077 | .537 / .535 |
| multi-scale local std | .179 | .116 | **.557** |
| **guided filter** | **.009** | **.027** | .515 / .485 |
| guided + multi-scale | .026 | .057 | .458 |

Two things worth stating plainly. The original detector was the **worst** of
everything tested -- rectifying to one side of the edge threw away half the
signal. And combining guided with multi-scale is *worse* than guided alone, so
the clever hybrid is not the answer.

The advantage shrinks as `grow` grows, because dilation floods the halo zone
regardless of detector:

| grow | halo: plain -> guided | white: plain -> guided |
|---|---|---|
| 0.15 (4px) | .592 -> .287 | 20.3% -> **41.8%** |
| 0.25 (8px) | .760 -> .456 | 50.7% -> 55.2% |
| 0.5 (15px) | .927 -> .739 | 62.8% -> 65.3% |
| 1.0 (30px) | .999 -> .986 | 72.9% -> 72.7% |

At every value the guided detector produces *more* white, not less.

Reference: [He et al., Guided Image
Filtering](https://www.sensetime.com/xo/profile/upload/2024/05/23/2012%20Guided%20Image%20Filtering_20240523184437A013.pdf)

## Radius: smaller is strictly better for protection

`grain_filter = 0`, at 1024px. Monotone on both fixtures:

| radius | px | spraycrete leak | degenerator leak | texture kept | degenerator protected |
|---|---|---|---|---|---|
| 0.25× | 5 | **.036** | **.003** | .412 / .417 | **99.3%** |
| 0.35× | 7 | .039 | .006 | .415 / .415 | 98.8% |
| 0.5× | 10 | .045 | .015 | .410 / .402 | 97.4% |
| 0.7× | 14 | .056 | .029 | .397 / .379 | 95.4% |
| **1.0× (auto)** | **20** | **.073** | **.050** | .377 / .350 | **92.6%** |
| 1.4× | 28 | .092 | .075 | .355 / .321 | 89.5% |
| 2.0× | 40 | .114 | .105 | .325 / .294 | 85.7% |

**The automatic radius is about 3× too coarse for this job.** Protection plateaus
below 0.3× while texture retention peaks around 0.25–0.35×, so that band is the sweet
spot — going finer buys nothing and starts costing texture.

In absolute terms: **5–7 px at 1024, 10–14 px at 2048.**

Note how much more the second fixture gains. The more flat area an image has, the more
the coarse default costs — which is exactly backwards, since large flat regions are
the whole reason to use this node.

## Defaults vs tuned

Tuned = radius 0.35×, `grain_filter` 0, `feather` 1.0:

| fixture | defaults | tuned | improvement |
|---|---|---|---|
| spraycrete | .115 | **.048** | 2.4× less leak |
| degenerator | .064 | **.011** | **6× less leak** |

Texture retention goes *up* at the same time (.298 → .405 and .305 → .404), so this is
not a trade-off — the defaults are simply mistuned for upscale protection.

## Grain suppression: off, unless the source is actually grainy

`grain_filter` defaults to 1.0 and is scaled by the same `base` as the radius, so it
smooths the image *before* anything is measured. At the auto radius on spraycrete:

| | flat leak | texture kept | protected |
|---|---|---|---|
| `grain_filter = 1.0` | .115 | .298 | 81.6% |
| `grain_filter = 0` | .073 | .377 | 86.5% |

60% more leak **and** less real texture. It earns its keep on grainy photographs and
JPEG sources; on ComfyUI generations and clean upscales, set it to 0.

## Feather: cheap, and worth it

A hard mask edge shows as a seam after an upscale pass. At 0.35× radius on spraycrete:

| `feather` | flat leak |
|---|---|
| 0.0 | .039 |
| 0.5 | .041 |
| 1.0 | .048 |
| 2.0 | .058 |

0.5–1.0 is a good trade. 2.0 starts eating the protection you tuned for.

## `strength` is alive but very uneven

Contrary to an earlier reading taken from a synthetic fixture, the slider works across
its whole range on real images — coverage climbs .078 → .308 from 0.4 to 2.0. But:

| strength | 0.4 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| texture | .129 | .261 | .297 | .326 | .341 | .348 | .354 | .359 |
| coverage | .078 | .181 | .219 | .255 | .276 | .287 | .297 | .308 |

0.4 → 1.0 moves texture by .168; 1.4 → 2.0 by .018. **Nearly all the useful control
sits below 1.2**, which is why the top of the range feels dead in use.

## Resolution stability

`base = min(W, H) / 52` keeps the mask consistent as you step up through an upscale
chain — the same image at 768px and 1536px agrees with the 1024px result to IoU
0.87–0.98. Below ~768px the `max(4, ...)` floor takes over instead of the law.

This is why the node scales its radius with resolution while the rest of the ecosystem
uses a fixed ~10px (WAS `radius` 10, KJNodes `radius` 1.0, Masquerade `radius` 10,
Photoshop frequency separation 3–15). For sharpening a fixed radius is right; for a
mask that must select the same regions at every step of an upscale chain, it is not.

Sources: [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui),
[Masquerade](https://github.com/BadCafeCode/masquerade-nodes-comfyui),
[KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
