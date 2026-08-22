# Settings for upscaling: what the measurements say

Reproduce from the repo:

```
python_embeded/python.exe tests/test_radius.py
```

Fixture is `tests/assets/spraycrete.png` — a real render of sprayed concrete. The
rough aggregate is genuine texture; the panels and the overcast sky are genuinely
flat. That is exactly the situation the node exists for: an upscaler will happily
invent detail in those panels and shift their colour if nothing stops it.

Flat and textured regions are separated by local standard deviation per 8px block
(bottom and top quartile). **The number that matters is the flat-area leak** — every
bit the mask lets through there is an upscaler free to hallucinate.

## Radius: smaller is strictly better for protection

`grain_suppress = 0`, at 1024px:

| radius | px | flat leak | texture kept | flat fully protected |
|---|---|---|---|---|
| 0.15× | 3 | .036 | .401 | 90.6% |
| 0.25× | 5 | **.035** | .414 | **90.7%** |
| 0.35× | 7 | .038 | **.417** | 90.4% |
| 0.5× | 10 | .044 | .411 | 89.6% |
| 0.7× | 14 | .056 | .398 | 88.3% |
| **1.0× (auto)** | **20** | **.073** | **.378** | **86.4%** |
| 1.4× | 28 | .093 | .355 | 84.2% |
| 2.0× | 39 | .115 | .325 | 81.7% |

Monotone. **The automatic radius is about 3× too coarse for this job.** Protection
plateaus below 0.3× while texture retention peaks around 0.25–0.35×, so that band is
the sweet spot and going finer buys nothing.

In absolute terms: **5–7 px at 1024, 10–14 px at 2048.**

## Grain suppression: off, unless the source is actually grainy

`entrauschen` defaults to 1.0 and is scaled by the same `base` as the radius, so it
smooths the image *before* anything is measured. At the auto radius:

| | flat leak | texture kept | flat fully protected |
|---|---|---|---|
| `entrauschen = 1.0` | .116 | .297 | 81.5% |
| `entrauschen = 0` | .073 | .378 | 86.4% |

It costs 60% more leak **and** removes real texture. It earns its keep on grainy
photographs and JPEG sources; on renders, clean upscales and Flux output, set it to 0.

## Feather: cheap, and worth it

A hard mask edge shows up as a seam after an upscale pass. Softening costs a little
protection — at 0.35× radius:

| `weichheit` | flat leak |
|---|---|
| 0.0 | .038 |
| 0.5 | .041 |
| 1.0 | .047 |
| 2.0 | .058 |

0.5–1.0 is a good trade. 2.0 starts eating into the protection you tuned for.

## Combined

| | flat leak | texture kept | flat fully protected |
|---|---|---|---|
| shipped defaults | .116 | .297 | 81.5% |
| **radius 0.35×, grain 0, feather 1.0** | **.047** | **.408** | **~90%** |

## `staerke` is alive but very uneven

Contrary to an earlier reading taken from a synthetic fixture, the slider works across
its whole range on real images — coverage climbs .078 → .308 from 0.4 to 2.0. But the
travel is badly distributed:

| staerke | 0.4 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| texture | .129 | .261 | .297 | .326 | .341 | .348 | .354 | .359 |
| coverage | .078 | .181 | .219 | .255 | .276 | .287 | .297 | .308 |

0.4 → 1.0 moves texture by .168; 1.4 → 2.0 moves it by .018. **Nearly all the useful
control sits below 1.2**, which is why the top of the range feels dead in use.

## Resolution stability

`base = min(W, H) / 52` keeps the mask consistent as you step up through an upscale
chain — the same image at 768px and 1536px agrees with the 1024px result to IoU
0.87–0.98. Below ~768px the `max(4, ...)` floor takes over instead of the law.

This is why the node scales its radius with resolution while the rest of the ecosystem
uses a fixed ~10px (WAS `radius` 10, KJNodes `radius` 1.0, Masquerade `radius` 10,
Photoshop frequency separation 3–15). For sharpening a fixed radius is right; for a
mask that has to select the same regions at every step of an upscale, it is not.

Sources: [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui),
[Masquerade](https://github.com/BadCafeCode/masquerade-nodes-comfyui),
[KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
