"""Generate the deterministic test chart used by the radius tests.

A photo makes a poor fixture: you never know exactly where the detail is, so any
score is an estimate. This chart has known ground truth -- each patch carries
texture at one known scale on a flat background, so "did the mask find the 8px
texture and ignore the flat field" is an exact question.

Run:  python tests/assets/make_testchart.py
"""
import numpy as np
from PIL import Image
import os

SIZE = 1024
SCALES = [4, 8, 16, 32, 64, 128]      # texture wavelength in px, one per patch
GRID = 3                               # 3x2 patches
PATCH = SIZE // 4


def _texture(n, scale, rng):
    """Band-limited noise at roughly one spatial scale."""
    small = rng.random((max(2, n // scale), max(2, n // scale)))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
    a = np.asarray(img).astype(np.float32) / 255.0
    return (a - a.mean()) / max(a.std(), 1e-6)


def build(seed=7):
    rng = np.random.default_rng(seed)

    # flat background: smooth vertical gradient, no high frequency at all
    y = np.linspace(0.75, 0.35, SIZE, dtype=np.float32)[:, None]
    canvas = np.repeat(y, SIZE, axis=1)

    boxes = []
    for i, scale in enumerate(SCALES):
        r, c = divmod(i, GRID)
        y0 = PATCH // 2 + r * (PATCH + PATCH // 2)
        x0 = PATCH // 2 + c * (PATCH + PATCH // 4)
        canvas[y0:y0 + PATCH, x0:x0 + PATCH] += _texture(PATCH, scale, rng) * 0.06
        boxes.append({"scale": scale, "y0": y0, "x0": x0, "size": PATCH})

    canvas = np.clip(canvas, 0.0, 1.0)
    rgb = np.repeat(canvas[:, :, None], 3, axis=2)
    return (rgb * 255).astype(np.uint8), boxes


if __name__ == "__main__":
    arr, boxes = build()
    here = os.path.dirname(os.path.abspath(__file__))
    Image.fromarray(arr).save(os.path.join(here, "testchart.png"), optimize=True)
    print(f"wrote testchart.png  {arr.shape[1]}x{arr.shape[0]}")
    for b in boxes:
        print(f"  scale {b['scale']:>3}px  at ({b['x0']},{b['y0']}) {b['size']}px")
