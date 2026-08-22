"""Render docs/detector-grid.jpg -- guided filter vs the original high pass.

Top row is the whole mask, bottom row zooms into the window band, which is the
case that matters: flat glass sitting inside a strong frame. A Gaussian
low-pass smears across that frame and the difference lights up the glass, so
an upscaler is free to invent detail on it. The guided filter does not.

Run:  python_embeded/python.exe docs/make_detector_grid.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from helpers import load_node, photo, variance_quartiles, halo_zone, build_mask

FIXTURE = "degenerator"
CROP = (300, 330, 760, 900)     # the window band in the cliff face
TILE_W = 420
GAP, PAD = 10, 20


def _f(sz, b=False):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/" + ("arialbd.ttf" if b else "arial.ttf"), sz)
    except OSError:
        return ImageFont.load_default()


def main():
    _, node = load_node()
    img = photo(name=FIXTURE)
    flat, tex, (H, W) = variance_quartiles(img)
    halo, _clean = halo_zone(img, flat, (H, W))
    src = (img[0].numpy() * 255).astype(np.uint8)

    cols = [("source image", None, (235, 235, 235))]
    for det, col in (("high pass", (240, 150, 120)), ("guided", (120, 230, 140))):
        cols.append((det, det, col))

    tiles, zooms, notes = [], [], []
    for label, det, _col in cols:
        if det is None:
            im = Image.fromarray(src)
            notes.append("the flat glass and the sky are what must stay protected")
        else:
            m = build_mask(node, img, grow=0.0, feather=0.0, grain=0.0, detector=det)
            mm = m[0][:H, :W]
            im = Image.fromarray((mm.clamp(0, 1).numpy() * 255).astype(np.uint8)).convert("RGB")
            notes.append(f"halo leak {float(mm[halo].mean()):.3f}    "
                         f"texture kept {float(mm[tex].mean()):.3f}")
        tiles.append(im)
        zooms.append(im.crop(CROP))

    h, w = img.shape[1], img.shape[2]
    th = round(TILE_W * h / w)
    zw, zh = CROP[2] - CROP[0], CROP[3] - CROP[1]
    zth = round(TILE_W * zh / zw)

    gw = PAD * 2 + 3 * TILE_W + 2 * GAP
    gh = PAD + 34 + 26 + th + 50 + 32 + zth + 26 + PAD
    canvas = Image.new("RGB", (gw, gh), (20, 20, 22))
    d = ImageDraw.Draw(canvas)

    d.text((PAD, PAD), "Detector — guided filter vs the original high pass",
           (245, 245, 245), _f(25, True))
    d.text((PAD, PAD + 32), "White = would be sampled.  grow 0, feather 0, grain_filter 0, "
                            "so only the detector differs.", (150, 150, 156), _f(13))

    y = PAD + 60
    for i, ((label, _det, col), t) in enumerate(zip(cols, tiles)):
        x = PAD + i * (TILE_W + GAP)
        canvas.paste(t.resize((TILE_W, th), Image.LANCZOS), (x, y))
        d.text((x, y + th + 7), label, col, _f(16, True))
        d.text((x, y + th + 27), notes[i], (150, 150, 156), _f(13))

    y += th + 54
    d.text((PAD, y), "zoom — the windows in the cliff face", (235, 235, 235), _f(17, True))
    y += 26
    for i, z in enumerate(zooms):
        canvas.paste(z.resize((TILE_W, zth), Image.LANCZOS),
                     (PAD + i * (TILE_W + GAP), y))

    out = os.path.join(HERE, "detector-grid.jpg")
    canvas.save(out, quality=92, optimize=True)
    print(f"wrote {out}  {canvas.size}  {os.path.getsize(out)//1024}kB")


if __name__ == "__main__":
    main()
