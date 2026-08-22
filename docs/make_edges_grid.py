"""Render docs/edges-grid.jpg -- grow x feather, at strength 1.0.

grow dilates the speckled detector output into continuous regions; feather
softens the boundary afterwards. They interact: without grow, feather crushes
the mask's peaks and it never reaches white at all.

Run:  python_embeded/python.exe docs/make_edges_grid.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from helpers import load_node, photo, build_mask

FIXTURE = "degenerator"
GROWS = [0.0, 0.5, 1.0]         # rows
FEATHERS = [0.0, 1.0, 2.0]      # cols
TILE_W = 380
GAP, PAD, LABEL_W = 10, 18, 132


def _f(sz, b=False):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/" + ("arialbd.ttf" if b else "arial.ttf"), sz)
    except OSError:
        return ImageFont.load_default()


def stats(m):
    return (float((m >= 0.98).float().mean()) * 100,
            float((m <= 0.02).float().mean()) * 100,
            float(m.max()), float(m.mean()))


def main():
    _, node = load_node()
    img = photo(name=FIXTURE)
    src = (img[0].numpy() * 255).astype(np.uint8)
    base = max(4, round(min(img.shape[1], img.shape[2]) / 52.0))  # the auto radius
    h, w = img.shape[1], img.shape[2]
    th = round(TILE_W * h / w)

    gw = PAD * 2 + LABEL_W + 3 * TILE_W + 2 * GAP
    gh = PAD + 32 + 24 + th + 50 + 46 + 3 * (th + 48) + 2 * GAP + PAD
    canvas = Image.new("RGB", (gw, gh), (20, 20, 22))
    d = ImageDraw.Draw(canvas)

    d.text((PAD, PAD), "grow x feather at strength 1.0", (245, 245, 245), _f(24, True))
    d.text((PAD, PAD + 30), "grow dilates the detector's speckle into continuous regions.  "
                            "feather softens the boundary afterwards.", (150, 150, 156), _f(13))

    y = PAD + 58
    canvas.paste(Image.fromarray(src).resize((TILE_W, th), Image.LANCZOS), (PAD + LABEL_W, y))
    d.text((PAD + LABEL_W, y + th + 7), "source image", (235, 235, 235), _f(15))

    m = build_mask(node, img, grow=1.0, feather=1.0, grain=1.0)
    a = (m[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    x = PAD + LABEL_W + TILE_W + GAP
    canvas.paste(Image.fromarray(a).convert("RGB").resize((TILE_W, th), Image.LANCZOS), (x, y))
    wt, bk, mx, mn = stats(m[0])
    d.text((x, y + th + 7), "node defaults (grow 1.0, feather 1.0)", (120, 230, 140), _f(15))
    d.text((x, y + th + 27), f"white {wt:.0f}%   black {bk:.0f}%   max {mx:.2f}",
           (150, 150, 156), _f(13))

    y += th + 50
    d.line((PAD, y, gw - PAD, y), fill=(58, 58, 64), width=1)
    y += 16
    for c, fe in enumerate(FEATHERS):
        x = PAD + LABEL_W + c * (TILE_W + GAP)
        d.text((x, y), f"feather {fe}", (235, 235, 235), _f(17, True))
        d.text((x, y + 21), f"blur {base * 0.5 * fe:.0f} px", (150, 150, 156), _f(13))
    d.text((PAD, y), "grow", (150, 150, 156), _f(17, True))
    y += 46

    for gr in GROWS:
        d.text((PAD, y + th // 2 - 22), f"{gr}", (235, 235, 235), _f(17, True))
        d.text((PAD, y + th // 2), f"dilate {int(round(base * 1.5 * gr))} px",
               (150, 150, 156), _f(13))
        for c, fe in enumerate(FEATHERS):
            mm = build_mask(node, img, grow=gr, feather=fe, grain=1.0)
            arr = (mm[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
            x = PAD + LABEL_W + c * (TILE_W + GAP)
            canvas.paste(Image.fromarray(arr).convert("RGB").resize((TILE_W, th),
                                                                    Image.LANCZOS), (x, y))
            wt, bk, mx, mn = stats(mm[0])
            d.text((x, y + th + 7), f"white {wt:.0f}%   black {bk:.0f}%",
                   (200, 200, 206), _f(15))
            col = (240, 150, 120) if mx < 0.95 else (150, 150, 156)
            d.text((x, y + th + 26), f"max {mx:.2f}   mean {mn:.2f}", col, _f(13))
        y += th + 48 + GAP

    out = os.path.join(HERE, "edges-grid.jpg")
    canvas.save(out, quality=90, optimize=True)
    print(f"wrote {out}  {canvas.size}  {os.path.getsize(out)//1024}kB")


if __name__ == "__main__":
    main()
