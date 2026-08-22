"""Render the settings grid in docs/settings-grid.jpg.

Shows the node's actual MASK output -- black and white, as it comes out of the
mask socket -- across the two settings that change what gets selected:
radius_override (which size of structure counts as detail) and strength (how
much counts). Everything else stays at the node's shipped defaults.

No setting here is "the right one". Fine radius selects fine texture, coarse
radius selects broad shapes; low strength selects less, high selects more. Which
you want depends on the image and what you are after.

Run:  python_embeded/python.exe docs/make_grid.py
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
RADII = [7, 20, 40]              # rows: px. 20 is what auto picks at 1024
STRENGTH = [0.6, 1.0, 1.4]        # cols
TILE_W = 380
GAP = 10
PAD = 18
LABEL_W = 118


def _font(size, bold=False):
    names = ("arialbd.ttf",) if bold else ("arial.ttf",)
    for n in names:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/" + n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    _, node = load_node()
    img = photo(name=FIXTURE)
    img_np = (img[0].numpy() * 255).astype(np.uint8)
    h, w = img.shape[1], img.shape[2]
    tile_h = round(TILE_W * h / w)

    gw = PAD + LABEL_W + 3 * TILE_W + 2 * GAP + PAD
    gh = PAD + 34 + 22 + tile_h + 30 + 30 + 3 * (tile_h + 40) + 2 * GAP + PAD

    canvas = Image.new("RGB", (gw, gh), (20, 20, 22))
    d = ImageDraw.Draw(canvas)
    f_t = _font(24, True)
    f_h = _font(17, True)
    f_c = _font(15)
    f_s = _font(13)

    y = PAD
    d.text((PAD, y), "High Frequency Mask — mask output across settings",
           (245, 245, 245), f_t)
    y += 30
    d.text((PAD, y), "White = sampled.  Black = left alone.  All other settings at "
                     "the node's defaults (grow 0.5, feather 1.0, grain_filter 1.0).",
           (150, 150, 156), f_s)
    y += 26

    # source, plus the mask the node produces with no changes at all
    canvas.paste(Image.fromarray(img_np).resize((TILE_W, tile_h), Image.LANCZOS),
                 (PAD + LABEL_W, y))
    d.text((PAD + LABEL_W, y + tile_h + 7), "source image", (235, 235, 235), f_c)

    m = build_mask(node, img, sensitivity=1.0, grow=0.5, feather=1.0, grain=1.0)
    arr = (m[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(arr).convert("RGB").resize((TILE_W, tile_h), Image.LANCZOS),
                 (PAD + LABEL_W + TILE_W + GAP, y))
    d.text((PAD + LABEL_W + TILE_W + GAP, y + tile_h + 7),
           "straight out of the box (radius auto = 20px)", (235, 235, 235), f_c)
    d.text((PAD + LABEL_W + TILE_W + GAP, y + tile_h + 26),
           f"brightest value {float(m.max()):.2f}   mean {float(m.mean()):.2f}",
           (150, 150, 156), f_s)

    y += tile_h + 50
    d.line((PAD, y, gw - PAD, y), fill=(58, 58, 64), width=1)
    y += 14

    for c, st in enumerate(STRENGTH):
        d.text((PAD + LABEL_W + c * (TILE_W + GAP), y), f"strength {st}",
               (235, 235, 235), f_h)
    d.text((PAD, y), "radius", (150, 150, 156), f_h)
    y += 28

    for px in RADII:
        note = {7: "fine structures", 20: "auto at 1024px", 40: "broad shapes"}[px]
        d.text((PAD, y + tile_h // 2 - 14), f"{px} px", (235, 235, 235), f_h)
        d.text((PAD, y + tile_h // 2 + 8), note, (150, 150, 156), f_s)
        for c, st in enumerate(STRENGTH):
            mm = build_mask(node, img, sensitivity=st, grow=0.5, feather=1.0,
                            grain=1.0, radius_override=px)
            a = (mm[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
            x = PAD + LABEL_W + c * (TILE_W + GAP)
            canvas.paste(Image.fromarray(a).convert("RGB").resize((TILE_W, tile_h),
                                                                  Image.LANCZOS), (x, y))
            d.text((x, y + tile_h + 7),
                   f"max {float(mm.max()):.2f}   mean {float(mm.mean()):.2f}",
                   (170, 170, 176), f_c)
        y += tile_h + 40 + GAP

    out = os.path.join(HERE, "settings-grid.jpg")
    canvas.save(out, quality=90, optimize=True)
    print(f"wrote {out}  {canvas.size}  {os.path.getsize(out)//1024}kB")


if __name__ == "__main__":
    main()
