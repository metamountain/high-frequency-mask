"""Shared test plumbing: load the node without a running ComfyUI server."""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(HERE, "assets")

# ComfyUI generations -- the kind of image this node actually runs on, not
# photographs. Both have large genuinely flat regions (sky, snow, smooth
# concrete panels) next to genuine texture, which is the situation an upscaler
# gets wrong: it invents detail in the flat parts and shifts their colour.
IMAGES = {
    "spraycrete": "sprayed concrete facade -- rough aggregate vs smooth panels and overcast sky",
    "degenerator": "rock-and-concrete tower -- cliff texture vs flat sky, snow field and panels",
}

# The node imports folder_paths, which only resolves from the ComfyUI root.
_comfy = os.path.abspath(os.path.join(ROOT, "..", ".."))
if _comfy not in sys.path:
    sys.path.insert(0, _comfy)


def load_node():
    spec = importlib.util.spec_from_file_location("hfm", os.path.join(ROOT, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.HighFrequencyMask()


def photo(min_edge=1024, name="spraycrete"):
    """One of the ComfyUI test generations as an IMAGE tensor (1, H, W, 3)."""
    im = Image.open(os.path.join(ASSETS, name + ".jpg")).convert("RGB")
    if min(im.size) != min_edge:
        s = min_edge / min(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).unsqueeze(0)


def variance_quartiles(img, block=8):
    """Split the image into genuinely flat and genuinely textured regions.

    Local standard deviation per block; returns the bottom and top quartile.
    Flat regions are what the mask has to protect.
    """
    g = img.mean(-1)[0]
    h, w = g.shape
    h2, w2 = h // block * block, w // block * block
    b = (g[:h2, :w2]
         .reshape(h2 // block, block, w2 // block, block)
         .permute(0, 2, 1, 3)
         .reshape(h2 // block, w2 // block, -1))
    sd = b.std(-1)
    q1 = torch.quantile(sd.flatten(), 0.25)
    q3 = torch.quantile(sd.flatten(), 0.75)
    up = lambda t: F.interpolate(t.float()[None, None], size=(h2, w2),
                                 mode="nearest")[0, 0].bool()
    return up(sd <= q1), up(sd >= q3), (h2, w2)


def build_mask(node, img, sensitivity=1.0, grow=0.0, feather=0.0,
               grain=1.0, invert=False, **kw):
    return node.build(img, sensitivity, grow, feather, grain, invert, **kw)["result"][0]


def protection(node, img=None, block=8, name="spraycrete", **kw):
    """The number that matters for upscaling.

    Returns (flat_leak, texture_level, fully_protected_fraction):
    how much the mask lets through in flat areas, how much it opens up over
    real texture, and what share of flat pixels are fully black.
    """
    if img is None:
        img = photo(name=name)
    flat, textured, (h, w) = variance_quartiles(img, block)
    m = build_mask(node, img, **kw)[0][:h, :w]
    return (float(m[flat].mean()),
            float(m[textured].mean()),
            float((m[flat] < 0.05).float().mean()))


def radius_px(mult, min_edge=1024):
    """The radius_override value corresponding to a multiple of the auto radius."""
    return max(1, round(max(4, round(min_edge / 52.0)) * mult))
