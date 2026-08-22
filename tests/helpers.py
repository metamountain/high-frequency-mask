"""Shared test plumbing: load the node without a running ComfyUI server."""
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHART = os.path.join(HERE, "assets", "testchart.png")

# The node imports folder_paths, which only resolves from the ComfyUI root.
_comfy = os.path.abspath(os.path.join(ROOT, "..", ".."))
if _comfy not in sys.path:
    sys.path.insert(0, _comfy)


def load_node():
    spec = importlib.util.spec_from_file_location("hfm", os.path.join(ROOT, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.HighFrequencyMask()


def chart(min_edge=1024):
    """The synthetic chart as a ComfyUI IMAGE tensor (1, H, W, 3)."""
    im = Image.open(CHART).convert("RGB")
    if min(im.size) != min_edge:
        s = min_edge / min(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).unsqueeze(0)


def patches(min_edge=1024):
    """Ground-truth patch boxes, scaled to the requested resolution."""
    sys.path.insert(0, os.path.join(HERE, "assets"))
    from make_testchart import build, SIZE
    _, boxes = build()
    k = min_edge / SIZE
    return [{"scale": b["scale"],
             "y0": round(b["y0"] * k), "x0": round(b["x0"] * k),
             "size": round(b["size"] * k)} for b in boxes]


def patch_mean(mask, box):
    return float(mask[0, box["y0"]:box["y0"] + box["size"],
                         box["x0"]:box["x0"] + box["size"]].mean())


def background_mean(mask, boxes):
    """Mask level outside every textured patch -- should be near zero."""
    keep = torch.ones_like(mask[0], dtype=torch.bool)
    for b in boxes:
        keep[b["y0"]:b["y0"] + b["size"], b["x0"]:b["x0"] + b["size"]] = False
    return float(mask[0][keep].mean())


def build_mask(node, img, sensitivity=1.0, grow=0.0, feather=0.0,
               grain=1.0, invert=False, **kw):
    return node.build(img, sensitivity, grow, feather, grain, invert, **kw)["result"][0]
