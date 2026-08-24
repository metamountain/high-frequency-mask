"""Regression targets.

Each XFAIL test documents a bug reproducible in the current __init__.py. They
are expected to fail today; when a fix lands, remove the name from XFAIL and
the suite enforces it. See CLAUDE.md for the analysis.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from helpers import load_node, photo, protection, build_mask, IMAGES

XFAIL = set()


def test_happy_path():
    """Flat sky, snow and panels stay dark; real texture opens up."""
    _, node = load_node()
    for name in IMAGES:
        leak, texture, _ = protection(node, name=name)
        assert leak < 0.15, f"{name}: flat areas should stay protected, got {leak:.3f}"
        assert texture > 0.25, f"{name}: texture should open up, got {texture:.3f}"
        assert texture - leak > 0.15, f"{name}: separation {texture - leak:.3f}"


def test_invert_swaps_the_regions():
    _, node = load_node()
    leak, texture, _ = protection(node)
    ileak, itexture, _ = protection(node, invert=True)
    assert itexture < ileak, "inverted mask should favour the flat regions"
    assert abs((texture + itexture) - 1.0) < 0.05, "invert should be 1 - mask"


def test_quantile_ceiling():
    """torch.quantile caps at 2**24 elements -- images >=~4096x4096 raise.

    Reachable on a normal upscale: 4096x4096 is 16.8M pixels.
    """
    _, node = load_node()
    build_mask(node, torch.rand(1, 4100, 4100, 3))


def test_reflect_pad_overflow():
    """_blur's reflect padding requires 1.5 * base * feather < min(W, H)."""
    _, node = load_node()
    build_mask(node, photo(512), feather=4.0, radius_override=120)


def test_latent_batch_mismatch():
    """noise_mask must match the latent batch when 1 < mask_batch < latent_batch."""
    _, node = load_node()
    img = photo(256).repeat(3, 1, 1, 1)
    latent = {"samples": torch.zeros(4, 4, 32, 32)}
    out = node.build(img, 1.0, 1.0, 1.0, 1.0, False, samples=latent)["result"][2]
    assert out["noise_mask"].shape[0] == out["samples"].shape[0], (
        f"noise_mask {tuple(out['noise_mask'].shape)} vs "
        f"latent {tuple(out['samples'].shape)}")


def test_sensitivity_is_compressed_at_the_top():
    """strength keeps working across its whole range, but very unevenly.

    Measured on this render: 0.4 -> 1.0 moves the texture level by 0.168,
    while 1.4 -> 2.0 moves it by 0.018. Nearly all the useful travel sits in
    the lower half of the slider. Not a bug, but the reason the top of the
    range feels dead in use.
    """
    _, node = load_node()
    lo = protection(node, sensitivity=1.0)[1] - protection(node, sensitivity=0.4)[1]
    hi = protection(node, sensitivity=2.0)[1] - protection(node, sensitivity=1.4)[1]
    assert hi > 0.0, "the top of the range should still do something"
    assert lo > hi * 4, f"expected heavy compression: low {lo:.3f} vs high {hi:.3f}"
