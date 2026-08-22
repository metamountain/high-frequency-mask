"""Regression targets.

Each XFAIL test documents a bug that is reproducible in the current __init__.py.
They are expected to fail today; when a fix lands, flip XFAIL to False and the
suite enforces it. See CLAUDE.md for the analysis behind each one.
"""
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import load_node, chart, build_mask

XFAIL = {"quantile_ceiling", "reflect_pad_overflow", "latent_batch_mismatch",
         "sensitivity_saturates"}


def test_happy_path():
    """The chart's textured patches score higher than its flat background."""
    from helpers import patches, patch_mean, background_mean
    _, node = load_node()
    mask = build_mask(node, chart())
    boxes = patches()
    bg = background_mean(mask, boxes)
    best = max(patch_mean(mask, b) for b in boxes)
    assert bg < 0.10, f"flat background should stay dark, got {bg:.3f}"
    assert best > 0.30, f"textured patch should light up, got {best:.3f}"
    assert best - bg > 0.25, f"separation too weak: {best - bg:.3f}"


def test_quantile_ceiling():
    """torch.quantile caps at 2**24 elements -- images >=~4096x4096 raise."""
    _, node = load_node()
    big = torch.rand(1, 4100, 4100, 3)
    build_mask(node, big)          # RuntimeError: input tensor is too large


def test_reflect_pad_overflow():
    """_blur's reflect padding requires 1.5 * base * feather < min(W, H)."""
    _, node = load_node()
    build_mask(node, chart(512), feather=4.0, radius_override=120)


def test_latent_batch_mismatch():
    """noise_mask must match the latent batch when 1 < mask_batch < latent_batch."""
    _, node = load_node()
    img = chart(256).repeat(3, 1, 1, 1)
    latent = {"samples": torch.zeros(4, 4, 32, 32)}
    out = node.build(img, 1.0, 1.0, 1.0, 1.0, False, samples=latent)["result"][2]
    assert out["noise_mask"].shape[0] == out["samples"].shape[0], (
        f"noise_mask {tuple(out['noise_mask'].shape)} vs "
        f"latent {tuple(out['samples'].shape)}")


def test_sensitivity_saturates():
    """The sensitivity slider stops doing anything above ~1.4."""
    from helpers import patches, patch_mean
    _, node = load_node()
    img, boxes = chart(), patches()
    def score(s):
        m = build_mask(node, img, sensitivity=s)
        return sum(patch_mean(m, b) for b in boxes) / len(boxes)
    assert abs(score(2.0) - score(1.4)) > 0.01, (
        f"1.4 -> {score(1.4):.4f}, 2.0 -> {score(2.0):.4f}: slider is dead up here")
