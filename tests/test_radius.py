"""What the settings do, measured on the real render.

The job is protecting flat areas during an upscale pass. So the number that
matters is not "does the mask find texture" but "how much does it let through
where there is nothing to refine" -- every bit of leak there is an upscaler
free to invent detail and shift colour.

Run directly for the tables:  python tests/test_radius.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from helpers import load_node, photo, protection, radius_px, build_mask, IMAGES

XFAIL = set()
MULTS = [0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]


def test_smaller_radius_protects_flat_areas_better():
    """Monotone on every fixture: finer high-pass, less leak into flat regions."""
    _, node = load_node()
    for name in IMAGES:
        leaks = [protection(node, name=name, grain=0.0, radius_override=radius_px(m))[0]
                 for m in MULTS]
        for a, b, ma, mb in zip(leaks, leaks[1:], MULTS, MULTS[1:]):
            assert a <= b + 1e-3, f"{name}: {ma}x leaked {a:.3f}, {mb}x leaked {b:.3f}"


def test_defaults_leak_several_times_more_than_necessary():
    """The shipped defaults are tuned too coarse for upscale protection.

    Measured: 3x worse on the concrete facade, 10x worse on the image with a
    large flat sky and snow field -- the more flat area, the more it costs.
    """
    _, node = load_node()
    for name in IMAGES:
        default_leak = protection(node, name=name)[0]
        tuned_leak = protection(node, name=name, grain=0.0,
                                radius_override=radius_px(0.35))[0]
        assert default_leak > tuned_leak * 2.5, (
            f"{name}: default {default_leak:.3f} vs tuned {tuned_leak:.3f}")


def test_protection_plateaus_below_a_third():
    """Going finer than ~0.25x buys nothing, so that is the useful floor."""
    _, node = load_node()
    at_015 = protection(node, grain=0.0, radius_override=radius_px(0.15))
    at_035 = protection(node, grain=0.0, radius_override=radius_px(0.35))
    assert abs(at_015[2] - at_035[2]) < 0.02, "protection should be flat here"
    assert at_035[1] >= at_015[1], "but the coarser end should keep more texture"


def test_grain_suppression_hurts_a_clean_render():
    """grain_filter defaults to 1.0 and costs protection on noise-free sources."""
    _, node = load_node()
    on = protection(node, radius_override=radius_px(1.0))
    off = protection(node, grain=0.0, radius_override=radius_px(1.0))
    assert off[0] < on[0], f"grain=0 should leak less: {off[0]:.3f} vs {on[0]:.3f}"
    assert off[1] > on[1], "and keep more real texture"


def test_feather_costs_protection_but_stays_usable():
    """Soft edges avoid upscale seams; the price is a little extra leak."""
    _, node = load_node()
    hard = protection(node, grain=0.0, feather=0.0, radius_override=radius_px(0.35))
    soft = protection(node, grain=0.0, feather=1.0, radius_override=radius_px(0.35))
    assert soft[0] > hard[0], "feathering must let a little more through"
    assert soft[0] < 0.08, f"but not this much: {soft[0]:.3f}"


def test_auto_radius_is_resolution_stable():
    """base = min(W,H)/52 gives a consistent mask across upscale steps."""
    _, node = load_node()
    out = {}
    for edge in (768, 1024, 1536):
        m = build_mask(node, photo(edge), grain=0.0)
        out[edge] = F.interpolate(m[None], size=(256, 256), mode="area")[0, 0]
    for edge in (768, 1536):
        inter = float(torch.min(out[edge], out[1024]).sum())
        union = float(torch.max(out[edge], out[1024]).sum())
        assert inter / max(union, 1e-6) > 0.80, f"{edge}px drifts from 1024px"


if __name__ == "__main__":
    _, node = load_node()
    for grain in (1.0, 0.0):
        print(f"\ngrain_suppress = {grain}")
        print(f"  {'radius':>7} {'px':>4} | {'flat leak':>10} {'texture':>8} {'protected':>10}")
        for m in MULTS:
            leak, tex, prot = protection(node, grain=grain, radius_override=radius_px(m))
            print(f"  {m:>7} {radius_px(m):>4} | {leak:>10.3f} {tex:>8.3f} {prot*100:>9.1f}%")
