"""Validation on a real render.

The generated chart proves the node behaves correctly on known input. This file
checks it still behaves on something photographic -- sprayed concrete, where the
rough aggregate is genuine texture and the panels and overcast sky are genuinely
flat, with soft gradients and render noise that the chart does not have.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import load_node, photo, variance_quartiles, build_mask

XFAIL = set()


def _separation(node, **kw):
    img = photo()
    flat, textured, (h, w) = variance_quartiles(img)
    mask = build_mask(node, img, **kw)[0][:h, :w]
    return float(mask[textured].mean()), float(mask[flat].mean())


def test_texture_separates_from_flat():
    """Thresholds are set from measurement: 0.297 / 0.116 at defaults."""
    _, node = load_node()
    hot, cold = _separation(node)
    assert cold < 0.15, f"flat panels and sky should stay dark, got {cold:.3f}"
    assert hot > 0.25, f"concrete aggregate should light up, got {hot:.3f}"
    assert hot - cold > 0.15, f"separation too weak: {hot - cold:.3f}"


def test_grain_suppression_costs_separation_on_a_clean_render():
    """entrauschen defaults to 1.0, which is wrong for noise-free sources.

    On this render it smooths away real aggregate texture before the analysis
    ever sees it: separation 0.181 with it, 0.305 without -- and the flat
    regions leak *more*, not less (0.116 vs 0.073).
    """
    _, node = load_node()
    on_hot, on_cold = _separation(node)
    off_hot, off_cold = _separation(node, grain=0.0)
    assert (off_hot - off_cold) > (on_hot - on_cold) + 0.10, (
        f"expected grain=0 to win clearly: "
        f"{off_hot - off_cold:.3f} vs {on_hot - on_cold:.3f}")
    assert off_cold < on_cold, "grain suppression should not increase flat-region leak"


def test_invert_swaps_the_regions():
    _, node = load_node()
    hot, cold = _separation(node)
    ihot, icold = _separation(node, invert=True)
    assert ihot < icold, "inverted mask should favour the flat regions"
    assert abs((hot + ihot) - 1.0) < 0.05, "invert should be 1 - mask"


def test_finer_radius_helps_on_fine_aggregate():
    """The concrete grain is fine, so a sub-1.0 radius should beat the default."""
    _, node = load_node()
    def sep(mult):
        base = max(4, round(1024 / 52.0))
        ro = max(1, round(base * mult)) if mult != 1.0 else 0
        hot, cold = _separation(node, grain=0.0, radius_override=ro)
        return hot - cold
    assert sep(0.5) > sep(1.0), f"0.5x {sep(0.5):.3f} vs 1.0x {sep(1.0):.3f}"


if __name__ == "__main__":
    _, node = load_node()
    hot, cold = _separation(node)
    print(f"textured {hot:.3f}   flat {cold:.3f}   separation {hot - cold:.3f}")
