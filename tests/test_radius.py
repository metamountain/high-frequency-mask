"""Radius characterisation. Regenerates the numbers in docs/radius-study.md.

Slow-ish (a few seconds). Run directly for the full tables:
    python tests/test_radius.py
"""
import numpy as np
import torch
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import load_node, chart, patches, patch_mean, background_mean, build_mask

MULTS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]


def _mask(node, img, mult, grain):
    h, w = img.shape[1], img.shape[2]
    base = max(4, round(min(w, h) / 52.0))
    ro = max(1, int(round(base * mult))) if mult != 1.0 else 0
    return build_mask(node, img, grain=grain, radius_override=ro)


def sweep(grain=0.0, min_edge=1024):
    """separation per (texture scale, radius multiplier)."""
    _, node = load_node()
    img, boxes = chart(min_edge), patches(min_edge)
    rows = {}
    for mu in MULTS:
        m = _mask(node, img, mu, grain)
        bg = background_mean(m, boxes)
        for b in boxes:
            rows.setdefault(b["scale"], {})[mu] = patch_mean(m, b) - bg
    return rows


def test_radius_optimum_tracks_texture_scale():
    """Finer texture must peak at a smaller radius. This is why it needs a slider."""
    rows = sweep(grain=0.0)
    peaks = {s: max(v, key=v.get) for s, v in rows.items()}
    fine = peaks[min(peaks)]
    coarse = peaks[max(peaks)]
    assert fine < coarse, f"expected monotone peak shift, got {peaks}"


def test_whole_slider_range_is_live_without_grain_coupling():
    """With grain decoupled, no multiplier in 0.25-4.0 is uniformly useless."""
    rows = sweep(grain=0.0)
    for mu in MULTS:
        best = max(rows[s][mu] for s in rows)
        assert best > 0.15, f"multiplier {mu} is dead everywhere (best {best:.3f})"


def test_grain_coupling_costs_the_top_of_the_range():
    """base drives grain suppression too, which over-smooths at high radius."""
    off, on = sweep(grain=0.0), sweep(grain=1.0)
    fine = min(off)
    assert off[fine][4.0] > on[fine][4.0], "expected coupling to hurt at 4.0"


def test_auto_radius_is_resolution_stable():
    """base = min(W,H)/52 should give the same mask at 768 and 1536."""
    _, node = load_node()
    out = {}
    for edge in (768, 1024, 1536):
        m = build_mask(node, chart(edge), grain=0.0)
        out[edge] = F.interpolate(m[None], size=(256, 256), mode="area")[0, 0]
    for edge in (768, 1536):
        inter = float(torch.min(out[edge], out[1024]).sum())
        union = float(torch.max(out[edge], out[1024]).sum())
        iou = inter / max(union, 1e-6)
        assert iou > 0.80, f"{edge}px drifts from 1024px reference: IoU {iou:.3f}"


if __name__ == "__main__":
    for grain, label in ((0.0, "grain_suppress = 0 (radius isolated)"),
                         (1.0, "grain_suppress = 1 (as shipped)")):
        rows = sweep(grain=grain)
        print("=" * 78)
        print(label)
        print("=" * 78)
        print(f"  {'texture':9s}" + "".join(f"{m:>7}" for m in MULTS) + "   peak")
        for s in sorted(rows):
            v = rows[s]
            print(f"  {str(s) + 'px':9s}" + "".join(f"{v[m]:7.3f}" for m in MULTS)
                  + f"   {max(v, key=v.get)}")
        print()
