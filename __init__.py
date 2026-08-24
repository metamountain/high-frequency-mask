"""
High Frequency Mask — ComfyUI custom node

White = textured, gets sampled. Black = flat, stays as it is.
Texture is found with an edge-preserving guided filter, so strong edges do not
spill a halo of false detail into the flat areas beside them.
Four sliders: strength, grow, feather, grain_filter. The mask previews in the node.
An optional LATENT input gets the mask attached as its noise mask.

All torch -> runs on the GPU and handles whole batches.
"""

import base64
import io
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

try:                                    # run on the GPU where possible
    import comfy.model_management as _mm
    def _device():
        return _mm.get_torch_device()
except Exception:
    def _device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- ops

def _gauss1d(sigma, device, dtype):
    r = max(1, int(round(sigma * 3)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).view(1, 1, -1)


def _blur(x, sigma):
    """Separable Gaussian, B1HW."""
    if sigma < 0.3:
        return x
    k = _gauss1d(sigma, x.device, x.dtype)
    r = k.shape[-1] // 2
    cap = min(x.shape[-2], x.shape[-1]) - 1     # reflect pad needs pad < dim
    if r > cap:
        if cap < 1:
            return x
        mid = k.shape[-1] // 2
        k = k[:, :, mid - cap: mid + cap + 1]
        k = k / k.sum()
        r = cap
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x


def _box(x, r):
    """Separable box mean. Separable so a large radius stays cheap at 4K."""
    k = 2 * r + 1
    x = F.avg_pool2d(F.pad(x, (r, r, 0, 0), mode="reflect"), (1, k), stride=1)
    return F.avg_pool2d(F.pad(x, (0, 0, r, r), mode="reflect"), (k, 1), stride=1)


def _guided(g, r, eps=0.01):
    """Self-guided edge-preserving low-pass (He et al., Guided Image Filtering).

    A Gaussian low-pass smears straight across a strong edge, so the high-pass
    difference lights up the flat ground on BOTH sides of it -- the halo that
    puts a glow into sky beside a cliff and across the glass inside a window
    frame. A guided filter follows the edge instead, so nothing is left over
    there and genuinely flat surfaces stay protected.
    """
    mean = _box(g, r)
    var = (_box(g * g, r) - mean * mean).clamp(min=0)
    a = var / (var + eps)
    b = mean - a * mean
    return _box(a, r) * g + _box(b, r)


def _dilate(x, px):
    """Maximum filter via max_pool2d, stepped for large kernels."""
    if px < 1:
        return x
    rest = int(px)
    while rest > 0:
        step = min(rest, 12)
        x = F.max_pool2d(x, kernel_size=2 * step + 1, stride=1, padding=step)
        rest -= step
    return x


def _erode(x, px):
    return -_dilate(-x, px)


_QUANTILE_MAX = 2 ** 24 - 1    # torch.quantile's hard ceiling on input elements


def _quantile(flat, q):
    """torch.quantile as used here, safe past its 2**24-element ceiling.

    A strided subsample keeps the estimate representative -- millions of
    points remain even after subsampling a 4K+ image -- while staying under
    the limit that otherwise raises RuntimeError.
    """
    n = flat.numel()
    if n > _QUANTILE_MAX:
        step = -(-n // _QUANTILE_MAX)          # ceil division
        flat = flat[::step]
    return torch.quantile(flat.float(), q)


# ---------------------------------------------------------------- node

class HighFrequencyMask:

    DESCRIPTION = (
        "Protects flat areas during an upscale or refine pass. White where the image "
        "already has texture, black where it is flat -- feed it to the sampler as a "
        "noise mask and skies, walls, panels and skin are left alone.\n\n"
        "HOW TO WIRE IT. Send the mask output into Set Latent Noise Mask, together with the latent you are about to resample, and feed that into the sampler. That is the intended use. The node can also attach the mask itself if you connect a latent to the samples input, but Set Latent Noise Mask is clearer to read in a graph and is what these settings were tuned against.\n\n"
        "WHAT IT FIXES. Upscalers invent detail in surfaces that should stay smooth "
        "and drift the colour while doing it. Flux Klein is especially prone to both. "
        "Masking the flat regions out stops the model touching them at all, so they "
        "keep their original tone and stay clean.\n\n"
        "SETTINGS FOR UPSCALING. The shipped auto radius is tuned too coarse for this "
        "job -- it lets roughly three times more through in flat areas than needed. "
        "Set radius_override to about a third of the automatic radius (5-7 at 1024px, "
        "10-14 at 2048px); leave grain_filter at its default 0 unless the source is "
        "grainy or JPEG. Measured on ComfyUI generations that cuts the flat-area leak by 2x to 6x -- the "
        "more flat area, the bigger the win -- while keeping MORE real texture. Add "
        "feather 0.5-1.0 so the mask edge does "
        "not leave a seam, and grow 0.5-1.0 so real detail keeps a margin.\n\n"
        "Going finer than a third buys nothing -- protection plateaus there while "
        "texture retention starts dropping.\n\n"
        "CAN'T MAKE IT WEAKER? grow re-dilates the mask as fast as strength shrinks "
        "it, so pulling strength down alone often does nothing visible once grow is "
        "above 0. opacity is the fix: it caps the mask after grow and feather have "
        "run, so it reliably weakens the result even when the other sliders fight "
        "each other.\n\n"
        "ONE THING THAT MISLEADS. black_override and white_override are measured on "
        "the HIGH-PASS image, not on brightness: useful values sit near 0-30, not "
        "0-255, and setting either one switches off the automatic levels so strength "
        "stops mattering.\n\n"
        "The info output reports the values actually used, so you can read them off a "
        "good frame and pin them for a whole batch or video."
    )

    SEARCH_ALIASES = [
        # what it is for
        "upscale mask", "upscale detail", "upscale protection", "protect flat areas",
        "colour shift", "color shift", "too much detail", "over detail",
        "detail control", "refine mask", "hires fix mask", "flux klein",
        "flat area mask", "sky mask", "skin mask", "smooth area",
        # what it is
        "detail mask", "texture mask", "high pass", "highpass", "high frequency",
        "high frequency mask", "frequency separation", "structure mask",
        "selective sampling", "noise mask",
        # German
        # German search terms, so the node is findable either way
        "hochfrequenz", "detailmaske", "strukturmaske", "texturmaske",
        "flaechen schuetzen", "farbverschiebung",
    ]

    OUTPUT_TOOLTIPS = (
        "The finished mask. White is sampled, black stays original. Send this into Set Latent Noise Mask together with your latent.",
        "The same mask as an IMAGE, for chaining into Preview, Save or a compositor.",
        "The input latent with the mask attached as its noise mask. Only meaningful "
        "when the samples input is connected.",
        "Text readout of the values actually used: resolution, radius, grow and blur "
        "in px, the black and white points, and the mask level at the top of the frame.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to analyse. Only its texture is used, never its colour."}),
                "strength": ("FLOAT", {"default": 1.00, "min": 0.40, "max": 2.00, "step": 0.02,
                                       "tooltip": "How much of the image counts as detail. Higher = more gets sampled, less stays flat. Has little effect above about 1.4."}),
                "grow": ("FLOAT", {"default": 0.50, "min": -1.00, "max": 4.00, "step": 0.05,
                                   "tooltip": "Expands the white areas so detail keeps a safety margin. Negative values shrink the mask instead."}),
                "feather": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 4.00, "step": 0.05,
                                      "tooltip": "Softness of the edge between white and black. Higher = longer fade and no visible mask border."}),
                "grain_filter": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 4.00, "step": 0.05,
                                           "tooltip": "Smooths grain and JPEG artifacts before analysis, so a noisy sky is still recognised as flat. Default 0 suits clean renders and upscales. Raise it for grainy or JPEG sources -- on a noise-free source, smoothing before analysis removes real texture and weakens the mask."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Swaps black and white: protects the detail and samples the flat areas instead."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "samples": ("LATENT", {"tooltip": "Optional. If connected, the mask is resized to the latent and attached as its noise mask."}),
                "radius_override": ("INT", {"default": 0, "min": 0, "max": 400,
                                            "tooltip": "High-pass radius in px, deciding which size of structure counts as detail. 0 = automatic from the image size (min(W,H)/52). Smaller finds finer texture. Very large values combined with a high feather will error."}),
                "black_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0,
                                             "tooltip": "Below this, an area counts as flat. Measured on the HIGH-PASS image, not on brightness -- typical values are 0 to 10. 0 means automatic. Setting this or white_override disables strength."}),
                "white_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0,
                                             "tooltip": "Above this, an area counts as full detail. Also high-pass contrast -- typical values are 8 to 40. 0 means automatic. Setting black_override alone falls back to 255 here and yields a near-empty mask."}),
                # NOTE: detector is deliberately LAST. ComfyUI matches a saved
                # workflow's widget values to widgets by position, so inserting a
                # new one in the middle shifts every value after it -- that is how
                # radius_override's 0 ended up in detector. New widgets go on the
                # end so older graphs keep lining up.
                "detector": (["guided", "high pass"], {"default": "guided",
                                                       "tooltip": "How texture is found. 'guided' uses an "
                                                       "edge-preserving low-pass, so strong edges do not "
                                                       "spill a halo into the flat areas next to them -- "
                                                       "sky beside a cliff, glass inside a window frame. "
                                                       "It finds more real texture AND protects flat areas "
                                                       "better. 'high pass' is the original plain Gaussian "
                                                       "difference, kept so older results can be reproduced."}),
                # NOTE: also last, for the same reason -- appended after detector
                # so it never shifts an existing widget's position.
                "opacity": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.02,
                                      "tooltip": "Caps how strong the mask can get, applied after everything "
                                      "else. 1.0 = full effect. Lower this when grow/strength alone will not "
                                      "make the mask weak enough -- grow dilates white back in as fast as "
                                      "strength shrinks it, so this is the only control that reliably "
                                      "attenuates the whole mask."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("mask", "mask_preview", "latent", "info")
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = "mask"

    def build(self, image, strength, grow, feather, grain_filter, invert,
              samples=None, detector="guided", radius_override=0,
              black_override=0.0, white_override=0.0, opacity=1.0, unique_id=None):

        if unique_id is not None:
            _remember(unique_id, image)

        src_dev = image.device
        dev = _device()
        x = image.movedim(-1, 1).to(device=dev, dtype=torch.float32)   # B,C,H,W
        g = x.mean(dim=1, keepdim=True)                      # B,1,H,W  greyscale
        B, _, H, W = g.shape

        # Pixel sizes scale with resolution, so 816 and 1632 behave the same
        base = radius_override if radius_override > 0 else max(4, int(round(min(W, H) / 52.0)))

        # Pre-smooth against sensor grain / JPEG before anything is measured
        if grain_filter > 0:
            g = _blur(g, sigma=max(0.3, base / 12.0 * grain_filter))

        if detector == "guided":
            # Edge-preserving low-pass, and both sides of the difference kept.
            # Measured against the plain version: 2-3x less leak into flat areas
            # that sit next to a strong edge, while finding MORE real texture
            # (0.515 vs 0.350) -- rectifying to one side threw half of it away.
            hp = ((g - _guided(g, max(1, int(round(base / 3.0))))).abs() * 4.0).clamp(0, 1)
        else:
            # Original: image minus Gaussian low-pass, negative half dropped
            hp = ((g - _blur(g, sigma=base / 2.0)).clamp(min=0) * 3.0).clamp(0, 1)

        masks = []
        stats = []
        for i in range(B):
            h = hp[i:i + 1]
            flat = h.flatten()
            if black_override > 0 or white_override > 0:
                blk = black_override / 255.0
                wht = max(blk + 0.03, (white_override / 255.0) if white_override > 0 else 1.0)
            else:
                pw = float(np.clip(99.0 - (strength - 0.40) * (99.0 - 62.0) / 1.60, 62.0, 99.0))
                pb = float(np.clip(68.0 + (1.00 - strength) * 8.0, 50.0, 90.0))
                blk = _quantile(flat, pb / 100.0).item()
                wht = max(blk + 0.03, _quantile(flat, pw / 100.0).item())

            m = ((h - blk) / max(wht - blk, 1e-6)).clamp(0, 1)

            px = int(round(base * 1.5 * abs(grow)))
            if px >= 1:
                m = _dilate(m, px) if grow > 0 else _erode(m, px)

            sig = base * 0.5 * feather
            if sig >= 0.3:
                m = _blur(m, sig).clamp(0, 1)

            if invert:
                m = 1.0 - m

            if opacity < 1.0:
                m = m * opacity

            masks.append(m)
            stats.append((float(m[0, 0, : max(1, H // 6)].mean()), float(m.mean()),
                          blk * 255.0, wht * 255.0))

        mask = torch.cat(masks, 0)[:, 0].to(src_dev)          # B,H,W

        s0 = stats[0]
        info = (f"{W}x{H} x{B} | {detector} | radius {base} | grow {int(round(base * 1.5 * grow))}px | "
                f"blur {base * 0.5 * feather:.0f}px | black {s0[2]:.0f} white {s0[3]:.0f} | "
                f"top {s0[0]:.3f} | mean {s0[1]:.3f}")
        if opacity < 1.0:
            info += f" | opacity {opacity:.2f}"
        if B > 1:
            info += " | " + " ".join(f"[{i + 1}] top {s[0]:.3f}" for i, s in enumerate(stats))

        preview = mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        if samples is not None:
            out_latent = samples.copy()
            sh = samples["samples"].shape
            mm = F.interpolate(mask.unsqueeze(1), size=(sh[2], sh[3]),
                               mode="bilinear", align_corners=False).squeeze(1)
            if mm.shape[0] != sh[0]:
                if mm.shape[0] < sh[0]:
                    reps = -(-sh[0] // mm.shape[0])        # ceil division
                    mm = mm.repeat(reps, 1, 1)[:sh[0]]
                else:
                    mm = mm[:sh[0]]
            out_latent["noise_mask"] = mm
        else:
            out_latent = {"samples": torch.zeros((1, 4, 8, 8))}

        return {"ui": {"images": self._ui(mask)},
                "result": (mask, preview, out_latent, info)}

    def _ui(self, mask):
        try:
            d = folder_paths.get_temp_directory()
            os.makedirs(d, exist_ok=True)
            out = []
            for i in range(min(mask.shape[0], 4)):
                a = (mask[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                name = f"hfmask_{np.random.randint(0, 1_000_000):06d}.png"
                Image.fromarray(a).convert("RGB").save(os.path.join(d, name), compress_level=4)
                out.append({"filename": name, "subfolder": "", "type": "temp"})
            return out
        except Exception:
            return []


# ---------------------------------------------------------------- auto button
#
# The button in the node needs something to measure, and the only place the
# image exists is inside build(). So each run stashes a small copy, keyed by the
# node's id, and the /hfmask/auto route works from that. Before the graph has
# ever run there is nothing cached -- the route says so rather than guessing.

_LAST_IMAGE = {}
_CACHE_EDGE = 512


def _png_b64(arr):
    """Encode an HxW or HxWx3 uint8 array as a base64 PNG, for the live preview."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _remember(node_id, image):
    """Keep a small copy of the last input, for the auto button and live preview."""
    try:
        x = image[:1].movedim(-1, 1).float()
        h, w = x.shape[-2], x.shape[-1]
        orig_edge = min(h, w)
        if orig_edge > _CACHE_EDGE:
            k = _CACHE_EDGE / orig_edge
            x = F.interpolate(x, size=(max(1, round(h * k)), max(1, round(w * k))),
                              mode="area")
        _LAST_IMAGE[str(node_id)] = {"img": x.movedim(1, -1).cpu(), "orig_edge": orig_edge}
        if len(_LAST_IMAGE) > 32:                      # do not grow without bound
            _LAST_IMAGE.pop(next(iter(_LAST_IMAGE)))
    except Exception:
        pass


def _estimate_noise(img):
    """Grain level, read off the flattest parts of the image.

    A blurred copy is subtracted to leave only fine detail, then the local
    deviation is taken over the calmest tenth of the picture -- whatever is
    left there is grain or compression, not content.
    """
    g = img.movedim(-1, 1).mean(dim=1, keepdim=True).float()
    resid = g - _blur(g, 1.0)
    k = 8
    h2, w2 = resid.shape[-2] // k * k, resid.shape[-1] // k * k
    if h2 < k or w2 < k:
        return 0.0
    b = (resid[..., :h2, :w2]
         .reshape(1, 1, h2 // k, k, w2 // k, k)
         .permute(0, 1, 2, 4, 3, 5)
         .reshape(-1, k * k))
    sd = b.std(dim=1)
    calm = torch.quantile(sd.float(), 0.10)
    return float(calm) * 255.0


def _suggest(img, grow, feather, detector):
    """Pick grain_filter from measured noise, then solve strength for coverage.

    The target is the mask profile that reads well in practice -- roughly three
    quarters of the frame open, a solid quarter held back. Strength is bisected
    rather than derived because the autolevel is a quantile, which has no
    closed form.
    """
    node = HighFrequencyMask()
    noise = _estimate_noise(img)
    grain = 0.0 if noise < 0.6 else min(2.0, round(noise / 0.9, 2))

    target = 0.72
    lo, hi = 0.40, 2.00

    def coverage(st):
        return float(node.build(img, st, grow, feather, grain, False,
                                detector=detector)["result"][0].mean())

    # The slider cannot always reach the target: with a generous grow the mask is
    # already past it at minimum strength. Say so rather than clamping silently.
    at_lo, at_hi = coverage(lo), coverage(hi)
    if at_lo > target:
        strength, note = lo, ("already above target at minimum strength — "
                              "lower grow for a tighter mask")
    elif at_hi < target:
        strength, note = hi, ("cannot reach target at maximum strength — "
                              "raise grow or lower the radius")
    else:
        note = ""
        for _ in range(9):
            mid = (lo + hi) / 2.0
            if coverage(mid) < target:
                lo = mid
            else:
                hi = mid
        strength = round((lo + hi) / 2.0, 2)

    m = node.build(img, strength, grow, feather, grain, False,
                   detector=detector)["result"][0]
    return {
        "strength": strength,
        "grain_filter": grain,
        "note": note,
        "noise": round(noise, 2),
        "white": round(float((m >= 0.98).float().mean()) * 100, 1),
        "black": round(float((m <= 0.02).float().mean()) * 100, 1),
        "mean": round(float(m.mean()), 3),
    }


try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.post("/hfmask/auto")
    async def _hfmask_auto(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        node_id = str(data.get("node_id", ""))
        cached = _LAST_IMAGE.get(node_id)
        if cached is None:
            return web.json_response(
                {"ok": False,
                 "message": "No image yet — run the graph once, then press again."})
        try:
            out = _suggest(cached["img"],
                           float(data.get("grow", 0.5)),
                           float(data.get("feather", 1.0)),
                           str(data.get("detector", "guided")))
            out["ok"] = True
            return web.json_response(out)
        except Exception as exc:
            return web.json_response({"ok": False, "message": f"{type(exc).__name__}: {exc}"})

    @server.PromptServer.instance.routes.post("/hfmask/preview")
    async def _hfmask_preview(request):
        """Re-render the mask from the cached image at the current widget values.

        Backs the node's live preview: the browser posts here on every slider
        move and gets a mask PNG, an overlay PNG (source tinted where the mask
        would sample) and the white/black/mean stats back. radius_override is
        an absolute px value tuned for the full-resolution image, so it is
        rescaled here to match the smaller cached copy -- otherwise the
        preview would look far coarser than the real run.
        """
        try:
            data = await request.json()
        except Exception:
            data = {}
        node_id = str(data.get("node_id", ""))
        cached = _LAST_IMAGE.get(node_id)
        if cached is None:
            return web.json_response(
                {"ok": False,
                 "message": "No image yet — run the graph once, then press again."})
        try:
            img = cached["img"]
            cached_edge = min(img.shape[1], img.shape[2])
            scale = cached_edge / max(1, cached["orig_edge"])

            radius_in = float(data.get("radius_override", 0) or 0)
            radius = max(1, int(round(radius_in * scale))) if radius_in > 0 else 0

            node = HighFrequencyMask()
            out = node.build(
                img,
                float(data.get("strength", 1.0)),
                float(data.get("grow", 0.5)),
                float(data.get("feather", 1.0)),
                float(data.get("grain_filter", 0.0)),
                bool(data.get("invert", False)),
                detector=str(data.get("detector", "guided")),
                radius_override=radius,
                black_override=float(data.get("black_override", 0.0)),
                white_override=float(data.get("white_override", 0.0)),
                opacity=float(data.get("opacity", 1.0)),
            )["result"]
            mask = out[0][0].clamp(0, 1)

            mask_np = (mask.numpy() * 255).astype(np.uint8)
            src = img[0].clamp(0, 1).numpy()
            tint = np.array([1.0, 0.18, 0.18], dtype=np.float32)
            alpha = (mask.numpy() * 0.55)[..., None]
            comp = ((src * (1 - alpha) + tint * alpha).clip(0, 1) * 255).astype(np.uint8)

            return web.json_response({
                "ok": True,
                "mask_png": _png_b64(mask_np),
                "overlay_png": _png_b64(comp),
                "white": round(float((mask >= 0.98).float().mean()) * 100, 1),
                "black": round(float((mask <= 0.02).float().mean()) * 100, 1),
                "mean": round(float(mask.mean()), 3),
            })
        except Exception as exc:
            return web.json_response({"ok": False, "message": f"{type(exc).__name__}: {exc}"})
except Exception:
    pass


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {"HighFrequencyMask": HighFrequencyMask}
NODE_DISPLAY_NAME_MAPPINGS = {"HighFrequencyMask": "High Frequency Mask"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
