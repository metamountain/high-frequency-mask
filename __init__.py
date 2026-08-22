"""
High Frequency Mask — ComfyUI custom node

White = textured, gets sampled. Black = flat, stays as it is.
Four sliders: strength, grow, feather, grain_filter. The mask previews in the node.
An optional LATENT input gets the mask attached as its noise mask.

All torch -> runs on the GPU and handles whole batches.
"""

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
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x


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


# ---------------------------------------------------------------- node

class HighFrequencyMask:

    DESCRIPTION = (
        "Protects flat areas during an upscale or refine pass. White where the image "
        "already has texture, black where it is flat -- feed it to the sampler as a "
        "noise mask and skies, walls, panels and skin are left alone.\n\n"
        "WHAT IT FIXES. Upscalers invent detail in surfaces that should stay smooth "
        "and drift the colour while doing it. Flux Klein is especially prone to both. "
        "Masking the flat regions out stops the model touching them at all, so they "
        "keep their original tone and stay clean.\n\n"
        "SETTINGS FOR UPSCALING. The shipped defaults are tuned too coarse for this "
        "job -- they let roughly three times more through in flat areas than needed. "
        "Set radius_override to about a third of the automatic radius (5-7 at 1024px, "
        "10-14 at 2048px) and grain_filter to 0 unless the source is grainy or JPEG. "
        "Measured on ComfyUI generations that cuts the flat-area leak by 2x to 6x -- the "
        "more flat area, the bigger the win -- while keeping MORE real texture. Add "
        "feather 0.5-1.0 so the mask edge does "
        "not leave a seam, and grow 0.5-1.0 so real detail keeps a margin.\n\n"
        "Going finer than a third buys nothing -- protection plateaus there while "
        "texture retention starts dropping.\n\n"
        "TWO THINGS THAT MISLEAD. grain_filter defaults to 1.0, which smooths away real "
        "texture on clean renders before it is ever measured. And black_override and "
        "white_override are measured on the HIGH-PASS image, not on brightness: useful "
        "values sit near 0-30, not 0-255, and setting either one switches off the "
        "automatic levels so strength stops mattering.\n\n"
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
        "The finished mask. White is sampled, black stays original.",
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
                "grow": ("FLOAT", {"default": 1.00, "min": -1.00, "max": 4.00, "step": 0.05,
                                   "tooltip": "Expands the white areas so detail keeps a safety margin. Negative values shrink the mask instead."}),
                "feather": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 4.00, "step": 0.05,
                                      "tooltip": "Softness of the edge between white and black. Higher = longer fade and no visible mask border."}),
                "grain_filter": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 4.00, "step": 0.05,
                                           "tooltip": "Smooths grain and JPEG artifacts before analysis, so a noisy sky is still recognised as flat. Set to 0 for clean renders and upscales -- on a noise-free source it removes real texture and weakens the mask."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Swaps black and white: protects the detail and samples the flat areas instead."}),
            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "Optional. If connected, the mask is resized to the latent and attached as its noise mask."}),
                "radius_override": ("INT", {"default": 0, "min": 0, "max": 400,
                                            "tooltip": "High-pass radius in px, deciding which size of structure counts as detail. 0 = automatic from the image size (min(W,H)/52). Smaller finds finer texture. Very large values combined with a high feather will error."}),
                "black_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0,
                                             "tooltip": "Below this, an area counts as flat. Measured on the HIGH-PASS image, not on brightness -- typical values are 0 to 10. 0 means automatic. Setting this or white_override disables strength."}),
                "white_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0,
                                             "tooltip": "Above this, an area counts as full detail. Also high-pass contrast -- typical values are 8 to 40. 0 means automatic. Setting black_override alone falls back to 255 here and yields a near-empty mask."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("mask", "mask_preview", "latent", "info")
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = "mask"

    def build(self, image, strength, grow, feather, grain_filter, invert,
              samples=None, radius_override=0, black_override=0.0, white_override=0.0):

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

        # High-pass: image minus low-pass, negative half dropped (bright edge side only)
        hp = (g - _blur(g, sigma=base / 2.0)).clamp(min=0) * 3.0
        hp = hp.clamp(0, 1)

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
                blk = torch.quantile(flat.float(), pb / 100.0).item()
                wht = max(blk + 0.03, torch.quantile(flat.float(), pw / 100.0).item())

            m = ((h - blk) / max(wht - blk, 1e-6)).clamp(0, 1)

            px = int(round(base * 1.5 * abs(grow)))
            if px >= 1:
                m = _dilate(m, px) if grow > 0 else _erode(m, px)

            sig = base * 0.5 * feather
            if sig >= 0.3:
                m = _blur(m, sig).clamp(0, 1)

            if invert:
                m = 1.0 - m

            masks.append(m)
            stats.append((float(m[0, 0, : max(1, H // 6)].mean()), float(m.mean()),
                          blk * 255.0, wht * 255.0))

        mask = torch.cat(masks, 0)[:, 0].to(src_dev)          # B,H,W

        s0 = stats[0]
        info = (f"{W}x{H} x{B} | radius {base} | grow {int(round(base * 1.5 * grow))}px | "
                f"blur {base * 0.5 * feather:.0f}px | black {s0[2]:.0f} white {s0[3]:.0f} | "
                f"top {s0[0]:.3f} | mean {s0[1]:.3f}")
        if B > 1:
            info += " | " + " ".join(f"[{i + 1}] top {s[0]:.3f}" for i, s in enumerate(stats))

        preview = mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        if samples is not None:
            out_latent = samples.copy()
            sh = samples["samples"].shape
            mm = F.interpolate(mask.unsqueeze(1), size=(sh[2], sh[3]),
                               mode="bilinear", align_corners=False).squeeze(1)
            if mm.shape[0] != sh[0]:
                mm = mm[:1].repeat(sh[0], 1, 1) if mm.shape[0] == 1 else mm[:sh[0]]
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


NODE_CLASS_MAPPINGS = {"HighFrequencyMask": HighFrequencyMask}
NODE_DISPLAY_NAME_MAPPINGS = {"HighFrequencyMask": "High Frequency Mask"}
