"""
High Frequency Mask — ComfyUI custom node

weiss = Struktur wird gesampelt, schwarz = Flaeche bleibt original.
Drei Regler: staerke, groesse, weichheit. Maske erscheint direkt im Node.
Optionaler LATENT-Eingang setzt die Noise-Mask gleich mit.

Alles in torch -> laeuft auf der GPU und verarbeitet ganze Batches.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

try:                                    # auf der GPU rechnen, wenn moeglich
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
    """separabler Gauss, B1HW"""
    if sigma < 0.3:
        return x
    k = _gauss1d(sigma, x.device, x.dtype)
    r = k.shape[-1] // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x


def _dilate(x, px):
    """Maximum-Filter ueber max_pool2d, kachelweise fuer grosse Kernel"""
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "staerke": ("FLOAT", {"default": 1.00, "min": 0.40, "max": 2.00, "step": 0.02,
                                      "tooltip": "wieviel als Detail gilt. hoeher = mehr wird gesampelt"}),
                "groesse": ("FLOAT", {"default": 1.00, "min": -1.00, "max": 4.00, "step": 0.05,
                                      "tooltip": "Maske ausdehnen. negativ = schrumpfen"}),
                "weichheit": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 4.00, "step": 0.05,
                                        "tooltip": "Kantenweichheit der Uebergaenge"}),
                "entrauschen": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 4.00, "step": 0.05,
                                          "tooltip": "Vorabglaettung gegen Korn und JPEG. hoeher = Himmel wird "
                                                     "zuverlaessiger als Flaeche erkannt"}),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "samples": ("LATENT",),
                "radius_override": ("INT", {"default": 0, "min": 0, "max": 400,
                                            "tooltip": "0 = automatisch aus der Bildgroesse"}),
                "black_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0}),
                "white_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("mask", "mask_preview", "latent", "info")
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = "mask"

    def build(self, image, staerke, groesse, weichheit, entrauschen, invert,
              samples=None, radius_override=0, black_override=0.0, white_override=0.0):

        src_dev = image.device
        dev = _device()
        x = image.movedim(-1, 1).to(device=dev, dtype=torch.float32)   # B,C,H,W
        g = x.mean(dim=1, keepdim=True)                      # B,1,H,W  graustufen
        B, _, H, W = g.shape

        # Pixelmasse skalieren mit der Aufloesung: gleiche Wirkung bei 816 wie bei 1632
        base = radius_override if radius_override > 0 else max(4, int(round(min(W, H) / 52.0)))

        # Vorabglaettung gegen Sensorrauschen / JPEG, bevor analysiert wird
        if entrauschen > 0:
            g = _blur(g, sigma=max(0.3, base / 12.0 * entrauschen))

        # Hochpass: Bild minus Tiefpass, negative Haelfte verworfen (nur helle Kantenseite)
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
                pw = float(np.clip(99.0 - (staerke - 0.40) * (99.0 - 62.0) / 1.60, 62.0, 99.0))
                pb = float(np.clip(68.0 + (1.00 - staerke) * 8.0, 50.0, 90.0))
                blk = torch.quantile(flat.float(), pb / 100.0).item()
                wht = max(blk + 0.03, torch.quantile(flat.float(), pw / 100.0).item())

            m = ((h - blk) / max(wht - blk, 1e-6)).clamp(0, 1)

            px = int(round(base * 1.5 * abs(groesse)))
            if px >= 1:
                m = _dilate(m, px) if groesse > 0 else _erode(m, px)

            sig = base * 0.5 * weichheit
            if sig >= 0.3:
                m = _blur(m, sig).clamp(0, 1)

            if invert:
                m = 1.0 - m

            masks.append(m)
            stats.append((float(m[0, 0, : max(1, H // 6)].mean()), float(m.mean()),
                          blk * 255.0, wht * 255.0))

        mask = torch.cat(masks, 0)[:, 0].to(src_dev)          # B,H,W

        s0 = stats[0]
        info = (f"{W}x{H} x{B} | radius {base} | grow {int(round(base * 1.5 * groesse))}px | "
                f"blur {base * 0.5 * weichheit:.0f}px | black {s0[2]:.0f} white {s0[3]:.0f} | "
                f"oben {s0[0]:.3f} | mittel {s0[1]:.3f}")
        if B > 1:
            info += " | " + " ".join(f"[{i + 1}] oben {s[0]:.3f}" for i, s in enumerate(stats))

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
