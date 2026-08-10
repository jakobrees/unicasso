"""Ink extraction prototype: split a color image into an INK layer (near-black
structure, the asciify target) and a PAINT layer (ink deleted, holes filled by
normalized convolution -- the color field for fg/bg fitting).

    python -m unicasso.output.ink_extract IMG [IMG ...] --out DIR \
        [--thr 0.30] [--ramp 0.10] [--chroma-max 0.25] [--fill-sigma 8] [--kuwahara 0]

Per image writes <stem>_ink.png (white bg, soft ink mask applied to darkness),
<stem>_paint.png (filled color field) and <stem>_panel.png (original | ink | paint).

Soft mask (no hard steps -- binarized edges are aliasing bait for recon):
    m = smoothstep((thr - L) / ramp) [* smoothstep((chroma_max - C) / ramp_c)]
L = perceptual-ish lightness (sqrt of relative luminance), C = max(RGB)-min(RGB).
--chroma-max off (default) = dark-of-any-hue counts as ink; set it to require
NEUTRAL darks only (dark navy jacket -> paint, not ink).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def smoothstep(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def gauss_blur(x, sigma):
    """x (C,H,W) -> separable gaussian blur, replicate padding."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma * sigma))
    k = k / k.sum()
    x = x[None]
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(x.shape[1], 1, -1, 1), groups=x.shape[1])
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1), groups=x.shape[1])
    return x[0]


def kuwahara(x, radius):
    """Cheap box Kuwahara on (C,H,W): per pixel take the mean of the quadrant
    window with the lowest luminance variance (painterly flattening)."""
    C, H, W = x.shape
    lum = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])[None]
    r = radius
    means, varis = [], []
    for dy, dx in ((-r, -r), (-r, 0), (0, -r), (0, 0)):
        pad = (max(0, dx + r), max(0, -(dx)), max(0, dy + r), max(0, -(dy)))
        sl = lambda t: F.avg_pool2d(F.pad(t[None], (r, r, r, r), mode="replicate"),
                                    r + 1, stride=1)[0][:, max(0, -dy):H + max(0, -dy),
                                                        max(0, -dx):W + max(0, -dx)]
        mL = sl(lum); m2L = sl(lum * lum)
        varis.append((m2L - mL * mL)[0])
        means.append(sl(x))
    v = torch.stack(varis)                     # (4,H,W)
    pick = v.argmin(dim=0)                     # (H,W)
    out = torch.zeros_like(x)
    for q in range(4):
        out += means[q] * (pick == q).float()[None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="out/ink_extract")
    ap.add_argument("--thr", type=float, default=0.30, help="lightness below this = ink (soft)")
    ap.add_argument("--ramp", type=float, default=0.10, help="softness of the ink cut")
    ap.add_argument("--chroma-max", type=float, default=None,
                    help="require NEUTRAL darks: chroma above this disqualifies ink (off = any hue)")
    ap.add_argument("--fill-sigma", type=float, default=8.0,
                    help="normalized-convolution fill radius for the paint layer")
    ap.add_argument("--kuwahara", type=int, default=0, help="painterly flattening radius (0=off)")
    ap.add_argument("--max-width", type=int, default=1400, help="downscale wide inputs for preview")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for path in args.images:
        im = Image.open(path).convert("RGB")
        if im.width > args.max_width:
            im = im.resize((args.max_width, im.height * args.max_width // im.width), Image.LANCZOS)
        rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
        Y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        L = Y.sqrt()                                            # cheap perceptual lightness
        m = smoothstep((args.thr - L) / args.ramp + 0.5)        # soft ink mask (1 = ink)
        if args.chroma_max is not None:
            Cc = rgb.max(0).values - rgb.min(0).values
            m = m * smoothstep((args.chroma_max - Cc) / args.ramp + 0.5)

        # INK layer: white paper, ink darkness kept where masked (asciify target)
        ink_img = 1.0 - m * (1.0 - L)
        # PAINT layer: blur/Kuwahara of the WHOLE original -- no ink subtraction
        # (deleting+filling invents content inside large ink masses; the per-cell
        # fg/bg fit separates ink from paint at fitting time via the mask anyway)
        paint = gauss_blur(rgb, args.fill_sigma)
        if args.kuwahara > 0:
            paint = kuwahara(paint, args.kuwahara)

        stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
        to8 = lambda t: (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8) \
            if t.dim() == 3 else (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(to8(ink_img)).save(f"{args.out}/{stem}_ink.png")
        Image.fromarray(to8(paint)).save(f"{args.out}/{stem}_paint.png")
        panel = np.concatenate([to8(rgb), np.stack([to8(ink_img)] * 3, -1), to8(paint)], axis=1)
        Image.fromarray(panel).save(f"{args.out}/{stem}_panel.png")
        print(f"{stem}: ink coverage {float(m.mean()):.3f} -> {args.out}/{stem}_panel.png")


if __name__ == "__main__":
    main()
