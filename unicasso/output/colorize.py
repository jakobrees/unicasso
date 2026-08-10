"""Colorize an asciify output: per-cell fg/bg fitted against a color field.

    python -m unicasso.output.colorize --txt OUT.txt --image ORIGINAL [--paint PAINT.png] \
        --out-png colored.png [--out-ans colored.ans] [--fg tint|paint|ink|dark] [...tweaks]

Closed-form MSE fit per cell (mask = the placed glyph's ink bitmap):
    fg = weighted mean of the color field under the mask
    bg = weighted mean under (1 - mask)
render = bg + (fg - bg) * mask. Space cells: fg := bg.

--fg tint   (default): local paint HUE, luminance clamped dark (--fg-lightness)
--fg paint           : fg fitted on the PAINT field -> strokes tinted by local color
--fg ink             : fg fitted on the ORIGINAL image -> strokes keep the ink's own
                       (dark) color, paint only shows in backgrounds
--fg dark            : constant ink color (--fg-color)

Tweak knobs (all post-fit, cheap to re-run):
    --sat S           chroma multiplier around each color's gray axis (1 = neutral)
    --gamma G         gamma on both layers (1 = off)
    --bg-lift L       lighten backgrounds toward white by fraction L
    --min-contrast D  if |fg-bg| luminance gap < D, push fg darker to D (legibility)
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image


def load_field(path, W, H):
    im = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0)      # (H,W,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True)
    ap.add_argument("--image", required=True, help="color original (fg source in ink mode)")
    ap.add_argument("--paint", default=None, help="paint field (default: blurred original)")
    ap.add_argument("--paint-sigma", type=float, default=8.0)
    ap.add_argument("--fg", choices=["paint", "ink", "dark", "tint"], default="tint",
                    help="paint = fitted on paint field (tinted strokes); ink = fitted on "
                         "original (keeps source ink color); dark = constant --fg-color; "
                         "tint = local paint HUE clamped dark to --fg-lightness (default)")
    ap.add_argument("--fg-color", default="141414", help="(dark mode) hex ink color")
    ap.add_argument("--fg-lightness", type=float, default=0.18,
                    help="(tint mode) luminance ceiling for stroke colors")
    ap.add_argument("--paint-kernel", choices=["gauss", "cell", "none"], default="cell",
                    help="internal paint blur: gauss = isotropic; cell = cell-shaped box "
                         "(the active kit's cell size) with gaussian rolloff -- the matched "
                         "prefilter for per-cell bg sampling (kills inter-cell color banding)")
    ap.add_argument("--sat", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--bg-lift", type=float, default=0.0)
    ap.add_argument("--min-contrast", type=float, default=0.0)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-ans", default=None)
    args = ap.parse_args()

    from unicasso.substrate import glyphs as G
    ink, chars = G.load_glyphs(device=torch.device("cpu"))
    masks = ink[:, 0]                                                # (N,CH,CW) ink=1
    CH, CW = masks.shape[1], masks.shape[2]
    cidx = {c: i for i, c in enumerate(chars)}

    lines = open(args.txt, encoding="utf-8").read().rstrip("\n").split("\n")
    GW = max(len(l) for l in lines)
    GH = len(lines)
    grid = torch.zeros(GH, GW, dtype=torch.long)
    sp = cidx.get(" ", 0)
    for r, l in enumerate(lines):
        for c, ch in enumerate(l):
            grid[r, c] = cidx.get(ch, sp)
    W, H = GW * CW, GH * CH

    field_img = load_field(args.image, W, H)
    import torch.nn.functional as F

    def sep_blur(img_hw3, ky, kx):
        x = img_hw3.permute(2, 0, 1)[None]
        ry, rx = (len(ky) - 1) // 2, (len(kx) - 1) // 2
        x = F.pad(x, (0, 0, ry, ry), mode="replicate")
        x = F.conv2d(x, ky.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
        x = F.pad(x, (rx, rx, 0, 0), mode="replicate")
        x = F.conv2d(x, kx.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        return x[0].permute(1, 2, 0)

    def gauss_k(sigma):
        r = max(1, int(3 * sigma))
        k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
        return k / k.sum()

    def cell_k(size, roll=3.0):
        # flat-top box of the cell dimension with gaussian rolloff: the matched
        # prefilter for box-sampling at cell pitch
        g = gauss_k(roll)
        box = torch.ones(size)
        k = torch.conv1d(box.view(1, 1, -1), g.view(1, 1, -1), padding=len(g) // 2)[0, 0]
        if len(k) % 2 == 0:                     # odd length so separable padding is exact
            k = torch.cat([k, k.new_zeros(1)])
        return k / k.sum()

    if args.paint:
        paint = load_field(args.paint, W, H)
    elif args.paint_kernel == "none":
        paint = field_img                       # raw original: per-cell fit does the averaging
    elif args.paint_kernel == "cell":
        paint = sep_blur(field_img, cell_k(CH), cell_k(CW))
    else:
        g = gauss_k(args.paint_sigma)
        paint = sep_blur(field_img, g, g)

    fg_src = field_img if args.fg in ("ink", "tint") else paint
    if args.fg == "tint":                       # hue from a LIGHT blur (strokes are thin;
        fg_src = sep_blur(field_img, gauss_k(2.5), gauss_k(2.5))   # cell blur would wash them)
    M = masks[grid]                                                  # (GH,GW,CH,CW)
    pc = paint.view(GH, CH, GW, CW, 3).permute(0, 2, 1, 3, 4)        # (GH,GW,CH,CW,3)
    fc = fg_src.view(GH, CH, GW, CW, 3).permute(0, 2, 1, 3, 4)
    msum = M.sum(dim=(2, 3)).clamp_min(1e-6)[..., None]
    fg = (fc * M[..., None]).sum(dim=(2, 3)) / msum                  # (GH,GW,3)
    wb = (1.0 - M)
    bg = (pc * wb[..., None]).sum(dim=(2, 3)) / wb.sum(dim=(2, 3)).clamp_min(1e-6)[..., None]
    empty = M.sum(dim=(2, 3)) < 1.0
    if args.fg == "dark":
        c = torch.tensor([int(args.fg_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
        fg = c.expand_as(fg).clone()
    elif args.fg == "tint":                     # local hue, luminance clamped dark
        lum = (0.2126 * fg[..., 0] + 0.7152 * fg[..., 1] + 0.0722 * fg[..., 2]).clamp_min(1e-4)
        scale = (args.fg_lightness / lum).clamp(max=1.0)
        fg = fg * scale[..., None]
    fg[empty] = bg[empty]

    def tweak(col):
        if args.sat != 1.0:
            g = col.mean(-1, keepdim=True)
            col = (g + (col - g) * args.sat)
        if args.gamma != 1.0:
            col = col.clamp(0, 1) ** args.gamma
        return col.clamp(0, 1)
    fg, bg = tweak(fg), tweak(bg)
    if args.bg_lift > 0:
        bg = bg + (1.0 - bg) * args.bg_lift
    if args.min_contrast > 0:
        lum = lambda c: 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
        gap = lum(bg) - lum(fg)
        need = (gap < args.min_contrast) & ~empty
        scale = ((lum(bg) - args.min_contrast).clamp(0, 1) / lum(fg).clamp_min(1e-4)).clamp(0, 1)
        fg = torch.where(need[..., None], fg * scale[..., None], fg)

    # render: (GH,GW,CH,CW,3) = bg + (fg-bg)*mask
    cellimg = bg[:, :, None, None, :] + (fg - bg)[:, :, None, None, :] * M[..., None]
    img = cellimg.permute(0, 2, 1, 3, 4).reshape(H, W, 3)
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    Image.fromarray((img.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(args.out_png)
    print("wrote", args.out_png)

    if args.out_ans:
        f8 = (fg.clamp(0, 1) * 255).to(torch.uint8)
        b8 = (bg.clamp(0, 1) * 255).to(torch.uint8)
        with open(args.out_ans, "w", encoding="utf-8") as f:
            for r in range(GH):
                for c in range(GW):
                    fr, fgn, fb = (int(v) for v in f8[r, c])
                    br, bgn, bb = (int(v) for v in b8[r, c])
                    ch = chars[grid[r, c]]
                    f.write(f"\x1b[38;2;{fr};{fgn};{fb}m\x1b[48;2;{br};{bgn};{bb}m{ch}")
                f.write("\x1b[0m\n")
        print("wrote", args.out_ans, "(view with: cat)")


if __name__ == "__main__":
    main()
