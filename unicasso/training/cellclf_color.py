"""Color pipeline through the cell classifier: photo -> ink layer -> glyphs -> ANSI.

python -m unicasso.training.cellclf_color --images "/path/to/dir" --width 60

Per image, one sheet row of four panels:
    input image | extracted ink structure | predicted glyphs | full color render
Ink extraction = the engine's --color decomposition (engine/color.py:decompose):
per-cell Lab principal-axis clustering, minority cluster = ink, gated by Lab
separation (jnd) and separation/within-spread ratio; nomination_target() folds it
into the white=1 structure map that drives glyph choice. Glyphs = a dense-trained
checkpoint under framing-ensemble inference. Color = engine.color.fit_fg_bg:
closed-form per-cell MSE fit of fg/bg with the predicted glyph's ink bitmap as
the mask, composited as bg + (fg-bg)*mask.
Also writes <stem>.ans (24-bit ANSI, `cat`-able in a terminal).
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image, ImageFont, ImageOps

from unicasso.adapter.corrupt import CorruptionSampler
from unicasso.engine.color import decompose, fit_fg_bg, nomination_target
from unicasso.substrate import glyphs as G
from unicasso.substrate import raster
from unicasso.training.cellclf_ensemble import predict_run
from unicasso.training.cellclf_sheet import caption, render_grid, to_img
from unicasso.training.cellclf_widths import discover_models, load_any


def glyph_bg_dist(ink_bitmaps, ch, cw):
    """(N, P) ink masks -> (N, P) distance of each pixel to the nearest ink pixel,
    normalized per glyph to max 1. Ink-free glyphs (space) get uniform 1."""
    N, P = ink_bitmaps.shape
    ys, xs = torch.meshgrid(torch.arange(ch, dtype=torch.float32),
                            torch.arange(cw, dtype=torch.float32), indexing="ij")
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)          # (P, 2)
    out = torch.ones(N, P)
    for i in range(N):
        pts = coords[ink_bitmaps[i] > 0.5]
        if len(pts) == 0:
            continue
        d = torch.cdist(coords, pts).min(dim=1).values                     # (P,)
        out[i] = d / d.max().clamp_min(1e-6)
    return out


def fit_fg_bg_distweighted(cell_rgb, mask, bg_dist, pow_=1.0, ridge=1e-3):
    """fit_fg_bg, but bg votes are weighted by (distance to the glyph's ink)^pow --
    pixels hugging the stroke (antialiasing/misalignment contamination) barely count.
    pow_=0 reproduces the plain closed-form MSE fit. fg is unchanged."""
    w = mask.clamp(0, 1)
    wb = (1 - w) * bg_dist.clamp_min(1e-6) ** pow_
    mean = cell_rgb.mean(1)
    fg = ((w[:, :, None] * cell_rgb).sum(1) + ridge * mean) \
        / (w.sum(1, keepdim=True) + ridge).clamp_min(1e-6)
    bg = ((wb[:, :, None] * cell_rgb).sum(1) + ridge * mean) \
        / (wb.sum(1, keepdim=True) + ridge).clamp_min(1e-6)
    return fg, bg


def mc_push(fg, bg, mc=0.12):
    """Engine _mc_push, identical arithmetic: legibility floor on the luminance
    gap. Only fg moves; applied BEFORE k so k=1 reproduces the floored base."""
    if mc <= 0:
        return fg
    lum = lambda t: 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
    gap = lum(fg) - lum(bg)
    push = torch.where(gap <= 0, -1.0, 1.0) * (mc - gap.abs()).clamp_min(0)
    return fg + push[..., None]


def orient_ink_dark(dec):
    """Flip each cell's ink so it covers the DARKER cluster (lineart convention),
    instead of decompose's minority convention. Returns the modified dec.
    ink = sig*gate, so a flip maps ink -> gate - ink; fg/bg swap accordingly."""
    fg_L = (0.2126 * dec["fg"][:, 0] + 0.7152 * dec["fg"][:, 1] + 0.0722 * dec["fg"][:, 2])
    bg_L = (0.2126 * dec["bg"][:, 0] + 0.7152 * dec["bg"][:, 1] + 0.0722 * dec["bg"][:, 2])
    flip = fg_L > bg_L                                     # fg (ink side) is the brighter one
    dec["ink"][flip] = dec["gate"][flip, None] - dec["ink"][flip]
    fg = dec["fg"].clone()
    dec["fg"][flip] = dec["bg"][flip]
    dec["bg"][flip] = fg[flip]
    return dec


def ansi_txt(pred, chars, fg, bg, gh, gw):
    lines = []
    for y in range(gh):
        parts = []
        for x in range(gw):
            i = y * gw + x
            f = (fg[i] * 255).round().astype(int)
            b = (bg[i] * 255).round().astype(int)
            parts.append(f"\x1b[38;2;{f[0]};{f[1]};{f[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m"
                         + chars[pred[y, x]])
        lines.append("".join(parts) + "\x1b[0m")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="directory (or single file) of color images")
    p.add_argument("--model", default="tf5x3_dense_b3_h4@2000")
    p.add_argument("--ensemble-mode", default="prob", choices=["prob", "logprob", "center"])
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--color-ramp", type=float, default=0.35, help="engine --color-ramp")
    p.add_argument("--color-jnd", type=float, default=4.0, help="engine --color-jnd")
    p.add_argument("--color-ratio", type=float, default=1.5, help="engine --color-ratio")
    p.add_argument("--color-source", default="fit", choices=["fit", "cluster", "blend"],
                   help="fit = MSE fg/bg at the predicted glyph mask; cluster = decompose's "
                        "own cluster colors (glyph-independent, punchier); blend = 50/50")
    p.add_argument("--bg-dist-pow", type=float, default=0.0,
                   help="weight bg pixels by (distance to the glyph's ink)^pow in the color fit; "
                        "0 = plain closed-form MSE (contaminated near strokes), 1-2 = cleaner bg")
    p.add_argument("--ink-polarity", default="minority", choices=["minority", "dark"],
                   help="minority = engine convention (per-cell smaller cluster); dark = flip "
                        "so ink is always the darker cluster (matches lineart training)")
    p.add_argument("--cache", default="runs/cellclf/cache")
    p.add_argument("--vae-ckpt", default="weights/vae_dejavu/model.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    meta = json.load(open(os.path.join(G.repo_path(args.cache), "meta.json")))
    ch, cw = meta["cell_h"], meta["cell_w"]
    sampler = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu", profile="dejavu")
    ink_bitmaps = (1.0 - sampler.bitmaps.cpu()).reshape(sampler.N, -1)   # (N, P) ink=1
    bg_dist_all = glyph_bg_dist(ink_bitmaps, meta["cell_h"], meta["cell_w"]) \
        if args.bg_dist_pow > 0 else None
    model, cfg = load_any(dict(discover_models())[args.model], meta, device)
    try:
        font = ImageFont.truetype(G.repo_path("fonts/DejaVuSansMono.ttf"), 13)
    except OSError:
        font = ImageFont.load_default()
    zero_emb = np.zeros(meta["clip_dim"], dtype=np.float32)

    if os.path.isdir(args.images):
        paths = sorted(sum((glob.glob(os.path.join(args.images, e))
                            for e in ("*.jpg", "*.jpeg", "*.png", "*.webp")), []))
    else:
        paths = [args.images]
    out_dir = args.out or os.path.join(G.REPO_ROOT, "runs", "cellclf", "color_sheets")
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w0, h0 = im.size
            gh = raster.grid_height_for_aspect(w0, h0, args.width, cw, ch, 0)
            im = im.resize((args.width * cw, gh * ch), Image.LANCZOS)
        rgb = np.asarray(im, np.float32) / 255.0                          # (H,W,3)
        dec = decompose(torch.from_numpy(rgb), gh, args.width, ch, cw,
                        ramp=args.color_ramp, jnd=args.color_jnd, ratio=args.color_ratio)
        if args.ink_polarity == "dark":
            dec = orient_ink_dark(dec)
        ink_layer = nomination_target(dec).numpy()                        # white=1
        ink_u8 = np.clip((1.0 - ink_layer) * 255.0, 0, 255).astype(np.uint8)

        pred = predict_run(model, cfg, ink_u8, zero_emb, meta, device, args.ensemble_mode)

        C = dec["cell_rgb"]                                               # (M,P,3)
        idx = torch.from_numpy(pred.ravel())
        mask = ink_bitmaps[idx]                                           # (M,P)
        if args.bg_dist_pow > 0:
            fg, bg = fit_fg_bg_distweighted(C, mask, bg_dist_all[idx], pow_=args.bg_dist_pow)
        else:
            fg, bg = fit_fg_bg(C, mask)
        if args.color_source == "cluster":
            fg, bg = dec["fg"], dec["bg"]
        elif args.color_source == "blend":
            fg, bg = 0.5 * dec["fg"] + 0.5 * fg, 0.5 * dec["bg"] + 0.5 * bg
        color_cells = bg[:, None, :] + (fg - bg)[:, None, :] * mask[:, :, None]   # (M,P,3)
        color_img = color_cells.view(gh, args.width, ch, cw, 3).permute(0, 2, 1, 3, 4) \
            .reshape(gh * ch, args.width * cw, 3).clamp(0, 1)
        color_u8 = (color_img.numpy() * 255).astype(np.uint8)

        stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
        with open(os.path.join(out_dir, stem + ".ans"), "w") as f:
            f.write(ansi_txt(pred, sampler.chars, fg.numpy(), bg.numpy(), gh, args.width))
        with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join("".join(sampler.chars[g] for g in row) for row in pred) + "\n")

        panels = [caption(im, f"{stem}  input", font),
                  caption(to_img((ink_layer * 255).astype(np.uint8)),
                          "extracted ink structure", font),
                  caption(to_img(render_grid(sampler, pred)),
                          f"glyphs  ({args.model} {args.ensemble_mode})", font),
                  caption(Image.fromarray(color_u8), "color render (MSE fg/bg fit)", font)]
        gut = 8
        sheet = Image.new("RGB", (sum(x.width for x in panels) + gut * (len(panels) + 1),
                                  max(x.height for x in panels) + 2 * gut), (255, 255, 255))
        x = gut
        for pane in panels:
            sheet.paste(pane, (x, gut)); x += pane.width + gut
        sheet.save(os.path.join(out_dir, stem + ".png"))
        print("wrote", os.path.join(out_dir, stem + ".png"), flush=True)


if __name__ == "__main__":
    main()
