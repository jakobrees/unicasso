"""Re-fit fg/bg for an existing .txt by closed-form MSE against the original image.

The glyphs are taken as FIXED -- whatever the run chose stays exactly as it is. Only the
colors are replaced, by the per-cell least-squares solution:

    minimize  || bg + (fg - bg)*m - c ||^2   over the cell
      =>  fg = mean of c weighted by m,   bg = mean of c weighted by (1 - m)

where m is the placed glyph's own ink bitmap and c is the original image's cell. This is
the same closed-form fit that initializes the per-slot color leaves during a run -- so this
tool answers "what would the colors be if CLIP had never moved them off the MSE optimum".

Useful when a color run's glyphs are good but its colors drifted: CLIP optimizes a
perceptual objective, and the colors it likes are not the colors that minimize pixel error.

    python -m unicasso.output.recolor --txt OUT.txt --image ORIGINAL.jpg --out-png recolored.png

Legibility knob (--min-contrast, default 0.12): the pure MSE optimum sets fg~bg wherever the
image is locally smooth, which makes the glyph disappear and the result read as a soft photo.
A floor on the fg/bg luminance gap keeps the characters legible as texture at a small cost in
pixel error (e.g. 0.0395 -> 0.0417 at 0.12, -> 0.0459 at 0.25 on a sample image).
    --min-contrast D   if the fg/bg luminance gap is under D, push fg away until it is D
    --sat S            chroma multiplier around each color's gray axis (1 = untouched)
"""
import argparse
import os
import unicodedata

import numpy as np
import torch
from PIL import Image

from unicasso.engine.color import fit_fg_bg


def load_txt(path):
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    while lines and not lines[-1]:
        lines.pop()
    gw = max(len(l) for l in lines)
    lines = [l.ljust(gw) for l in lines]
    return lines, len(lines), gw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--txt", required=True, help="the ASCII to keep (glyphs are not changed)")
    ap.add_argument("--image", required=True, help="original color image to fit against")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-ans", default=None)
    ap.add_argument("--compare", default=None,
                    help="write a side-by-side vs this render (e.g. the run's own _color.png)")
    ap.add_argument("--min-contrast", type=float, default=0.12,
                    help="minimum fg/bg luminance gap (0 = pure MSE fit, which makes "
                         "glyphs vanish in smooth regions)")
    ap.add_argument("--sat", type=float, default=1.0)
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="regularizer for near-empty / near-full masks")
    ap.add_argument("--proportion", type=float, default=0.0,
                    help="blend the fit toward the cell's TRUE color populations (2-means over "
                         "the cell, which is area/proportion aware) instead of only the colors "
                         "the glyph mask happens to land on. 0 = off, 1 = equal weight with the "
                         "fit, large = pure population colors. Same convex form as --ridge.")
    args = ap.parse_args()

    from unicasso.substrate import glyphs as G
    ink, chars = G.load_glyphs(device=torch.device("cpu"))       # (N,1,CH,CW) ink=1, list[str]
    ink = ink.squeeze(1) if ink.dim() == 4 else ink              # -> (N,CH,CW)
    N, CH, CW = ink.shape
    idx_of = {c: i for i, c in enumerate(chars)}

    lines, GH, GW = load_txt(args.txt)
    M = GH * GW
    missing = set()
    gid = torch.zeros(M, dtype=torch.long)
    for i, row in enumerate(lines):
        for j, ch in enumerate(row):
            k = idx_of.get(ch)
            if k is None:
                missing.add(ch)
                k = idx_of.get(" ", 0)
            gid[i * GW + j] = k
    if missing:
        names = ", ".join(f"{c!r}({unicodedata.name(c, '?')})" for c in sorted(missing)[:6])
        print(f"WARNING: {len(missing)} char(s) not in the charset, rendered as space: {names}")

    im = Image.open(args.image).convert("RGB").resize((GW * CW, GH * CH), Image.LANCZOS)
    img = torch.from_numpy(np.asarray(im, np.float32) / 255.0)                    # (H,W,3)
    cells = img.reshape(GH, CH, GW, CW, 3).permute(0, 2, 1, 3, 4).reshape(M, CH * CW, 3)

    mask = ink.reshape(N, CH * CW)[gid]                                           # (M,P) ink=1
    fg, bg = fit_fg_bg(cells, mask, ridge=args.ridge)

    if args.proportion > 0:
        # The plain fit only ever sees the colors the GLYPH MASK lands on: a small bright highlight
        # that the ink happens to miss is absorbed into bg as a faint tint and disappears. The cell's
        # own 2-means split knows the real color populations and their areas, so anchor toward it.
        # Convex blend, identical in form to the ridge term -- t=0 is the pure fit, t=1 the populations.
        from unicasso.engine.color import decompose
        d = decompose(img, GH, GW, CH, CW)
        pf, pb = d["fg"], d["bg"]
        # Orientation: 2-means calls the MINORITY cluster fg, but the placed glyph may have landed
        # on the majority color. Pair each population color with the fitted value it is nearer to,
        # otherwise the blend drags fg and bg toward each other and muddies the cell.
        keep = ((fg - pf).pow(2).sum(-1) + (bg - pb).pow(2).sum(-1)) <= \
               ((fg - pb).pow(2).sum(-1) + (bg - pf).pow(2).sum(-1))
        pf2 = torch.where(keep[:, None], pf, pb)
        pb2 = torch.where(keep[:, None], pb, pf)
        t = args.proportion / (1.0 + args.proportion)
        fg = (1 - t) * fg + t * pf2
        bg = (1 - t) * bg + t * pb2
        print(f"proportion anchor: t={t:.3f} toward per-cell 2-means populations "
              f"({int((~keep).sum())}/{M} cells needed a fg/bg swap)")

    if args.sat != 1.0:
        for t in (fg, bg):
            g = t.mean(-1, keepdim=True)
            t.copy_((g + (t - g) * args.sat).clamp(0, 1))
    if args.min_contrast > 0:
        lum = lambda t: 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]
        gap = lum(fg) - lum(bg)
        need = args.min_contrast - gap.abs()
        push = torch.where(gap <= 0, -1.0, 1.0) * need.clamp_min(0)
        fg = (fg + push[:, None]).clamp(0, 1)

    rec = bg[:, None, :] + (fg - bg)[:, None, :] * mask[:, :, None]                # (M,P,3)
    out = rec.view(GH, GW, CH, CW, 3).permute(0, 2, 1, 3, 4).reshape(GH * CH, GW * CW, 3)
    resid = (out - img).abs().mean()
    n_ink = int((mask.sum(1) > 0).sum())
    print(f"{GH}x{GW} cells ({n_ink} with ink, {M - n_ink} blank), "
          f"refit residual (mean |err|): {float(resid):.4f}")

    Image.fromarray((out.clamp(0, 1).numpy() * 255).astype(np.uint8), "RGB").save(args.out_png)
    print(f"Saved {args.out_png}")

    ans = args.out_ans or (os.path.splitext(args.out_png)[0] + ".ans")
    f8 = (fg.clamp(0, 1).view(GH, GW, 3).numpy() * 255).astype(np.uint8)
    b8 = (bg.clamp(0, 1).view(GH, GW, 3).numpy() * 255).astype(np.uint8)
    with open(ans, "w", encoding="utf-8") as fh:
        for i in range(GH):
            fh.write("".join(
                f"\x1b[38;2;{f8[i, j, 0]};{f8[i, j, 1]};{f8[i, j, 2]}m"
                f"\x1b[48;2;{b8[i, j, 0]};{b8[i, j, 1]};{b8[i, j, 2]}m" + lines[i][j]
                for j in range(GW)) + "\x1b[0m\n")
    print(f"Saved {ans}")

    if args.compare:
        other = Image.open(args.compare).convert("RGB").resize((GW * CW, GH * CH), Image.LANCZOS)
        panel = Image.new("RGB", (GW * CW * 3 + 24, GH * CH), (255, 255, 255))
        panel.paste(im, (0, 0))
        panel.paste(other, (GW * CW + 12, 0))
        panel.paste(Image.open(args.out_png), (2 * (GW * CW + 12), 0))
        cp = os.path.splitext(args.out_png)[0] + "_compare.png"
        panel.save(cp)
        o = torch.from_numpy(np.asarray(other, np.float32) / 255.0)
        print(f"Saved {cp}  (original | run colors | MSE refit)")
        print(f"  run-color residual {float((o - img).abs().mean()):.4f} "
              f"vs MSE-refit residual {float(resid):.4f}")


if __name__ == "__main__":
    main()
