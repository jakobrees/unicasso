"""Let CLIP choose the fg/bg CONTRAST on top of the closed-form MSE color fit.

The glyphs and the fitted colors are both fixed. The only free parameter is a contrast
multiplier that pushes fg and bg apart around the cell's midpoint:

    mid   = (fg0 + bg0) / 2                       -- the cell's mean color, MSE-optimal
    fg(k) = mid + k * (fg0 - mid)
    bg(k) = mid + k * (bg0 - mid)

k = 1 is the pure MSE fit; k = 0 collapses the cell to a flat block of its mean color;
k > 1 exaggerates the glyph against its background. Crucially the cell's MEAN color is
invariant in k, so this cannot change hue or brightness -- it can only trade pixel error
for glyph legibility, which is exactly the axis where MSE and perception disagree.

    --per-cell off : one global k for the whole image
    --per-cell on  : k per cell (M free scalars) -- CLIP decides WHERE glyphs should read

Pure CLIP by default (--recon-weight 0): the point is to see what the perceptual metric
wants when pixel error is not holding it back.

    python -m unicasso.output.contrast_opt --txt OUT.txt --image ORIG.jpg \
        --out-png k.png --iters 250 [--per-cell] [--tv 0.01]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from unicasso.engine.color import fit_fg_bg
from unicasso.output.recolor import load_txt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--txt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-ans", default=None)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--per-cell", action="store_true", help="one k per cell instead of one global k")
    ap.add_argument("--k-init", type=float, default=1.0, help="1.0 = start at the pure MSE fit")
    ap.add_argument("--k-max", type=float, default=4.0, help="clamp on k (0 = flat cell)")
    ap.add_argument("--tv", type=float, default=0.0,
                    help="(per-cell) total-variation smoothness on the k field")
    ap.add_argument("--recon-weight", type=float, default=0.0,
                    help="pixel anchor; 0 = pure CLIP (the interesting case)")
    ap.add_argument("--clip-weight", type=float, default=1.0)
    ap.add_argument("--clip-model", default="RN101")
    ap.add_argument("--clip-pretrained", default="openai")
    ap.add_argument("--clip-adapter", default=None)
    ap.add_argument("--clip-aug", type=int, default=8)
    ap.add_argument("--clip-crop-scale", type=float, nargs=2, default=(0.4, 0.9))
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--color-source", default="blend", choices=["fit", "cluster", "blend"],
                    help="base fg0/bg0: plain MSE fit, decompose cluster colors, or the "
                         "engine-default mix (--color-cluster-alpha)")
    ap.add_argument("--color-cluster-alpha", type=float, default=0.5)
    ap.add_argument("--color-bg-dist-pow", type=float, default=1.0,
                    help="bg-fit votes ~ (distance to the glyph's ink)^pow; 0 = plain")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    from unicasso.substrate import glyphs as G
    ink, chars = G.load_glyphs(device=torch.device("cpu"))
    ink = ink.squeeze(1) if ink.dim() == 4 else ink
    N, CH, CW = ink.shape
    idx_of = {c: i for i, c in enumerate(chars)}

    lines, GH, GW = load_txt(args.txt)
    M = GH * GW
    gid = torch.tensor([idx_of.get(ch, 0) for row in lines for ch in row], dtype=torch.long)

    im = Image.open(args.image).convert("RGB").resize((GW * CW, GH * CH), Image.LANCZOS)
    img = torch.from_numpy(np.asarray(im, np.float32) / 255.0).to(dev)
    cells = img.reshape(GH, CH, GW, CW, 3).permute(0, 2, 1, 3, 4).reshape(M, CH * CW, 3)
    mask = ink.reshape(N, CH * CW)[gid].to(dev)                          # (M,P) ink=1
    bg_w = None
    if args.color_bg_dist_pow > 0:
        from unicasso.engine.color import glyph_bg_dist
        bg_w = glyph_bg_dist(ink.reshape(N, CH * CW), CH, CW).to(dev)[gid] \
            .pow(args.color_bg_dist_pow)
    fg0, bg0 = fit_fg_bg(cells, mask, ridge=args.ridge, bg_w=bg_w)       # the MSE optimum
    if args.color_source != "fit":
        # engine-default shipped colors: mix in the decomposition's own cluster colors
        from unicasso.engine.color import decompose
        dec = decompose(img, GH, GW, CH, CW)
        a = 1.0 if args.color_source == "cluster" else args.color_cluster_alpha
        fg0 = a * dec["fg"].to(dev) + (1 - a) * fg0
        bg0 = a * dec["bg"].to(dev) + (1 - a) * bg0
    mid = 0.5 * (fg0 + bg0)
    dfg, dbg = fg0 - mid, bg0 - mid

    k = nn.Parameter(torch.full((M,) if args.per_cell else (1,), float(args.k_init), device=dev))
    opt = torch.optim.AdamW([k], lr=args.lr)

    from unicasso.engine.clip_loss import CLIPPerceptualLoss
    print(f"Loading CLIP ({args.clip_model}) ...")
    clipper = CLIPPerceptualLoss(dev, model_name=args.clip_model, pretrained=args.clip_pretrained,
                                 n_aug=args.clip_aug, crop_scale=tuple(args.clip_crop_scale),
                                 adapter=args.clip_adapter)

    def render(kv):
        kk = kv[:, None] if args.per_cell else kv
        fg = mid + kk * dfg
        bg = mid + kk * dbg
        fg = fg + (fg.clamp(0, 1) - fg).detach()       # STE: ship valid colors, keep the gradient
        bg = bg + (bg.clamp(0, 1) - bg).detach()
        cell = bg[:, None, :] + (fg - bg)[:, None, :] * mask[:, :, None]
        return cell.view(GH, GW, CH, CW, 3).permute(0, 2, 1, 3, 4).reshape(GH * CH, GW * CW, 3)

    with torch.no_grad():
        r0 = render(k.detach())
        base_clip = float(clipper(r0, img))
        base_res = float((r0 - img).abs().mean())
    print(f"start (k={args.k_init}): clip {base_clip:.4f}  residual {base_res:.4f}")

    bar = tqdm(range(args.iters))
    for it in bar:
        r = render(k)
        loss = args.clip_weight * clipper(r, img)
        post = {"clip": f"{float(loss):.4f}"}
        if args.recon_weight > 0:
            rc = ((r - img) ** 2).mean()
            loss = loss + args.recon_weight * rc
            post["recon"] = f"{float(rc):.4f}"
        if args.per_cell and args.tv > 0:
            kf = k.view(GH, GW)
            tv = (kf[:, 1:] - kf[:, :-1]).pow(2).mean() + (kf[1:] - kf[:-1]).pow(2).mean()
            loss = loss + args.tv * tv
            post["tv"] = f"{float(tv):.4f}"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            k.data.clamp_(0.0, args.k_max)
        post["k"] = f"{float(k.mean()):.3f}"
        bar.set_postfix(post)

    with torch.no_grad():
        r1 = render(k)
        end_clip = float(clipper(r1, img))
        end_res = float((r1 - img).abs().mean())
        kv = k.detach().cpu()
    print(f"end: clip {base_clip:.4f} -> {end_clip:.4f}   residual {base_res:.4f} -> {end_res:.4f}")
    if args.per_cell:
        q = torch.quantile(kv, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        print(f"k per cell: mean {float(kv.mean()):.3f}  "
              f"p05 {q[0]:.2f} p25 {q[1]:.2f} med {q[2]:.2f} p75 {q[3]:.2f} p95 {q[4]:.2f}  "
              f"({int((kv < 0.5).sum())} cells pushed toward flat, {int((kv > 1.5).sum())} exaggerated)")
    else:
        print(f"global k: {float(kv):.4f}")

    Image.fromarray((r1.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), "RGB").save(args.out_png)
    print(f"Saved {args.out_png}")

    with torch.no_grad():
        kk = kv[:, None].to(dev) if args.per_cell else kv.to(dev)
        fgf = (mid + kk * dfg).clamp(0, 1).cpu().numpy()
        bgf = (mid + kk * dbg).clamp(0, 1).cpu().numpy()
    ans = args.out_ans or (os.path.splitext(args.out_png)[0] + ".ans")
    f8 = (fgf.reshape(GH, GW, 3) * 255).astype(np.uint8)
    b8 = (bgf.reshape(GH, GW, 3) * 255).astype(np.uint8)
    with open(ans, "w", encoding="utf-8") as fh:
        for i in range(GH):
            fh.write("".join(
                f"\x1b[38;2;{f8[i,j,0]};{f8[i,j,1]};{f8[i,j,2]}m"
                f"\x1b[48;2;{b8[i,j,0]};{b8[i,j,1]};{b8[i,j,2]}m" + lines[i][j]
                for j in range(GW)) + "\x1b[0m\n")
    print(f"Saved {ans}")

    if args.per_cell:                      # the k field itself: where CLIP wants glyphs to read
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        m = ax.imshow(kv.view(GH, GW).numpy(), cmap="magma", vmin=0, vmax=max(2.0, float(kv.max())))
        ax.set_title("per-cell contrast k (bright = CLIP wants the glyph to read)")
        fig.colorbar(m, ax=ax, fraction=0.046)
        kp = os.path.splitext(args.out_png)[0] + "_kfield.png"
        fig.savefig(kp, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"Saved {kp}")


if __name__ == "__main__":
    main()
