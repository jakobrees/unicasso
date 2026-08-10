"""Pick which glyphs the VAE symmetry term (--sym-weight) applies to.

    GLYPHVAE_FONT=sfmono python -m unicasso.curation.sym_selector          # -> kits/sfmono/sym_chars.txt

Shows the CURRENT charset (profile) page by page; GREEN frame = included in the
symmetry views (flips + shifts + classifier), RED + X = excluded. Click to toggle.

Initial selection = auto-detected symmetry evidence (unless resuming from --out):
  * h/v mirror twin: flip(bitmap) matches itself or another charset glyph (IoU > --iou)
  * off-center ink: |centroid - center| > --offc (position-informative for the
    translation spectrum: bars, stubs, corners)
  * blanks are never auto-selected (a blank's centroid target is undefined)
Hover shows the evidence: twin char + IoU per flip axis, centroid offset, ink.

Keys:  ←/→ or j/k  page    a  include all on page    x  exclude all on page    s  save
Saves to --out (font_kit format: char<TAB>U+XXXX<TAB>NAME) for train_vae --sym-chars.
"""
import argparse
import os
import unicodedata

import numpy as np
import torch

from unicasso.substrate import glyphs as G
from unicasso.substrate import raster as train


def auto_evidence(ink, iou_thresh, offc_thresh):
    """ink: (N,H,W) 0..1. Returns per-glyph dict lists: selected, reason strings."""
    N, H, W = ink.shape
    b = (ink > 0.5).float().reshape(N, -1)                      # binarized (N,P)
    area = b.sum(1)
    fh = (torch.flip(ink, dims=[-1]) > 0.5).float().reshape(N, -1)
    fv = (torch.flip(ink, dims=[-2]) > 0.5).float().reshape(N, -1)

    def best_iou(fl):
        inter = fl @ b.t()                                      # (N,N)
        union = fl.sum(1)[:, None] + area[None, :] - inter
        iou = inter / union.clamp_min(1.0)
        iou[fl.sum(1) == 0] = 0.0                               # blank flips match nothing
        v, j = iou.max(dim=1)
        return v, j

    hv, hj = best_iou(fh)
    vv, vj = best_iou(fv)
    yg = ((torch.arange(H) + 0.5) / H)[None, :, None]
    xg = ((torch.arange(W) + 0.5) / W)[None, None, :]
    m = ink.sum(dim=(1, 2))
    cy = (ink * yg).sum(dim=(1, 2)) / m.clamp_min(1e-6) - 0.5
    cx = (ink * xg).sum(dim=(1, 2)) / m.clamp_min(1e-6) - 0.5
    offc = torch.maximum(cy.abs(), cx.abs())
    blank = m < 2.0
    sel = (~blank) & ((hv > iou_thresh) | (vv > iou_thresh) | (offc > offc_thresh))
    return dict(hv=hv, hj=hj, vv=vv, vj=vj, cy=cy, cx=cx, ink=m / (H * W),
                blank=blank, auto=sel)


def main():
    p = argparse.ArgumentParser(description="pick glyphs for the VAE symmetry term")
    p.add_argument("--out", default=None, help="default <active kit>/sym_chars.txt")
    p.add_argument("--iou", type=float, default=0.75, help="mirror-twin IoU threshold")
    p.add_argument("--offc", type=float, default=0.08, help="off-center centroid threshold")
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--rows", type=int, default=7)
    args = p.parse_args()

    ink_t, chars = G.load_glyphs(device="cpu", pad=0)
    ink = ink_t[:, 0]                                            # (N,H,W) 1=ink
    N = len(chars)
    H, W = ink.shape[-2:]
    ev = auto_evidence(ink, args.iou, args.offc)

    if args.out is None:
        args.out = os.path.join(G.kit_dir(), "sym_chars.txt")
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            selected = {l.split("\t")[0] for l in f if l.rstrip("\n")}
        included = torch.tensor([c in selected for c in chars])
        print(f"[sym-sel] resumed from {args.out}: {int(included.sum())}/{N} included")
    else:
        included = ev["auto"].clone()
        print(f"[sym-sel] auto-init: {int(included.sum())}/{N} included "
              f"(h-twin/v-twin IoU>{args.iou} or off-center>{args.offc}; blanks excluded)")

    import matplotlib
    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    for km in ("keymap.back", "keymap.forward", "keymap.save", "keymap.xscale", "keymap.yscale"):
        if km in plt.rcParams:
            plt.rcParams[km] = []
    from PIL import Image, ImageDraw, ImageFont

    per_page = args.cols * args.rows
    n_pages = (N + per_page - 1) // per_page
    s = args.scale
    label_h = 16
    tile_w, tile_h = W * s + 8, H * s + label_h + 8
    try:
        lab_font = ImageFont.truetype(train.PRINTER_FONT, 11)
    except OSError:
        lab_font = ImageFont.load_default()

    def page_image(pg):
        img = Image.new("RGB", (args.cols * tile_w, args.rows * tile_h), (250, 250, 250))
        d = ImageDraw.Draw(img)
        for i in range(pg * per_page, min((pg + 1) * per_page, N)):
            k = i - pg * per_page
            yy, xx = divmod(k, args.cols)
            xx, yy = xx * tile_w + 4, yy * tile_h + 4
            cell = ((1.0 - ink[i]).numpy() * 255).astype(np.uint8)
            big = Image.fromarray(cell).resize((W * s, H * s), Image.NEAREST)
            img.paste(big.convert("RGB"), (xx, yy))
            d.text((xx, yy + H * s + 2), f"{ord(chars[i]):04X}", fill=(120, 120, 120),
                   font=lab_font)
        return np.asarray(img)

    state = {"page": 0, "hover": None}
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)
    im = ax.imshow(page_image(0))
    ax.set_axis_off()
    frames = []

    def rebuild_frames():
        for f_ in frames:
            f_.remove()
        frames.clear()
        pg = state["page"]
        for i in range(pg * per_page, min((pg + 1) * per_page, N)):
            k = i - pg * per_page
            yy, xx = divmod(k, args.cols)
            x0, y0 = xx * tile_w + 3, yy * tile_h + 3
            ok = bool(included[i])
            rect = mp.Rectangle((x0, y0), W * s + 2, H * s + 2, fill=False,
                                color="#2c8a3d" if ok else "#c03030", lw=2 if ok else 1.2)
            ax.add_patch(rect)
            frames.append(rect)
            if not ok:
                l1, = ax.plot([x0, x0 + W * s + 2], [y0, y0 + H * s + 2],
                              color="#c03030", lw=1, alpha=0.55)
                l2, = ax.plot([x0, x0 + W * s + 2], [y0 + H * s + 2, y0],
                              color="#c03030", lw=1, alpha=0.55)
                frames.extend([l1, l2])

    def show_page():
        im.set_data(page_image(state["page"]))
        rebuild_frames()
        fig.suptitle(f"page {state['page'] + 1}/{n_pages}   included "
                     f"{int(included.sum())}/{N}   "
                     f"[click toggle | ←/→ page | a/x page-all | s save]", fontsize=11)
        fig.canvas.draw_idle()

    def tile_at(event):
        if event.xdata is None or event.inaxes is not ax:
            return None
        c, r = int(event.xdata // tile_w), int(event.ydata // tile_h)
        if not (0 <= c < args.cols and 0 <= r < args.rows):
            return None
        i = state["page"] * per_page + r * args.cols + c
        return i if i < N else None

    def on_click(event):
        i = tile_at(event)
        if i is None:
            return
        included[i] = not bool(included[i])
        show_page()

    def on_move(event):
        i = tile_at(event)
        if i is None or i == state["hover"]:
            return
        state["hover"] = i
        parts = [f"U+{ord(chars[i]):04X} '{chars[i]}'",
                 unicodedata.name(chars[i], ""), f"ink {float(ev['ink'][i]):.3f}"]
        if ev["blank"][i]:
            parts.append("BLANK")
        else:
            parts.append(f"h~'{chars[int(ev['hj'][i])]}'({float(ev['hv'][i]):.2f})")
            parts.append(f"v~'{chars[int(ev['vj'][i])]}'({float(ev['vv'][i]):.2f})")
            parts.append(f"offc({float(ev['cy'][i]):+.2f},{float(ev['cx'][i]):+.2f})")
        parts.append("INCLUDED" if included[i] else "excluded")
        ax.set_title("   ".join(parts), fontsize=10)
        fig.canvas.draw_idle()

    def save(*_):
        keep = [(ord(chars[i]), chars[i]) for i in range(N) if included[i]]
        with open(args.out, "w", encoding="utf-8") as f:
            for cp, ch in sorted(keep):
                f.write(f"{ch}\tU+{cp:04X}\t{unicodedata.name(ch, f'U+{cp:04X}')}\n")
        print(f"[sym-sel] saved {len(keep)} glyphs -> {args.out}")

    def on_key(event):
        if event.key in ("right", "k"):
            state["page"] = (state["page"] + 1) % n_pages
        elif event.key in ("left", "j"):
            state["page"] = (state["page"] - 1) % n_pages
        elif event.key in ("a", "x"):
            pg = state["page"]
            for i in range(pg * per_page, min((pg + 1) * per_page, N)):
                included[i] = event.key == "a"
        elif event.key == "s":
            save()
            return
        else:
            return
        show_page()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show_page()
    plt.show()
    save()


if __name__ == "__main__":
    main()
