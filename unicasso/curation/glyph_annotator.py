"""Interactive glyph annotator -> glyph_labels.json for train_vae.

Two jobs, one tool, over a grid of all glyphs:
  * QUALITY override: the computed quality (coh^(1-lin_w)*lin^lin_w) ranks some good glyphs low.
    Hover a glyph and scroll (or [ / ]) to set its quality by hand; r resets to computed. Cell
    background is tinted by the EFFECTIVE quality (red=low -> green=high) so the ranking is visible.
  * CATEGORY tags: visually-similar groups (vert/horiz/diag/curve/box/...). Hover + press a digit to
    toggle membership; shown as a colored barcode along the top of each cell. (Organizes the latent
    space -- a relational pull term, NOT a frequency lever.)

Saves {categories:[...], glyphs:{char:{quality, manual, categories:[...]}}} -> resumable.

  python -m unicasso.curation.glyph_annotator                       # pad 4, default categories
  python -m unicasso.curation.glyph_annotator --categories vert,horiz,diag,curve,box,corner,dot,dense
"""
import argparse
import json
import os

import numpy as np
import torch

from unicasso.substrate import glyphs as G
from unicasso.substrate import orientation as O

DEFAULT_CATEGORIES = ["vert", "horiz", "diag", "curve", "box", "corner", "dot", "dense", "letter", "sparse"]


def compute_quality(ink, sigma, lin_w):
    """Mirror train_vae's quality: coh^(1-lin_w) * lin^lin_w on the (padded) glyph rasters."""
    _, g_coh, _ = O.glyph_orientations(ink, sigma=sigma)
    g_elong, _, _ = O.glyph_linearity(ink)
    cc = g_coh.clamp(0, 1).cpu().numpy()
    ll = g_elong.clamp(0, 1).cpu().numpy()
    return (cc ** (1.0 - lin_w) * ll ** lin_w).astype(np.float32)


def kit_dir():
    """The active font profile's kit directory (the bundled dejavu kit when unset)."""
    prof = os.environ.get("GLYPHVAE_FONT") or "dejavu"
    path = G.PROFILE_ALIASES.get(prof, prof)
    return os.path.dirname(os.path.join(G.REPO_ROOT, path)
                           if not os.path.isabs(path) else path)


def load_kit_meta(kit):
    """char -> {ink, clip, name, block} from the kit's candidates.tsv (unique rows)."""
    meta = {}
    tsv = os.path.join(kit, "candidates.tsv")
    if not os.path.exists(tsv):
        return meta
    with open(tsv, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            v = dict(zip(header, line.rstrip("\n").split("\t")))
            if not v["dup_of"]:
                meta[v["char"] or " "] = dict(ink=float(v["ink"]), clip=float(v["clip"]),
                                              name=v["name"], block=v["block"])
    return meta


def main():
    p = argparse.ArgumentParser(description="interactive glyph quality + category annotator")
    p.add_argument("--out", default=None,
                   help="labels json; default = <kit>/glyph_labels.json for the active "
                        "font profile (GLYPHVAE_FONT, dejavu when unset)")
    p.add_argument("--init-from", default=None,
                   help="seed manual qualities + categories from another labels json for "
                        "overlapping chars (default: the sfmono kit's glyph_labels.json "
                        "when annotating a different kit for the first time)")
    p.add_argument("--pad", type=int, default=4, help="match train_vae --pad so computed quality lines up")
    p.add_argument("--struct-sigma", type=float, default=0.0)
    p.add_argument("--lin-weight", type=float, default=0.70, help="computed-quality linearity exponent")
    p.add_argument("--cols", type=int, default=0, help="columns (0 = auto from --aspect)")
    p.add_argument("--aspect", type=float, default=2.5, help="grid width:height (5:2 = 2.5); accounts for the 2:1 cells")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                   help="comma-separated category names (max 10 -> hotkeys 1..9,0)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()][:10]

    kit = kit_dir()
    meta = load_kit_meta(kit) if kit else {}
    classic_labels = os.path.join(G.REPO_ROOT, "kits", "sfmono", "glyph_labels.json")
    if args.out is None:
        args.out = os.path.join(kit, "glyph_labels.json") if kit else classic_labels

    ink, chars = G.load_glyphs(device=args.device, pad=args.pad)        # (N,1,H,W) 1=ink
    N = len(chars)
    npink = ink[:, 0].cpu().numpy()                                     # (N,H,W) 1=ink
    H, W = npink.shape[-2:]
    q_computed = compute_quality(ink, args.struct_sigma, args.lin_weight)
    q_manual = np.full(N, np.nan, np.float32)                           # NaN = use computed
    cat_mat = np.zeros((N, len(cats)), bool)                            # membership

    # Seed / resume (match by char): first --init-from (another font's labels), then
    # --out on top (this file's own previous session wins).
    idx = {c: i for i, c in enumerate(chars)}
    init_from = args.init_from
    if init_from is None and kit and os.path.abspath(args.out) != os.path.abspath(classic_labels) \
            and os.path.exists(classic_labels):
        init_from = classic_labels
    sources = []
    if init_from and os.path.exists(init_from) \
            and os.path.abspath(init_from) != os.path.abspath(args.out):
        sources.append(("seeded", init_from))
    if os.path.exists(args.out):
        sources.append(("resumed", args.out))
    n_seed = 0
    for verb, src in sources:
        with open(src) as f:
            prev = json.load(f)
        for ch, rec in prev.get("glyphs", {}).items():
            if ch not in idx:
                continue
            i = idx[ch]
            if rec.get("manual"):
                q_manual[i] = rec.get("quality", q_computed[i])
                n_seed += verb == "seeded"
            for cn in rec.get("categories", []):
                if cn in cats:
                    cat_mat[i, cats.index(cn)] = True
        print(f"{verb} from {src}")
    if n_seed:
        print(f"  ({n_seed} manual qualities carried over -- review, they were set on the old font)")

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for km in ("save", "home", "back", "forward", "pan", "zoom", "yscale", "xscale"):
        matplotlib.rcParams[f"keymap.{km}"] = []
    cmap = plt.get_cmap("RdYlGn")
    catcmap = plt.get_cmap("tab10")
    catcol = [np.array(catcmap(j)[:3]) for j in range(len(cats))]
    gap = 1
    CH, CW = H + gap, W + gap
    # cols so the composite is ~--aspect (w:h): (cols*CW)/(rows*CH)=aspect, rows~=N/cols
    cols = args.cols if args.cols > 0 else max(1, round((args.aspect * N * CH / CW) ** 0.5))
    rows = (N + cols - 1) // cols

    def q_eff(i):
        return q_computed[i] if np.isnan(q_manual[i]) else q_manual[i]

    def build():
        img = np.full((rows * CH, cols * CW, 3), 0.25, np.float32)      # gray separators
        for n in range(N):
            r, c = divmod(n, cols)
            y0, x0 = r * CH, c * CW
            bg = np.array(cmap(float(q_eff(n)))[:3])
            cell = bg[None, None, :] * (1.0 - npink[n][..., None])      # ink -> black
            tb = max(2, H // 10)                                        # category barcode height
            if cat_mat[n].any():
                seg = max(1, W // len(cats))
                for j in range(len(cats)):
                    if cat_mat[n, j]:
                        cell[:tb, j * seg:(j + 1) * seg, :] = catcol[j]
            img[y0:y0 + H, x0:x0 + W] = cell
        return img

    fig, ax = plt.subplots(figsize=(15, 15 / args.aspect))
    im = ax.imshow(build(), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    hl = Rectangle((0, 0), W, H, fill=False, edgecolor="cyan", lw=2)
    ax.add_patch(hl)
    legend = "  ".join(f"{(j + 1) % 10}:{cats[j]}" for j in range(len(cats)))
    fig.suptitle(f"hover+scroll/[ ] quality | r reset | digit toggle category | s save\n{legend}",
                 fontsize=9)
    status = ax.set_title("", fontsize=9)
    state = {"hover": None}

    def cell_at(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return None
        c = int(ev.xdata // CW); r = int(ev.ydata // CH)
        if not (0 <= c < cols and 0 <= r < rows):
            return None
        if (ev.xdata - c * CW) >= W or (ev.ydata - r * CH) >= H:
            return None
        n = r * cols + c
        return n if n < N else None

    def refresh_status(n):
        if n is None:
            status.set_text(""); return
        m = "*" if not np.isnan(q_manual[n]) else ""
        cs = ", ".join(cats[j] for j in range(len(cats)) if cat_mat[n, j]) or "-"
        extra = ""
        md = meta.get(chars[n])
        if md:
            extra = (f"   {md['name']} [{md['block']}]  ink {md['ink']:.3f}"
                     + (f"  clip {md['clip']:.0%}" if md["clip"] > 0.005 else ""))
        status.set_text(f"[{n}] '{chars[n]}' U+{ord(chars[n]):04X}  quality={q_eff(n):.2f}{m}  "
                        f"(computed {q_computed[n]:.2f})  cats: {cs}{extra}")

    def on_move(ev):
        n = cell_at(ev); state["hover"] = n
        if n is not None:
            r, c = divmod(n, cols)
            hl.set_xy((c * CW, r * CH)); hl.set_visible(True)
        else:
            hl.set_visible(False)
        refresh_status(n); fig.canvas.draw_idle()

    def redraw_full():
        im.set_data(build()); fig.canvas.draw_idle()

    def bump(n, d):
        q_manual[n] = float(np.clip(q_eff(n) + d, 0.0, 1.0))

    def on_scroll(ev):
        n = state["hover"]
        if n is None:
            return
        bump(n, 0.05 if ev.button == "up" else -0.05)
        refresh_status(n); redraw_full()

    def on_key(ev):
        n = state["hover"]
        if ev.key == "s":
            save(); return
        if n is None:
            return
        if ev.key == "]":
            bump(n, 0.05); refresh_status(n); redraw_full()
        elif ev.key == "[":
            bump(n, -0.05); refresh_status(n); redraw_full()
        elif ev.key == "r":
            q_manual[n] = np.nan; refresh_status(n); redraw_full()
        elif ev.key in "1234567890":
            j = (int(ev.key) - 1) % 10
            if j < len(cats):
                cat_mat[n, j] = not cat_mat[n, j]; refresh_status(n); redraw_full()

    def save(*_):
        out = {"categories": cats, "glyphs": {}}
        for i, ch in enumerate(chars):
            out["glyphs"][ch] = {
                "quality": round(float(q_eff(i)), 4),
                "manual": bool(not np.isnan(q_manual[i])),
                "categories": [cats[j] for j in range(len(cats)) if cat_mat[i, j]],
            }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        nm = int((~np.isnan(q_manual)).sum()); nc = int(cat_mat.any(1).sum())
        print(f"saved {args.out}  ({nm} manual-quality, {nc} categorized)")

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", save)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
