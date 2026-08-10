"""Paged click-to-curate glyph selector for a font_kit charset.

    python -m unicasso.curation.glyph_selector                    # active kit (GLYPHVAE_FONT) -> <kit>/charset_curated.txt
    python -m unicasso.curation.glyph_selector --kit kits/sfmono --out custom.txt

Shows every unique glyph of the kit (charset_all.txt) page by page. GREEN frame =
allowed, RED frame + X = disallowed. Click a tile to toggle. Initial selection =
--out if it already exists (resume curating), else the profile's charset_file.

Keys:  ←/→ or j/k  page    a  allow all on page    x  disallow all on page
       B  toggle the hovered glyph's whole Unicode block    s  save
Hover shows cp / name / block / ink / clip. Saves to --out on close (font_kit
format: char<TAB>U+XXXX<TAB>NAME), which profile.json's charset_file can point at.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from unicasso.substrate.font_kit import render_glyph, block_of


def load_kit(kit):
    with open(os.path.join(kit, "profile.json")) as f:
        prof = json.load(f)
    rows = []
    with open(os.path.join(kit, "candidates.tsv"), encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            v = dict(zip(header, line.rstrip("\n").split("\t")))
            if v["dup_of"]:
                continue
            rows.append(dict(cp=int(v["cp"][2:], 16), char=v["char"] or " ",
                             name=v["name"], block=v["block"],
                             ink=float(v["ink"]), clip=float(v["clip"])))
    rows.sort(key=lambda r: r["cp"])
    return prof, rows


def read_charset(path):
    chars = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n"):
                chars.add(line.split("\t")[0])
    return chars


def main():
    ap = argparse.ArgumentParser(description="click-to-curate glyph selector")
    ap.add_argument("--kit", default=None, help="default: the active kit (GLYPHVAE_FONT)")
    ap.add_argument("--out", default=None, help="default <kit>/charset_curated.txt")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--rows", type=int, default=7)
    args = ap.parse_args()
    if args.kit is None:
        from unicasso.substrate import glyphs as G
        args.kit = G.kit_dir()
    out = args.out or os.path.join(args.kit, "charset_curated.txt")

    prof, rows = load_kit(args.kit)
    ch_, cw_ = prof["cell_h"], prof["cell_w"]
    font = ImageFont.truetype(prof["font"], prof["size"])
    for r in rows:
        r["cell"], _ = render_glyph(font, r["char"], ch_, cw_, prof["baseline_y"])

    init = out if os.path.exists(out) else prof["charset_file"]
    allowed = read_charset(init)
    print(f"[selector] {len(rows)} glyphs, {sum(r['char'] in allowed for r in rows)} "
          f"allowed (init from {init})")

    import matplotlib
    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    for km in ("keymap.back", "keymap.forward", "keymap.save", "keymap.xscale",
               "keymap.yscale", "keymap.all_axes" if "keymap.all_axes" in plt.rcParams else "keymap.save"):
        if km in plt.rcParams:
            plt.rcParams[km] = []

    per_page = args.cols * args.rows
    n_pages = (len(rows) + per_page - 1) // per_page
    s = args.scale
    label_h = 16
    tile_w, tile_h = cw_ * s + 8, ch_ * s + label_h + 8
    try:
        lab_font = ImageFont.truetype(prof["font"], 11)
    except OSError:
        lab_font = ImageFont.load_default()

    def page_image(p):
        img = Image.new("RGB", (args.cols * tile_w, args.rows * tile_h), (250, 250, 250))
        d = ImageDraw.Draw(img)
        for i, r in enumerate(rows[p * per_page:(p + 1) * per_page]):
            yy, xx = divmod(i, args.cols)
            xx, yy = xx * tile_w + 4, yy * tile_h + 4
            big = Image.fromarray(255 - r["cell"]).resize((cw_ * s, ch_ * s), Image.NEAREST)
            img.paste(big.convert("RGB"), (xx, yy))
            tag = f"{r['cp']:04X}" + ("!" if r["clip"] > 0.35 else "")
            d.text((xx, yy + ch_ * s + 2), tag, fill=(120, 120, 120), font=lab_font)
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
        p = state["page"]
        for i, r in enumerate(rows[p * per_page:(p + 1) * per_page]):
            yy, xx = divmod(i, args.cols)
            x0, y0 = xx * tile_w + 3, yy * tile_h + 3
            ok = r["char"] in allowed
            col = "#2c8a3d" if ok else "#c03030"
            rect = mp.Rectangle((x0, y0), cw_ * s + 2, ch_ * s + 2, fill=False,
                                color=col, lw=2 if ok else 1.2)
            ax.add_patch(rect)
            frames.append(rect)
            if not ok:
                l1, = ax.plot([x0, x0 + cw_ * s + 2], [y0, y0 + ch_ * s + 2],
                              color="#c03030", lw=1, alpha=0.55)
                l2, = ax.plot([x0, x0 + cw_ * s + 2], [y0 + ch_ * s + 2, y0],
                              color="#c03030", lw=1, alpha=0.55)
                frames.extend([l1, l2])

    def show_page():
        im.set_data(page_image(state["page"]))
        rebuild_frames()
        n_ok = sum(r["char"] in allowed for r in rows)
        fig.suptitle(f"page {state['page'] + 1}/{n_pages}   allowed {n_ok}/{len(rows)}   "
                     f"[click toggle | ←/→ page | a/x page-all | B block | s save]",
                     fontsize=11)
        fig.canvas.draw_idle()

    def tile_at(event):
        if event.xdata is None or event.inaxes is not ax:
            return None
        c, r = int(event.xdata // tile_w), int(event.ydata // tile_h)
        if not (0 <= c < args.cols and 0 <= r < args.rows):
            return None
        i = state["page"] * per_page + r * args.cols + c
        return i if i < len(rows) else None

    def on_click(event):
        i = tile_at(event)
        if i is None:
            return
        r = rows[i]
        if r["char"] in allowed:
            allowed.discard(r["char"])
        else:
            allowed.add(r["char"])
        show_page()

    def on_move(event):
        i = tile_at(event)
        if i is None or i == state["hover"]:
            return
        state["hover"] = i
        r = rows[i]
        ax.set_title(f"U+{r['cp']:04X} '{r['char']}'  {r['name']}   [{r['block']}]   "
                     f"ink {r['ink']:.3f}  clip {r['clip']:.1%}   "
                     f"{'ALLOWED' if r['char'] in allowed else 'disallowed'}",
                     fontsize=10)
        fig.canvas.draw_idle()

    def save():
        keep = sorted((r for r in rows if r["char"] in allowed), key=lambda r: r["cp"])
        with open(out, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(f"{r['char']}\tU+{r['cp']:04X}\t{r['name']}\n")
        print(f"[selector] saved {len(keep)} glyphs -> {out}")

    def on_key(event):
        if event.key in ("right", "k"):
            state["page"] = (state["page"] + 1) % n_pages
        elif event.key in ("left", "j"):
            state["page"] = (state["page"] - 1) % n_pages
        elif event.key in ("a", "x"):
            for r in rows[state["page"] * per_page:(state["page"] + 1) * per_page]:
                (allowed.add if event.key == "a" else allowed.discard)(r["char"])
        elif event.key == "B" and state["hover"] is not None:
            blk = rows[state["hover"]]["block"]
            members = [r for r in rows if r["block"] == blk]
            turn_on = not all(r["char"] in allowed for r in members)
            for r in members:
                (allowed.add if turn_on else allowed.discard)(r["char"])
            print(f"[selector] block '{blk}': {'allowed' if turn_on else 'disallowed'} "
                  f"({len(members)} glyphs)")
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
