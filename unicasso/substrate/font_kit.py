"""Font onboarding kit: calibrate a terminal font into the pipeline's cell raster.

    python -m unicasso.substrate.font_kit --out kits/sfmono     # default font: SF Mono @ 14pt
    python -m unicasso.substrate.font_kit --font /path/f.otf --terminal-size 16 --out kits/mykit/

Does, in order:
  1. CALIBRATE: model Terminal.app's cell at --terminal-size (cell_w = round(advance),
     cell_h = ceil(ascent)+ceil(descent) points -- measured behavior) to pick the
     raster cell shape (default 2px/pt = the retina pixel grid), then fit the font
     width-exactly (advance == cell width, line-box top == cell top; isotropic), and
     verify seam continuity of box glyphs and block h-touch across tiled cells.
  2. EXTRACT: every cmap'd codepoint minus traps -- control/format chars, combining marks
     (zero-width, would overdraw), zero-advance glyphs, non-space whitespace.
  3. RENDER + FILTER: raster all candidates on an overdraw canvas; measure ink, clipping
     (ink outside the cell box) and bitmap-identity duplicates (keep lowest codepoint).
  4. EMIT (to --out):
       profile.json            font path/size/offset/cell -- consumed by glyphs.load_glyphs
       candidates.tsv          cp / char / block / name / ink / clip% / dup_of / recommended
       charset_recommended.txt the suggested working set, one char per line (curate = delete lines)
       charset_all.txt         every kept candidate, same format
       sheet_NN.png            contact sheets (cell border + hex label; green = recommended)
       seam_test.png           3x3 tilings of | - + and full block, for continuity eyeballing
"""
import argparse
import json
import os
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = ("/System/Applications/Utilities/Terminal.app/Contents/Resources/Fonts/"
                "SF-Mono-Regular.otf")

# (start, end, name) -- compact block table covering what monospace fonts ship.
BLOCKS = [
    (0x0020, 0x007E, "ASCII"),
    (0x00A0, 0x00FF, "Latin-1"),
    (0x0100, 0x017F, "Latin Ext-A"),
    (0x0180, 0x024F, "Latin Ext-B"),
    (0x0250, 0x02AF, "IPA"),
    (0x02B0, 0x02FF, "Spacing Modifiers"),
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x1E00, 0x1EFF, "Latin Ext Additional"),
    (0x2000, 0x206F, "General Punctuation"),
    (0x2070, 0x209F, "Super/Subscripts"),
    (0x20A0, 0x20CF, "Currency"),
    (0x2100, 0x214F, "Letterlike"),
    (0x2150, 0x218F, "Number Forms"),
    (0x2190, 0x21FF, "Arrows"),
    (0x2200, 0x22FF, "Math Operators"),
    (0x2300, 0x23FF, "Misc Technical"),
    (0x2400, 0x243F, "Control Pictures"),
    (0x2500, 0x257F, "Box Drawing"),
    (0x2580, 0x259F, "Block Elements"),
    (0x25A0, 0x25FF, "Geometric Shapes"),
    (0x2600, 0x26FF, "Misc Symbols"),
    (0x2700, 0x27BF, "Dingbats"),
    (0x27C0, 0x2BFF, "Arrows/Math Ext"),
    (0xE000, 0xF8FF, "PUA"),
    (0xFB00, 0xFB4F, "Ligatures"),
]

# Blocks whose (clean, non-dup) members go into the recommended working set. Letters and
# digits ride along via ASCII/Latin-1/Greek -- the codebook includes letters; the
# optimizer can exclude them separately at glyph-selection time.
RECOMMENDED_BLOCKS = {
    "ASCII", "Latin-1", "Greek", "General Punctuation", "Super/Subscripts",
    "Letterlike", "Number Forms", "Arrows", "Math Operators", "Misc Technical",
    "Box Drawing", "Block Elements", "Geometric Shapes",
}


def block_of(cp):
    for lo, hi, name in BLOCKS:
        if lo <= cp <= hi:
            return name
    return "Other"


def render_glyph(font, ch, cell_h, cell_w, baseline_y, over=None):
    """Raster ch into an overdraw canvas; return (cell uint8 ink 0..255, clip_frac).

    Placement is TERMINAL-FAITHFUL: baseline-anchored at baseline_y inside the cell
    (= the font's line box aligned to the cell), then CLIPPED to the cell. Box-drawing
    glyphs overdraw the line box by design (so they seam across cells) -- for them a
    nonzero clip_frac is intent, not damage.
    """
    if over is None:
        over = max(cell_h, cell_w)
    img = Image.new("L", (cell_w + 2 * over, cell_h + 2 * over), 255)
    ImageDraw.Draw(img).text((over, over + baseline_y), ch, font=font, fill=0,
                             anchor="ls")
    a = 255 - np.asarray(img, dtype=np.int32)          # ink space
    cell = a[over:over + cell_h, over:over + cell_w]
    total, inside = int(a.sum()), int(cell.sum())
    clip = 0.0 if total == 0 else (total - inside) / total
    return cell.astype(np.uint8), clip


def bbox_of(cell, thresh=128):
    ys, xs = np.nonzero(cell >= thresh)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def terminal_cell_pt(font_path, term_size):
    """Terminal.app's cell for this font at point size term_size.

    Measured behavior (macOS Terminal, verified empirically via scripted window
    resizes at sizes 12-16): cell_w = round(advance), cell_h = ceil(ascent) +
    ceil(descent), each in points. The rounding slack pads the glyph's line box,
    so e.g. SF Mono 14 -> 9x18pt cell around a 8.65x16.71pt line box.
    """
    import math
    from fontTools.ttLib import TTFont
    tt = TTFont(font_path)
    upm = tt["head"].unitsPerEm
    hh = tt["hhea"]
    adv_u = tt["hmtx"][tt.getBestCmap()[0x4D]][0]
    asc_pt = hh.ascent * term_size / upm
    desc_pt = -hh.descent * term_size / upm
    adv_pt = adv_u * term_size / upm
    cw = round(adv_pt)
    ch = math.ceil(asc_pt) + math.ceil(desc_pt)
    return dict(cw=cw, ch=ch, adv_pt=adv_pt, asc_pt=asc_pt, desc_pt=desc_pt,
                asc_ceil=math.ceil(asc_pt), gap=hh.lineGap, upm=upm)


def calibrate(font_path, term_size, cell_h, cell_w):
    """Width-exact isotropic fit of the font into the cell_h x cell_w raster.

    Size is chosen so the ADVANCE exactly equals cell_w (adjacent blocks touch, as
    they visually do in the terminal -- the sub-pixel advance slack is invisible
    there) and the line-box TOP is aligned to the cell top (baseline = ascent), so
    '|'-family glyphs reach row 0 via their designed overdraw and runs connect. The
    vertical rounding slack all sits at the cell bottom; block elements keep their
    real top stripe. Glyphs stay isotropic -- no stretch, no folding.
    """
    from fontTools.ttLib import TTFont
    tc = terminal_cell_pt(font_path, term_size)
    print(f"[calibrate] terminal cell @ {term_size}pt: {tc['cw']}x{tc['ch']}pt "
          f"(advance {tc['adv_pt']:.3f}, line box {tc['asc_pt']:.2f}+{tc['desc_pt']:.2f}"
          f"={tc['asc_pt'] + tc['desc_pt']:.2f}pt, aspect {tc['cw'] / tc['ch']:.4f})")
    if abs(cell_w / cell_h - tc["cw"] / tc["ch"]) > 1e-6:
        print(f"[calibrate] WARNING: raster {cell_w}x{cell_h} aspect differs from the "
              f"terminal cell {tc['cw']}/{tc['ch']} = {tc['cw'] / tc['ch']:.4f}")
    tt = TTFont(font_path)
    upm = tt["head"].unitsPerEm
    hh = tt["hhea"]
    adv_u = tt["hmtx"][tt.getBestCmap()[0x4D]][0]
    size = cell_w * upm / adv_u                     # advance == cell_w exactly
    baseline_y = hh.ascent * size / upm             # line-box top == cell top
    line_px = (hh.ascent - hh.descent) * size / upm
    if line_px > cell_h + 0.5:
        print(f"[calibrate] WARNING: line box {line_px:.1f}px overflows cell_h={cell_h} "
              f"-- descenders will clip harder than in the terminal")
    print(f"[calibrate] raster: size={size:.3f}px, baseline_y={baseline_y:.2f}px, "
          f"line box {line_px:.2f}px of {cell_h} (bottom slack "
          f"{cell_h - line_px:.2f}px), cell {cell_w}x{cell_h}px")

    font = ImageFont.truetype(font_path, size)
    for ch, label in [("█", "FULL BLOCK"), ("│", "BOX VERT"), ("M", "M")]:
        cell, clip = render_glyph(font, ch, cell_h, cell_w, baseline_y)
        bb = bbox_of(cell)
        t, b, l, r = bb
        print(f"[calibrate] '{ch}' {label}: rows {t}..{b} cols {l}..{r} "
              f"of {cell_h}x{cell_w}, clip={clip:.1%}")
    return size, baseline_y


# The paste-test pattern: terminal_test.txt gets exactly these lines, seam_test.png is
# the kit raster of the same lines -- cat one, open the other, they must look alike.
TEST_PATTERN = [
    "██████████  ▄▄▄▄▄▄▄▄▄▄  ░░▒▒▓▓██  │ ║ ┃",
    "██████████  ▀▀▀▀▀▀▀▀▀▀  ░░▒▒▓▓██  │ ║ ┃",
    "██████████  ▄▀▄▀▄▀▄▀▄▀  ░░▒▒▓▓██  │ ║ ┃",
    "──────────  ┌──┬──┐  ╔══╦══╗  ┼┼┼┼┼┼┼",
    "            │  │  │  ║  ║  ║  ┼┼┼┼┼┼┼",
    "            └──┴──┘  ╚══╩══╝  ┼┼┼┼┼┼┼",
]


def seam_report(font, cell_h, cell_w, baseline_y, out_png):
    """Render TEST_PATTERN (the terminal_test.txt content) + numeric seam checks."""
    msgs = []
    for ch in "│─█":
        cell, _ = render_glyph(font, ch, cell_h, cell_w, baseline_y)
        if ch == "│":
            top, bot = cell[0], cell[-1]
            ok = (top >= 128).any() and (bot >= 128).any()
            msgs.append(f"  '│' seam: top-row ink {(top >= 128).sum()}px, "
                        f"bottom-row ink {(bot >= 128).sum()}px -> {'OK' if ok else 'BROKEN'}")
        elif ch == "─":
            lft, rgt = cell[:, 0], cell[:, -1]
            ok = (lft >= 128).any() and (rgt >= 128).any()
            msgs.append(f"  '─' seam: left-col ink {(lft >= 128).sum()}px, "
                        f"right-col ink {(rgt >= 128).sum()}px -> {'OK' if ok else 'BROKEN'}")
        else:
            bb = bbox_of(cell)
            gap_top, gap_bot = bb[0], cell_h - 1 - bb[1]
            lft, rgt = (cell[:, 0] >= 128).sum(), (cell[:, -1] >= 128).sum()
            msgs.append(f"  '█' h-touch: left-col ink {lft}px, right-col ink {rgt}px -> "
                        f"{'OK' if lft and rgt else 'BROKEN'}; v-stripe {gap_top}px top / "
                        f"{gap_bot}px bottom (the terminal really shows this)")

    ncols = max(len(l) for l in TEST_PATTERN)
    canvas = np.zeros((cell_h * len(TEST_PATTERN), cell_w * ncols), np.uint8)
    cache = {}
    for r, line in enumerate(TEST_PATTERN):
        for c, ch in enumerate(line.ljust(ncols)):
            if ch not in cache:
                cache[ch] = render_glyph(font, ch, cell_h, cell_w, baseline_y)[0]
            canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cache[ch]
    Image.fromarray(255 - canvas).save(out_png)
    print("[seam]")
    print("\n".join(msgs))
    print(f"[seam] -> {out_png} (raster of terminal_test.txt -- compare with `cat`)")


def extract_candidates(font_path):
    from fontTools.ttLib import TTFont
    tt = TTFont(font_path)
    cmap = tt.getBestCmap()
    hmtx = tt["hmtx"]
    keep, skipped = [], {}
    for cp in sorted(cmap):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if cp == 0x20:
            keep.append(cp)
            continue
        if cat in ("Cc", "Cf", "Mn", "Me", "Mc", "Zl", "Zp", "Zs", "Cs"):
            skipped[cat] = skipped.get(cat, 0) + 1
            continue
        if hmtx[cmap[cp]][0] == 0:
            skipped["zero-adv"] = skipped.get("zero-adv", 0) + 1
            continue
        keep.append(cp)
    print(f"[extract] {len(cmap)} cmap'd -> {len(keep)} candidates "
          f"(skipped: {', '.join(f'{k}:{v}' for k, v in sorted(skipped.items()))})")
    return keep


def build_sheets(rows, cell_h, cell_w, out_dir, font_path, scale=4, cols=16,
                 max_rows_per_sheet=26):
    """rows: list of dicts with keys cp/char/cell/recommended/dup_of (dups excluded)."""
    label_h = 14
    tile_w, tile_h = cell_w * scale + 6, cell_h * scale + label_h + 6
    try:
        label_font = ImageFont.truetype(font_path, 11)
    except OSError:
        label_font = ImageFont.load_default()

    # group by block, block header bands
    items = []                                   # ('hdr', name) | ('glyph', row)
    cur = None
    for r in rows:
        b = block_of(r["cp"])
        if b != cur:
            items.append(("hdr", b))
            cur = b
        items.append(("glyph", r))

    # paginate: walk items into grid positions
    sheets, grid, y = [], [], 0
    x = 0
    row_buf = []
    def flush_row():
        nonlocal row_buf, y, grid
        if row_buf:
            grid.append(("row", row_buf))
            row_buf = []
            y += 1
    for kind, val in items:
        if kind == "hdr":
            flush_row()
            grid.append(("hdr", val))
            y += 1
        else:
            row_buf.append(val)
            if len(row_buf) == cols:
                flush_row()
        if y >= max_rows_per_sheet:
            flush_row()
            sheets.append(grid)
            grid, y = [], 0
    flush_row()
    if grid:
        sheets.append(grid)

    hdr_h = 22
    paths = []
    for si, grid in enumerate(sheets):
        H = sum(hdr_h if k == "hdr" else tile_h for k, _ in grid) + 8
        W = cols * tile_w + 8
        img = Image.new("RGB", (W, H), (250, 250, 250))
        d = ImageDraw.Draw(img)
        yy = 4
        for kind, val in grid:
            if kind == "hdr":
                d.rectangle([0, yy, W, yy + hdr_h - 4], fill=(230, 234, 245))
                d.text((8, yy + 3), val, fill=(40, 40, 80), font=label_font)
                yy += hdr_h
                continue
            for i, r in enumerate(val):
                xx = 4 + i * tile_w
                cell = 255 - r["cell"]
                big = np.asarray(Image.fromarray(cell).resize(
                    (cell_w * scale, cell_h * scale), Image.NEAREST))
                img.paste(Image.fromarray(big).convert("RGB"), (xx + 3, yy + 3))
                d.rectangle([xx + 2, yy + 2, xx + 3 + cell_w * scale,
                             yy + 3 + cell_h * scale], outline=(140, 170, 230))
                col = (20, 130, 40) if r["recommended"] else (150, 150, 150)
                tag = f"{r['cp']:04X}"
                if r["clip"] > 0.35:
                    tag += "!"
                d.text((xx + 3, yy + 4 + cell_h * scale), tag, fill=col, font=label_font)
            yy += tile_h
        p = os.path.join(out_dir, f"sheet_{si:02d}.png")
        img.save(p)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description="font onboarding kit")
    ap.add_argument("--font", default=DEFAULT_FONT)
    ap.add_argument("--terminal-size", type=float, default=14,
                    help="the terminal's font point size; sets the rounded cell shape")
    ap.add_argument("--cell", nargs=2, type=int, default=None, metavar=("H", "W"),
                    help="raster cell px; default = 2px/pt of the terminal cell "
                         "(the exact retina pixel grid)")
    ap.add_argument("--out", required=True, help="kit output dir, e.g. kits/scp (refusing a default: it would overwrite a shipped kit)")
    ap.add_argument("--clip-flag", type=float, default=0.35,
                    help="flag (and exclude from recommended) glyphs losing more ink than "
                         "this to cell clipping; box glyphs sit ~13%% by design")
    args = ap.parse_args()
    if args.cell is None:
        tc = terminal_cell_pt(args.font, args.terminal_size)
        cell_h, cell_w = 2 * tc["ch"], 2 * tc["cw"]
    else:
        cell_h, cell_w = args.cell
    os.makedirs(args.out, exist_ok=True)
    print(f"[kit] {args.font}\n[kit] cell {cell_w}x{cell_h}px (aspect {cell_w / cell_h:.4f})")

    size, baseline_y = calibrate(args.font, args.terminal_size, cell_h, cell_w)
    font = ImageFont.truetype(args.font, size)
    seam_report(font, cell_h, cell_w, baseline_y, os.path.join(args.out, "seam_test.png"))

    cps = extract_candidates(args.font)
    rows, seen = [], {}
    for cp in cps:
        ch = chr(cp)
        cell, clip = render_glyph(font, ch, cell_h, cell_w, baseline_y)
        ink = float((cell.astype(np.float32) / 255.0).mean())
        if ink == 0.0 and cp != 0x20:
            continue                                # renders blank (tofu-less empties)
        key = cell.tobytes()
        dup_of = seen.get(key)
        if dup_of is None and cp != 0x20:
            seen[key] = cp
        rows.append(dict(cp=cp, char=ch, cell=cell, ink=ink, clip=clip, dup_of=dup_of,
                         block=block_of(cp),
                         name=unicodedata.name(ch, f"U+{cp:04X}")))
    uniq = [r for r in rows if r["dup_of"] is None]
    for r in uniq:
        r["recommended"] = (r["block"] in RECOMMENDED_BLOCKS and r["clip"] <= args.clip_flag)
    rec = [r for r in uniq if r["recommended"]]
    n_clip = sum(1 for r in uniq if r["clip"] > args.clip_flag)
    print(f"[filter] {len(rows)} rendered, {len(rows) - len(uniq)} bitmap-dups folded, "
          f"{n_clip} clip-flagged (> {args.clip_flag:.0%}) -> {len(uniq)} unique, "
          f"{len(rec)} recommended")

    # per-block summary
    from collections import Counter
    cu, cr = Counter(r["block"] for r in uniq), Counter(r["block"] for r in rec)
    print(f"[blocks] {'block':<22}{'unique':>7}{'recomm.':>9}")
    for lo, hi, name in BLOCKS + [(0, 0, "Other")]:
        if cu.get(name):
            print(f"[blocks] {name:<22}{cu[name]:>7}{cr.get(name, 0):>9}")

    # emits
    with open(os.path.join(args.out, "candidates.tsv"), "w", encoding="utf-8") as f:
        f.write("cp\tchar\tblock\tname\tink\tclip\tdup_of\trecommended\n")
        for r in rows:
            dup = "" if r["dup_of"] is None else f"U+{r['dup_of']:04X}"
            f.write(f"U+{r['cp']:04X}\t{r['char']}\t{r['block']}\t{r['name']}\t"
                    f"{r['ink']:.4f}\t{r['clip']:.4f}\t{dup}\t"
                    f"{int(r.get('recommended', False))}\n")
    for fname, sel in [("charset_all.txt", uniq), ("charset_recommended.txt", rec)]:
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            for r in sel:
                f.write(f"{r['char']}\tU+{r['cp']:04X}\t{r['name']}\n")
    curated = os.path.join(args.out, "charset_curated.txt")
    with open(os.path.join(args.out, "profile.json"), "w") as f:
        json.dump(dict(font=args.font, terminal_size=args.terminal_size,
                       size=size, baseline_y=baseline_y,
                       cell_h=cell_h, cell_w=cell_w,
                       charset_file=curated if os.path.exists(curated)
                       else os.path.join(args.out, "charset_recommended.txt")), f,
                  indent=2)

    # paste test: cat this in the real terminal and compare with seam_test.png (the
    # kit raster of the SAME lines) -- box lines should connect, block rows stripe.
    with open(os.path.join(args.out, "terminal_test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(TEST_PATTERN) + "\n")

    paths = build_sheets(uniq, cell_h, cell_w, args.out, args.font)
    print(f"[emit] profile.json, candidates.tsv, charset_all.txt ({len(uniq)}), "
          f"charset_recommended.txt ({len(rec)}), terminal_test.txt, "
          f"{len(paths)} sheets -> {args.out}/")


if __name__ == "__main__":
    main()
