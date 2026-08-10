"""Fit the UAC1 order-2 tables on the corpus and emit the shipped binary.

    python -m unicasso.artcode.fit_tables                  # write tables/uac1_sfmono.bin
    python -m unicasso.artcode.fit_tables --report         # held-out size measurement
    python -m unicasso.artcode.fit_tables --report --min-context-count 4

The report is a real measurement, not an entropy estimate: it refits the model on
4/5 of the corpus and runs the actual encoder over the held-out fifth, counting
the bytes that come out. Folds are group-disjoint via data/corpus/groups.json so
variants of the same source photo never straddle the split -- without that the
numbers are optimistic by a wide margin.
"""
import argparse
import collections
import glob
import json
import os
import re

from .codec import canonicalize, encode_text
from .model import build_from_counts

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CHARSET = os.path.join(REPO_ROOT, "kits", "sfmono", "charset_curated.txt")
DEFAULT_PROFILE = os.path.join(REPO_ROOT, "kits", "sfmono", "profile.json")
DEFAULT_CORPUS = os.path.join(REPO_ROOT, "data", "corpus", "txts")
DEFAULT_GROUPS = os.path.join(REPO_ROOT, "data", "corpus", "groups.json")
DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tables", "uac1_sfmono.bin")


def read_charset(path=DEFAULT_CHARSET):
    """One glyph per line, char in the first tab-separated field (font_kit format)."""
    chars = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n"):
                chars.append(line.split("\t")[0])
    return "".join(chars)


def read_cell(path=DEFAULT_PROFILE):
    """(cell_w, cell_h) from a font-kit profile -- the aspect a renderer must match."""
    with open(path) as f:
        p = json.load(f)
    return int(p["cell_w"]), int(p["cell_h"])


def load_corpus(txt_dir=DEFAULT_CORPUS, groups_path=DEFAULT_GROUPS):
    """[(name, group, canonical_text)] -- group keys keep photo variants together."""
    p2g = {}
    if os.path.exists(groups_path):
        with open(groups_path) as f:
            groups = json.load(f)
        for gname, members in groups.items():
            if gname.startswith("_"):
                continue
            for m in members:
                p2g[m] = gname
    out = []
    for path in sorted(glob.glob(os.path.join(txt_dir, "*.txt"))):
        name = os.path.splitext(os.path.basename(path))[0]
        parent = re.sub(r"_w\d+(_clip\d+)?$", "", name)
        with open(path, encoding="utf-8") as f:
            out.append((name, p2g.get(parent, parent), canonicalize(f.read())))
    return out


def count_texts(texts, chars):
    """Raw (order-0, order-1, order-2) counts over the coding alphabet.

    Mirrors the raster-order context walk in codec.encode_text: context values are
    glyph ids, OOV for a literal-escaped cell, BORDER off the top/left edge.
    """
    index = {c: i for i, c in enumerate(chars)}
    g = len(chars)
    esc_lit, oov, border, card = g, g, g + 1, g + 2
    c0 = collections.Counter()
    c1 = collections.defaultdict(collections.Counter)
    c2 = collections.defaultdict(collections.Counter)
    for text in texts:
        lines = text.split("\n")[:-1] if text else []
        w = len(lines[0]) if lines else 0
        prev = [border] * w
        for row in lines:
            left = border
            cur = [0] * w
            for j, ch in enumerate(row):
                sym = index.get(ch)
                if sym is None:
                    c0[esc_lit] += 1
                    c1[left][esc_lit] += 1
                    c2[left * card + prev[j]][esc_lit] += 1
                    cur[j] = oov
                else:
                    c0[sym] += 1
                    c1[left][sym] += 1
                    c2[left * card + prev[j]][sym] += 1
                    cur[j] = sym
                left = cur[j]
            prev = cur
    return c0, c1, c2


def fit(texts, chars, min_context_count=80, cell=(18, 36)):
    c0, c1, c2 = count_texts(texts, chars)
    return build_from_counts(chars, c0, c1, c2, min_context_count=min_context_count,
                             cell_w=cell[0], cell_h=cell[1])


def _pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def report(chars, corpus, min_context_count, folds=5):
    keys = sorted({g for _, g, _ in corpus})
    fold_of = {k: i % folds for i, k in enumerate(keys)}
    sizes, bpc, rows = [], [], []
    for fd in range(folds):
        train = [t for _, g, t in corpus if fold_of[g] != fd]
        test = [(n, t) for n, g, t in corpus if fold_of[g] == fd]
        model = fit(train, chars, min_context_count)
        for name, text in test:
            payload = encode_text(text, chars, model)
            lines = text.split("\n")[:-1]
            cells = len(lines) * (len(lines[0]) if lines else 0)
            sizes.append(len(payload))
            bpc.append(8 * len(payload) / max(cells, 1))
            rows.append((name, len(lines[0]) if lines else 0, len(lines),
                         cells, len(payload)))
    import math
    print(f"\nheld-out, {folds}-fold group-disjoint, real encoder output "
          f"(min_context_count={min_context_count}):")
    print(f"  bits/cell   mean {8*sum(sizes)/sum(r[3] for r in rows):.3f}   "
          f"median {_pct(bpc,.5):.3f}   p90 {_pct(bpc,.9):.3f}")
    print(f"  payload B   min {min(sizes)}   p25 {_pct(sizes,.25)}   "
          f"med {_pct(sizes,.5)}   p75 {_pct(sizes,.75)}   "
          f"p90 {_pct(sizes,.9)}   max {max(sizes)}")
    b32 = [math.ceil(s * 8 / 5) for s in sizes]
    print(f"  base32 chr  min {min(b32)}   med {_pct(b32,.5)}   "
          f"p90 {_pct(b32,.9)}   max {max(b32)}   (QR V40-L alnum ceiling 4296)")
    worst = max(rows, key=lambda r: r[4])
    print(f"  worst piece {worst[0]} {worst[1]}x{worst[2]} -> {worst[4]} B "
          f"({math.ceil(worst[4]*8/5)} base32 chars)")
    return sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--charset", default=DEFAULT_CHARSET)
    ap.add_argument("--profile", default=DEFAULT_PROFILE,
                    help="font-kit profile.json; supplies the cell aspect renderers must match")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--groups", default=DEFAULT_GROUPS)
    ap.add_argument("--out", default=DEFAULT_TABLE)
    ap.add_argument("--min-context-count", type=int, default=80,
                    help="drop contexts seen fewer times than this (table size vs ratio)")
    ap.add_argument("--report", action="store_true",
                    help="held-out measurement instead of writing the table")
    args = ap.parse_args()

    chars = read_charset(args.charset)
    cell = read_cell(args.profile)
    corpus = load_corpus(args.corpus, args.groups)
    print(f"charset {len(chars)} glyphs, corpus {len(corpus)} files, "
          f"{len({g for _, g, _ in corpus})} groups")

    if args.report:
        report(chars, corpus, args.min_context_count)
        return

    model = fit([t for _, _, t in corpus], chars, args.min_context_count, cell)
    blob = model.to_bytes()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"contexts: L1 {len(model.l1)}, L2 {len(model.l2)}")
    print(f"cell {model.cell_w}x{model.cell_h} px (aspect {model.cell_h / model.cell_w:.4f})")
    print(f"wrote {args.out}  ({len(blob)} bytes, {len(blob)/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
