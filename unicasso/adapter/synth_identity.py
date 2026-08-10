"""Synthetic IDENTITY pairs: glyph grids whose target image IS their own render.

    GLYPHVAE_FONT=sfmono python -m unicasso.adapter.synth_identity \
        --n 20 --out "<corpus>/txts" --img-out "<corpus>/images" [--seed 0]

PRINCIPLED GLYPH MODEL: every box-drawing character is, per Unicode, a 4-tuple of
per-arm types (up, down, left, right) in {none, light, heavy, double} -- and the mixed
glyphs (┽ ╂ ┿ ╞ ╤ ...) are exactly the connectors between differing types. We parse
each font glyph's signature FROM ITS UNICODE NAME ("BOX DRAWINGS LEFT HEAVY AND RIGHT
VERTICAL LIGHT" -> l=heavy, r=u=d=light), invert to signature->glyph, and compose
shapes that each carry a type: rectangles, rectilinear paths (corner-dense L/Z/U/
staircases), loose segments. A heavy path crossing a light box lands on the cell as
(u=light, d=light, l=heavy, r=heavy) and looks up ┿-class connectors -- tiling is
correct by construction, including across type changes. Signatures with no Unicode
glyph (e.g. heavy+double mixes) relax through a defined chain (double<->heavy swap ->
uniform-majority -> light). Arc corners ╭╮╰╯ substitute on pure-light corners of
"round"-flagged shapes.

The .png target is the exact render of the .txt: a zero-domain-gap pair (regularizer
with a true d-floor at 0, and a crisp substrate for the spur/break connection
corruptions). find_pairs matches stem 'synthid_*'; corpus_split routes new parents to
train.
"""
import argparse
import os
import unicodedata

import numpy as np
import torch
from PIL import Image

from unicasso.adapter.corrupt import CorruptionSampler
from unicasso.adapter.clip_adapt import VAE_DEFAULT

U, D, L, R = 0, 1, 2, 3
DIR_TOKENS = {"UP": (U,), "DOWN": (D,), "LEFT": (L,), "RIGHT": (R,),
              "VERTICAL": (U, D), "HORIZONTAL": (L, R)}
W_TOKENS = {"LIGHT": 1, "SINGLE": 1, "HEAVY": 2, "DOUBLE": 3}
ROUND_MAP = {"┌": "╭", "┐": "╮", "└": "╰", "┘": "╯"}


def parse_box(ch):
    """char -> ((w_u,w_d,w_l,w_r), is_arc) from its Unicode name; None if not a plain
    box-drawing connector (dashed/diagonal/block variants excluded)."""
    if not (0x2500 <= ord(ch) <= 0x257F):
        return None
    name = unicodedata.name(ch, "")
    if not name.startswith("BOX DRAWINGS "):
        return None
    body = name[len("BOX DRAWINGS "):]
    if "DASH" in body or "DIAGONAL" in body:
        return None
    arc = body.startswith("ARC ") or " ARC " in body
    body = body.replace("ARC ", "")
    sig = [0, 0, 0, 0]
    w = 1                                     # leading weight distributes over AND-segments
    for seg in body.split(" AND "):
        dirs = []
        for tok in seg.split():
            if tok in W_TOKENS:
                w = W_TOKENS[tok]
            elif tok in DIR_TOKENS:
                dirs += list(DIR_TOKENS[tok])
        for d_ in dirs:
            sig[d_] = w
    if not any(sig):
        return None
    return tuple(sig), arc


def build_tables(char_to_idx):
    """font -> {signature: char} (plain) + {signature: char} (arc corners)."""
    sig2, arc2 = {}, {}
    for ch in char_to_idx:
        p = parse_box(ch)
        if p:
            (arc2 if p[1] else sig2).setdefault(p[0], ch)
    return sig2, arc2


def lookup(sig, sig2):
    """Exact signature, else a defined relaxation chain (Unicode has no heavy+double
    mixes, doubles lack stubs, etc.): swap double<->heavy, uniform-majority, light."""
    cands = [sig,
             tuple(2 if w == 3 else w for w in sig),
             tuple(3 if w == 2 else w for w in sig)]
    nz = [w for w in sig if w]
    maj = max(set(nz), key=nz.count)
    cands.append(tuple(maj if w else 0 for w in sig))
    cands.append(tuple(1 if w else 0 for w in sig))
    for c in cands:
        if c in sig2:
            return sig2[c]
    return None


def compose(rng, GH, GW, weights):
    """-> (GH,GW) grid of {dir: weight} dicts. Each shape carries its own type; unions
    at crossings produce mixed signatures (the connector glyphs). First writer wins on
    a same-direction collision (collinear overlap)."""
    arms = [[{} for _ in range(GW)] for _ in range(GH)]
    rounds = torch.zeros(GH, GW, dtype=torch.bool)          # cells drawn by a round shape

    def put(y, x, d_, w):
        arms[y][x].setdefault(d_, w)

    def hseg(y, x0, x1, w):
        for x in range(x0, x1 + 1):
            if x > x0:
                put(y, x, L, w)
            if x < x1:
                put(y, x, R, w)

    def vseg(x, y0, y1, w):
        for y in range(y0, y1 + 1):
            if y > y0:
                put(y, x, U, w)
            if y < y1:
                put(y, x, D, w)

    def mark_round(cells):
        for y, x in cells:
            rounds[y, x] = True

    def shape_w():
        return int(weights[int(rng.integers(len(weights)))])

    for _ in range(int(rng.integers(2, 5))):                # paths: the corner factories
        w = shape_w()
        rnd = w == 1 and rng.random() < 0.4
        y = int(rng.integers(2, GH - 2)); x = int(rng.integers(2, GW - 2))
        horiz = bool(rng.random() < 0.5)
        for _ in range(int(rng.integers(3, 8))):
            if horiz:
                nx = int(np.clip(x + int(rng.integers(3, 16))
                                 * (1 if rng.random() < 0.5 else -1), 0, GW - 1))
                if nx != x:
                    hseg(y, min(x, nx), max(x, nx), w)
                    if rnd:
                        mark_round([(y, x), (y, nx)])
                    x = nx
            else:
                ny = int(np.clip(y + int(rng.integers(2, 9))
                                 * (1 if rng.random() < 0.5 else -1), 0, GH - 1))
                if ny != y:
                    vseg(x, min(y, ny), max(y, ny), w)
                    if rnd:
                        mark_round([(y, x), (ny, x)])
                    y = ny
            horiz = not horiz
    for _ in range(int(rng.integers(1, 4))):                # rectangles (closed loops)
        w = shape_w()
        rnd = w == 1 and rng.random() < 0.4
        y0 = int(rng.integers(0, GH - 4)); x0 = int(rng.integers(0, GW - 6))
        y1 = y0 + int(rng.integers(3, min(GH - y0, 14))) - 1
        x1 = x0 + int(rng.integers(5, min(GW - x0, 26))) - 1
        hseg(y0, x0, x1, w); hseg(y1, x0, x1, w)
        vseg(x0, y0, y1, w); vseg(x1, y0, y1, w)
        if rnd:
            mark_round([(y0, x0), (y0, x1), (y1, x0), (y1, x1)])
    for _ in range(int(rng.integers(0, 3))):                # loose segments -> T/X hits
        w = shape_w()
        if rng.random() < 0.5:
            y = int(rng.integers(0, GH)); x0 = int(rng.integers(0, GW - 4))
            hseg(y, x0, x0 + int(rng.integers(3, min(GW - x0, 20))), w)
        else:
            x = int(rng.integers(0, GW)); y0 = int(rng.integers(0, GH - 3))
            vseg(x, y0, y0 + int(rng.integers(2, min(GH - y0, 12))), w)
    return arms, rounds


def emit_grid(arms, rounds, s, sig2):
    """arm-type grid -> glyph index grid via signature lookup (+ arc corners)."""
    GH, GW = len(arms), len(arms[0])
    grid = torch.full((GH, GW), s.space, dtype=torch.long)
    for y in range(GH):
        for x in range(GW):
            a = arms[y][x]
            if not a:
                continue
            sig = (a.get(U, 0), a.get(D, 0), a.get(L, 0), a.get(R, 0))
            ch = lookup(sig, sig2)
            if ch and bool(rounds[y][x] if isinstance(rounds, list) else rounds[y, x]) \
                    and ch in ROUND_MAP and ROUND_MAP[ch] in s.char_to_idx:
                ch = ROUND_MAP[ch]
            if ch:
                grid[y, x] = s.char_to_idx[ch]
    return grid


def make_patch(rng, gh, gw, s):
    """One small shape (rect or rectilinear path, single weight) as a (gh,gw) glyph
    grid -- the whitespace-decoration unit for the trainer's --decorate-frac
    augmentation. Identity property: its target patch IS its render."""
    if not hasattr(s, "_synthid_sig2"):
        s._synthid_sig2 = build_tables(s.char_to_idx)[0]
    w = int(rng.choice([1, 1, 2, 3]))
    arms = [[{} for _ in range(gw)] for _ in range(gh)]

    def put(y, x, d_):
        arms[y][x].setdefault(d_, w)

    def hseg(y, x0, x1):
        for x in range(x0, x1 + 1):
            if x > x0:
                put(y, x, L)
            if x < x1:
                put(y, x, R)

    def vseg(x, y0, y1):
        for y in range(y0, y1 + 1):
            if y > y0:
                put(y, x, U)
            if y < y1:
                put(y, x, D)

    if rng.random() < 0.5 and gh >= 4 and gw >= 5:            # closed rect
        y0 = int(rng.integers(0, gh - 3)); x0 = int(rng.integers(0, gw - 4))
        y1 = int(rng.integers(y0 + 2, gh)); x1 = int(rng.integers(x0 + 3, gw))
        hseg(y0, x0, x1); hseg(y1, x0, x1); vseg(x0, y0, y1); vseg(x1, y0, y1)
    else:                                                     # corner-dense path
        y = int(rng.integers(0, gh)); x = int(rng.integers(0, gw))
        horiz = bool(rng.random() < 0.5)
        for _ in range(int(rng.integers(2, 5))):
            if horiz:
                nx = int(rng.integers(0, gw))
                if nx != x:
                    hseg(y, min(x, nx), max(x, nx)); x = nx
            else:
                ny = int(rng.integers(0, gh))
                if ny != y:
                    vseg(x, min(y, ny), max(y, ny)); y = ny
            horiz = not horiz
    rounds = [[False] * gw for _ in range(gh)]
    return emit_grid(arms, rounds, s, s._synthid_sig2)


def main():
    ap = argparse.ArgumentParser(description="synthetic identity txt+png pairs")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True, help="dir for .txt grids")
    ap.add_argument("--img-out", default=None, help="dir for .png targets (default: --out)")
    ap.add_argument("--mix-prob", type=float, default=0.5,
                    help="probability an image uses TWO weights (mixed connectors at "
                         "crossings); else single-weight")
    ap.add_argument("--vae-ckpt", default=VAE_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    img_out = args.img_out or args.out
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(img_out, exist_ok=True)

    s = CorruptionSampler(args.vae_ckpt, device="cpu", seed=args.seed)
    rng = np.random.default_rng(args.seed)
    inv = {v: k for k, v in s.char_to_idx.items()}
    sig2, arc2 = build_tables(s.char_to_idx)
    print(f"font box-glyph signatures: {len(sig2)} plain + {len(arc2)} arc")

    made = 0
    for i in range(args.n):
        GH = int(rng.integers(22, 32)); GW = int(rng.integers(44, 64))
        if rng.random() < args.mix_prob:
            weights = [1, int(rng.choice([2, 3]))]           # light + one accent type
        else:
            weights = [int(rng.choice([1, 1, 2, 3]))]
        arms, rounds = compose(rng, GH, GW, weights)
        grid = emit_grid(arms, rounds, s, sig2)
        if int((grid != s.space).sum()) < 40:                # degenerate draw
            continue
        tag = "mix" if len(weights) > 1 else {1: "light", 2: "heavy", 3: "double"}[weights[0]]
        stem = f"synthid_{args.seed:02d}_{i:03d}_{tag}"
        with open(os.path.join(args.out, stem + ".txt"), "w") as f:
            f.write("\n".join("".join(inv[int(g)] for g in row) for row in grid.tolist()))
        canvas = (s.render(grid).numpy() * 255).astype(np.uint8)
        Image.fromarray(canvas, "L").save(os.path.join(img_out, stem + ".png"))
        made += 1
    print(f"wrote {made} identity pairs -> {args.out} (txt) / {img_out} (png)")


if __name__ == "__main__":
    main()
