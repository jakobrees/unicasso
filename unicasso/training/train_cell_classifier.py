"""Per-cell glyph classifier ladder over the dataset_v1 cell cache.

python -m unicasso.training.train_cell_classifier --variant tf5x3 --clip-token --heads 4

Distills the optimizer's per-cell decisions into a feedforward model. Every variant
reads the same cache (see cell_data.py) and reports the same metrics; the headline
number is accuracy on the cells where the optimizer DISAGREED with greedy
nearest-glyph snapping -- the only place context can show up.

Variants:
    cell    per-cell CNN, (1,40,24) input (cell + real 3/4-px neighbor margins)
    cnn3x3  one CNN over the whole 3x3-cell window (context via convolution)
    cnn5x3  same, 5 cells wide x 3 tall (physically ~square receptive field)
    tf3x3   shared per-cell CNN token encoder + 1 pre-LN transformer block
    tf5x3   same, 5x3 window
Add --clip-token to prepend a down-projected global CLIP image embedding
(any tf variant). Window pixels come from the run image, so token margins and
window borders contain REAL neighbor ink; off-frame is white.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unicasso.substrate import glyphs as G

VARIANTS = {
    # name: (rows, cols, transformer)
    "cell":   (1, 1, False),
    "cnn3x3": (3, 3, False),
    "cnn5x3": (3, 5, False),
    "tf3x3":  (3, 3, True),
    "tf5x3":  (3, 5, True),
}


# --------------------------------------------------------------------------- data

class CellCache:
    """Loads the whole cache into RAM (uint8) and serves window batches.

    viz_parents: parent names whose runs go to a separate 'viz' split (held out
    of training WITHOUT polluting val, so val metrics stay comparable).
    window_labels: fetch() returns labels for EVERY window cell (dense head),
    off-grid cells = -1.
    """

    def __init__(self, cache_dir, rows, cols, pad_h, pad_w, cell_h, cell_w,
                 window_labels=False, viz_parents=()):
        self.meta = json.load(open(os.path.join(cache_dir, "meta.json")))
        self.chars = self.meta["chars"]
        self.space = self.chars.index(" ")
        self.ch, self.cw = cell_h, cell_w
        self.ph, self.pw = pad_h, pad_w
        self.rh, self.rw = rows // 2, cols // 2          # window half-extent in cells
        self.win_h = rows * cell_h + 2 * pad_h
        self.win_w = cols * cell_w + 2 * pad_w
        # pad every run image once, by the window reach (cells) + token margin (px)
        py = self.rh * cell_h + pad_h
        px = self.rw * cell_w + pad_w
        self.off_y, self.off_x = py, px

        self.window_labels = window_labels
        self.ink, self.labels, self.greedy, self.clip = [], [], [], []
        self.labels_pad = []      # labels padded with -1, for dense window fetches
        self.stems = []
        self.index = {"train": [], "val": [], "viz": []}
        for ri, r in enumerate(self.meta["runs"]):
            d = np.load(os.path.join(cache_dir, r["stem"] + ".npz"))
            self.ink.append(np.pad(d["ink"], ((py, py), (px, px))))   # 0 = white
            self.labels.append(d["labels"].astype(np.int64))
            if window_labels:
                self.labels_pad.append(np.pad(d["labels"].astype(np.int64),
                                              ((self.rh, self.rh), (self.rw, self.rw)),
                                              constant_values=-1))
            self.greedy.append(d["greedy"].astype(np.int64))
            self.clip.append(d["clip"].astype(np.float32))
            self.stems.append(r["stem"])
            gh, gw = d["labels"].shape
            ys, xs = np.mgrid[0:gh, 0:gw]
            idx = np.stack([np.full(gh * gw, ri), ys.ravel(), xs.ravel()], axis=1)
            split = "viz" if r["parent"] in viz_parents else r["split"]
            self.index[split].append(idx)
        self.index = {k: (np.concatenate(v).astype(np.int32) if v
                          else np.zeros((0, 3), np.int32)) for k, v in self.index.items()}
        self.clip_dim = self.clip[0].shape[0]

    def fetch(self, idx):
        """idx (B,3) [run,y,x] -> window uint8 (B,win_h,win_w), labels, greedy, clip.

        window_labels=False: labels (B,) = center cell. True: (B, rows*cols)
        row-major over the window, off-grid = -1.
        """
        B = idx.shape[0]
        R, C = 2 * self.rh + 1, 2 * self.rw + 1
        out = np.empty((B, self.win_h, self.win_w), dtype=np.uint8)
        lab = np.empty((B, R * C) if self.window_labels else B, dtype=np.int64)
        gre = np.empty(B, dtype=np.int64)
        emb = np.empty((B, self.clip_dim), dtype=np.float32)
        for i, (ri, y, x) in enumerate(idx):
            top = self.off_y + y * self.ch - self.rh * self.ch - self.ph
            left = self.off_x + x * self.cw - self.rw * self.cw - self.pw
            out[i] = self.ink[ri][top:top + self.win_h, left:left + self.win_w]
            if self.window_labels:
                lab[i] = self.labels_pad[ri][y:y + R, x:x + C].ravel()
            else:
                lab[i] = self.labels[ri][y, x]
            gre[i] = self.greedy[ri][y, x]
            emb[i] = self.clip[ri]
        return out, lab, gre, emb

    def clamp_shift(self, idx, dy, dx):
        """Window centers shifted by (dy,dx), clamped into each run's grid.
        Returns (shifted_idx, actual_dy (B,), actual_dx (B,))."""
        out = idx.copy()
        for i, (ri, y, x) in enumerate(idx):
            gh, gw = self.labels[ri].shape
            out[i, 1] = min(max(y + dy, 0), gh - 1)
            out[i, 2] = min(max(x + dx, 0), gw - 1)
        return out, out[:, 1] - idx[:, 1], out[:, 2] - idx[:, 2]


def _gauss1d(sigma, device=None):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur_img(img, sigma):
    """Separable gaussian blur. img (B,3,H,W) or (H,W,3)."""
    hw = img.dim() == 3
    if hw:
        img = img.permute(2, 0, 1)[None]
    k = _gauss1d(sigma, img.device)
    r = (k.numel() - 1) // 2
    c = img.shape[1]
    img = F.conv2d(img, k.view(1, 1, -1, 1).expand(c, 1, -1, 1),
                   padding=(r, 0), groups=c)
    img = F.conv2d(img, k.view(1, 1, 1, -1).expand(c, 1, 1, -1),
                   padding=(0, r), groups=c)
    return img[0].permute(1, 2, 0) if hw else img


def corrupt_color_batch(xc, rng, n_smudge=3, n_blob=5, blur_p=0.5, ink=None):
    """Colored corruptions on a window batch. xc (B,3,H,W) in [0,1] ->
    (corrupted, alpha (B,H,W)).

    alpha is the per-pixel corruption strength and doubles as the ABSTAIN
    supervision: a corrupted pixel's color represents neither layer, so the
    mask should route the color fit around it. Strengths are deliberately
    capped (faint elongated smudges ~0.08-0.2, particles ~0.5-0.8) -- targets
    stay clean, and an over-corrupted input would train an insensitive model.
    `ink` (B,H,W, ink=1): when given, HALF of each family lands with its
    center on an ink pixel -- junk touching the strokes is the case the fit
    must actually survive; the rest stays uniform. The blur field corrupts
    colors only where the image has edges (flat regions blur to themselves),
    so its alpha is gated by local gradient."""
    B, _, H, W = xc.shape
    dev = xc.device
    yy = torch.arange(H, device=dev).view(1, 1, H, 1).float()
    xx = torch.arange(W, device=dev).view(1, 1, 1, W).float()

    def rt(*s):
        return torch.from_numpy(rng.random(s).astype(np.float32)).to(dev)

    def centers(n):
        cy, cx = rt(B, n) * H, rt(B, n) * W          # (B,n) uniform
        if ink is not None:
            w = (ink.reshape(B, -1) > 0.15).float() + 1e-6
            pick = torch.multinomial(w, n, replacement=True)      # (B,n)
            on = rt(B, n) < 0.5                       # half anchored to ink
            cy = torch.where(on, (pick // W).float(), cy)
            cx = torch.where(on, (pick % W).float(), cx)
        return cy, cx

    out = xc.clone()
    alpha = torch.zeros(B, H, W, device=dev)
    for n, r_lo, r_hi, a_lo, a_hi, hard in (
            (n_smudge, 2.0, 8.0, 0.08, 0.2, False),
            (n_blob, 1.0, 3.5, 0.5, 0.8, True)):
        if n <= 0:
            continue
        cy, cx = centers(n)
        rad = r_lo + rt(B, n) * (r_hi - r_lo)
        amp = (a_lo + rt(B, n) * (a_hi - a_lo)) * (rt(B, n) < 0.7).float()
        col = rt(B, n, 3)
        dy = yy - cy[..., None, None]
        dx = xx - cx[..., None, None]                 # (B,n,H,W)
        if hard:
            d2 = dy ** 2 + dx ** 2
            g = torch.exp(-(d2 / (rad[..., None, None] ** 2)) ** 3)
        else:
            # smudges are STREAKS: anisotropic gaussian along a random axis
            # (axis ratio 1.5-3, area ~preserved: a*b = rad^2)
            th = rt(B, n) * math.pi
            stretch = 1.5 + rt(B, n) * 1.5
            ct = torch.cos(th)[..., None, None]
            st = torch.sin(th)[..., None, None]
            u = dy * ct + dx * st
            v = -dy * st + dx * ct
            a2 = (rad * stretch)[..., None, None] ** 2
            b2 = (rad / stretch)[..., None, None] ** 2
            g = torch.exp(-0.5 * (u ** 2 / a2 + v ** 2 / b2))
        a = amp[..., None, None] * g                  # (B,n,H,W)
        for j in range(a.shape[1]):                   # composite one by one
            aj = a[:, j][:, None]
            out = out * (1 - aj) + col[:, j, :, None, None] * aj
            alpha = torch.maximum(alpha, a[:, j])
    if rng.random() < blur_p:                         # low-freq blur field
        sig = 1.0 + rng.random() * 1.5
        blurred = blur_img(out, sig)
        f = rt(B, 4, 4)[:, None]                      # (B,1,4,4) strength field
        f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        f = f * (0.4 + 0.6 * rng.random())
        lum = out.mean(1, keepdim=True)
        gx = F.pad(lum[..., 1:] - lum[..., :-1], (0, 1))
        gy = F.pad(lum[..., 1:, :] - lum[..., :-1, :], (0, 0, 0, 1))
        edge = ((gx.abs() + gy.abs()) * 4.0).clamp(0, 1)
        out = out * (1 - f) + blurred * f
        alpha = torch.maximum(alpha, (f * edge)[:, 0])
    return out.clamp(0, 1), alpha


# -------------------------------------------------------------------------- models

def conv_trunk(in_ch=1):
    """(B,in_ch,40,24) -> (B,1920). Also used on bigger window inputs by WindowCNN.
    in_ch=4 = ink + RGB color channels (the scratch color-conv variant)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
        nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),
    )


def head(in_dim, n_classes):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, n_classes))


class WindowCNN(nn.Module):
    """One CNN over the full window (context via convolution only)."""

    def __init__(self, win_h, win_w, n_classes):
        super().__init__()
        self.trunk = conv_trunk()
        self.extra = (nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1), nn.GELU())
                      if win_h > 60 else nn.Identity())
        with torch.no_grad():
            f = self.extra(self.trunk(torch.zeros(1, 1, win_h, win_w)))
        self.head = head(f.numel(), n_classes)

    def forward(self, x, clip_emb=None):
        f = self.extra(self.trunk(x))
        return self.head(f.flatten(1))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, t):
        h = self.ln1(t)
        t = t + self.attn(h, h, h, need_weights=False)[0]
        return t + self.mlp(self.ln2(t))


class MaskDecoder(nn.Module):
    """Per-pixel fg/bg/abstain mask logits for one cell patch (minimal U-Net:
    exactly one skip). The trunk's first-conv feature map carries the local
    pixel evidence a 96-dim token cannot store; the attention-contextualized
    cell token, broadcast over the map, carries the global context (which
    entity is fg here). Two thin convs + x2 upsample back to patch resolution.

    Zero-init output conv: the per-pixel softmax starts uniform (1/3 each),
    so the mask-weighted color fit begins as the plain cell mean and
    sharpens only as the loss asks for it -- surgery-safe like the other
    color-model extensions."""

    def __init__(self, dim, skip_ch=32, mid=32, flip=False):
        super().__init__()
        self.tok = nn.Linear(dim, skip_ch)
        self.c1 = nn.Conv2d(2 * skip_ch, mid, 3, padding=1)
        self.gn = nn.GroupNorm(8, mid)
        self.c2 = nn.Conv2d(mid, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if flip:
            self.add_flip()

    def add_flip(self):
        """Per-cell signed gain on the fg-vs-bg decision (FiLM-style).

        The token enters this decoder only as extra input CHANNELS to a linear
        conv, so its influence is ADDITIVE. Deciding "in this cell the darker
        cluster is the subject" requires INVERTING a spatial pattern on the
        strength of a cell-level fact, and no additive path can do that -- which
        is measurable, not hypothetical: given RGB alone the mask gets the
        bright-subject cells right 98.5% of the time and the dark-subject cells
        (69% of all cells) at chance, having learned the single rule "subject =
        the brighter side" and being unable to express its converse.

        s = 1 + Linear(tok), zero-init, so s == 1 exactly at creation and this is
        a no-op on an existing checkpoint. s < 0 swaps the cell's polarity."""
        self.flip = nn.Linear(self.tok.in_features, 1)
        nn.init.zeros_(self.flip.weight)
        nn.init.zeros_(self.flip.bias)

    def forward(self, tok, skip):
        """tok (P, dim) cell tokens; skip (P, skip_ch, th/2, tw/2)."""
        g = self.tok(tok)[:, :, None, None].expand(-1, -1, *skip.shape[2:])
        h = F.gelu(self.gn(self.c1(torch.cat([skip, g], dim=1))))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = F.gelu(self.c2(h))
        m = self.out(h)                                  # (P, 3, th, tw)
        if getattr(self, "flip", None) is not None:
            # scale only the fg-minus-bg CONTRAST; abstain is untouched, and the
            # fg/bg midpoint is preserved, so s=1 reconstructs m exactly
            s = (1.0 + self.flip(tok))[:, :, None]       # (P,1,1)
            mid = 0.5 * (m[:, 0] + m[:, 1])
            d = 0.5 * (m[:, 0] - m[:, 1]) * s
            m = torch.stack([mid + d, mid - d, m[:, 2]], dim=1)
        return m


class PxResBlock(nn.Module):
    """Pre-norm residual conv block, EXACTLY the identity at init.

    A plain conv stack cannot be identity-initialised because the GELU is not
    the identity, so deepening the per-pixel encoder would mean discarding its
    trained weights. In residual form, zeroing the second conv makes the whole
    block contribute 0, so depth can be appended to an already-trained encoder
    with no behavioural change at step 0 -- the same surgery discipline as the
    identity-init transformer blocks. Stride 1, pad 1: full cell resolution is
    preserved and the receptive field grows by 2 per conv."""

    def __init__(self, d, mid=None):
        super().__init__()
        mid = mid or d
        self.gn = nn.GroupNorm(8, d)
        self.c1 = nn.Conv2d(d, mid, 3, padding=1)
        self.c2 = nn.Conv2d(mid, d, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(self.gn(x))))


class MaskAttnDecoder(nn.Module):
    """Mask as CLASSIFICATION: each pixel's embedding dotted with one query per
    class, softmax over classes. The alternative to MaskDecoder's conv stack.

    Why this shape. The decision a mask has to make is "is this pixel nearer
    cluster A or cluster B, and which of those is foreground". That is a
    COMPARISON between a per-pixel vector and per-cell vectors -- multiplicative.
    MaskDecoder cannot do it: the token reaches it as extra input CHANNELS to a
    linear conv, so its influence is additive, and no additive path inverts a
    spatial pattern on the strength of a cell-level fact. Measured consequence:
    given RGB alone the conv head learns the single rule "foreground = brighter
    side", scores 98.5% on the cells where that holds and chance on the 69%
    where it does not, and 20x the learning rate does not move it.

    Here the logit IS an inner product, so the comparison is native, and
    "swap which side is foreground" is a LINEAR change in the query -- among the
    easiest things a network can learn, rather than something a conv stack has
    to invert.

    Two further consequences of the formulation, both wanted: pixel features are
    computed at FULL cell resolution, so the x2 nearest upsample that made every
    conv mask blocky at 2x2 is gone; and the class count is set by the number of
    queries, so fg/bg/abstain are symmetric rather than three channels of one
    conv.

    Note the queries come from the token, and the token is fed cell_feats, whose
    fg/bg slots hold decompose's own cluster colours in a fixed order. So this
    CAN learn to follow decompose's orientation. That is copying, but of one bit
    per cell rather than 648, and the pixel partition remains the model's own.
    """

    wants_rgb = True          # feed me the raw patch, not rgb_trunk features
    n_glyph = 0               # broadcast conditioning, off unless add_bcast ran
    bc_mean = False

    def __init__(self, dim, d=32, n_cls=3, depth=2, mid=None):
        super().__init__()
        self.d, self.n_cls, self.depth = d, n_cls, depth
        self.mid = mid = mid or d
        # per-pixel encoder: stride 1 and pad 1 throughout, so the map stays at
        # full cell resolution and every pixel keeps its own vector. Depth buys
        # receptive field (2 convs = 5x5, 4 = 9x9) -- the queries carry no
        # spatial extent, so ALL local structure the mask can use has to come
        # from here. It is also the cheapest part of the model to enlarge.
        layers, c_in = [], 3
        for i in range(max(1, depth)):
            last = (i == max(1, depth) - 1)
            layers.append(nn.Conv2d(c_in, d if last else mid, 3, padding=1))
            if not last:
                layers += [nn.GroupNorm(8, mid), nn.GELU()]
            c_in = mid
        self.px = nn.Sequential(*layers)
        self.px_res = nn.ModuleList()      # appended depth, identity at init
        self.q = nn.Linear(dim, n_cls * d)
        self.b = nn.Linear(dim, n_cls)                # per-class cell prior
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def add_px_blocks(self, n, mid=None):
        """Append n identity-init residual blocks to the per-pixel encoder."""
        for _ in range(max(0, n - len(self.px_res))):
            self.px_res.append(PxResBlock(self.d, mid))
        return len(self.px_res)

    def add_bcast(self, n_classes, n_glyph=8, use_mean=True, ph=0, pw=0):
        """BROADCAST CONDITIONING: extra INPUT channels to the per-pixel encoder,
        constant across the cell.

        Why the pixel encoder needs help it cannot give itself. `px` is a conv
        stack with a 5x5 receptive field at depth 2 (13x13 with two px_res
        blocks) over a 44x24 patch, and it has no positional input. So the single
        most useful per-pixel feature for an fg/bg decision -- *am I above or
        below this cell's average colour* -- is not merely unlearned, it is
        UNREPRESENTABLE: the mean is a statistic over the whole cell and no
        13x13 window can see it. Until now the only thing carrying cell-level
        information was the query, which can apply one linear rule to a feature
        space built blind to it.

        Two channel groups, both constant within a cell:

        `rgb - cell_mean` (3 ch, no parameters). Note WHY this is safe where
        dec["fg"]/dec["bg"] are not: the mean is INVARIANT to which cluster you
        call foreground, so it cannot leak polarity. The decompose cluster
        colours are not invariant, and putting them in the pixel path would
        re-open the copy route this whole head exists to close.

        `gemb` (n_glyph ch). A learned embedding read out by the glyph SOFTMAX,
        i.e. a convex combination of n_classes learned vectors -- the same
        mechanism as glyph_ctx, its own matrix, delivered to the pixel encoder
        instead of the token residual. It says "you are colouring a dense block"
        or "a thin stroke", which changes what fg has to be to land the right
        cell mean. Copy risk stays nil for the reason that governs this head:
        the vector is IDENTICAL at every pixel, and a conv with no positional
        input cannot turn a constant into a glyph-shaped mask.

        Surgery-safe: conv0 is widened and the new input columns are ZEROED, so
        the encoder is exactly its trained self at step 0. gemb is left at random
        init deliberately -- zeroing both sides of the product would leave
        neither with a gradient."""
        self.bc_mean = bool(use_mean)
        self.bc_ph, self.bc_pw = int(ph), int(pw)
        self.n_glyph = int(n_glyph)
        if self.n_glyph:
            self.gemb = nn.Linear(n_classes, self.n_glyph, bias=False)
        n_new = (3 if self.bc_mean else 0) + self.n_glyph
        if not n_new:
            return 0
        c0 = self.px[0]
        wide = nn.Conv2d(c0.in_channels + n_new, c0.out_channels, 3, padding=1)
        with torch.no_grad():
            wide.weight.zero_()
            wide.weight[:, :c0.in_channels].copy_(c0.weight)
            wide.bias.copy_(c0.bias)
        self.px[0] = wide
        return n_new

    def seed_gemb(self, ink_flat, ch, cw):
        """Warm-start the glyph embedding from the glyphs' actual GEOMETRY.

        Why this is not cosmetic. Read by a near-one-hot softmax, an
        (n_glyph x 358) embedding is really 358 nearly-disjoint weight columns,
        and column j only receives gradient on the steps where glyph j is
        chosen -- roughly 1/358 of them. That is exactly why glyph_ctx, which
        has the identical structure, finished runF contributing 0.10% of the
        token it is added to (per-glyph spread 0.0025) while mask_ctx, zero-init
        in the same branch off the same objective but fed a DENSE conv feature,
        reached 15.3%. A lookup table read one column at a time is the slowest
        possible thing to learn from scratch.

        So seed the first four dimensions with the quantities the feature exists
        to convey -- coverage, where the ink sits, and how it is spread -- each
        standardised across the glyph set. The table starts informative and
        training only has to refine it. Remaining dimensions keep their random
        init and are free to learn whatever else matters.

        ink_flat: (N, ch*cw) in [0,1], 1 = inked."""
        if not self.n_glyph:
            return 0
        N = ink_flat.shape[0]
        m = ink_flat.view(N, ch, cw)
        ys = torch.linspace(-1, 1, ch, device=ink_flat.device)[None, :, None]
        xs = torch.linspace(-1, 1, cw, device=ink_flat.device)[None, None, :]
        tot = m.sum((1, 2)).clamp_min(1e-6)
        cy, cx = (m * ys).sum((1, 2)) / tot, (m * xs).sum((1, 2)) / tot
        stats = [m.mean((1, 2)),                       # coverage: dense vs thin
                 cy, cx,                               # where the ink sits
                 ((m * (ys - cy[:, None, None]).pow(2)).sum((1, 2))
                  / tot).sqrt()]                       # vertical spread
        k = min(len(stats), self.n_glyph)
        with torch.no_grad():
            for i in range(k):
                s = stats[i]
                self.gemb.weight[i] = (s - s.mean()) / s.std().clamp_min(1e-6)
        return k

    def _bcast(self, rgb, gsoft):
        """rgb (P,3,th,tw) -> (P, 3+n_new, th, tw), the extras broadcast."""
        if not (self.bc_mean or self.n_glyph):
            return rgb
        chans = [rgb]
        if self.bc_mean:
            # the mean of the CELL CORE, not the padded patch: the core is what
            # gets coloured, and the pad belongs to the neighbours
            ph, pw = self.bc_ph, self.bc_pw
            core = rgb[:, :, ph:rgb.shape[2] - ph, pw:rgb.shape[3] - pw] \
                if (ph or pw) else rgb
            chans.append(rgb - core.mean((2, 3), keepdim=True))
        if self.n_glyph:
            if gsoft is None:      # caller has no glyph distribution (ablations,
                g = rgb.new_zeros(rgb.shape[0], self.n_glyph)   # ablations)
            else:
                g = self.gemb(gsoft)
            chans.append(g[:, :, None, None].expand(-1, -1, *rgb.shape[2:]))
        return torch.cat(chans, 1)

    def forward(self, tok, rgb, gsoft=None):
        """tok (P, dim); rgb (P, 3, th, tw) the cell's own pixels, [0,1];
        gsoft (P, n_classes) the centre cell's glyph softmax, or None."""
        f = self.px(self._bcast(rgb, gsoft))                   # (P,d,th,tw)
        for blk in self.px_res:
            f = blk(f)
        q = self.q(tok).view(-1, self.n_cls, self.d)           # (P,K,d)
        m = torch.einsum("pdhw,pkd->pkhw", f, q) * (self.d ** -0.5)
        return m + self.b(tok)[:, :, None, None]               # (P,K,th,tw)


def identity_block(dim, heads):
    """A pre-LN Block that is EXACTLY the identity at init.

    Pre-LN is t + attn(ln1(t)) then t + mlp(ln2(t)); zeroing BOTH residual
    branches' output projections makes each contribute 0, so the block passes t
    through untouched. That lets extra depth be bolted onto a trained checkpoint
    with no behavioural change at step 0 and earn its way in -- the same
    surgery-safe discipline as the zero-init color extensions."""
    b = Block(dim, heads)
    for m in (b.attn.out_proj, b.mlp[2]):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
    return b


class TokenTransformer(nn.Module):
    """Per-cell CNN tokens + n_blocks pre-LN transformer blocks.

    forward() -> center-token logits (the classifier interface everything evals);
    forward_all() -> (B, rows*cols, N) logits for every window cell, for the
    dense multi-cell objective and framing-ensemble inference.
    """

    def __init__(self, rows, cols, cell_h, cell_w, pad_h, pad_w, n_classes,
                 dim=64, heads=4, clip_dim=0, n_blocks=1, in_ch=1):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.in_ch = in_ch
        self.th, self.tw = cell_h + 2 * pad_h, cell_w + 2 * pad_w   # token patch 40x24
        self.sh, self.sw = cell_h, cell_w                           # patch stride
        self.trunk = conv_trunk(in_ch)
        with torch.no_grad():
            f = self.trunk(torch.zeros(1, in_ch, self.th, self.tw))
        self.proj = nn.Linear(f.numel(), dim)
        self.pos = nn.Parameter(torch.randn(rows * cols, dim) * 0.02)
        self.use_clip = clip_dim > 0
        if self.use_clip:
            self.clip_proj = nn.Linear(clip_dim, dim)
            self.clip_pos = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(n_blocks))
        self.ln_out = nn.LayerNorm(dim)
        self.head = head(dim, n_classes)
        self.n_extra = 1 if self.use_clip else 0
        self.center = (rows // 2) * cols + cols // 2 + self.n_extra

    def _embed(self, x, clip_emb, feats=None, mode=None, want_skip=False):
        """Per-window token embeddings: everything up to (not including) the blocks.
        want_skip=True also returns the trunk's first-conv feature map per patch
        (half resolution, local pixel evidence) for the mask decoder."""
        B = x.shape[0]
        in_ch = getattr(self, "in_ch", 1)
        if in_ch == 4 and x.shape[1] == 1:
            # ink-only input to a color-conv model: color channels = the visual
            # image (white paper, dark ink) so lineart tooling works unchanged
            x = torch.cat([x, (1.0 - x).repeat(1, 3, 1, 1)], dim=1)
        # window (B,C,H,W) -> overlapping per-cell patches (B*T,C,40,24)
        p = x.unfold(2, self.th, self.sh).unfold(3, self.tw, self.sw)
        p = p.reshape(B, x.shape[1], self.rows * self.cols, self.th, self.tw)
        p = p.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], self.th, self.tw)
        s1 = self.trunk[:3](p)                           # after conv1+GN+GELU
        t = self.proj(self.trunk[3:](s1).flatten(1)).view(B, -1, self.pos.shape[1])
        t = t + self.pos
        # color-model extensions (created by surgery in the color trainer; both
        # zero-init so the base model's behavior is the exact starting point)
        if feats is not None and getattr(self, "color_proj", None) is not None:
            t = t + self.color_proj(feats)               # feats (B, T, F) per cell
        if mode is not None and getattr(self, "mode_emb", None) is not None:
            t = t + self.mode_emb[mode][None, None, :]   # task token, added to all
        if self.use_clip:
            c = self.clip_proj(clip_emb).unsqueeze(1) + self.clip_pos
            t = torch.cat([c, t], dim=1)
        return (t, s1) if want_skip else t

    def _run_blocks(self, t, mix=None):
        """Block loop. mix = (ids, valid, kernel, n_cells, alpha) turns on
        cross-window residual mixing: after each block, the residual it produced
        on every cell token is replaced by (1-a)*own + a*(binomial-weighted mean
        of the residuals ALL overlapping windows produced for that same grid
        cell). Requires the batch to be one contiguous region (ids = per-token
        global cell index, as in framing-ensemble accumulation). The averaging
        is linear, so autograd routes each cell's gradient into every window
        covering it; framing-disagreement components are averaged out of the
        forward, so updates push the consensus state rather than per-framing
        fits."""
        if mix is None:
            for blk in self.blocks:
                t = blk(t)
            return self.ln_out(t)
        ids, valid, kernel, n_cells, alpha = mix
        B, _, d = t.shape
        w = kernel[None, :] * valid.float()               # (B, T_cells)
        flat_ids = ids.reshape(-1)
        wsum = torch.zeros(n_cells, device=t.device, dtype=t.dtype)
        wsum.index_add_(0, flat_ids, w.reshape(-1))
        wsum = wsum.clamp_min(1e-8)
        for blk in self.blocks:
            r = blk(t) - t                                # the block's residual
            rc = r[:, self.n_extra:]
            acc = torch.zeros(n_cells, d, device=t.device, dtype=t.dtype)
            acc.index_add_(0, flat_ids, (rc * w[..., None]).reshape(-1, d))
            ravg = (acc / wsum[:, None])[ids]             # (B, T_cells, d)
            rmix = torch.where(valid[..., None],
                               (1 - alpha) * rc + alpha * ravg, rc)
            t = t + torch.cat([r[:, :self.n_extra], rmix], dim=1)
        return self.ln_out(t)

    def _tokens(self, x, clip_emb, feats=None, mode=None, mix=None,
                want_skip=False):
        if want_skip:
            t, s1 = self._embed(x, clip_emb, feats, mode, want_skip=True)
            return self._run_blocks(t, mix), s1
        return self._run_blocks(self._embed(x, clip_emb, feats, mode), mix)

    def rgb_features(self, x, center_only=False, raw=False):
        """SEPARATE RGB encoder for the mask branch -- not a skip connection.

        `s1` was a genuine skip: it came from trunk[:3] of the same stack whose
        deeper layers produce the token, so it bypassed the rest of that stack.
        This shares no path with the token at all. Two things follow, both
        wanted: the decompose ink never reaches the mask's spatial evidence (so
        there is no per-pixel fg/bg answer sitting in it to copy), and the mask
        objective's gradient never flows back through the glyph stack (so it
        cannot smooth the features the glyph head reads).

        The cost is that it is trained ONLY by the mask objective, the weakest
        signal in the run -- hence the warm start from the shared trunk's first
        conv, which already extracts edges from these photos.

        x: (B, 4, win_h, win_w). Returns (B*T, 32, th/2, tw/2), same shape s1 had.

        center_only: return ONLY the window's center-cell patch, (B, 32, ...).
        Every caller feeds mask_dec `sk.view(B, T, ...)[:, center]` and discards
        the other 14, so computing them is 15x the conv and 15x the activation
        memory for nothing. The center patch is a plain slice of x -- the unfold
        starts token t at (t//cols * sh, t%cols * sw) -- so this is bit-exact,
        not an approximation."""
        B = x.shape[0]
        if center_only:
            y0, x0 = (self.rows // 2) * self.sh, (self.cols // 2) * self.sw
            p = x[:, 1:4, y0:y0 + self.th, x0:x0 + self.tw]
            if raw:                       # the pixels themselves, full res
                return p
            return None if getattr(self, "rgb_trunk", None) is None \
                else self.rgb_trunk(p)
        if getattr(self, "rgb_trunk", None) is None:
            return None
        p = x[:, 1:4].unfold(2, self.th, self.sh).unfold(3, self.tw, self.sw)
        p = p.reshape(B, 3, self.rows * self.cols, self.th, self.tw)
        p = p.permute(0, 2, 1, 3, 4).reshape(-1, 3, self.th, self.tw)
        return self.rgb_trunk(p)

    def add_mask_blocks(self, n, heads):
        """Grow the mask branch to n blocks, identity-init so step 0 is
        unchanged. build_model sizes the stack from the checkpoint, so without
        this there is no way to go from a trained 2 to a 3."""
        cur = list(getattr(self, "mask_blocks", []) or [])
        dim = self.pos.shape[1]
        for _ in range(max(0, n - len(cur))):
            cur.append(identity_block(dim, heads))
        self.mask_blocks = nn.ModuleList(cur)
        return len(cur)

    def add_aux_block(self, heads):
        """One extra block whose only job is to predict the NON-centre glyphs.

        With the framing kernel, 81.3% of the CE gradient trains neighbour
        prediction, which shapes the shared token toward the neighbours'
        identities rather than the centre's. Moving that signal behind its own
        block keeps it as a regulariser on the tokens while letting the main
        head specialise on the centre.

        Discarded at inference ONLY when glyph_ctx_aux is off. With it on this
        block is the calibrated readout for the 14 non-centre cells that
        glyph_ctx feeds to the colouriser, so it is load-bearing -- see
        glyph_readout."""
        self.aux_block = Block(self.pos.shape[1], heads)
        return self.aux_block

    def aux_logits(self, t):
        """Glyph logits down the auxiliary branch (all cells; the caller
        weights the centre to zero)."""
        return self.head(self.aux_block(t))

    def glyph_readout(self, t):
        """Glyph logits for EVERY cell of the window, each through the readout
        it was actually trained under.

        This exists because --centre-only and --glyph-ctx interact badly, and
        did so silently through all of runF. --centre-only sets kern_main to a
        one-hot on the centre, so `head` receives CE gradient on the CENTRE
        TOKEN ONLY; the neighbours are trained through head(aux_block(t)), a
        different readout, which is the entire reason aux_block exists. But
        glyph_ctx read all 15 distributions off head(t) -- so 14 of the 15
        glyph vectors handed to the colouriser came from a head applied to
        tokens it was never asked to be correct on. Not noise (the tokens are
        trained and the head is shared) but uncalibrated, and nothing in the
        objective could report it.

        Centre from the main head, neighbours from the auxiliary one. Costs one
        extra block forward over 15 tokens. Note the consequence for deployment:
        with this on, aux_block is LOAD-BEARING at inference and can no longer
        be dropped -- lite.py builds it from the state dict, and cfg carries the
        flag so the two cannot diverge."""
        lg = self.head(t[:, self.n_extra:])
        if not (getattr(self, "glyph_ctx_aux", False)
                and getattr(self, "aux_block", None) is not None):
            return lg
        la = self.head(self.aux_block(t))[:, self.n_extra:]
        c = self.center - self.n_extra
        return torch.cat([la[:, :c], lg[:, c:c + 1], la[:, c + 1:]], dim=1)

    def mask_center(self, x, t, s1=None, glogits=None, ctx=None,
                    center_rgb=None):
        """Mask logits for the CENTER cell of every window -- the whole mask
        branch, in one place.

        The trainer, Lite (two paths) and lite_color each had their own copy of
        this five-line sequence, and a divergence between them is silent: the
        weights would simply be read through a structure they never trained
        under. Whether the decoder wants raw pixels (MaskAttnDecoder) or encoded
        features (MaskDecoder) is now decided here, once."""
        if getattr(self, "mask_dec", None) is None:
            return None
        need_g = (getattr(self, "glyph_ctx_proj", None) is not None
                  or getattr(self.mask_dec, "n_glyph", 0))
        gs = None
        if need_g:
            lg = glogits if glogits is not None else self.glyph_readout(t)
            gs = lg.softmax(-1)
        adds = [(getattr(self, "mask_ctx_inject", 0),
                 ctx if ctx is not None else self.mask_ctx(x))]
        if getattr(self, "glyph_ctx_proj", None) is not None:
            g = self.glyph_ctx_proj(gs)
            if self.n_extra:
                g = torch.cat([g.new_zeros(g.shape[0], self.n_extra, g.shape[2]),
                               g], dim=1)
            adds.append((getattr(self, "glyph_ctx_inject", 1), g))
        tm = self.mask_tokens(t, adds)[:, self.center]
        if getattr(self.mask_dec, "wants_rgb", False):
            gsc = (gs[:, self.center - self.n_extra]
                   if gs is not None and self.mask_dec.n_glyph else None)
            rgb = (center_rgb if center_rgb is not None else
                   self.rgb_features(x, center_only=True, raw=True))
            return self.mask_dec(tm, rgb, gsc)
        sk = self.rgb_features(x, center_only=True)
        if sk is None:                        # no separate encoder: use the skip
            tc = (self.rows // 2) * self.cols + self.cols // 2
            sk = s1.view(x.shape[0], self.rows * self.cols, *s1.shape[1:])[:, tc]
        return self.mask_dec(tm, sk)

    def add_mask_ctx(self, dim):
        """A colour encoder PRIVATE to the mask branch: its own conv trunk over
        the RGB patch, projected to a per-cell vector and added into the mask
        branch's residual stream.

        Why it is needed: the query is built from a token whose gradient is
        dominated by glyph CE, so it mostly encodes WHICH CHARACTER goes here.
        cell_feats does inject decompose's two cluster colours, but as 11 numbers
        added at the trunk input and then pushed through three glyph-driven
        blocks. This gives the query colour evidence that never passed through
        the glyph stack at all.

        Zero-init projection, so it contributes exactly 0 at step 0."""
        self.mask_ctx_trunk = conv_trunk(3)
        with torch.no_grad():
            n = self.mask_ctx_trunk(torch.zeros(1, 3, self.th, self.tw)).numel()
        self.mask_ctx_proj = nn.Linear(n, dim)
        nn.init.zeros_(self.mask_ctx_proj.weight)
        nn.init.zeros_(self.mask_ctx_proj.bias)
        return self.mask_ctx_proj

    def cell_embed(self, p):
        """Per-cell trunk+proj embedding on (N, C, th, tw) patches -- the
        position-free half of _embed. A cell's patch is an absolute crop of the
        image (the window unfold strides by exactly one cell), so it is
        IDENTICAL in all rows*cols windows that cover the cell: encode each
        cell once, gather into windows, and the forward is bit-equal at 1/15th
        of the trunk work. Lite's cached inference path builds windows as
        cell_embed[ids] + pos (+ color_proj(feats) + mode_emb) -> _run_blocks."""
        return self.proj(self.trunk(p).flatten(1))

    def cell_ctx(self, p_rgb):
        """Per-cell mask_ctx embedding on (N, 3, th, tw) patches -- the same
        cacheable factoring for the mask branch's private colour encoder."""
        if getattr(self, "mask_ctx_trunk", None) is None:
            return None
        return self.mask_ctx_proj(self.mask_ctx_trunk(p_rgb).flatten(1))

    def mask_ctx(self, x):
        """(B, 4, win_h, win_w) -> (B, rows*cols, dim), one colour summary per
        cell of the window -- per cell, not per window, so the mask blocks'
        attention can compare a cell's colours against its neighbours'."""
        if getattr(self, "mask_ctx_trunk", None) is None:
            return None
        B = x.shape[0]
        p = x[:, 1:4].unfold(2, self.th, self.sh).unfold(3, self.tw, self.sw)
        p = p.reshape(B, 3, self.rows * self.cols, self.th, self.tw)
        p = p.permute(0, 2, 1, 3, 4).reshape(-1, 3, self.th, self.tw)
        f = self.mask_ctx_trunk(p).flatten(1)
        return self.mask_ctx_proj(f).view(B, self.rows * self.cols, -1)

    def add_glyph_ctx(self, dim, n_classes):
        """Tell the colouriser WHICH GLYPH it is colouring.

        The mask currently has no idea whether it is choosing colours for a
        dense block or a thin stroke, and those need different fg to land the
        same cell mean. Linear(n_classes -> dim) on the glyph SOFTMAX, added
        into the mask branch's residual stream. Softmax rather than one-hot for
        two reasons: it is smooth, and it opens a gradient path from the colour
        loss back into the glyph head, which nothing else provides.

        Uses the PER-WINDOW logits, not the framing-ensembled ones. The ensemble
        needs every covering window, so depending on it would force the mask to
        be computed in a second pass in Lite and lite_color -- and the ensemble
        is just an average of these anyway.

        Copy risk is bounded: this vector has no spatial extent, so it can say
        'dense' or 'sparse' but cannot paint a glyph-shaped mask."""
        self.glyph_ctx_proj = nn.Linear(n_classes, dim)
        nn.init.zeros_(self.glyph_ctx_proj.weight)
        nn.init.zeros_(self.glyph_ctx_proj.bias)
        return self.glyph_ctx_proj

    def mask_tokens(self, t, adds=()):
        """Private depth for the MASK branch (created by surgery; identity at
        init). The glyph head reads the shared blocks' output, the mask decoder
        reads this.

        Why they must differ: without it both read the same token, and since the
        glyph CE carries most of the gradient that token is dominated by glyph
        IDENTITY. Handed a token that effectively says "this cell is glyph g",
        the cheapest valid thing the decoder can emit is g's own shape -- which
        is also the honest optimum of a pure colour objective, since fitting fg
        on exactly the painted pixels minimises pixel error. The result is a mask
        that is a redundant re-encoding of the glyph. This gives the branch room
        to represent something else: which photo pixels belong to the entity."""
        # adds: iterable of (depth, tensor). depth 0 = at the input, so every
        # block's attention can share it across the 3x5 window; depth n = after
        # block n, which keeps it out of the earlier blocks' reasoning. Sources
        # are injected at DIFFERENT depths deliberately rather than summed into
        # one vector at one point.
        blocks = list(getattr(self, "mask_blocks", []) or [])
        adds = [(d, v) for d, v in adds if v is not None]
        for d, v in adds:
            if d <= 0 or not blocks:
                t = t + v
        for i, blk in enumerate(blocks):
            t = blk(t)
            for d, v in adds:
                if d == i + 1:
                    t = t + v
        return t          # NO trailing LayerNorm: the shared path already applied
        # ln_out, and a second LayerNorm renormalises those tokens (measured 0.60
        # max delta) -- which would break identity-at-init. Pre-LN blocks with
        # residuals already return to the input space, so this is exact.

    def mask_logits(self, t, s1):
        """Per-pixel fg/bg/abstain logits for every window cell.

        t: post-block tokens (B, n_extra+T, dim); s1: the _embed skip
        (B*T, 32, th/2, tw/2). Conv MaskDecoder path only -- MaskAttnDecoder
        wants raw RGB and goes through mask_center. Returns (B, T, 3, th, tw) patch-resolution
        logits -- callers softmax over dim 2 and crop the cell core
        [pad_h:pad_h+CH, pad_w:pad_w+CW] before fitting colors."""
        tc = t[:, self.n_extra:]
        B, T, d = tc.shape
        m = self.mask_dec(tc.reshape(B * T, d), s1)
        return m.view(B, T, 3, self.th, self.tw)

    def forward(self, x, clip_emb=None, feats=None, mode=None):
        return self.head(self._tokens(x, clip_emb, feats, mode)[:, self.center])

    def forward_all(self, x, clip_emb=None, feats=None, mode=None):
        t = self._tokens(x, clip_emb, feats, mode)[:, self.n_extra:]  # cell tokens
        return self.head(t)                                  # (B, rows*cols, N)


# ----------------------------------------------------------- dense objective helpers

def binomial_kernel(rows, cols, device):
    """Separable binomial weights over the window, scaled so the center = 1."""
    from math import comb
    v = torch.tensor([comb(rows - 1, i) for i in range(rows)], dtype=torch.float32)
    h = torch.tensor([comb(cols - 1, j) for j in range(cols)], dtype=torch.float32)
    w = torch.outer(v / v[rows // 2], h / h[cols // 2])
    return w.flatten().to(device)                     # (rows*cols,)


def kernel_ce(logits, y, kernel, ce_w):
    """Center-weighted dense CE. logits (B,T,N), y (B,T) with -1 = off-grid."""
    B, T, N = logits.shape
    flat_y = y.reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, N), flat_y.clamp(min=0),
                         weight=ce_w, reduction="none")
    valid = (flat_y >= 0).float()
    wt = kernel.repeat(B) * ce_w[flat_y.clamp(min=0)] * valid
    return (ce * kernel.repeat(B) * valid).sum() / wt.sum().clamp_min(1e-8)


def shift_token_maps(rows, cols, device):
    """For each shift (sy,sx) in {-1,0,1}^2: (idxA, idxB) — token indices that
    refer to the SAME grid cell in the base framing and the shifted framing."""
    maps = {}
    for sy in (-1, 0, 1):
        for sx in (-1, 0, 1):
            a, b2 = [], []
            for i in range(rows):
                for j in range(cols):
                    ii, jj = i - sy, j - sx
                    if 0 <= ii < rows and 0 <= jj < cols:
                        a.append(i * cols + j)
                        b2.append(ii * cols + jj)
            maps[(sy, sx)] = (torch.tensor(a, device=device),
                              torch.tensor(b2, device=device))
    return maps


def consistency_loss(l1, l2, ady, adx, maps):
    """Symmetric KL between the two framings' predictions on shared cells.
    l1/l2 (B,T,N); ady/adx (B,) actual per-sample shifts after grid clamping."""
    total, count = l1.new_zeros(()), 0
    for (sy, sx), (ia, ib) in maps.items():
        g = (ady == sy) & (adx == sx)
        if not bool(g.any()):
            continue
        pa = F.log_softmax(l1[g][:, ia], dim=-1)
        pb = F.log_softmax(l2[g][:, ib], dim=-1)
        kl_ab = (pa.exp() * (pa - pb)).sum(-1)
        kl_ba = (pb.exp() * (pb - pa)).sum(-1)
        total = total + 0.5 * (kl_ab + kl_ba).mean() * float(g.sum())
        count += int(g.sum())
    return total / max(count, 1)


# ------------------------------------------------------- beyond-the-dataset objectives

class SynthRenderer:
    """Letter-target synthetic batches: take a real label window, re-render it with
    per-glyph affine jitter + a global sub-cell phase shift, and train against the
    letters themselves. Labels are exact by construction (off-grid cells render as
    space and are labeled space), and the stream never repeats."""

    def __init__(self, cache, ink_bm, rows, cols, ch, cw, ph, pw, device,
                 rand_frac=0.05, jit_px=2.0, jit_deg=4.0, jit_scale=0.08,
                 phase_frac=0.2):
        self.bm = ink_bm.to(device)                       # (N, ch, cw) ink=1
        self.N = ink_bm.shape[0]
        self.rows, self.cols = rows, cols
        self.ch, self.cw, self.ph, self.pw = ch, cw, ph, pw
        self.R2, self.C2 = rows + 2, cols + 2             # one extra cell ring
        self.win_h = rows * ch + 2 * ph
        self.win_w = cols * cw + 2 * pw
        self.device = device
        self.rand_frac, self.phase_frac = rand_frac, phase_frac
        self.jit_px, self.jit_deg, self.jit_scale = jit_px, jit_deg, jit_scale
        ry, rx = rows // 2 + 1, cols // 2 + 1
        self.lp = [np.pad(l, ((ry, ry), (rx, rx)), constant_values=cache.space)
                   for l in cache.labels]

    def sample(self, idx_batch, rng, want_color=False):
        """want_color: also sample per-cell fg/bg, composite a color canvas, and
        return (x_ink, y, x_color (B,3,H,W), fg (B,T,3), bg (B,T,3))."""
        B = len(idx_batch)
        T2 = self.R2 * self.C2
        labs = np.empty((B, T2), np.int64)
        for i, (ri, y, x) in enumerate(idx_batch):
            labs[i] = self.lp[ri][y:y + self.R2, x:x + self.C2].ravel()
        labs = torch.from_numpy(labs).to(self.device)
        if self.rand_frac > 0:                            # rare-tail sprinkle
            m = torch.rand(B, T2, device=self.device) < self.rand_frac
            labs[m] = torch.randint(self.N, (int(m.sum()),), device=self.device)
        g = self.bm[labs.view(-1)]                        # (B*T2, ch, cw)
        n = g.shape[0]
        ang = (torch.rand(n, device=self.device) * 2 - 1) * math.radians(self.jit_deg)
        sc = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * self.jit_scale
        tx = (torch.rand(n, device=self.device) * 2 - 1) * self.jit_px / (self.cw / 2)
        ty = (torch.rand(n, device=self.device) * 2 - 1) * self.jit_px / (self.ch / 2)
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.zeros(n, 2, 3, device=self.device)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
        theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
        grid = F.affine_grid(theta, (n, 1, self.ch, self.cw), align_corners=False)
        gj = F.grid_sample(g[:, None], grid, align_corners=False, padding_mode="zeros")
        canvas = gj.view(B, self.R2, self.C2, self.ch, self.cw) \
            .permute(0, 1, 3, 2, 4).reshape(B, self.R2 * self.ch, self.C2 * self.cw)
        canvas = canvas * (0.75 + 0.25 * torch.rand(B, 1, 1, device=self.device))
        if rng.random() < 0.5:                            # stroke softening, whole batch
            k = torch.tensor([0.25, 0.5, 0.25], device=self.device)
            k2 = torch.outer(k, k)[None, None]
            canvas = F.conv2d(canvas[:, None], k2, padding=1)[:, 0]
        col_canvas = fgc = bgc = None
        if want_color:
            # window-level palette + per-cell jitter; enforce fg/bg separation
            fg0 = torch.rand(B, 1, 1, 3, device=self.device)
            bg0 = torch.rand(B, 1, 1, 3, device=self.device)
            near = (fg0 - bg0).norm(dim=-1, keepdim=True) < 0.25
            bg0 = torch.where(near, (bg0 + 0.5) % 1.0, bg0)
            fgc = (fg0 + 0.15 * (torch.rand(B, self.R2, self.C2, 3,
                                            device=self.device) - 0.5)).clamp(0, 1)
            bgc = (bg0 + 0.15 * (torch.rand(B, self.R2, self.C2, 3,
                                            device=self.device) - 0.5)).clamp(0, 1)
            w = canvas.view(B, self.R2, self.ch, self.C2, self.cw) \
                .permute(0, 1, 3, 2, 4).clamp(0, 1)                    # (B,R2,C2,ch,cw)
            col = bgc[:, :, :, None, None, :] \
                + (fgc - bgc)[:, :, :, None, None, :] * w[..., None]
            col_canvas = col.permute(0, 1, 3, 2, 4, 5) \
                .reshape(B, self.R2 * self.ch, self.C2 * self.cw, 3)
        # global phase shift < phase_frac of a cell, then crop out the padded window
        max_dy = int(self.phase_frac * self.ch)
        max_dx = int(self.phase_frac * self.cw)
        base_t, base_l = self.ch - self.ph, self.cw - self.pw
        outs, couts = [], []
        for i in range(B):
            t = base_t + int(rng.integers(-max_dy, max_dy + 1))
            l = base_l + int(rng.integers(-max_dx, max_dx + 1))
            outs.append(canvas[i, t:t + self.win_h, l:l + self.win_w])
            if want_color:
                couts.append(col_canvas[i, t:t + self.win_h, l:l + self.win_w])
        x = torch.stack(outs)[:, None].clamp(0, 1)        # (B,1,win_h,win_w) ink=1
        y = labs.view(B, self.R2, self.C2)[:, 1:1 + self.rows, 1:1 + self.cols]
        y = y.reshape(B, self.rows * self.cols)
        if not want_color:
            return x, y
        xc = torch.stack(couts).permute(0, 3, 1, 2)       # (B,3,win_h,win_w)
        sl_r, sl_c = slice(1, 1 + self.rows), slice(1, 1 + self.cols)
        fg_t = fgc[:, sl_r, sl_c].reshape(B, -1, 3)
        bg_t = bgc[:, sl_r, sl_c].reshape(B, -1, 3)
        return x, y, xc, fg_t, bg_t


def st_sample_mosaic(logits, x, bm_white, rows, cols, ch, cw, ph, pw):
    """Stochastically sample the window distributions (Gumbel-max = categorical),
    render with a straight-through gradient, and tile render + matching targets
    into one mosaic pair for a single crop-augmented CLIP call.
    logits (Bs,T,N); x (Bs,1,win_h,win_w) ink=1. Returns (render, target, nonblank)."""
    Bs, T, N = logits.shape
    p = logits.softmax(-1)
    u = torch.rand_like(logits).clamp_(1e-9, 1 - 1e-9)
    g = (logits - torch.log(-torch.log(u))).argmax(-1)            # (Bs,T)
    hard = F.one_hot(g, N).float()
    st = hard + p - p.detach()                                    # ST estimator
    win = torch.einsum("btn,np->btp", st, bm_white)               # (Bs,T,ch*cw) white=1
    win = win.view(Bs, rows, cols, ch, cw).permute(0, 1, 3, 2, 4) \
        .reshape(Bs, rows * ch, cols * cw)
    tgt = 1.0 - x[:, 0, ph:ph + rows * ch, pw:pw + cols * cw]     # white=1
    k = math.ceil(math.sqrt(Bs))
    if k * k > Bs:                                                # pad with white tiles
        padn = k * k - Bs
        win = torch.cat([win, win.new_ones(padn, rows * ch, cols * cw)])
        tgt = torch.cat([tgt, tgt.new_ones(padn, rows * ch, cols * cw)])
    mos_r = win.view(k, k, rows * ch, cols * cw).permute(0, 2, 1, 3) \
        .reshape(k * rows * ch, k * cols * cw)
    mos_t = tgt.view(k, k, rows * ch, cols * cw).permute(0, 2, 1, 3) \
        .reshape(k * rows * ch, k * cols * cw)
    space_frac = float((bm_white[g.view(-1)].min(dim=1).values > 0.999).float().mean())
    return mos_r, mos_t, 1.0 - space_frac


def grad_norm_of(grads):
    return float(torch.sqrt(sum((g * g).sum() for g in grads if g is not None)))


@torch.no_grad()
def eval_clip_render(model, cache, stems, clipper, bm_white, device, use_clip,
                     batch=1024):
    """Label-free val metric: CLIP perceptual loss of the ARGMAX render of whole
    held-out grids vs their ink images, under seeded (reproducible) crops."""
    import random as pyrandom
    model.eval()
    ch, cw = cache.ch, cache.cw
    state = pyrandom.getstate()
    pyrandom.seed(1234)
    losses = []
    for stem in stems:
        ri = cache.stems.index(stem)
        gh, gw = cache.labels[ri].shape
        ys, xs = np.mgrid[0:gh, 0:gw]
        idx = np.stack([np.full(gh * gw, ri), ys.ravel(), xs.ravel()],
                       axis=1).astype(np.int32)
        pred = np.empty(gh * gw, dtype=np.int64)
        for i in range(0, len(idx), batch):
            w, _, _, emb = cache.fetch(idx[i:i + batch])
            xw = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
            ce = torch.from_numpy(emb).to(device) if use_clip else None
            pred[i:i + batch] = model(xw, ce).argmax(dim=1).cpu().numpy()
        r = bm_white[torch.from_numpy(pred).to(device)] \
            .view(gh, gw, ch, cw).permute(0, 2, 1, 3).reshape(gh * ch, gw * cw)
        ink = cache.ink[ri][cache.off_y:cache.off_y + gh * ch,
                            cache.off_x:cache.off_x + gw * cw]
        t = 1.0 - torch.from_numpy(ink.astype(np.float32) / 255.0).to(device)
        losses.append(float(clipper(r, t)))
    pyrandom.setstate(state)
    model.train()
    return float(np.mean(losses))


# ------------------------------------------------------------------------ optimizer

def _newton_schulz5(g, steps=5, eps=1e-7):
    """Orthogonalize g (approximate UV^T of its SVD) via 5 quintic NS iterations."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + eps)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    return x.T if tall else x


class MuonWithAdamW(torch.optim.Optimizer):
    """Muon (orthogonalized momentum) for hidden 2D matrices; AdamW for the rest.
    Groups carry use_muon=True/False. Update size is spectral (direction, not
    magnitude): whitened steps keep serving small-but-consistent directions."""

    def __init__(self, muon_params, adamw_params, muon_lr=0.02, muon_momentum=0.95,
                 adamw_lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01):
        groups = [dict(params=list(muon_params), use_muon=True, lr=muon_lr,
                       momentum=muon_momentum, weight_decay=weight_decay),
                  dict(params=list(adamw_params), use_muon=False, lr=adamw_lr,
                       betas=betas, weight_decay=weight_decay)]
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            if gr["use_muon"]:
                for p in gr["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "mom" not in st:
                        st["mom"] = torch.zeros_like(p)
                    buf = st["mom"].mul_(gr["momentum"]).add_(p.grad)
                    g = p.grad.add(buf, alpha=gr["momentum"])       # nesterov
                    u = _newton_schulz5(g.reshape(g.shape[0], -1)).view_as(g)
                    scale = max(1.0, g.shape[0] / g.reshape(g.shape[0], -1).shape[1]) ** 0.5
                    p.mul_(1 - gr["lr"] * gr["weight_decay"])
                    p.add_(u, alpha=-gr["lr"] * scale)
            else:
                for p in gr["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                        st["t"] = 0
                    st["t"] += 1
                    b1, b2 = gr["betas"]
                    st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                    st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    mh = st["m"] / (1 - b1 ** st["t"])
                    vh = st["v"] / (1 - b2 ** st["t"])
                    p.mul_(1 - gr["lr"] * gr["weight_decay"])
                    p.addcdiv_(mh, vh.sqrt().add_(1e-8), value=-gr["lr"])


def split_muon_params(model):
    """Hidden 2D matrices (attention/MLP/token-proj) -> Muon; trunk convs,
    embeddings/positions, norms, biases, and the class head -> AdamW."""
    muon, adamw = [], []
    for n, q in model.named_parameters():
        if not q.requires_grad:
            continue
        hidden = (q.ndim >= 2 and ("blocks." in n or n.startswith("proj.")))
        (muon if hidden else adamw).append(q)
    return muon, adamw


# ------------------------------------------------------------------------ training

def evaluate(model, cache, split, device, batch, space, use_clip, max_cells=None,
             ce_weight=None):
    model.eval()
    idx = cache.index[split]
    if max_cells is not None and len(idx) > max_cells:
        idx = idx[np.random.default_rng(0).permutation(len(idx))[:max_cells]]
    n = len(idx)
    correct = np.zeros(n, bool)
    labs = np.empty(n, np.int64)
    gres = np.empty(n, np.int64)
    loss_sum = loss_bw_sum = bw_den = 0.0
    tc = cache.rh * (2 * cache.rw + 1) + cache.rw     # center slot in window labels
    with torch.no_grad():
        for i in range(0, n, batch):
            w, lab, gre, emb = cache.fetch(idx[i:i + batch])
            if cache.window_labels:
                lab = lab[:, tc]
            x = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
            ce = torch.from_numpy(emb).to(device) if use_clip else None
            logits = model(x, ce)
            y = torch.from_numpy(lab).to(device)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
            if ce_weight is not None:
                wt = ce_weight[y]
                loss_bw_sum += float((F.cross_entropy(logits, y, reduction="none") * wt).sum())
                bw_den += float(wt.sum())
            pred = logits.argmax(dim=1).cpu().numpy()
            correct[i:i + batch] = pred == lab
            labs[i:i + batch] = lab
            gres[i:i + batch] = gre
    model.train()
    nb = labs != space
    dis = labs != gres                      # optimizer disagreed with greedy snap
    m = dict(
        val_loss=loss_sum / n,
        val_loss_blankweighted=(loss_bw_sum / bw_den) if ce_weight is not None else None,
        top1=float(correct.mean()),
        nonblank_top1=float(correct[nb].mean()),
        greedy_top1=float((gres == labs).mean()),
        greedy_nonblank_top1=float((gres[nb] == labs[nb]).mean()),
        agree_set_top1=float(correct[~dis].mean()),
        disagree_set_top1=float(correct[dis].mean()),
        disagree_frac=float(dis.mean()),
        disagree_nonblank_top1=float(correct[dis & nb].mean()),
        n_cells=int(n),
    )
    # macro over classes with enough support (the tail is noise otherwise)
    accs = [float(correct[labs == c].mean()) for c in np.unique(labs)
            if (labs == c).sum() >= 20]
    m["balanced_top1_min20"] = float(np.mean(accs))
    m["n_classes_min20"] = len(accs)
    return m


# ------------------------------------------------------------------ live monitoring

@torch.no_grad()
def render_stems(model, cache, stems, device, use_clip, sampler, path, batch=1024):
    """Forward-pass the given runs and save a 2-up grid of rendered predictions."""
    from PIL import Image, ImageDraw
    model.eval()
    panels = []
    for stem in stems:
        ri = cache.stems.index(stem)
        gh, gw = cache.labels[ri].shape
        ys, xs = np.mgrid[0:gh, 0:gw]
        idx = np.stack([np.full(gh * gw, ri), ys.ravel(), xs.ravel()], axis=1).astype(np.int32)
        pred = np.empty(gh * gw, dtype=np.int64)
        for i in range(0, len(idx), batch):
            w, _, _, emb = cache.fetch(idx[i:i + batch])
            x = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
            ce = torch.from_numpy(emb).to(device) if use_clip else None
            pred[i:i + batch] = model(x, ce).argmax(dim=1).cpu().numpy()
        pred = pred.reshape(gh, gw)
        match = float((pred == cache.labels[ri]).mean())
        img = (sampler.render(torch.from_numpy(pred)).cpu().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(img).convert("RGB")
        bar = Image.new("RGB", (im.width, 20), (240, 240, 240))
        ImageDraw.Draw(bar).text((5, 4), f"{stem}  ({match:.1%} match)", fill=(0, 0, 0))
        pane = Image.new("RGB", (im.width, im.height + 20), (255, 255, 255))
        pane.paste(bar, (0, 0)); pane.paste(im, (0, 20))
        panels.append(pane)
    model.train()
    cols = 2
    rows_n = (len(panels) + cols - 1) // cols
    W = max(p.width for p in panels)
    H = max(p.height for p in panels)
    sheet = Image.new("RGB", (cols * W + 8, rows_n * H + 8), (255, 255, 255))
    for k, pane in enumerate(panels):
        sheet.paste(pane, ((k % cols) * (W + 8), (k // cols) * (H + 8)))
    sheet.save(path)


def live_chart(hist, ev_hist, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=120, facecolor="white")
    s = np.array(hist["step"]); L = np.array(hist["loss"])
    axes[0].plot(s, L, color="#2a78d6", lw=0.6, alpha=0.3)
    if len(L) > 100:
        k = np.ones(100) / 100
        axes[0].plot(s[99:], np.convolve(L, k, "valid"), color="#2a78d6", lw=2,
                     label="train loss (100-step mean)")
    else:
        axes[0].plot([], [], color="#2a78d6", lw=2, label="train loss")
    if any(c > 0 for c in hist["cons"]):
        axes[0].plot(s, np.array(hist["cons"]), color="#1baf7a", lw=0.6, alpha=0.5,
                     label="consistency term")
    if any(c > 0 for c in hist.get("clip", [])):
        axes[0].plot(s, np.array(hist["clip"]), color="#9b59b6", lw=0.6, alpha=0.5,
                     label="clip term (ST sample)")
    vclip = np.array(ev_hist.get("val_clip", []), dtype=float)
    if len(vclip) and not np.all(np.isnan(vclip)):
        axes[0].plot(ev_hist["step"], vclip, color="#9b59b6", lw=2, marker="s",
                     ms=3, label="val clip (argmax render)")
    if ev_hist["step"]:
        axes[0].plot(ev_hist["step"], ev_hist["val_loss_bw"], color="#eb6834", lw=2,
                     marker="o", ms=3, label="val loss (blank-weighted)")
    axes[0].set_title("loss", loc="left"); axes[0].legend(frameon=False, fontsize=8)
    if ev_hist["step"]:
        for key, c, lbl in [("top1", "#2a78d6", "val top-1"),
                            ("nonblank_top1", "#eb6834", "val non-blank"),
                            ("disagree_set_top1", "#1baf7a", "val disagree-set")]:
            axes[1].plot(ev_hist["step"], ev_hist[key], color=c, lw=2, label=lbl)
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("validation accuracy", loc="left")
    for ax in axes:
        ax.grid(True, color="#eeeeee", lw=0.8)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlabel("step", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="runs/cellclf/cache")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--clip-token", action="store_true",
                   help="prepend the down-projected global CLIP embedding (tf variants only)")
    p.add_argument("--dim", type=int, default=64, help="transformer model width")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--blocks", type=int, default=1, help="transformer depth (tf variants)")
    p.add_argument("--dense", action="store_true",
                   help="predict EVERY window cell (binomial center-weighted CE), not just the center")
    p.add_argument("--consistency-weight", type=float, default=0.0,
                   help="symmetric-KL between two 1-cell-shifted framings on their overlapping "
                        "cells (requires --dense); makes adjacent-cell errors coherent")
    p.add_argument("--consistency-decay", type=float, default=1.0,
                   help="end-of-run consistency weight as a fraction of the start, linear "
                        "ramp (0.25 = decay to a quarter; stabilizes early, frees late "
                        "commitment). 1.0 = constant")
    p.add_argument("--render-frac", type=float, default=0.0,
                   help="fraction of steps trained on synthetic letter-render batches "
                        "(jittered re-renders of real label windows; requires --dense)")
    p.add_argument("--in-ch", type=int, default=1, choices=[1, 4],
                   help="4 = color-ready trunk from step 0: [ink, RGB] input. "
                        "Plain batches feed 1ch (the _embed shim expands); "
                        "--color-frac batches feed COLORED line art / letters so "
                        "the RGB channels are alive before any color surgery")
    p.add_argument("--color-frac", type=float, default=0.5,
                   help="(--in-ch 4) fraction of batches with colorized input: "
                        "per-window fg-on-bg palette (separation enforced), "
                        "labels unchanged")
    p.add_argument("--mask-dec", action="store_true",
                   help="(--in-ch 4) attach the MaskDecoder and BOOTSTRAP it "
                        "from step 0: dense plain-CE toward (ink, 1-ink, "
                        "corruption-alpha) on every colorized batch. Plain CE "
                        "= forced commitment -- the free-abstain equilibrium "
                        "that drained fg mass has no home here. The color "
                        "chain inherits a WORKING decoder instead of birthing "
                        "one at surgery time")
    p.add_argument("--mask-ce-weight", type=float, default=1.0)
    p.add_argument("--color-corrupt", type=float, default=0.5,
                   help="(--in-ch 4) probability a colorized SYNTH batch gets "
                        "light colored corruptions (smudges/particles/blur); "
                        "dense cache pairs stay corruption-free so the "
                        "consistency loss compares like with like")
    p.add_argument("--clip-weight-target", type=float, default=0.0,
                   help="CLIP-through-the-sampler term: target ratio ||g_clip||/||g_ce|| "
                        "(e.g. 0.1); lambda recalibrated dynamically. Requires --dense")
    p.add_argument("--clip-sub", type=int, default=64,
                   help="windows per step entering the CLIP mosaic (unbiased subsample)")
    p.add_argument("--clip-aug", type=int, default=8, help="CLIP crops per mosaic")
    p.add_argument("--clip-start", type=int, default=500,
                   help="steps of pure CE before the CLIP term activates "
                        "(lets the space economy establish first)")
    p.add_argument("--clip-recal-every", type=int, default=100,
                   help="recalibrate lambda from measured grad norms every N clip steps")
    p.add_argument("--eval-clip", action="store_true",
                   help="report the label-free val metric (CLIP loss of argmax renders "
                        "of held-out grids) even when the CLIP train term is off")
    p.add_argument("--holdout-parents", default="",
                   help="comma-separated parent names moved to a 'viz' split: held out of "
                        "training without changing val")
    p.add_argument("--train-on-all", action="store_true",
                   help="RELEASE mode: fold val+viz into training (eval numbers become "
                        "informational only); train to a fixed step count")
    p.add_argument("--live-every", type=int, default=0,
                   help="every N steps rewrite live_loss.png and live_render.png in the run dir")
    p.add_argument("--live-stems", default="",
                   help="comma-separated run stems rendered through the model at live updates")
    p.add_argument("--vae-ckpt", default="weights/vae_dejavu/model.pt",
                   help="for the glyph renderer used by --live-stems")
    p.add_argument("--profile", default="dejavu",
                   help="font kit profile (dejavu / sfmono); must match the cache + vae")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="save ckpt_step<N>.pt every N steps (early termination stays usable)")
    p.add_argument("--optim", default="adamw", choices=["adamw", "muon"],
                   help="muon = orthogonalized momentum on hidden 2D matrices "
                        "(direction-not-magnitude updates), AdamW on the rest")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--blank-weight", type=float, default=0.2,
                   help="CE weight on the space class (train_vae crossmodal idiom)")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-cells", type=int, default=40000, help="subsample for mid-run evals")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.clip_token and not VARIANTS[args.variant][2]:
        p.error("--clip-token requires a tf variant")
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cache_dir = G.repo_path(args.cache)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    rows, cols, is_tf = VARIANTS[args.variant]
    if args.dense and not is_tf:
        p.error("--dense requires a tf variant")
    if (args.render_frac > 0 or args.clip_weight_target > 0) and not args.dense:
        p.error("--render-frac / --clip-weight-target require --dense")
    viz_parents = tuple(s for s in args.holdout_parents.split(",") if s)
    args.pad_h, args.pad_w = meta["pad_h"], meta["pad_w"]   # into vars(args) ->
    args.cell_h, args.cell_w = meta["cell_h"], meta["cell_w"]   # self-describing ckpts
    cache = CellCache(cache_dir, rows, cols,
                      meta["pad_h"], meta["pad_w"], meta["cell_h"], meta["cell_w"],
                      window_labels=args.dense, viz_parents=viz_parents)
    if args.train_on_all:
        cache.index["train"] = np.concatenate(
            [cache.index[s] for s in ("train", "val", "viz") if len(cache.index[s])])
    N = len(cache.chars)

    if is_tf:
        model = TokenTransformer(rows, cols, meta["cell_h"], meta["cell_w"],
                                 meta["pad_h"], meta["pad_w"], N, dim=args.dim,
                                 heads=args.heads, n_blocks=args.blocks,
                                 clip_dim=cache.clip_dim if args.clip_token else 0,
                                 in_ch=args.in_ch)
        if args.mask_dec:
            if args.in_ch != 4:
                p.error("--mask-dec requires --in-ch 4")
            model.mask_dec = MaskDecoder(args.dim)
    else:
        model = WindowCNN(cache.win_h, cache.win_w, N)
    model = model.to(device).train()
    n_params = sum(x.numel() for x in model.parameters())

    name = args.name or (args.variant
                         + ("_dense" if args.dense else "")
                         + (f"_clip" if args.clip_token else "")
                         + (f"_b{args.blocks}" if is_tf and args.blocks > 1 else "")
                         + (f"_h{args.heads}" if is_tf else "")
                         + ("_c4" if args.in_ch == 4 else "")
                         + (f"_rf{int(args.render_frac * 100)}" if args.render_frac > 0 else "")
                         + (f"_ct{int(args.clip_weight_target * 100)}"
                            if args.clip_weight_target > 0 else "")
                         + ("_muon" if args.optim == "muon" else ""))
    outdir = os.path.join(G.REPO_ROOT, "runs", "cellclf", name)
    os.makedirs(outdir, exist_ok=True)
    print(f"[{name}] {args.variant} window {rows}x{cols} cells | {n_params/1e6:.2f}M params "
          f"| {len(cache.index['train'])} train / {len(cache.index['val'])} val "
          f"/ {len(cache.index['viz'])} viz cells | {device}")

    live_stems = [s for s in args.live_stems.split(",") if s]
    sampler = None
    need_glyphs = args.render_frac > 0 or args.clip_weight_target > 0 or args.eval_clip
    if (args.live_every and live_stems) or need_glyphs:
        from unicasso.adapter.corrupt import CorruptionSampler
        sampler = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu",
                                    profile=args.profile)

    bm_white = ink_bm = clipper = synth = None
    clip_val_stems = []
    if need_glyphs:
        ink_bm = 1.0 - sampler.bitmaps.cpu().float()             # (N, ch, cw) ink=1
        bm_white = (1.0 - ink_bm).reshape(sampler.N, -1).to(device)   # white=1, flat
        from unicasso.engine.clip_loss import CLIPPerceptualLoss
        clipper = CLIPPerceptualLoss(torch.device(device), model_name="RN101",
                                     pretrained="openai", n_aug=args.clip_aug,
                                     crop_scale=(0.4, 0.9),
                                     batch_aug=(device == "cuda"))
        clip_val_stems = [r["stem"] for r in cache.meta["runs"]
                          if r["split"] == "val"][:4]
    if args.render_frac > 0:
        synth = SynthRenderer(cache, ink_bm, rows, cols,
                              meta["cell_h"], meta["cell_w"],
                              meta["pad_h"], meta["pad_w"], device)

    ce_w = torch.ones(N, device=device)
    ce_w[cache.space] = args.blank_weight
    if args.optim == "muon":
        mu, ad = split_muon_params(model)
        print(f"  muon: {sum(q.numel() for q in mu)/1e3:.0f}k params orthogonalized, "
              f"{sum(q.numel() for q in ad)/1e3:.0f}k on AdamW")
        opt = MuonWithAdamW(mu, ad, muon_lr=args.muon_lr,
                            muon_momentum=args.muon_momentum, adamw_lr=args.lr,
                            weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    def lr_lambda(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tr_idx = cache.index["train"]
    hist = {"step": [], "loss": [], "ce": [], "cons": [], "acc": [],
            "clip": [], "lam": []}
    ev_hist = {"step": [], "val_loss": [], "val_loss_bw": [], "top1": [],
               "nonblank_top1": [], "disagree_set_top1": [], "val_clip": []}
    if args.dense:
        kernel = binomial_kernel(rows, cols, device)
        maps = shift_token_maps(rows, cols, device)
        shifts = [(sy, sx) for sy in (-1, 0, 1) for sx in (-1, 0, 1) if (sy, sx) != (0, 0)]
        tc = (rows // 2) * cols + cols // 2

    def record_eval(step, m):
        ev_hist["step"].append(step)
        ev_hist["val_loss"].append(m["val_loss"])
        ev_hist["val_loss_bw"].append(m["val_loss_blankweighted"])
        ev_hist["top1"].append(m["top1"])
        ev_hist["nonblank_top1"].append(m["nonblank_top1"])
        ev_hist["disagree_set_top1"].append(m["disagree_set_top1"])
        ev_hist["val_clip"].append(m.get("val_clip_render", float("nan")))
    in4 = args.in_ch == 4

    def colorize_ink(x, pal=None, corrupt=False):
        """(B,1,H,W) ink -> 4ch [ink, fg-on-bg RGB]. Labels untouched -- the
        art under the paint is the same. pal reuses a palette (shifted dense
        pairs must agree so the consistency loss compares like with like)."""
        B_ = x.shape[0]
        if pal is None:
            fg = torch.rand(B_, 3, 1, 1, device=x.device)
            bg = torch.rand(B_, 3, 1, 1, device=x.device)
            near = (fg - bg).abs().mean(1, keepdim=True) < 0.15
            bg = torch.where(near, (bg + 0.5) % 1.0, bg)
        else:
            fg, bg = pal
        col = bg + (fg - bg) * x
        ca = None
        if corrupt:
            col, ca = corrupt_color_batch(col, rng, n_smudge=2, n_blob=3,
                                          blur_p=0.3, ink=x[:, 0])
        return torch.cat([x, col], dim=1), (fg, bg), ca

    def mask_ce_base(mlog, xw, ca):
        """Dense plain-CE mask bootstrap: fg = the clean rendered ink, bg =
        complement, abstain = corruption alpha (zero on uncorrupted batches).
        PLAIN CE = forced per-pixel commitment -- no free abstain during
        bootstrap; the wrong-vote flexibility belongs to later phases."""
        B_ = xw.shape[0]

        def cores(img):
            u = img.unfold(2, model.th, model.sh).unfold(3, model.tw, model.sw)
            u = u.reshape(B_, 1, rows * cols, model.th, model.tw)[:, 0]
            return u[..., cache.ph:cache.ph + cache.ch,
                     cache.pw:cache.pw + cache.cw]

        tf = cores(xw[:, :1])
        ta = cores(ca[:, None]) if ca is not None else torch.zeros_like(tf)
        tgt = torch.stack([tf * (1 - ta), (1 - tf) * (1 - ta), ta], dim=2)
        lp = F.log_softmax(mlog[:, :, :, cache.ph:cache.ph + cache.ch,
                                cache.pw:cache.pw + cache.cw], dim=2)
        cepix = -(tgt * lp).sum(2).mean((-1, -2))          # (B,T)
        return ((cepix * kernel[None, :]).sum(1) / kernel.sum()).mean()

    t0 = time.time()
    perm, pi = rng.permutation(len(tr_idx)), 0
    lam = None                                   # dynamic CLIP weight, set at recal
    clip_nb = -1.0                               # sampled-render non-blank frac (monitor)
    for step in range(args.steps):
        if pi + args.batch > len(perm):
            perm, pi = rng.permutation(len(tr_idx)), 0
        b = tr_idx[perm[pi:pi + args.batch]]; pi += args.batch
        is_synth = synth is not None and rng.random() < args.render_frac
        do_col = in4 and rng.random() < args.color_frac
        if is_synth:
            calpha = None
            if do_col:
                x, y, xc, _, _ = synth.sample(b, rng, want_color=True)
                if rng.random() < args.color_corrupt:
                    xc, calpha = corrupt_color_batch(xc, rng, n_smudge=2,
                                                     n_blob=3, blur_p=0.3,
                                                     ink=x[:, 0])
                x = torch.cat([x, xc], dim=1)
            else:
                x, y = synth.sample(b, rng)
            ce = (torch.zeros(len(b), cache.clip_dim, device=device)
                  if args.clip_token else None)
            if args.mask_dec and do_col:
                t_, s1_ = model._tokens(x, ce, want_skip=True)
                l1 = model.head(t_[:, model.n_extra:])
                mloss = mask_ce_base(model.mask_logits(t_, s1_), x, calpha)
            else:
                l1 = model.forward_all(x, ce)
                mloss = None
            ce_loss = kernel_ce(l1, y, kernel, ce_w)
            cons = l1.new_zeros(())
            loss = ce_loss
            if mloss is not None:
                loss = loss + args.mask_ce_weight * mloss
            logits, y_c = l1[:, tc], y[:, tc]
        else:
            w, lab, _, emb = cache.fetch(b)
            x = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
            y = torch.from_numpy(lab).to(device)
            ce = torch.from_numpy(emb).to(device) if args.clip_token else None
            pal = None
            if do_col:                  # palette only: pairs must match
                x, pal, _ = colorize_ink(x)
            if args.dense:
                sy, sx = shifts[int(rng.integers(len(shifts)))]
                b2, ady, adx = cache.clamp_shift(b, sy, sx)
                w2, lab2, _, _ = cache.fetch(b2)
                x2 = torch.from_numpy(w2).to(device).float().div_(255).unsqueeze(1)
                y2 = torch.from_numpy(lab2).to(device)
                if do_col:
                    x2, _, _ = colorize_ink(x2, pal=pal)
                if args.mask_dec and do_col:
                    t_, s1_ = model._tokens(x, ce, want_skip=True)
                    l1 = model.head(t_[:, model.n_extra:])
                    mloss = mask_ce_base(model.mask_logits(t_, s1_), x, None)
                else:
                    l1 = model.forward_all(x, ce)
                    mloss = None
                l2 = model.forward_all(x2, ce)
                ce_loss = 0.5 * (kernel_ce(l1, y, kernel, ce_w) + kernel_ce(l2, y2, kernel, ce_w))
                if args.consistency_weight > 0:
                    cons = consistency_loss(l1, l2,
                                            torch.from_numpy(ady.astype(np.int64)).to(device),
                                            torch.from_numpy(adx.astype(np.int64)).to(device),
                                            maps)
                else:
                    cons = l1.new_zeros(())
                cw_now = args.consistency_weight * (
                    1.0 + (args.consistency_decay - 1.0)
                    * step / max(1, args.steps - 1))
                loss = ce_loss + cw_now * cons
                if mloss is not None:
                    loss = loss + args.mask_ce_weight * mloss
                logits, y_c = l1[:, tc], y[:, tc]
            else:
                logits = model(x, ce)
                y_c = y
                loss = F.cross_entropy(logits, y_c, weight=ce_w)
                ce_loss, cons = loss, loss.new_zeros(())
        clip_l = None
        if (args.clip_weight_target > 0 and not is_synth
                and (step + 1) >= args.clip_start):
            Bs = min(args.clip_sub, l1.shape[0])
            mos_r, mos_t, clip_nb = st_sample_mosaic(
                l1[:Bs], x[:Bs, :1], bm_white, rows, cols,
                cache.ch, cache.cw, cache.ph, cache.pw)
            clip_l = clipper(mos_r, mos_t)
        opt.zero_grad(set_to_none=True)
        recal = clip_l is not None and (
            lam is None or (step + 1 - args.clip_start) % args.clip_recal_every == 0)
        if recal:
            params = [q for q in model.parameters() if q.requires_grad]
            g_ce = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            g_cl = torch.autograd.grad(clip_l, params, allow_unused=True)
            n_ce, n_cl = grad_norm_of(g_ce), grad_norm_of(g_cl)
            lam_new = args.clip_weight_target * n_ce / max(n_cl, 1e-8)
            lam = lam_new if lam is None else 0.9 * lam + 0.1 * lam_new
            for q, a, c in zip(params, g_ce, g_cl):
                gg = a.clone() if a is not None else None
                if c is not None:
                    gg = lam * c if gg is None else gg + lam * c
                q.grad = gg
            print(f"  [lam@{step+1}] ||g_ce|| {n_ce:.3e} ||g_clip|| {n_cl:.3e} "
                  f"-> lam {lam:.4f} (nb {clip_nb:.2f})", flush=True)
        elif clip_l is not None:
            (loss + lam * clip_l).backward()
        else:
            loss.backward()
        opt.step()
        sched.step()
        acc = float((logits.argmax(1) == y_c).float().mean())
        hist["step"].append(step + 1); hist["loss"].append(float(loss))
        hist["ce"].append(float(ce_loss)); hist["cons"].append(float(cons))
        hist["acc"].append(acc)
        hist["clip"].append(float(clip_l) if clip_l is not None else 0.0)
        hist["lam"].append(lam if lam is not None else 0.0)
        if (step + 1) % 100 == 0:
            cl = f" clip {hist['clip'][-1]:.3f} lam {hist['lam'][-1]:.3f} nb {clip_nb:.2f}" \
                if args.clip_weight_target > 0 else ""
            print(f"  step {step+1}/{args.steps} loss {float(loss):.3f} "
                  f"(ce {float(ce_loss):.3f} cons {float(cons):.3f}{cl}) batch-acc {acc:.3f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            m = evaluate(model, cache, "val", device, 2048, cache.space,
                         args.clip_token, max_cells=args.eval_cells, ce_weight=ce_w)
            if clipper is not None:
                m["val_clip_render"] = eval_clip_render(
                    model, cache, clip_val_stems, clipper, bm_white, device,
                    args.clip_token)
            record_eval(step + 1, m)
            vc = (f" val-clip {m['val_clip_render']:.4f}"
                  if "val_clip_render" in m else "")
            print(f"  [val@{step+1}] loss {m['val_loss']:.3f} top1 {m['top1']:.3f} "
                  f"nonblank {m['nonblank_top1']:.3f} "
                  f"disagree {m['disagree_set_top1']:.3f}{vc}", flush=True)
        if args.live_every and (step + 1) % args.live_every == 0:
            live_chart(hist, ev_hist, os.path.join(outdir, "live_loss.png"), name)
            if sampler is not None:
                render_stems(model, cache, live_stems, device, args.clip_token,
                             sampler, os.path.join(outdir, "live_render.png"))
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            torch.save({"state_dict": model.state_dict(), "config": vars(args),
                        "chars": cache.chars, "variant": args.variant, "step": step + 1},
                       os.path.join(outdir, f"ckpt_step{step + 1:05d}.pt"))

    final = evaluate(model, cache, "val", device, 2048, cache.space, args.clip_token,
                     ce_weight=ce_w)
    if clipper is not None:
        final["val_clip_render"] = eval_clip_render(
            model, cache, clip_val_stems, clipper, bm_white, device, args.clip_token)
    record_eval(args.steps, final)
    final.update(train_minutes=round((time.time() - t0) / 60, 1),
                 n_params=n_params, config=vars(args), name=name)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=1)
    np.savez(os.path.join(outdir, "loss_hist.npz"),
             **{k: np.array(v) for k, v in hist.items()},
             **{"ev_" + k: np.array(v, dtype=float) for k, v in ev_hist.items()})
    torch.save({"state_dict": model.state_dict(), "config": vars(args),
                "chars": cache.chars, "variant": args.variant}, os.path.join(outdir, "model.pt"))
    with open(os.path.join(G.REPO_ROOT, "runs", "cellclf", "summary.jsonl"), "a") as f:
        f.write(json.dumps({k: v for k, v in final.items() if k != "config"}) + "\n")
    vc = (f" | val-clip {final['val_clip_render']:.4f}"
          if "val_clip_render" in final else "")
    print(f"[{name}] FINAL top1 {final['top1']:.4f} | nonblank {final['nonblank_top1']:.4f} | "
          f"balanced {final['balanced_top1_min20']:.4f} | "
          f"disagree-set {final['disagree_set_top1']:.4f} (frac {final['disagree_frac']:.3f}) | "
          f"greedy top1 {final['greedy_top1']:.4f}{vc}")


if __name__ == "__main__":
    main()
