"""unicasso-lite: distilled per-cell glyph models. Instant photo -> ANSI.

The lite models replace the swarm optimizer with a single forward pass
(~0.1 ms/cell on Apple Silicon; a w90 photo renders in ~0.35 s end to end):

    python -m unicasso.lite photo.jpg --width 60           # 24-bit ANSI to stdout
    python -m unicasso.lite photo.jpg -w 80 --out img.ans  # write a cat-able file
    python -m unicasso.lite drawing.png --line             # monochrome ASCII art (plain text)

Library use:

    from unicasso.lite import Lite
    lite = Lite("color")                  # or Lite("line")
    out = lite.render("photo.jpg", width=60)
    print(out.ans)                        # out.txt / out.glyphs / out.render also set

Pipeline (color, v2): decompose (per-cell Lab two-color clustering) -> ink
structure + RGB -> transformer glyph classifier and a per-pixel fg/bg/abstain
mask off the same tokens -> closed-form fg/bg fit through the mask -> ANSI.
The v1 models (framing-ensemble glyph read, distance-weighted colour fit and a
learned per-cell contrast k) load the same way; each checkpoint records how it
is meant to be read and Lite follows it.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

from unicasso.substrate import glyphs as G, raster

K_BIAS = math.log(1.0 / 3.0)                  # k-head: 4*sigmoid(raw+K_BIAS), k(0)=1
DEFAULT_WEIGHTS = {
    ("color", "dejavu"): "weights/lite/unicasso-lite-color.pt",
    ("line", "dejavu"): "weights/lite/unicasso-lite-line.pt",
    ("color", "sfmono"): "weights/lite/unicasso-lite-color-sfmono.pt",
    ("line", "sfmono"): "weights/lite/unicasso-lite-line-sfmono.pt",
}


def _grid_windows(img, gh, gw, ch, cw, ph, pw, rows, cols, pad_val=0,
                  centers=None, edge=False):
    py, px = (rows // 2) * ch + ph, (cols // 2) * cw + pw
    spec = ((py, py), (px, px)) + (((0, 0),) if img.ndim == 3 else ())
    pad = np.pad(img, spec, mode="edge") if edge else \
        np.pad(img, spec, constant_values=pad_val)
    wh, ww = rows * ch + 2 * ph, cols * cw + 2 * pw
    if centers is None:
        centers = [(y, x) for y in range(gh) for x in range(gw)]
    out = np.empty((len(centers), wh, ww) + ((3,) if img.ndim == 3 else ()), img.dtype)
    for i, (y, x) in enumerate(centers):
        t = py + y * ch - (rows // 2) * ch - ph
        l = px + x * cw - (cols // 2) * cw - pw
        out[i] = pad[t:t + wh, l:l + ww]
    return out


def _token_maps(gh, gw, rows, cols, centers=None):
    offs = [(t // cols - rows // 2, t % cols - cols // 2) for t in range(rows * cols)]
    if centers is None:
        ys, xs = np.mgrid[0:gh, 0:gw]
        cy, cx = ys.ravel(), xs.ravel()
    else:
        c = np.asarray(centers)
        cy, cx = c[:, 0], c[:, 1]
    yy = cy[:, None] + np.array([o[0] for o in offs])[None]
    xx = cx[:, None] + np.array([o[1] for o in offs])[None]
    valid = (yy >= 0) & (yy < gh) & (xx >= 0) & (xx < gw)
    return np.where(valid, yy * gw + xx, 0).astype(np.int64), valid


def _strided_centers(gh, gw, s):
    """Window centers at every s-th cell, last row/col always included; the 5x3
    windows still cover every grid cell for s <= 3 (rows) / s <= 5 (cols)."""
    ys = sorted(set(list(range(0, gh, s)) + [gh - 1]))
    xs = sorted(set(list(range(0, gw, s)) + [gw - 1]))
    return [(y, x) for y in ys for x in xs]


def _binomial_kernel(rows, cols):
    from math import comb
    v = torch.tensor([comb(rows - 1, i) for i in range(rows)], dtype=torch.float32)
    h = torch.tensor([comb(cols - 1, j) for j in range(cols)], dtype=torch.float32)
    return torch.outer(v / v[rows // 2], h / h[cols // 2]).flatten()


def _conf_fit(pm, C, mode="topk", frac=0.5, weight="prob", ridge=0.0,
              count="auto", temp=1.0):
    """Confidence-gated color fit -- an alternative READ of the same 3-way mask.

    `mask_fit` lets every pixel vote with its probability, so each color is a
    mean over confident and undecided pixels alike. With the abstain channel
    sitting near-unused in practice, nothing is ever actually excluded: the
    half-decided pixels along every stroke boundary pull fg and bg toward each
    other, and the cell ships washed out. Here each LAYER keeps a subset of the
    pixels and everything else is dropped exactly as abstain would be (no vote
    in either color). Two ways to pick that subset:

      topk   : deterministic -- the highest-probability pixels of that layer.
      sample : weighted sampling WITHOUT replacement at that layer's own
               probabilities, drawn to the same size. Implemented as Gumbel-
               top-k (perturb the log-probs with Gumbel noise, then take the
               top k), which is exactly repeated proportional draws without
               replacement, and degenerates to `topk` as the noise vanishes.
               Every pixel with mass can be drawn, so the selection is not
               self-reinforcing the way a fixed top-k is -- the read a
               sampling-based training objective would actually see. `temp`
               sharpens the distribution before drawing (p^(1/temp)): 1 = the
               layer's own probabilities, -> 0 recovers `topk` exactly, so it
               dials continuously between exploring and concentrating.

    How many pixels each layer keeps (`count`):
      fixed  : frac * P, the same budget for both layers regardless of how the
               cell is split.
      argmax : frac * (pixels this layer would win outright). A layer that owns
               a thin stroke keeps a few pixels; one that owns the field keeps
               many; a layer that wins nothing keeps nothing and falls back to
               the cell mean.
      auto   : 'fixed' for topk, 'argmax' for sample.

    pm (M,K,ch,cw) softmaxed probs, channel 0 = fg, 1 = bg. C (M,P,3) the cell's
    OWN pixels, P = ch*cw, row-major -- callers pass the core crop, never the
    padded patch, since the margin is a neighbour's territory.
    `weight`: 'prob' keeps the probability as the vote weight within the kept
    set, 'uniform' gives every kept pixel an equal vote.
    `ridge` (pixel-mass units) pulls a layer toward the plain cell mean; 0 =
    no pull, and a layer that keeps nothing falls back to the mean outright.
    """
    M, K = pm.shape[0], pm.shape[1]
    p = pm.reshape(M, K, -1)
    P = p.shape[2]
    mean = C.mean(1)
    if count == "auto":
        count = "argmax" if mode == "sample" else "fixed"
    am = p.argmax(1) if count == "argmax" else None
    sel = []
    for c in (0, 1):
        pc = p[:, c]
        if count == "argmax":
            kc = (frac * (am == c).sum(1).to(p.dtype)).round().long().clamp(0, P)
        else:
            kc = pc.new_full((M,), max(1, min(P, int(round(frac * P))))).long()
        if mode == "sample":     # Gumbel-top-k == proportional draws, no replacement
            u = torch.rand_like(pc).clamp_(1e-20, 1.0)
            keys = pc.clamp_min(1e-20).log() / max(temp, 1e-6) - (-u.log()).log()
        else:
            keys = pc
        rank = keys.argsort(1, descending=True).argsort(1)
        sel.append((rank < kc[:, None]).to(p.dtype))
    out = []
    for c in (0, 1):
        w = sel[c] if weight == "uniform" else sel[c] * p[:, c]
        s = w.sum(1, keepdim=True)
        col = ((w[..., None] * C).sum(1) + ridge * mean) / (s + ridge).clamp_min(1e-8)
        out.append(torch.where(s + ridge > 1e-6, col, mean))   # empty layer -> mean
    return out[0], out[1]


@dataclass
class LiteResult:
    glyphs: np.ndarray            # (gh, gw) int glyph indices
    txt: str                      # plain glyph text
    ans: str | None               # 24-bit ANSI (color model only)
    fg: np.ndarray | None         # (M, 3) cell foreground colors
    bg: np.ndarray | None
    k: np.ndarray | None          # (gh, gw) learned contrast field
    render: np.ndarray | None     # (H, W, 3) float render (color) or None
    masks: np.ndarray | None = None   # (M, 3, ch, cw) fg/bg/abstain probs
    # (mask models only -- the colorize pass's per-pixel layer assignment)


class Lite:
    def __init__(self, kind_or_path="color", device=None, font=None):
        """kind_or_path: "color" / "line" (resolved via `font`) or a ckpt path.
        font: "dejavu" (default) or "sfmono"; falls back to $GLYPHVAE_FONT.
        A checkpoint path ignores `font` -- its config names its own kit."""
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        font = font or os.environ.get("GLYPHVAE_FONT", "dejavu")
        path = DEFAULT_WEIGHTS.get((kind_or_path, font), kind_or_path)
        self.ckpt_path = G.repo_path(path)
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.chars = ck["chars"]
        self.color = bool(cfg.get("color_model"))
        # how the glyph head should be READ. v2 colour models train the main
        # head at the centre position only (neighbours go through an auxiliary
        # head), so the 15-window framing ensemble would average in positions
        # that were never supervised; they stamp render_ensemble="center".
        # v1 models (and the line models) were trained under the ensemble.
        self.render_ensemble = cfg.get("render_ensemble") or "mean"
        from unicasso.adapter.corrupt import CorruptionSampler
        profile = cfg.get("profile") or font   # font already defaulted above
        vae = {"dejavu": "weights/vae_dejavu/model.pt",
               "sfmono": "weights/vae_sfmono/model.pt"}[profile]
        # which colorize STRUCTURE this checkpoint was trained under:
        #   'two' -- glyph pass, then the chosen glyphs re-rendered into ch0 and
        #            a SECOND forward for the mask (the historic path)
        #   'one' -- one pass on [decompose ink, photo RGB]; head gives glyphs and
        #            mask_dec reads the SAME tokens. No glyph in the input, so the
        #            mask cannot copy the silhouette out of the skip.
        self.mask_forward = cfg.get("mask_forward", "two")
        self.align_blend = bool(cfg.get("align_blend"))
        self.mask_margin = bool(cfg.get("mask_margin_fit"))
        self.profile = profile
        sampler = CorruptionSampler(G.repo_path(vae), device="cpu",
                                    profile=profile)
        self.ch, self.cw = sampler.CH, sampler.CW
        pads = {"dejavu": (3, 4), "sfmono": (4, 3)}[profile]   # kit token margins
        self.ph = cfg.get("pad_h", pads[0])
        self.pw = cfg.get("pad_w", pads[1])
        self.rows, self.cols = 3, 5
        self.ink_flat = (1.0 - sampler.bitmaps.cpu().float()) \
            .reshape(sampler.N, -1).to(self.device)
        from unicasso.training.train_cell_classifier import TokenTransformer
        N = len(self.chars)
        m = TokenTransformer(self.rows, self.cols, self.ch, self.cw,
                             self.ph, self.pw, N, dim=cfg.get("dim", 64),
                             heads=cfg.get("heads", 4),
                             n_blocks=cfg.get("blocks", 3),
                             in_ch=cfg.get("in_ch", 1))
        sd = ck["state_dict"]
        if self.color:
            dim = cfg.get("dim", 64)
            if cfg.get("feat_dim", 0):
                m.color_proj = nn.Linear(cfg["feat_dim"], dim)
            if "mode_emb" in sd:
                m.mode_emb = nn.Parameter(
                    torch.zeros(sd["mode_emb"].shape[0], dim))
            if "k_head.weight" in sd:      # legacy contrast head; the joint
                m.k_head = nn.Linear(dim, 1)  # trainer never calls it and a
            if "col_head.weight" in sd:      # random-init model has none
                m.col_head = nn.Linear(dim, 6)
            # NB the two heads share no parameter names, so the outer test
            # must accept EITHER. Gating on mask_dec.out.weight alone silently
            # built no decoder at all for an attention checkpoint -- which
            # RollingPool would have hit at its first refresh, since it makes
            # warm-start grids through Lite.
            if cfg.get("mask_attn") or "mask_dec.q.weight" in sd:
                from unicasso.training.train_cell_classifier import \
                    MaskAttnDecoder
                m.mask_dec = MaskAttnDecoder(dim, d=cfg.get("mask_attn_dim", 32),
                                             depth=cfg.get("mask_attn_depth", 2),
                                             mid=cfg.get("mask_attn_mid"))
                npx = cfg.get("mask_attn_extra", 0) or sum(
                    1 for k in sd if k.startswith("mask_dec.px_res.")
                    and k.endswith(".c1.weight"))
                if npx:
                    m.mask_dec.add_px_blocks(npx, cfg.get("mask_attn_mid"))
                # broadcast conditioning widens conv0 -- rebuild before loading
                ng = cfg.get("mask_bcast_glyph")
                if ng is None:
                    ng = sd["mask_dec.gemb.weight"].shape[0] \
                        if "mask_dec.gemb.weight" in sd else 0
                bm = cfg.get("mask_bcast_mean")
                if bm is None:
                    bm = ("mask_dec.px.0.weight" in sd
                          and sd["mask_dec.px.0.weight"].shape[1] - 3 - ng == 3)
                if ng or bm:
                    m.mask_dec.add_bcast(N, int(ng or 0), bool(bm),
                                         cfg.get("pad_h", 0), cfg.get("pad_w", 0))
            elif "mask_dec.out.weight" in sd:
                from unicasso.training.train_cell_classifier import MaskDecoder
                m.mask_dec = MaskDecoder(
                    dim, flip=bool(cfg.get("mask_flip"))
                    or "mask_dec.flip.weight" in sd)
            if any(k.startswith("rgb_trunk.") for k in sd):
                m.rgb_trunk = nn.Sequential(          # separate mask-branch
                    nn.Conv2d(3, 32, 3, 2, 1),        # encoder (not a skip)
                    nn.GroupNorm(8, 32), nn.GELU())
            nmb = sum(1 for k in sd if k.startswith("mask_blocks.")
                      and k.endswith("ln1.weight"))
            if nmb:            # private mask-branch depth (Run B checkpoints)
                from unicasso.training.train_cell_classifier import Block
                m.mask_blocks = nn.ModuleList(
                    Block(dim, cfg.get("heads", 4)) for _ in range(nmb))
        if cfg.get("mask_ctx") or "mask_ctx_proj.weight" in sd:
            m.add_mask_ctx(cfg.get("dim", 64))
            m.mask_ctx_inject = cfg.get("mask_ctx_inject", 0)
        if cfg.get("glyph_ctx") or "glyph_ctx_proj.weight" in sd:
            m.add_glyph_ctx(cfg.get("dim", 64), N)
            m.glyph_ctx_inject = cfg.get("glyph_ctx_inject", 1)
        if cfg.get("aux_block") or "aux_block.ln1.weight" in sd:
            # Usually a training-only branch (the non-centre glyph regulariser),
            # present so a strict load succeeds. But under glyph_ctx_aux it IS
            # called at inference: it is the calibrated readout for the 14
            # neighbour cells that glyph_ctx feeds to the colouriser.
            from unicasso.training.train_cell_classifier import Block
            m.aux_block = Block(dim if self.color else cfg.get("dim", 64),
                                cfg.get("heads", 4))
        if cfg.get("glyph_ctx_aux"):
            m.glyph_ctx_aux = True
        m.load_state_dict(sd)
        self.model = m.to(self.device).eval()
        self.N = N
        self.kernel = _binomial_kernel(self.rows, self.cols).to(self.device)
        if self.color:
            from unicasso.training.cellclf_color import glyph_bg_dist
            self.bg_dist = glyph_bg_dist(self.ink_flat.cpu(), self.ch, self.cw) \
                .to(self.device)

    @torch.no_grad()
    def _predict(self, ink_u8, gh, gw, feats=None, mode=None, ban_idx=None,
                 stride=1, ensemble=None, temp=1.0, rgb01=None):
        ensemble = ensemble or self.render_ensemble
        M = gh * gw                                     # cells in the grid
        centers = None if stride <= 1 else _strided_centers(gh, gw, stride)
        w = _grid_windows(ink_u8, gh, gw, self.ch, self.cw, self.ph, self.pw,
                          self.rows, self.cols, centers=centers)
        x = torch.from_numpy(w).to(self.device).float().div_(255).unsqueeze(1)
        if rgb01 is not None and getattr(self.model, "in_ch", 1) == 4:
            cwin = _grid_windows(rgb01, gh, gw, self.ch, self.cw, self.ph,
                                 self.pw, self.rows, self.cols, edge=True,
                                 centers=centers)
            x = torch.cat([x, torch.from_numpy(cwin).to(self.device)
                           .permute(0, 3, 1, 2).float()], dim=1)
        ids, valid = _token_maps(gh, gw, self.rows, self.cols, centers=centers)
        ids_t = torch.from_numpy(ids).to(self.device)
        valid_t = torch.from_numpy(valid).to(self.device)
        feats_tok = (feats[ids_t] * valid_t[:, :, None].float()
                     if feats is not None else None)
        B = x.shape[0]
        lo, tt = [], []
        for i in range(0, B, 1024):
            t = self.model._tokens(x[i:i + 1024], None,
                                   feats_tok[i:i + 1024]
                                   if feats_tok is not None else None, mode)
            lo.append(self.model.head(t[:, self.model.n_extra:]))
            tt.append(t)
        logits, tok = torch.cat(lo), torch.cat(tt)
        wtok = self.kernel[None, :] * valid_t.float()
        wsum = torch.zeros(M, device=self.device)
        wsum.index_add_(0, ids_t.reshape(-1), wtok.reshape(-1))
        wsum = wsum.clamp_min(1e-8)
        probs = logits.softmax(-1)
        if ensemble == "center":                        # no ensemble: own window only
            if stride > 1:
                raise ValueError("ensemble='center' needs stride 1 "
                                 "(every cell must have its centered window)")
            tc = (self.rows // 2) * self.cols + self.cols // 2
            scores = probs[:, tc]                       # windows are in cell order
        else:                                           # mean / gmean / sample
            src = probs.clamp_min(1e-9).log() if ensemble == "gmean" else probs
            acc = torch.zeros(M, self.N, device=self.device)
            acc.index_add_(0, ids_t.reshape(-1),
                           (src * wtok[:, :, None]).reshape(-1, self.N))
            scores = acc / wsum[:, None]
            if ensemble == "gmean":                     # back to prob space
                scores = scores.softmax(-1)
        if ban_idx is not None and ban_idx.numel():     # never snap a banned glyph
            scores[:, ban_idx] = 0.0
        if stride <= 1:                                 # windows are in cell order
            ct = tok[:, self.model.center]
        else:                                           # per-cell accumulated state
            cells = tok[:, self.model.n_extra:]
            accs = torch.zeros(M, cells.shape[-1], device=self.device)
            accs.index_add_(0, ids_t.reshape(-1),
                            (cells * wtok[:, :, None]).reshape(-1, cells.shape[-1]))
            ct = accs / wsum[:, None]
        if ensemble == "sample":                        # stochastic decode
            p = scores.clamp_min(0) ** (1.0 / max(temp, 1e-3))
            pred = torch.multinomial(p.clamp_min(1e-12), 1)[:, 0]
        else:
            pred = scores.argmax(-1)
        return pred, ct

    def _txt(self, pred):
        return "\n".join("".join(self.chars[g] for g in row) for row in pred) + "\n"

    @torch.no_grad()
    def _one_forward(self, ink_u8, gh, gw, feats, rgb01, ban_idx=None,
                     ensemble=None, temp=1.0):
        """ONE pass: glyphs AND mask off the same tokens, same input.

        Channel 0 is the decompose ink, never a glyph render -- so unlike
        `_masks` there is no silhouette in the input for the decoder to trace.
        The mask is still glyph-CONDITIONED, just on the model's own belief
        carried in the tokens rather than on an externally supplied glyph.
        Returns (pred, pm_core (M,3,ch,cw), pm_full (M,3,th,tw), C_full
        (M,th*tw,3)) -- the padded patch and its photo pixels are returned too so
        diagnostics see the same tuple `_masks` gives them."""
        M = gh * gw
        w = _grid_windows(ink_u8, gh, gw, self.ch, self.cw, self.ph, self.pw,
                          self.rows, self.cols)
        x = torch.from_numpy(w).to(self.device).float().div_(255).unsqueeze(1)
        if getattr(self.model, "in_ch", 1) == 4:
            cwin = _grid_windows(rgb01, gh, gw, self.ch, self.cw, self.ph,
                                 self.pw, self.rows, self.cols, edge=True)
            x = torch.cat([x, torch.from_numpy(cwin).to(self.device)
                           .permute(0, 3, 1, 2).float()], dim=1)
        ids, valid = _token_maps(gh, gw, self.rows, self.cols)
        ids_t = torch.from_numpy(ids).to(self.device)
        valid_t = torch.from_numpy(valid).to(self.device)
        feats_tok = feats[ids_t] * valid_t[:, :, None].float()
        tci = (self.rows // 2) * self.cols + self.cols // 2
        lo, pms = [], []
        for i in range(0, M, 1024):
            t, s1 = self.model._tokens(x[i:i + 1024], None,
                                       feats_tok[i:i + 1024], 1, want_skip=True)
            n = t.shape[0]
            lo.append(self.model.head(t[:, self.model.n_extra:]))
            pms.append(self.model.mask_center(x[i:i + 1024], t, s1))
        logits = torch.cat(lo)
        mlog = torch.cat(pms)
        # framing ensemble over the covering windows, exactly as _predict does
        wtok = self.kernel[None, :] * valid_t.float()
        wsum = torch.zeros(M, device=self.device)
        wsum.index_add_(0, ids_t.reshape(-1), wtok.reshape(-1))
        probs = logits.softmax(-1)
        src = probs.clamp_min(1e-9).log() if ensemble == "gmean" else probs
        acc = torch.zeros(M, self.N, device=self.device)
        acc.index_add_(0, ids_t.reshape(-1),
                       (src * wtok[:, :, None]).reshape(-1, self.N))
        scores = acc / wsum.clamp_min(1e-8)[:, None]
        if ensemble == "gmean":
            scores = scores.softmax(-1)
        elif ensemble == "center":
            scores = probs[:, tci]
        if ban_idx is not None and ban_idx.numel():
            scores[:, ban_idx] = 0.0
        if ensemble == "sample":
            pr = scores.clamp_min(0) ** (1.0 / max(temp, 1e-3))
            g = torch.multinomial(pr.clamp_min(1e-12), 1)[:, 0]
        else:
            g = scores.argmax(-1)
        pm = mlog[:, :, self.ph:self.ph + self.ch,
                  self.pw:self.pw + self.cw].softmax(1)
        th_, tw_ = self.ch + 2 * self.ph, self.cw + 2 * self.pw
        C_full = (x[:, 1:4, self.ch:self.ch + th_, 2 * self.cw:2 * self.cw + tw_]
                  .permute(0, 2, 3, 1).reshape(M, th_ * tw_, 3)
                  if x.shape[1] >= 4 else None)
        return g, pm, mlog.softmax(1), C_full

    @torch.no_grad()
    def _khat(self, ct):
        """Per-cell contrast from the legacy k_head; k=1 (neutral, exactly what
        a zero head gives through 4*sigmoid(0+K_BIAS)) when the model has none."""
        kh = getattr(self.model, "k_head", None)
        if kh is None:
            return torch.ones(ct.shape[0], device=ct.device)
        return 4.0 * torch.sigmoid(kh(ct)[:, 0] + K_BIAS)

    def _masks(self, g, rgb01, gh, gw, feats):
        """Colorize pass (mode 2): ink channel = the chosen glyphs' own render,
        RGB = the photo. The masks say which photo pixels belong to the entity
        each glyph committed to -- the color fit reads those pixels directly.
        Returns (pm (M,3,ch,cw) core-cropped softmax probs, khat (M,))."""
        M = gh * gw
        ink_img = self.ink_flat[g].view(gh, gw, self.ch, self.cw) \
            .permute(0, 2, 1, 3).reshape(gh * self.ch, gw * self.cw)
        ink_u8 = np.clip(ink_img.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        w = _grid_windows(ink_u8, gh, gw, self.ch, self.cw, self.ph, self.pw,
                          self.rows, self.cols)
        x = torch.from_numpy(w).to(self.device).float().div_(255).unsqueeze(1)
        cwin = _grid_windows(rgb01, gh, gw, self.ch, self.cw, self.ph, self.pw,
                             self.rows, self.cols, edge=True)
        x = torch.cat([x, torch.from_numpy(cwin).to(self.device)
                       .permute(0, 3, 1, 2).float()], dim=1)
        ids, valid = _token_maps(gh, gw, self.rows, self.cols)
        ids_t = torch.from_numpy(ids).to(self.device)
        valid_t = torch.from_numpy(valid).to(self.device)
        feats_tok = feats[ids_t] * valid_t[:, :, None].float()
        tci = (self.rows // 2) * self.cols + self.cols // 2
        cts, pms = [], []
        for i in range(0, M, 1024):
            t, s1 = self.model._tokens(x[i:i + 1024], None,
                                       feats_tok[i:i + 1024], 2,
                                       want_skip=True)
            pms.append(self.model.mask_center(x[i:i + 1024], t, s1))
            cts.append(self.model.mask_tokens(
                t, [(getattr(self.model, "mask_ctx_inject", 0),
                     self.model.mask_ctx(x[i:i + 1024]))])[:, self.model.center])
        ct = torch.cat(cts)
        mlog = torch.cat(pms)
        pm_core = mlog[:, :, self.ph:self.ph + self.ch,
                       self.pw:self.pw + self.cw].softmax(1)
        khat = self._khat(ct)
        fit_pm, fit_C = pm_core, None
        if self.mask_margin:
            # margin fit: un-cropped mask votes neighbor pixels into the fit
            th_, tw_ = self.ch + 2 * self.ph, self.cw + 2 * self.pw
            fit_pm = mlog.softmax(1)
            fit_C = x[:, 1:4, self.ch:self.ch + th_,
                      2 * self.cw:2 * self.cw + tw_] \
                .permute(0, 2, 3, 1).reshape(M, th_ * tw_, 3)
        return pm_core, khat, fit_pm, fit_C

    def _read_grid(self, glyphs):
        """(gh,gw) int array, or a path / text block of glyph rows -> int array."""
        if isinstance(glyphs, np.ndarray):
            return glyphs.astype(np.int64)
        txt = glyphs
        if isinstance(glyphs, (str, os.PathLike)) and os.path.exists(glyphs):
            with open(glyphs, encoding="utf-8") as f:
                txt = f.read()
        rows = [r for r in txt.split("\n") if r != ""]
        w = max(len(r) for r in rows)
        rows = [r.ljust(w) for r in rows]          # ragged tail -> spaces
        idx = {c: i for i, c in enumerate(self.chars)}
        miss = {c for r in rows for c in r} - idx.keys()
        if miss:
            raise ValueError(f"grid uses {len(miss)} glyph(s) outside this "
                             f"model's charset: {''.join(sorted(miss))!r}")
        return np.array([[idx[c] for c in r] for r in rows], np.int64)

    @torch.no_grad()
    def recolor(self, image, glyphs, color_path="topk", topk_frac=0.25,
                topk_weight="prob", topk_count="auto", sample_temp=1.0,
                fit_ridge=0.0, k_scale=0.0, flat_thresh=0.0):
        """Color a grid the model did NOT choose, using the model's own masks.

        `glyphs`: (gh,gw) int array, or a path / text block of glyph rows (the
        engine's --output-text format). The grid fixes the geometry -- the image
        is resized to it, the HARD-SNAPPED glyph ink goes into the ink channel,
        the photo into the RGB channels, and the mask decoder's fg/bg masks give
        the colors. Exactly the colorize pass lite runs on its own predictions,
        pointed at someone else's glyphs.

        This is the primitive a CLIP refinement loop needs: it lets the engine
        judge candidate grids in the colors the lite model would ACTUALLY give
        them, so refinement moves inside the color space the model can express
        instead of chasing colors the model could never reproduce. The refined
        glyphs that come back are then color-aware by construction.

        color_path 'legacy' colors the same grid the old way (cluster blend +
        dist-weighted fit) -- the A/B against the learned masks.
        """
        if not self.color:
            raise ValueError("recolor needs a color model")
        g_np = self._read_grid(glyphs)
        gh, gw = g_np.shape
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        im = ImageOps.exif_transpose(image).convert("RGB") \
            .resize((gw * self.cw, gh * self.ch), Image.LANCZOS)
        from unicasso.engine.color import decompose
        from unicasso.training.cellclf_color import ansi_txt, fit_fg_bg_distweighted
        rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
        dec = decompose(rgb, gh, gw, self.ch, self.cw)
        feats = torch.cat([dec["gate"][:, None], (dec["sep"] / 20.0)[:, None],
                           dec["fg"], dec["bg"], dec["cell_rgb"].mean(1)],
                          dim=1).float().to(self.device)
        rgb01 = np.asarray(im, np.float32) / 255.0
        g = torch.from_numpy(g_np.reshape(-1)).to(self.device)
        pm = None
        if color_path != "legacy":
            if getattr(self.model, "mask_dec", None) is None:
                raise ValueError("recolor: this checkpoint has no mask decoder "
                                 "-- use color_path='legacy'")
            from unicasso.training.cellclf_color_train import mask_fit
            pm, khat, fit_pm, fit_C = self._masks(g, rgb01, gh, gw, feats)
            if color_path in ("topk", "sample"):
                fg0, bg0 = _conf_fit(pm, dec["cell_rgb"].to(self.device),
                                     mode=color_path, frac=topk_frac,
                                     weight=topk_weight, ridge=fit_ridge,
                                     count=topk_count, temp=sample_temp)
            else:
                C = fit_C if fit_C is not None else dec["cell_rgb"].to(self.device)
                fg0, bg0 = mask_fit(fit_pm, C, ridge=fit_ridge)
        else:
            # legacy here reads k = 1: the k head hangs off the mode-1 glyph
            # token, which recolor never computes (the grid is given, not
            # predicted). k is on its way out anyway -- see --k-scale.
            khat = torch.ones(gh * gw, device=self.device)
            mask_g = self.ink_flat[g]
            C = dec["cell_rgb"].to(self.device)
            fg_f, bg_f = fit_fg_bg_distweighted(C, mask_g, self.bg_dist[g], pow_=1.0)
            dfg, dbg = dec["fg"].to(self.device), dec["bg"].to(self.device)
            if self.align_blend:
                di = dec["ink"].to(self.device)
                mm = mask_g - mask_g.mean(1, keepdim=True)
                ii = di - di.mean(1, keepdim=True)
                ac = (mm * ii).sum(1) / (mm.norm(dim=1) * ii.norm(dim=1)).clamp_min(1e-6)
                sw = (ac < 0)[:, None]
                dfg, dbg = torch.where(sw, dbg, dfg), torch.where(sw, dfg, dbg)
            fg0, bg0 = 0.5 * dfg + 0.5 * fg_f, 0.5 * dbg + 0.5 * bg_f
        kf = 1.0 + k_scale * (khat - 1.0)
        mid = 0.5 * (fg0 + bg0)
        fg = (mid + kf[:, None] * (fg0 - mid)).clamp(0, 1)
        bg = (mid + kf[:, None] * (bg0 - mid)).clamp(0, 1)
        if flat_thresh > 0 and " " in self.chars:
            flat = (fg - bg).abs().mean(1) < flat_thresh
            g = torch.where(flat, torch.full_like(g, self.chars.index(" ")), g)
        mask = self.ink_flat[g]
        cell = bg[:, None, :] + (fg - bg)[:, None, :] * mask[:, :, None]
        render = cell.view(gh, gw, self.ch, self.cw, 3) \
            .permute(0, 2, 1, 3, 4).reshape(gh * self.ch, gw * self.cw, 3) \
            .cpu().numpy()
        pred = g.view(gh, gw).cpu().numpy()
        return LiteResult(pred, self._txt(pred),
                          ansi_txt(pred, self.chars, fg.cpu().numpy(),
                                   bg.cpu().numpy(), gh, gw),
                          fg.cpu().numpy(), bg.cpu().numpy(),
                          kf.view(gh, gw).cpu().numpy(), render,
                          masks=None if pm is None else pm.cpu().numpy())

    @torch.no_grad()
    def render(self, image, width=60, ban=None, stride=1,
               ensemble=None, temp=1.0, color_path="auto", topk_frac=0.5,
               topk_weight="prob", topk_count="auto", sample_temp=1.0,
               fit_ridge=1.0, k_scale=1.0, flat_thresh=0.0,
               mask_forward=None):
        """image: path or PIL.Image. Color model -> full LiteResult with .ans;
        line model -> glyph text from the grayscale image.

        ban: iterable of characters to exclude from the output (never snapped);
        the classifier's logits for those glyphs are masked before the argmax.
        stride: forward windows only at every s-th cell (the dense head covers
        the cells in between) -- ~stride^2 fewer forwards, fewer framings/cell.
        color_path: how a mask model's masks are turned into two colors --
        'auto' = mask-weighted mean over every pixel (`mask_fit`), falling back
        to legacy if the masks look degenerate; 'mask' = the same read with no
        fallback; 'topk' = core pixels only, each layer keeping its top
        `topk_frac` most confident pixels (see `_conf_fit`); 'sample' = core
        pixels only, the same selection drawn stochastically without
        replacement at the layer's own probabilities; 'legacy' = the old recipe
        (cluster blend + dist-weighted fit + align swap) -- the A/B diagnostic.
        topk_weight: 'prob' or 'uniform' vote weight inside the kept set.
        topk_count: how big each layer's kept set is -- 'fixed' (frac * all core
        pixels) or 'argmax' (frac * the pixels that layer wins outright);
        'auto' = fixed for topk, argmax for sample.
        sample_temp: sharpen the layer probabilities before the draw; 1 = as
        predicted, -> 0 makes 'sample' converge on 'topk'.
        fit_ridge: pull toward the plain cell mean, in pixel-mass units (the
        historic mask_fit value is 1.0; 0 removes the pull entirely).
        k_scale: rescale the learned contrast, k = 1 + k_scale*(k_hat - 1);
        0 disables k, 1 ships it as trained.
        flat_thresh: |fg-bg| below which a cell is called empty and shipped as
        space instead of a phantom glyph.
        mask_forward: 'two' = the historic path (glyph pass, then the chosen
        glyphs re-rendered into channel 0 and a SECOND forward for the mask);
        'one' = a single pass where head and mask_dec read the same tokens off
        [decompose ink, photo RGB], so no glyph silhouette is ever in the input.
        None = whatever the checkpoint was trained under.
        ensemble: how the covering windows' predictions combine per cell --
        'mean' (kernel-weighted arithmetic mean of probs, the default),
        'gmean' (geometric mean: sharper, a window that rules a glyph out
        vetoes it), 'center' (no ensemble: own centered window only),
        'sample' (draw from the mean distribution at temperature `temp`)."""
        ban_idx = None
        if ban:
            bs = set(ban)
            ban_idx = torch.tensor([i for i, c in enumerate(self.chars) if c in bs],
                                   device=self.device, dtype=torch.long)
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        im = ImageOps.exif_transpose(image).convert("RGB")
        gh = raster.grid_height_for_aspect(im.width, im.height, width,
                                           self.cw, self.ch, 0)
        im = im.resize((width * self.cw, gh * self.ch), Image.LANCZOS)
        if not self.color:
            gray = np.asarray(im.convert("L"), np.float32) / 255.0
            ink_u8 = np.clip((1.0 - gray) * 255.0, 0, 255).astype(np.uint8)
            g, _ = self._predict(ink_u8, gh, width, ban_idx=ban_idx,
                                 stride=stride, ensemble=ensemble, temp=temp)
            pred = g.view(gh, width).cpu().numpy()
            # black ink on white paper, so --png works for the line model too
            render = (1.0 - self.ink_flat[g.view(-1)]).view(gh, width, self.ch, self.cw) \
                .permute(0, 2, 1, 3).reshape(gh * self.ch, width * self.cw) \
                .cpu().numpy()[:, :, None].repeat(3, axis=2)
            return LiteResult(pred, self._txt(pred), None, None, None, None, render)
        from unicasso.engine.color import decompose, nomination_target
        from unicasso.training.cellclf_color import ansi_txt, fit_fg_bg_distweighted
        rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
        dec = decompose(rgb, gh, width, self.ch, self.cw)
        ink_u8 = np.clip((1.0 - nomination_target(dec).numpy()) * 255.0,
                         0, 255).astype(np.uint8)
        mean = dec["cell_rgb"].mean(1)
        feats = torch.cat([dec["gate"][:, None], (dec["sep"] / 20.0)[:, None],
                           dec["fg"], dec["bg"], mean], dim=1).float().to(self.device)
        rgb01 = np.asarray(im, np.float32) / 255.0
        mf = mask_forward or self.mask_forward
        one_fwd = (mf == "one" and getattr(self.model, "mask_dec", None) is not None
                   and color_path != "legacy")
        if one_fwd:
            g, pm_one, _pm_full, _C_full = self._one_forward(
                ink_u8, gh, width, feats, rgb01, ban_idx=ban_idx,
                ensemble=ensemble, temp=temp)
            ct = None
        else:
            g, ct = self._predict(ink_u8, gh, width, feats=feats, mode=1,
                                  ban_idx=ban_idx, stride=stride,
                                  ensemble=ensemble, temp=temp, rgb01=rgb01)
        if getattr(self.model, "mask_dec", None) is not None \
                and color_path != "legacy":
            # mask path: colors = the mask-weighted fit on the photo pixels,
            # k on top around its own midpoint. No cluster blend, no distance
            # weighting, no alignment swap -- the mask subsumes all three.
            from unicasso.training.cellclf_color_train import mask_fit
            if one_fwd:
                pm, fit_pm, fit_C = pm_one, pm_one, None
                khat = torch.ones(gh * width, device=self.device)
            else:
                pm, khat, fit_pm, fit_C = self._masks(g, rgb01, gh, width, feats)
            if color_path in ("topk", "sample"):
                # confidence-gated read: the CORE pixels only (the padded margin
                # is a neighbour's territory and colors badly there), and only
                # the pixels each layer is actually sure about get a vote
                C = dec["cell_rgb"].to(self.device)
                fgm, bgm = _conf_fit(pm, C, mode=color_path, frac=topk_frac,
                                     weight=topk_weight, ridge=fit_ridge,
                                     count=topk_count, temp=sample_temp)
            else:
                C = fit_C if fit_C is not None else dec["cell_rgb"].to(self.device)
                fgm, bgm = mask_fit(fit_pm, C, ridge=fit_ridge)
            # judge degeneracy on INK cells only: flat cells correctly have
            # fg == bg (one color), so a global median would condemn any
            # flat-heavy photo even with perfect masks
            sep_all = (fgm - bgm).abs().mean(1)
            inky = dec["ink"].mean(1).to(self.device) > 0.03
            sep_med = float(sep_all[inky].median()) if int(inky.sum()) >= 8 \
                else float(sep_all.median())
            if color_path == "auto" and sep_med < 0.03:
                # degenerate masks (fg ~ bg on the cells that matter): an
                # immature mask decoder would render mush -- use legacy
                print(f"lite: mask fit degenerate (median fg/bg sep "
                      f"{sep_med:.3f} < 0.03), using legacy coloring",
                      file=sys.stderr)
                return self.render(image, width=width, ban=ban, stride=stride,
                                   ensemble=ensemble, temp=temp,
                                   color_path="legacy", fit_ridge=fit_ridge,
                                   k_scale=k_scale, flat_thresh=flat_thresh)
            kf = 1.0 + k_scale * (khat - 1.0)
            mid = 0.5 * (fgm + bgm)
            fg = (mid + kf[:, None] * (fgm - mid)).clamp(0, 1)
            bg = (mid + kf[:, None] * (bgm - mid)).clamp(0, 1)
            # fg == bg renders any glyph invisible: such a cell is EMPTY --
            # ship the space glyph instead of a phantom character
            if " " in self.chars:
                flat = (fg - bg).abs().mean(1) < flat_thresh
                g = torch.where(flat,
                                torch.full_like(g, self.chars.index(" ")), g)
            mask = self.ink_flat[g]
            cell = bg[:, None, :] + (fg - bg)[:, None, :] * mask[:, :, None]
            render = cell.view(gh, width, self.ch, self.cw, 3) \
                .permute(0, 2, 1, 3, 4).reshape(gh * self.ch, width * self.cw, 3) \
                .cpu().numpy()
            pred = g.view(gh, width).cpu().numpy()
            return LiteResult(pred, self._txt(pred),
                              ansi_txt(pred, self.chars, fg.cpu().numpy(),
                                       bg.cpu().numpy(), gh, width),
                              fg.cpu().numpy(), bg.cpu().numpy(),
                              kf.view(gh, width).cpu().numpy(), render,
                              masks=pm.cpu().numpy())
        khat = self._khat(ct)
        khat = 1.0 + k_scale * (khat - 1.0)
        mask = self.ink_flat[g]
        C = dec["cell_rgb"].to(self.device)
        fg_f, bg_f = fit_fg_bg_distweighted(C, mask, self.bg_dist[g], pow_=1.0)
        dfg = dec["fg"].to(self.device)
        dbg = dec["bg"].to(self.device)
        if self.align_blend:            # swap cluster colors where the cluster's
            di = dec["ink"].to(self.device)   # ink map anti-aligns with the glyph
            mm = mask - mask.mean(1, keepdim=True)
            ii = di - di.mean(1, keepdim=True)
            acorr = (mm * ii).sum(1) / (mm.norm(dim=1) * ii.norm(dim=1)).clamp_min(1e-6)
            sw = (acorr < 0)[:, None]
            dfg, dbg = torch.where(sw, dbg, dfg), torch.where(sw, dfg, dbg)
        fg0 = 0.5 * dfg + 0.5 * fg_f
        bg0 = 0.5 * dbg + 0.5 * bg_f
        mid = 0.5 * (fg0 + bg0)
        fg = (mid + khat[:, None] * (fg0 - mid)).clamp(0, 1)
        bg = (mid + khat[:, None] * (bg0 - mid)).clamp(0, 1)
        cell = bg[:, None, :] + (fg - bg)[:, None, :] * mask[:, :, None]
        render = cell.view(gh, width, self.ch, self.cw, 3) \
            .permute(0, 2, 1, 3, 4).reshape(gh * self.ch, width * self.cw, 3) \
            .cpu().numpy()
        pred = g.view(gh, width).cpu().numpy()
        return LiteResult(pred, self._txt(pred),
                          ansi_txt(pred, self.chars, fg.cpu().numpy(),
                                   bg.cpu().numpy(), gh, width),
                          fg.cpu().numpy(), bg.cpu().numpy(),
                          khat.view(gh, width).cpu().numpy(), render)


def refine(lite, image_path, result, width, iters, lead=1.0, extra=(),
           color_mode="frozen", anchor=0.0, anchor_mc=0.12,
           anchor_start=0.0, anchor_start_frac=0.0, anchor_end_frac=1.0):
    """Polish a lite result with `iters` iterations of the full engine (swarm
    mode), warm-started from the grid via --init-text with a slot-0 W lead of
    `lead`. A mask-model result also carries its masks into the run; how COLOR is
    then arbitrated depends on `color_mode`:
      'frozen' (default): colors held at the lite masks' fit (--color-mask), so
        the engine arbitrates pure shape under CLIP -- the EM E-step.
      'free': fg/bg warm-started from the mask fit (--color-mask-mode init) then
        optimized FREELY by CLIP and shipped as-is (no closed-form at the end);
        `anchor` > 0 adds the cosine-ramped L2 pull back toward the closed-form
        fit late in the run. Without a mask model this is plain --no-color-fit.
      'fit': closed-form MSE colors in the loop (no mask freeze).

    The schedule enters at the START OF THE SECOND W-temp cycle of the canonical
    recipe rather than iteration 0: one anneal from the cycle-2 reheat peak
    (0.66 = decay * the 1.0 start) to 0, with z-noise picked up at its mid-run
    cosine value (0.45). A refinement pass, not a fresh exploration -- the lite
    prediction is the incumbent, and the engine's births/boosts must beat it.

    Returns the refined .ans text (color model) or plain text (line model).
    Engine progress/banners stream to stderr. Needs the engine deps (open_clip)."""
    import subprocess
    import tempfile
    d = tempfile.mkdtemp(prefix="unicasso_refine_")
    init = os.path.join(d, "init.txt")
    with open(init, "w", encoding="utf-8") as f:
        f.write(result.txt)
    cmd = [sys.executable, "-m", "unicasso.engine.asciify", str(image_path),
           "--base-width", str(width), "--iters", str(iters),
           "--init-text", init, "--init-w-lead", str(lead),
           "--swarm-w-temp", "0.66", "--swarm-w-temp-cycles", "1",
           "--z-noise", "0.45", "--progress-every", "25",
           "--output", os.path.join(d, "refined.png")]
    ans = os.path.join(d, "refined.ans")
    if lite.color:
        cmd += ["--color", "--output-ans", ans]
        # anchor + its cosine schedule (end value, start value, ramp window)
        anchor_flags = (["--color-anchor", str(anchor),
                         "--color-anchor-mc", str(anchor_mc),
                         "--color-anchor-start", str(anchor_start),
                         "--color-anchor-start-frac", str(anchor_start_frac),
                         "--color-anchor-end-frac", str(anchor_end_frac)]
                        if anchor > 0 else [])
        if color_mode == "free" and result.fg is not None:
            # seed the free leaves at the colors lite SHIPPED (its own recipe),
            # so the refine polishes the lite output instead of re-deriving colors
            cf = os.path.join(d, "color_init.npz")
            np.savez_compressed(cf,
                                fg=result.fg.reshape(-1, 3).astype(np.float32),
                                bg=result.bg.reshape(-1, 3).astype(np.float32))
            cmd += ["--color-init", cf]
        have_masks = result.masks is not None
        if have_masks and color_mode in ("frozen", "free"):
            mf = os.path.join(d, "masks.npz")
            np.savez_compressed(
                mf, masks=result.masks.astype(np.float16),
                k=result.k.reshape(-1).astype(np.float32),
                gh=result.glyphs.shape[0], gw=result.glyphs.shape[1])
            cmd += ["--color-mask", mf]
            if color_mode == "free":
                # warm-start fg/bg from the lite mask fit, then let CLIP move
                # them FREELY and ship those colors (not the frozen fit)
                cmd += ["--color-mask-mode", "init"] + anchor_flags
        elif color_mode == "free":
            # no mask model: free learned colors under CLIP, no closed-form
            cmd += ["--no-color-fit"] + anchor_flags
        # color_mode == "fit": no mask freeze -> engine's default --color-fit
        #   (closed-form MSE colors in the loop)
    cmd += list(extra)
    env = dict(os.environ, GLYPHVAE_FONT=lite.profile)
    r = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"refine: engine run failed (exit {r.returncode}); "
                           f"warm-start grid kept at {init}")
    path = ans if lite.color else os.path.join(d, "refined.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def color_refine(lite, image_path, result, width, iters, rounds=5, lead=1.0,
                 read=None, extra=(), progress=True):
    """Color-AWARE CLIP refinement: alternate engine SHAPE refinement with
    LITE-MODEL coloring, so the engine only ever judges glyphs in the colors the
    model can actually produce.

    Each round recolors the CURRENT grid through the mask decoder (hard-snapped
    glyph ink -> ink channel, photo -> RGB, mode-2 pass), hands those colors to
    the engine as --color-mask, and lets it move shape only. Because the masks
    are recomputed for the grid the engine just produced, they never go stale --
    the failure mode of a one-shot --color-mask dump, where a cell whose glyph
    changed keeps colors fitted to the glyph it used to hold.

    The result is color-aware glyph refinement: the engine cannot buy CLIP score
    with colors the coloring model would never assign, so the refined grids it
    returns are reachable targets rather than aspirational ones.

    w-temp and z-noise ramp DOWN across rounds, so the whole sequence is one
    anneal rather than `rounds` reheats. `read` is the color-read kwargs dict
    (color_path/topk_*/sample_temp/fit_ridge/k_scale) -- pass exactly what you
    intend to ship, since that is what the engine will be optimizing against.
    Returns the final .ans, colored by the model from the final grid."""
    import subprocess
    import tempfile
    if getattr(lite.model, "mask_dec", None) is None:
        raise RuntimeError("--color-refine needs a checkpoint with a mask "
                           "decoder (the coloring is the mask decoder's)")
    read = dict(read or {})
    read.pop("flat_thresh", None)          # never rewrite the grid mid-loop
    d = tempfile.mkdtemp(prefix="unicasso_colorrefine_")
    per = max(1, iters // max(1, rounds))
    grid = result.txt
    for r in range(rounds):
        gp = os.path.join(d, f"grid{r}.txt")
        with open(gp, "w", encoding="utf-8") as f:
            f.write(grid)
        # 1. color THIS grid with the model's own masks
        cur = lite.recolor(image_path, gp, flat_thresh=0.0, **read)
        mf = os.path.join(d, f"m{r}.npz")
        np.savez_compressed(mf, masks=cur.masks.astype(np.float16),
                            k=np.ones(cur.glyphs.size, np.float32),
                            gh=cur.glyphs.shape[0], gw=cur.glyphs.shape[1])
        # 2. move SHAPE under those colors; one global anneal across the rounds
        f_ = 1.0 - r / max(1, rounds)
        out_txt = os.path.join(d, f"grid{r + 1}.txt")
        cmd = [sys.executable, "-m", "unicasso.engine.asciify", str(image_path),
               "--base-width", str(width), "--iters", str(per),
               "--init-text", gp, "--init-w-lead", str(lead),
               "--swarm-w-temp", f"{max(0.08, 0.66 * f_):.3f}",
               "--swarm-w-temp-cycles", "1",
               "--z-noise", f"{max(0.05, 0.45 * f_):.3f}",
               "--color", "--color-mask", mf,
               "--output", os.path.join(d, f"r{r}.png"),
               "--output-text", out_txt,
               "--progress-every", "25"] + list(extra)
        if progress:
            sys.stderr.write(f"== color-refine round {r + 1}/{rounds} "
                             f"({per} iters, w-temp {max(0.08, 0.66 * f_):.2f}) ==\n")
            sys.stderr.flush()
        env = dict(os.environ, GLYPHVAE_FONT=lite.profile)
        rc = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr, env=env)
        if rc.returncode != 0:
            raise RuntimeError(f"color-refine: engine round {r} failed "
                               f"(exit {rc.returncode}); grids kept in {d}")
        with open(out_txt, encoding="utf-8") as f:
            new = f.read()
        moved = sum(a != b for ra, rb in zip(grid.split("\n"), new.split("\n"))
                    for a, b in zip(ra, rb))
        if progress:
            sys.stderr.write(f"   {moved} cells moved\n")
            sys.stderr.flush()
        grid = new
    # final coloring: the model colors the grid it ended on
    gp = os.path.join(d, "final.txt")
    with open(gp, "w", encoding="utf-8") as f:
        f.write(grid)
    return lite.recolor(image_path, gp, flat_thresh=0.0, **read)


def auto_refine(lite, image_path, result, width, iters, extra=(), progress=True):
    """`--refine N`: polish a lite result with N iterations of the full
    optimizer, the way the v2 models' own training pools were refined. The
    path is chosen from the model, nothing to configure:

      line model            -> monochrome shape refinement, no colour.
      colour model with a   -> the engine runs IN the colours this model would
        mask decoder (v2)      paint (--color-lite, mode 'birth': every slot
                               carries the model's fg/bg for its own glyph as
                               constants, so the soft render, the probe and the
                               emission all agree; no colour leaves), and the
                               final grid is coloured by the model.
      colour model without  -> closed-form colours in the loop (--color-fit),
        one (v1)               the engine's own ANSI shipped.

    Schedule = the training recipe: warm-start from the lite grid with the
    incumbent's weight lead at 1.0, ONE W-temperature cycle from 0.66 (the
    second cycle of the canonical run, not a fresh exploration) and z-noise
    0.49. Returns (text, LiteResult-or-None): the refined .ans / plain text and,
    for the mask model, the recoloured result (its `render` feeds --png)."""
    import subprocess
    import tempfile
    d = tempfile.mkdtemp(prefix="unicasso_refine_")
    init = os.path.join(d, "init.txt")
    with open(init, "w", encoding="utf-8") as f:
        f.write(result.txt)
    out_txt = os.path.join(d, "refined.txt")
    out_ans = os.path.join(d, "refined.ans")
    cmd = [sys.executable, "-m", "unicasso.engine.asciify", str(image_path),
           "--base-width", str(width), "--iters", str(iters),
           "--init-text", init, "--init-w-lead", "1.0",
           "--swarm-w-temp", "0.66", "--swarm-w-temp-cycles", "1",
           "--z-noise", "0.49",
           "--output", os.path.join(d, "refined.png"), "--output-text", out_txt,
           "--progress-every", "25" if progress else "0"]
    mask_model = lite.color and getattr(lite.model, "mask_dec", None) is not None
    read = dict(color_path="blend", fit_ridge=1.0)   # what Lite ships by default
    if mask_model:
        cmd += ["--color", "--color-lite", lite.ckpt_path,
                "--color-lite-mode", "birth", "--color-lite-ink", "render",
                "--color-lite-path", "blend", "--color-lite-ridge", "1.0",
                "--color-lite-count", "argmax", "--color-lite-temp", "1.0"]
    elif lite.color:
        cmd += ["--color", "--color-fit", "--output-ans", out_ans]
    cmd += list(extra)
    if progress:
        sys.stderr.write(f"== refine: {iters} iterations of the optimizer, "
                         f"{'model colouring' if mask_model else 'closed-form colouring' if lite.color else 'monochrome'} ==\n")
        sys.stderr.flush()
    env = dict(os.environ, GLYPHVAE_FONT=lite.profile)
    r = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"refine: optimizer run failed (exit {r.returncode}); "
                           f"warm-start grid kept at {init}")
    with open(out_txt, encoding="utf-8") as f:
        grid = f.read()
    if mask_model:
        res = lite.recolor(image_path, out_txt, flat_thresh=0.0, **read)
        return res.ans, res
    # no model colouring: the engine's own render is the pixel output
    render = np.asarray(Image.open(os.path.join(d, "refined.png")).convert("RGB"),
                        np.float32) / 255.0
    pred = lite._read_grid(grid)
    if lite.color:
        with open(out_ans, encoding="utf-8") as f:
            ans = f.read()
        return ans, LiteResult(pred, grid, ans, None, None, None, render)
    return grid, LiteResult(pred, grid, None, None, None, None, render)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="input image (photo, or line art with --line)")
    p.add_argument("-w", "--width", type=int, default=0,
                   help="grid width in characters; height follows the aspect. 0 (default) "
                        "fills the terminal when printing to a screen, else 60")
    p.add_argument("--line", action="store_true",
                   help="use the line model: monochrome ASCII art, plain text, no color "
                        "(default: the color model, which emits 24-bit ANSI). Expects "
                        "LINE-ART input -- convert a photo first with unicasso.lineart")
    p.add_argument("--font", default=None, choices=["dejavu", "sfmono"],
                   help="which font kit's models to use (default: $GLYPHVAE_FONT or "
                        "dejavu). sfmono renders with Apple's SF Mono from the system "
                        "path -- macOS only unless you provide the font")
    p.add_argument("--weights", default=None,
                   help="override the weights path (default: the shipped color/line "
                        "model for --font)")
    p.add_argument("--out", default=None,
                   help="write the render here instead of stdout (stdout is clean ANSI/text, "
                        "safe to pipe or cat)")
    p.add_argument("--png", default=None, help="also save the pixel render as a PNG")
    p.add_argument("--out-text", default=None, metavar="TXT",
                   help="also write the PLAIN GLYPH GRID here (no ANSI, no color). "
                        "This is the engine's --init-text format: colors do not "
                        "need to survive the handoff, since --color-lite recolors "
                        "whatever grid it is given. Works for the color model too, "
                        "where stdout/--out carry ANSI")
    p.add_argument("--ban-chars", default="", help="characters to exclude from the output")
    p.add_argument("--ban-blocks", action="store_true", help="also exclude block chars ░▒▓█▄▌▐▀■")
    p.add_argument("--ban-letters", action="store_true",
                   help="also exclude all Unicode letters (A-Z a-z + accented)")
    p.add_argument("--stride", type=int, default=1, help=argparse.SUPPRESS)
    # The read (centre vs framing ensemble) is decided by the checkpoint --
    # each model is read the way it was trained -- so it is not a user knob.
    p.add_argument("--ensemble", default=None,
                   choices=["mean", "gmean", "center", "sample"], help=argparse.SUPPRESS)
    p.add_argument("--temp", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--device", default=None, help="torch device (default: auto -- cuda/mps/cpu)")
    p.add_argument("--refine", type=int, default=0, metavar="N",
                   help="polish the result with N iterations of the full optimizer "
                        "(needs the engine deps; ~minutes). Line art is refined in "
                        "shape only; a colour model is refined rendering in the "
                        "colours it paints itself. Below ~300 iterations it "
                        "likely doesn't help much")
    p.add_argument("--refine-legacy", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--color-refine", type=int, default=0, metavar="N", help=argparse.SUPPRESS)
    p.add_argument("--color-refine-rounds", type=int, default=5, help=argparse.SUPPRESS)
    p.add_argument("--color-refine-args", default="", help=argparse.SUPPRESS)
    p.add_argument("--refine-lead", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--refine-color", default="frozen",
                   choices=["frozen", "free", "fit"], help=argparse.SUPPRESS)
    p.add_argument("--refine-color-anchor", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--refine-color-anchor-start", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--refine-color-anchor-start-frac", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--refine-color-anchor-end-frac", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--refine-color-anchor-mc", type=float, default=0.12, help=argparse.SUPPRESS)
    p.add_argument("--refine-args", default="", help=argparse.SUPPRESS)
    p.add_argument("--color-text", default=None, metavar="GRID.txt", help=argparse.SUPPRESS)
    p.add_argument("--mask-forward", default=None, choices=["one", "two"], help=argparse.SUPPRESS)
    p.add_argument("--color-path", default="auto",
                   choices=["auto", "legacy", "mask", "topk", "sample"], help=argparse.SUPPRESS)
    p.add_argument("--topk-frac", type=float, default=0.5, help=argparse.SUPPRESS)
    p.add_argument("--topk-count", default="auto",
                   choices=["auto", "fixed", "argmax"], help=argparse.SUPPRESS)
    p.add_argument("--sample-temp", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--topk-weight", default="prob", choices=["prob", "uniform"], help=argparse.SUPPRESS)
    p.add_argument("--fit-ridge", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--k-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--flat-thresh", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--dump-masks", default=None, metavar="NPZ", help=argparse.SUPPRESS)
    args = p.parse_args()
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):     # loader banners off stdout
        lite = Lite(args.weights or ("line" if args.line else "color"),
                    device=args.device, font=args.font)
    width = args.width
    if width <= 0:                                    # 0 = fill the terminal when on a screen
        import shutil
        width = shutil.get_terminal_size((60, 20)).columns if sys.stdout.isatty() else 60
    ban = set(args.ban_chars) | (set("░▒▓█▄▌▐▀■") if args.ban_blocks else set())
    if args.ban_letters:
        import unicodedata
        ban |= {c for c in lite.chars if unicodedata.category(c).startswith("L")}
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.color_text:
        # coloring a GIVEN grid: the empty-cell substitution is off unless asked
        # for, since silently rewriting someone else's glyphs to space would
        # change the grid we were told to color
        ft = 0.0 if args.flat_thresh is None else args.flat_thresh
        cp = "topk" if args.color_path == "auto" else args.color_path
        out = lite.recolor(args.image, args.color_text, color_path=cp,
                           topk_frac=args.topk_frac, topk_weight=args.topk_weight,
                           topk_count=args.topk_count, sample_temp=args.sample_temp,
                           fit_ridge=args.fit_ridge, k_scale=args.k_scale,
                           flat_thresh=ft)
        text = out.ans if out.ans else out.txt
        if args.dump_masks and out.masks is not None:
            # masks for THIS grid -> the engine's --color-mask. k is shipped as
            # 1.0: the contrast head is not part of the coloring we are training
            np.savez_compressed(args.dump_masks,
                                masks=out.masks.astype(np.float16),
                                k=np.ones(out.glyphs.size, np.float32),
                                glyphs=out.glyphs.astype(np.int16),
                                gh=out.glyphs.shape[0], gw=out.glyphs.shape[1],
                                photo=str(args.image), width=out.glyphs.shape[1])
        if args.png and out.render is not None:
            Image.fromarray((out.render * 255).astype(np.uint8)).save(args.png)
        if args.out_text:
            with open(args.out_text, "w", encoding="utf-8") as f:
                f.write(out.txt)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
        return
    out = lite.render(args.image, width=width, ban=ban or None,
                      stride=args.stride, ensemble=args.ensemble,
                      temp=args.temp, color_path=args.color_path,
                      topk_frac=args.topk_frac, topk_weight=args.topk_weight,
                      topk_count=args.topk_count, sample_temp=args.sample_temp,
                      fit_ridge=args.fit_ridge,
                      k_scale=args.k_scale, mask_forward=args.mask_forward,
                      flat_thresh=0.04 if args.flat_thresh is None
                      else args.flat_thresh)
    text = out.txt if args.line or out.ans is None else out.ans
    if args.dump_masks and out.masks is not None:
        np.savez_compressed(args.dump_masks,
                            masks=out.masks.astype(np.float16),
                            k=out.k.reshape(-1).astype(np.float32),
                            glyphs=out.glyphs.astype(np.int16),
                            gh=out.glyphs.shape[0], gw=out.glyphs.shape[1],
                            photo=str(args.image), width=width)
    if args.color_refine > 0:
        import shlex
        sys.stderr.write("== lite ==\n")
        sys.stderr.flush()
        sys.stdout.write(text)
        sys.stdout.flush()
        read = dict(color_path=("topk" if args.color_path == "auto"
                                else args.color_path),
                    topk_frac=args.topk_frac, topk_weight=args.topk_weight,
                    topk_count=args.topk_count, sample_temp=args.sample_temp,
                    fit_ridge=args.fit_ridge, k_scale=args.k_scale)
        cr = color_refine(lite, args.image, out, width, args.color_refine,
                          rounds=args.color_refine_rounds,
                          lead=args.refine_lead, read=read,
                          extra=shlex.split(args.color_refine_args))
        sys.stderr.write(f"== color-refined ({args.color_refine} iters, "
                         f"{args.color_refine_rounds} rounds) ==\n")
        sys.stderr.flush()
        text = cr.ans if cr.ans else cr.txt
        out = cr                                  # --png ships the refined render
    if args.refine > 0:
        import shlex
        text, out = auto_refine(lite, args.image, out, width, args.refine,
                                extra=shlex.split(args.refine_args))
        # --png / --out-text below ship the refined grid
    if args.refine_legacy > 0:
        sys.stderr.write("== lite ==\n")
        sys.stderr.flush()
        sys.stdout.write(text)
        sys.stdout.flush()
        import shlex
        refined = refine(lite, args.image, out, width,
                         args.refine_legacy, lead=args.refine_lead,
                         color_mode=args.refine_color,
                         anchor=args.refine_color_anchor,
                         anchor_mc=args.refine_color_anchor_mc,
                         anchor_start=args.refine_color_anchor_start,
                         anchor_start_frac=args.refine_color_anchor_start_frac,
                         anchor_end_frac=args.refine_color_anchor_end_frac,
                         extra=shlex.split(args.refine_args))
        sys.stderr.write(f"== refined ({args.refine_legacy} iters) ==\n")
        sys.stderr.flush()
        text = refined                       # --out / stdout below ship the refined result
    if args.out_text:
        with open(args.out_text, "w", encoding="utf-8") as f:
            f.write(out.txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    if args.png and out.render is not None:
        Image.fromarray((out.render * 255).astype(np.uint8)).save(args.png)


if __name__ == "__main__":
    main()
