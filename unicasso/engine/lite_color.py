"""Lite-model coloring for the engine: fg/bg from the distilled mask decoder.

The engine's own colors are a closed-form MSE fit at the placed glyph. That fit
is optimal for pixel error but it is NOT the coloring the lite model ships, so a
refinement run optimizes toward colors the model can never reproduce -- the
engine buys CLIP score with a palette that does not survive distillation.

This module lets every place the engine needs a color ask the LITE MODEL what it
would paint, so refinement stays inside the reachable color space and the grids
that come back are color-aware by construction.

Three substitution points, escalating (engine flag --color-lite-mode):

  emit    : probe renders + emission. Candidates are MEASURED in the colors the
            model would give them, and the shipped image uses them. Cheapest;
            the soft render still runs on the engine's own colors, so what the
            gradient sees disagrees with what the probe measures.
  birth   : + per-slot colors at init / birth / reseed. Each slot then carries
            the model's colors for ITS OWN glyph as constants, so the soft
            render, the probe and emission all agree. No color leaves.
  forward : + re-queried every forward from each slot's current nearest glyph,
            so colors track slots as they travel through latent space.

fg/bg are always DETACHED -- nothing differentiates through the lite model. The
colors are a lookup, not a term.

Ink conventions (--color-lite-ink), i.e. what goes in the ink channel:

  render : the whole 5x3 window rendered from the CURRENT grid. This is what
           lite._masks / lite.recolor do, and what the v2 models' colour pools
           were coloured with during training -- but a cell's colors then depend on its
           neighbours' glyphs, so they must be re-evaluated whenever the grid
           around a cell moves.
  center : the photo's own ink everywhere, with ONLY the center cell replaced by
           the candidate glyph's bitmap. Colors become a pure function of
           (cell, glyph), independent of what the neighbours chose. This is the
           convention worth training toward; note that no current checkpoint
           has seen it, so it reads pessimistically today.
"""

import numpy as np
import torch

from unicasso.lite import (K_BIAS, Lite, _conf_fit, _grid_windows,
                           _token_maps)


class LiteColorer:
    """Answers "what colors would the lite model paint on glyph g in cell c".

    Geometry is fixed at construction from (gh, gw) and must match the engine's
    grid exactly -- the model's cell size defines the pixel layout.
    """

    def __init__(self, ckpt, image, gh, gw, device=None, font=None,
                 ink="render", read=None, k_scale=0.0, chunk=512, bg_fixed=None):
        self.lite = Lite(ckpt, device=device, font=font)
        # --color-fg: the paper colour; fg is refit through the model's mask
        # with bg pinned, the model's own bg colour is discarded
        self.bg_fixed = None if bg_fixed is None else torch.as_tensor(bg_fixed).float()
        if getattr(self.lite.model, "mask_dec", None) is None:
            raise ValueError(f"--color-lite: {ckpt} has no mask decoder")
        self.dev = self.lite.device
        self.ch, self.cw = self.lite.ch, self.lite.cw
        self.ph, self.pw = self.lite.ph, self.lite.pw
        self.rows, self.cols = self.lite.rows, self.lite.cols
        self.gh, self.gw, self.M = gh, gw, gh * gw
        self.ink_mode = ink
        self.read = dict(color_path="topk", frac=0.25, weight="prob",
                         count="argmax", ridge=0.0, temp=1.0)
        self.read.update(read or {})
        # k = 1 + k_scale*(k_hat - 1), v1 checkpoints only (the v2 models have
        # no k head; k = 1). DEFAULT 0 = the contrast head is not part
        # of this coloring at all: it was trained against the soft mask-weighted
        # mean, it is uncorrelated with the fit separation under the selection
        # read, and it can satisfy separation constraints by itself. Kept as a
        # knob only so its absence can be A/B'd rather than assumed.
        self.k_scale = k_scale
        # NO CACHE. A colorize forward is ~1000 cells/s, so re-evaluating is
        # cheap enough that memoizing is not worth its cost -- and the cache was
        # the memory bug: two tiny tensors per entry meant 47.9k live tensors for
        # 22.8k entries, climbing monotonically. Every query is now a fresh
        # evaluation, which is also strictly more correct under ink='render',
        # where a cell's colors change whenever a neighbour's glyph does.
        self.chunk = chunk
        self.n_query = self.n_forward = 0

        from PIL import Image, ImageOps
        from unicasso.engine.color import decompose, nomination_target
        im = image
        if not hasattr(im, "convert"):
            im = Image.open(im)
        im = ImageOps.exif_transpose(im).convert("RGB") \
            .resize((gw * self.cw, gh * self.ch), Image.LANCZOS)
        rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
        dec = decompose(rgb, gh, gw, self.ch, self.cw)
        self.cell_rgb = dec["cell_rgb"].to(self.dev)              # (M,P,3)
        self.feats = torch.cat(
            [dec["gate"][:, None], (dec["sep"] / 20.0)[:, None],
             dec["fg"], dec["bg"], dec["cell_rgb"].mean(1)], dim=1) \
            .float().to(self.dev)
        self.rgb01 = np.asarray(im, np.float32) / 255.0
        # the photo's own ink field -- the 'center' convention's context
        self.photo_ink = np.clip((1.0 - nomination_target(dec).numpy()) * 255.0,
                                 0, 255).astype(np.uint8)
        self.ink_flat = self.lite.ink_flat                        # (N,P) ink=1
        # The RGB windows never change -- the photo is fixed. Building them per
        # call was the memory bug: (n,116,96,3) float32 is ~1 GB at w60 and it
        # was reallocated every forward. Cached once as uint8 (the source IS
        # uint8, so this is lossless) and sliced per chunk.
        self.rgb_win = _grid_windows(
            np.asarray(im, np.uint8), gh, gw, self.ch, self.cw, self.ph,
            self.pw, self.rows, self.cols, edge=True)             # (M,wh,ww,3)
        # where the center cell sits inside a window
        self.cy0 = (self.rows // 2) * self.ch + self.ph
        self.cx0 = (self.cols // 2) * self.cw + self.pw

    # ------------------------------------------------------------ the query
    @torch.no_grad()
    def query(self, cells, glyphs, grid=None):
        """fg, bg (n,3) detached, for glyph `glyphs[i]` placed in cell `cells[i]`.

        Evaluated fresh every call -- see __init__ on why there is no cache.
        grid (M,) current glyph indices: required for ink='render' (the window
        context), ignored for ink='center'."""
        cells = cells.to(self.dev).long().reshape(-1)
        glyphs = glyphs.to(self.dev).long().reshape(-1)
        if self.ink_mode == "render" and grid is None:
            raise ValueError("--color-lite-ink render needs the current grid")
        self.n_query += cells.numel()
        return self._compute(cells, glyphs, grid)

    @torch.no_grad()
    def _compute(self, cells, glyphs, grid):
        """Build windows, run the colorize pass, fit. CHUNKED: the full (n,4,wh,ww)
        batch is never materialized -- at w60 that tensor alone is 1.3 GB, and in
        'forward' mode it would be reallocated every iteration."""
        self.n_forward += cells.numel()
        ink_img = None
        if self.ink_mode == "render":
            g2 = grid.to(self.dev).long().clone()
            g2[cells] = glyphs                        # the counterfactual grid
            img = self.ink_flat[g2].view(self.gh, self.gw, self.ch, self.cw) \
                .permute(0, 2, 1, 3) \
                .reshape(self.gh * self.ch, self.gw * self.cw)
            ink_img = np.clip(img.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            del img, g2
        else:
            ink_img = self.photo_ink                  # context = the photo itself
        m = self.lite.model
        tci = (self.rows // 2) * self.cols + self.cols // 2
        r = self.read
        cc = cells.cpu().numpy()
        fgs, bgs = [], []
        for i in range(0, cells.numel(), self.chunk):
            cs = cells[i:i + self.chunk]
            gs = glyphs[i:i + self.chunk]
            centers = [divmod(int(c), self.gw) for c in cc[i:i + self.chunk]]
            w = _grid_windows(ink_img, self.gh, self.gw, self.ch, self.cw,
                              self.ph, self.pw, self.rows, self.cols,
                              centers=centers)
            x = torch.from_numpy(w).to(self.dev).float().div_(255).unsqueeze(1)
            if self.ink_mode == "center":
                # only the center cell speaks for the candidate glyph; everything
                # around it stays the photo's own ink, so the answer depends on
                # (cell, glyph) alone, independent of the neighbours
                gi = self.ink_flat[gs].view(-1, self.ch, self.cw)
                x[:, 0, self.cy0:self.cy0 + self.ch,
                  self.cx0:self.cx0 + self.cw] = gi
            rows_ = np.asarray([cy * self.gw + cx for cy, cx in centers])
            cw_ = torch.from_numpy(self.rgb_win[rows_]).to(self.dev) \
                .permute(0, 3, 1, 2).float().div_(255)
            x = torch.cat([x, cw_], dim=1)
            del cw_
            ids, valid = _token_maps(self.gh, self.gw, self.rows, self.cols,
                                     centers=centers)
            ids_t = torch.from_numpy(ids).to(self.dev)
            valid_t = torch.from_numpy(valid).to(self.dev)
            ft = self.feats[ids_t] * valid_t[:, :, None].float()
            t, s1 = m._tokens(x, None, ft, 2, want_skip=True)
            mlog = m.mask_center(x, t, s1)
            t = m.mask_tokens(t, [(getattr(m, "mask_ctx_inject", 0),
                                   m.mask_ctx(x))])
            pm = mlog[:, :, self.ph:self.ph + self.ch,
                      self.pw:self.pw + self.cw].softmax(1)
            if r["color_path"] == "blend":
                from unicasso.training.cellclf_color_train import mask_fit
                fg, bg = mask_fit(pm, self.cell_rgb[cs],
                                  ridge=max(r["ridge"], 1e-3))
            else:
                fg, bg = _conf_fit(pm, self.cell_rgb[cs], mode=r["color_path"],
                                   frac=r["frac"], weight=r["weight"],
                                   ridge=r["ridge"], count=r["count"],
                                   temp=r["temp"])
            if self.bg_fixed is not None:
                from unicasso.engine.color import fit_fg_fixed_bg
                bgc = self.bg_fixed.to(fg.device)
                fg = fit_fg_fixed_bg(self.cell_rgb[cs], pm[:, 0].reshape(pm.shape[0], -1),
                                     bgc, ridge=max(r["ridge"], 1e-3)).clamp(0, 1)
                bg = bgc.expand_as(fg)
            if self.k_scale:
                khat = 4.0 * torch.sigmoid(m.k_head(t[:, m.center])[:, 0] + K_BIAS)
                kf = (1.0 + self.k_scale * (khat - 1.0))[:, None]
                if self.bg_fixed is not None:
                    fg = (bg + kf * (fg - bg)).clamp(0, 1)
                else:
                    mid = 0.5 * (fg + bg)
                    fg = (mid + kf * (fg - mid)).clamp(0, 1)
                    bg = (mid + kf * (bg - mid)).clamp(0, 1)
            fgs.append(fg.detach())
            bgs.append(bg.detach())
            del x, t, s1, mlog, pm, ft, ids_t, valid_t
        return torch.cat(fgs), torch.cat(bgs)

    def stats(self):
        return (f"lite-color: {self.n_query} queries, "
                f"{self.n_forward} cells evaluated")
