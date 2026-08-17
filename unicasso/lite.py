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
    print(out.ans)                        # out.txt / out.glyphs / out.k also set

Pipeline (color): decompose (per-cell Lab two-color clustering) -> ink structure
-> transformer glyph classifier w/ framing-ensemble accumulation -> closed-form
blend colors (0.5 cluster + 0.5 distance-weighted MSE fit at the chosen glyph's
mask) -> learned per-cell contrast k around each cell's own midpoint.
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
DEFAULT_WEIGHTS = {"color": "weights/lite/unicasso-lite-color.pt",
                   "line": "weights/lite/unicasso-lite-line.pt"}


def _grid_windows(img, gh, gw, ch, cw, ph, pw, rows, cols, pad_val=0):
    py, px = (rows // 2) * ch + ph, (cols // 2) * cw + pw
    spec = ((py, py), (px, px)) + (((0, 0),) if img.ndim == 3 else ())
    pad = np.pad(img, spec, constant_values=pad_val)
    wh, ww = rows * ch + 2 * ph, cols * cw + 2 * pw
    out = np.empty((gh * gw, wh, ww) + ((3,) if img.ndim == 3 else ()), img.dtype)
    i = 0
    for y in range(gh):
        for x in range(gw):
            t = py + y * ch - (rows // 2) * ch - ph
            l = px + x * cw - (cols // 2) * cw - pw
            out[i] = pad[t:t + wh, l:l + ww]; i += 1
    return out


def _token_maps(gh, gw, rows, cols):
    offs = [(t // cols - rows // 2, t % cols - cols // 2) for t in range(rows * cols)]
    ys, xs = np.mgrid[0:gh, 0:gw]
    yy = ys.ravel()[:, None] + np.array([o[0] for o in offs])[None]
    xx = xs.ravel()[:, None] + np.array([o[1] for o in offs])[None]
    valid = (yy >= 0) & (yy < gh) & (xx >= 0) & (xx < gw)
    return np.where(valid, yy * gw + xx, 0).astype(np.int64), valid


def _binomial_kernel(rows, cols):
    from math import comb
    v = torch.tensor([comb(rows - 1, i) for i in range(rows)], dtype=torch.float32)
    h = torch.tensor([comb(cols - 1, j) for j in range(cols)], dtype=torch.float32)
    return torch.outer(v / v[rows // 2], h / h[cols // 2]).flatten()


@dataclass
class LiteResult:
    glyphs: np.ndarray            # (gh, gw) int glyph indices
    txt: str                      # plain glyph text
    ans: str | None               # 24-bit ANSI (color model only)
    fg: np.ndarray | None         # (M, 3) cell foreground colors
    bg: np.ndarray | None
    k: np.ndarray | None          # (gh, gw) learned contrast field
    render: np.ndarray | None     # (H, W, 3) float render (color) or None


class Lite:
    def __init__(self, kind_or_path="color", device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        path = DEFAULT_WEIGHTS.get(kind_or_path, kind_or_path)
        ck = torch.load(G.repo_path(path), map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.chars = ck["chars"]
        self.color = bool(cfg.get("color_model"))
        from unicasso.adapter.corrupt import CorruptionSampler
        sampler = CorruptionSampler(G.repo_path("weights/vae_dejavu/model.pt"),
                                    device="cpu", profile="dejavu")
        self.ch, self.cw = sampler.CH, sampler.CW
        self.ph, self.pw = cfg.get("pad_h", 3), cfg.get("pad_w", 4)
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
        if self.color:
            dim = cfg.get("dim", 64)
            if cfg.get("feat_dim", 0):
                m.color_proj = nn.Linear(cfg["feat_dim"], dim)
            m.mode_emb = nn.Parameter(torch.zeros(2, dim))
            m.k_head = nn.Linear(dim, 1)
            m.col_head = nn.Linear(dim, 6)
        m.load_state_dict(ck["state_dict"])
        self.model = m.to(self.device).eval()
        self.N = N
        self.kernel = _binomial_kernel(self.rows, self.cols).to(self.device)
        if self.color:
            from unicasso.training.cellclf_color import glyph_bg_dist
            self.bg_dist = glyph_bg_dist(self.ink_flat.cpu(), self.ch, self.cw) \
                .to(self.device)

    @torch.no_grad()
    def _predict(self, ink_u8, gh, gw, feats=None, mode=None):
        M = gh * gw
        w = _grid_windows(ink_u8, gh, gw, self.ch, self.cw, self.ph, self.pw,
                          self.rows, self.cols)
        x = torch.from_numpy(w).to(self.device).float().div_(255).unsqueeze(1)
        ids, valid = _token_maps(gh, gw, self.rows, self.cols)
        ids_t = torch.from_numpy(ids).to(self.device)
        valid_t = torch.from_numpy(valid).to(self.device)
        feats_tok = (feats[ids_t] * valid_t[:, :, None].float()
                     if feats is not None else None)
        lo, to = [], []
        for i in range(0, M, 1024):
            t = self.model._tokens(x[i:i + 1024], None,
                                   feats_tok[i:i + 1024]
                                   if feats_tok is not None else None, mode)
            lo.append(self.model.head(t[:, self.model.n_extra:]))
            to.append(t[:, self.model.center])
        logits, ct = torch.cat(lo), torch.cat(to)
        probs = logits.softmax(-1)
        wtok = self.kernel[None, :] * valid_t.float()
        acc = torch.zeros(M, self.N, device=self.device)
        wsum = torch.zeros(M, device=self.device)
        acc.index_add_(0, ids_t.reshape(-1),
                       (probs * wtok[:, :, None]).reshape(-1, self.N))
        wsum.index_add_(0, ids_t.reshape(-1), wtok.reshape(-1))
        return (acc / wsum[:, None].clamp_min(1e-8)).argmax(-1), ct

    def _txt(self, pred):
        return "\n".join("".join(self.chars[g] for g in row) for row in pred) + "\n"

    @torch.no_grad()
    def render(self, image, width=60):
        """image: path or PIL.Image. Color model -> full LiteResult with .ans;
        line model -> glyph text from the grayscale image."""
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        im = ImageOps.exif_transpose(image).convert("RGB")
        gh = raster.grid_height_for_aspect(im.width, im.height, width,
                                           self.cw, self.ch, 0)
        im = im.resize((width * self.cw, gh * self.ch), Image.LANCZOS)
        if not self.color:
            gray = np.asarray(im.convert("L"), np.float32) / 255.0
            ink_u8 = np.clip((1.0 - gray) * 255.0, 0, 255).astype(np.uint8)
            g, _ = self._predict(ink_u8, gh, width)
            pred = g.view(gh, width).cpu().numpy()
            return LiteResult(pred, self._txt(pred), None, None, None, None, None)
        from unicasso.engine.color import decompose, nomination_target
        from unicasso.training.cellclf_color import ansi_txt, fit_fg_bg_distweighted
        rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
        dec = decompose(rgb, gh, width, self.ch, self.cw)
        ink_u8 = np.clip((1.0 - nomination_target(dec).numpy()) * 255.0,
                         0, 255).astype(np.uint8)
        mean = dec["cell_rgb"].mean(1)
        feats = torch.cat([dec["gate"][:, None], (dec["sep"] / 20.0)[:, None],
                           dec["fg"], dec["bg"], mean], dim=1).float().to(self.device)
        g, ct = self._predict(ink_u8, gh, width, feats=feats, mode=1)
        khat = 4.0 * torch.sigmoid(self.model.k_head(ct)[:, 0] + K_BIAS)
        mask = self.ink_flat[g]
        C = dec["cell_rgb"].to(self.device)
        fg_f, bg_f = fit_fg_bg_distweighted(C, mask, self.bg_dist[g], pow_=1.0)
        fg0 = 0.5 * dec["fg"].to(self.device) + 0.5 * fg_f
        bg0 = 0.5 * dec["bg"].to(self.device) + 0.5 * bg_f
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
    p.add_argument("--weights", default=None,
                   help="override the weights path (default: the shipped color/line model)")
    p.add_argument("--out", default=None,
                   help="write the render here instead of stdout (stdout is clean ANSI/text, "
                        "safe to pipe or cat)")
    p.add_argument("--png", default=None, help="also save the pixel render as a PNG (color model)")
    p.add_argument("--device", default=None, help="torch device (default: auto -- cuda/mps/cpu)")
    args = p.parse_args()
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):     # loader banners off stdout
        lite = Lite(args.weights or ("line" if args.line else "color"),
                    device=args.device)
    width = args.width
    if width <= 0:                                    # 0 = fill the terminal when on a screen
        import shutil
        width = shutil.get_terminal_size((60, 20)).columns if sys.stdout.isatty() else 60
    out = lite.render(args.image, width=width)
    text = out.txt if args.line or out.ans is None else out.ans
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    if args.png and out.render is not None:
        Image.fromarray((out.render * 255).astype(np.uint8)).save(args.png)


if __name__ == "__main__":
    main()
