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
DEFAULT_WEIGHTS = {
    ("color", "dejavu"): "weights/lite/unicasso-lite-color.pt",
    ("line", "dejavu"): "weights/lite/unicasso-lite-line.pt",
    ("color", "sfmono"): "weights/lite/unicasso-lite-color-sfmono.pt",
    ("line", "sfmono"): "weights/lite/unicasso-lite-line-sfmono.pt",
}


def _grid_windows(img, gh, gw, ch, cw, ph, pw, rows, cols, pad_val=0, centers=None):
    py, px = (rows // 2) * ch + ph, (cols // 2) * cw + pw
    spec = ((py, py), (px, px)) + (((0, 0),) if img.ndim == 3 else ())
    pad = np.pad(img, spec, constant_values=pad_val)
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
    def __init__(self, kind_or_path="color", device=None, font=None):
        """kind_or_path: "color" / "line" (resolved via `font`) or a ckpt path.
        font: "dejavu" (default) or "sfmono"; falls back to $GLYPHVAE_FONT.
        A checkpoint path ignores `font` -- its config names its own kit."""
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        font = font or os.environ.get("GLYPHVAE_FONT", "dejavu")
        path = DEFAULT_WEIGHTS.get((kind_or_path, font), kind_or_path)
        ck = torch.load(G.repo_path(path), map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.chars = ck["chars"]
        self.color = bool(cfg.get("color_model"))
        from unicasso.adapter.corrupt import CorruptionSampler
        profile = cfg.get("profile", "dejavu")
        vae = {"dejavu": "weights/vae_dejavu/model.pt",
               "sfmono": "weights/vae_sfmono/model.pt"}[profile]
        self.align_blend = bool(cfg.get("align_blend"))
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
                m.mode_emb = nn.Parameter(torch.zeros(2, dim))
            m.k_head = nn.Linear(dim, 1)
            if "col_head.weight" in sd:
                m.col_head = nn.Linear(dim, 6)
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
                 stride=1, ensemble="mean", temp=1.0):
        M = gh * gw                                     # cells in the grid
        centers = None if stride <= 1 else _strided_centers(gh, gw, stride)
        w = _grid_windows(ink_u8, gh, gw, self.ch, self.cw, self.ph, self.pw,
                          self.rows, self.cols, centers=centers)
        x = torch.from_numpy(w).to(self.device).float().div_(255).unsqueeze(1)
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
    def render(self, image, width=60, ban=None, stride=1,
               ensemble="mean", temp=1.0):
        """image: path or PIL.Image. Color model -> full LiteResult with .ans;
        line model -> glyph text from the grayscale image.

        ban: iterable of characters to exclude from the output (never snapped);
        the classifier's logits for those glyphs are masked before the argmax.
        stride: forward windows only at every s-th cell (the dense head covers
        the cells in between) -- ~stride^2 fewer forwards, fewer framings/cell.
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
        g, ct = self._predict(ink_u8, gh, width, feats=feats, mode=1,
                              ban_idx=ban_idx, stride=stride,
                              ensemble=ensemble, temp=temp)
        khat = 4.0 * torch.sigmoid(self.model.k_head(ct)[:, 0] + K_BIAS)
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
    p.add_argument("--png", default=None, help="also save the pixel render as a PNG (color model)")
    p.add_argument("--ban-chars", default="", help="characters to exclude from the output")
    p.add_argument("--ban-blocks", action="store_true", help="also exclude block chars ░▒▓█▄▌▐▀■")
    p.add_argument("--ban-letters", action="store_true",
                   help="also exclude all Unicode letters (A-Z a-z + accented)")
    p.add_argument("--stride", type=int, default=1,
                   help="window stride in cells: forward only every s-th window, the dense "
                        "head covers the rest (~s^2 speedup, fewer framings per cell)")
    p.add_argument("--ensemble", default="mean",
                   choices=["mean", "gmean", "center", "sample"],
                   help="per-cell combination of the covering windows' predictions: "
                        "mean = kernel-weighted arithmetic mean (default), gmean = "
                        "geometric mean (sharper, veto-like), center = no ensemble "
                        "(own window only, needs stride 1), sample = stochastic "
                        "draw from the mean distribution")
    p.add_argument("--temp", type=float, default=1.0,
                   help="sampling temperature for --ensemble sample (lower = greedier)")
    p.add_argument("--device", default=None, help="torch device (default: auto -- cuda/mps/cpu)")
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
    out = lite.render(args.image, width=width, ban=ban or None,
                      stride=args.stride, ensemble=args.ensemble, temp=args.temp)
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
