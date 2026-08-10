"""Per-cell two-color decomposition: the bridge between a color image and the
monochrome structure the nomination channels need.

The split this module implements:
  * NOMINATION side -- every discrete-search channel (grad/pixel/blend/join/ports/
    tone) asks a STRUCTURE question, answerable in one channel. `decompose` turns
    each cell into (ink, fg, bg): ink = soft membership of the minority color,
    which is exactly the semantics `bm_flat` already has, so those channels run
    unmodified on a color source.
  * LOSS side -- fg/bg are only the INITIALIZATION for the per-slot color leaves
    that CLIP/recon then optimize. Nothing here constrains the final colors.

Per cell, in Lab:
    v          = principal color axis (eigvec of the cell's color covariance)
    proj       = (lab - mean) @ v                       -- 1-D
    thr        = 1-D 2-means (Lloyd) split of proj
    fg cluster = the MINORITY side, bg = the majority
    ink        = sigmoid(s*(proj - thr) / (ramp * |mu_hi - mu_lo|))

fg=minority matters: the codebook is overwhelmingly low-coverage, and glyphs have
no complements, so letting neighbouring cells disagree about which color is "ink"
makes their structures inverses of each other and the seam tears.

Flat-cell gate (the confetti control -- 2-means will happily cluster JPEG noise
inside a region of flat skin):
    sep   = ||mu_fg - mu_bg||_Lab                       -- perceptual separation
    within= within-cluster spread along v
    gate  = smoothstep((sep - jnd)/jnd) * smoothstep((sep/within - ratio)/ratio)
    ink  *= gate
A gated-off cell is one flat color: ink 0, fg := bg. Downstream that reads as an
empty cell, so `empty_safe` and the whole anti-confetti stack keep working and
what they emit is a flat color block -- which is the correct answer.

Line art is the degenerate case and comes out unchanged: clusters {black, white},
minority = the strokes, ink = the strokes, gate wide open.

    python -m unicasso.engine.color IMG --grid 60x30 --out DIR [--palette 16]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# D65, sRGB
_M_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.00000, 1.08883])


def srgb_to_lab(rgb):
    """(..., 3) sRGB in [0,1] -> (..., 3) CIE Lab (L in [0,100])."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp_min(1e-8) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.to(rgb).t() / _WHITE.to(rgb)
    e = 216.0 / 24389.0
    k = 24389.0 / 27.0
    f = torch.where(xyz > e, xyz.clamp_min(1e-8) ** (1.0 / 3.0), (k * xyz + 16.0) / 116.0)
    return torch.stack([116.0 * f[..., 1] - 16.0,
                        500.0 * (f[..., 0] - f[..., 1]),
                        200.0 * (f[..., 1] - f[..., 2])], dim=-1)


def smoothstep(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def cells_rgb(img, gh, gw, ch, cw):
    """(H,W,3) -> (M, P, 3) cells, row-major, P = ch*cw."""
    H, W, _ = img.shape
    assert H >= gh * ch and W >= gw * cw, f"image {H}x{W} too small for {gh}x{gw} cells of {ch}x{cw}"
    x = img[:gh * ch, :gw * cw].reshape(gh, ch, gw, cw, 3)
    return x.permute(0, 2, 1, 3, 4).reshape(gh * gw, ch * cw, 3)


def _lloyd_1d(proj, iters=12):
    """1-D 2-means on (M,P). Returns thr (M,1), mu_lo (M,1), mu_hi (M,1)."""
    thr = torch.zeros(proj.shape[0], 1, device=proj.device, dtype=proj.dtype)
    mu_lo = proj.min(1, keepdim=True).values
    mu_hi = proj.max(1, keepdim=True).values
    for _ in range(iters):
        hi = (proj >= thr).to(proj.dtype)
        n_hi = hi.sum(1, keepdim=True)
        n_lo = (1 - hi).sum(1, keepdim=True)
        mu_hi = torch.where(n_hi > 0, (proj * hi).sum(1, keepdim=True) / n_hi.clamp_min(1), mu_hi)
        mu_lo = torch.where(n_lo > 0, (proj * (1 - hi)).sum(1, keepdim=True) / n_lo.clamp_min(1), mu_lo)
        thr = 0.5 * (mu_lo + mu_hi)
    return thr, mu_lo, mu_hi


@torch.no_grad()
def decompose(img, gh, gw, ch, cw, ramp=0.35, jnd=4.0, ratio=1.5, lloyd=12):
    """(H,W,3) in [0,1] -> dict with
        ink  (M,P)   soft minority-color membership in [0,1]  (1 = fg)
        fg   (M,3)   minority-cluster mean RGB
        bg   (M,3)   majority-cluster mean RGB
        sep  (M,)    Lab separation of the two clusters
        gate (M,)    flat-cell gate in [0,1] (already applied to ink)
        axis (M,3)   principal color axis in Lab (diagnostic)
    """
    dev = img.device
    C = cells_rgb(img, gh, gw, ch, cw)                              # (M,P,3)
    M, P, _ = C.shape
    lab = srgb_to_lab(C)                                            # (M,P,3)
    mu = lab.mean(1, keepdim=True)
    X = lab - mu
    cov = X.transpose(1, 2) @ X / max(P - 1, 1)                     # (M,3,3)
    # MPS has no eigh; a batched 3x3 on CPU is trivial and this is a one-time precompute
    evals, evecs = torch.linalg.eigh(cov.cpu())                     # ascending
    v = evecs[:, :, -1].to(cov.device)                              # (M,3) principal axis
    proj = (X * v[:, None, :]).sum(-1)                              # (M,P)

    thr, mu_lo, mu_hi = _lloyd_1d(proj, lloyd)
    hi = (proj >= thr)
    n_hi = hi.sum(1)
    minority_is_hi = n_hi <= (P - n_hi)                             # fg = the smaller side
    s = torch.where(minority_is_hi, 1.0, -1.0)[:, None]             # orient so fg -> +1

    spread = (mu_hi - mu_lo).abs().clamp_min(1e-6)                  # (M,1) axis separation
    ink = torch.sigmoid(s * (proj - thr) / (ramp * spread))         # (M,P)

    # cluster colors: soft-membership-weighted RGB means (= the closed-form fit at this mask)
    w = ink
    sw = w.sum(1, keepdim=True).clamp_min(1e-6)
    sw_c = (1 - w).sum(1, keepdim=True).clamp_min(1e-6)
    fg = (w[:, :, None] * C).sum(1) / sw                            # (M,3)
    bg = ((1 - w)[:, :, None] * C).sum(1) / sw_c

    sep = (srgb_to_lab(fg) - srgb_to_lab(bg)).norm(dim=-1)          # (M,) perceptual gap
    # within-cluster spread along the axis (noise floor); ratio-test kills noise clusters
    resid = proj - torch.where(hi, mu_hi, mu_lo)
    within = resid.pow(2).mean(1).sqrt().clamp_min(1e-6)
    # `within` lives in Lab-projection units, same as spread -> compare like with like
    gate = smoothstep((sep - jnd) / max(jnd, 1e-6)) \
        * smoothstep((spread[:, 0] / within - ratio) / max(ratio, 1e-6))

    ink = ink * gate[:, None]
    flat = gate < 1e-3
    fg = torch.where(flat[:, None], bg, fg)                         # flat cell: fg := bg
    return dict(ink=ink, fg=fg, bg=bg, sep=sep, gate=gate, axis=v,
                cell_rgb=C, gh=gh, gw=gw, ch=ch, cw=cw)


def nomination_target(d):
    """Decomposition -> (gh*ch, gw*cw) white=1 grayscale: the drop-in `target` for
    every nomination-side consumer (pix_fit, cell_ink, tone_gate, blur_T, tgt_cent,
    empty_safe). ink=1 (fg) renders as 0 (dark), matching bm_flat's convention."""
    gh, gw, ch, cw = d["gh"], d["gw"], d["ch"], d["cw"]
    g = (1.0 - d["ink"]).view(gh, gw, ch, cw).permute(0, 2, 1, 3).reshape(gh * ch, gw * cw)
    return g.contiguous()


@torch.no_grad()
def fit_fg_bg(cell_rgb, mask, ridge=1e-3):
    """Closed-form per-cell MSE fit of fg/bg for a given ink mask -- the init for the
    per-slot color leaves, and the emission-time fit.

    Minimizes ||bg + (fg-bg)*m - c||^2 over the cell:
        fg = weighted mean of c under m, bg = weighted mean under (1-m)
    cell_rgb (M,P,3), mask (M,P) in [0,1] -> fg (M,3), bg (M,3).
    `ridge` keeps a near-empty or near-full mask from producing a wild color."""
    w = mask.clamp(0, 1)
    sw = w.sum(1, keepdim=True)
    sc = (1 - w).sum(1, keepdim=True)
    mean = cell_rgb.mean(1)                                          # fallback for degenerate masks
    fg = ((w[:, :, None] * cell_rgb).sum(1) + ridge * mean) / (sw + ridge).clamp_min(1e-6)
    bg = (((1 - w)[:, :, None] * cell_rgb).sum(1) + ridge * mean) / (sc + ridge).clamp_min(1e-6)
    return fg, bg


@torch.no_grad()
def build_palette(colors, n, iters=25, seed=0):
    """k-means over (Q,3) RGB -> (n,3) palette. Lab-space distance, RGB centroids.
    Feeding the optimizer a QUANTIZED color space is the defense against the color
    shortcut: with fg/bg free and continuous, a cell can hit any tone by tinting a
    full block, which costs nothing and beats finding the tone structurally."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = colors.shape[0]
    n = min(n, Q)
    cen = colors[torch.randperm(Q, generator=g)[:n]].clone()
    lab_c = srgb_to_lab(colors)
    for _ in range(iters):
        d = torch.cdist(lab_c, srgb_to_lab(cen))                     # (Q,n)
        a = d.argmin(1)
        for k in range(n):
            m = a == k
            if m.any():
                cen[k] = colors[m].mean(0)
    return cen


def snap_to_palette(c, pal):
    """(...,3) -> nearest palette entry (Lab distance). Returns (colors, indices)."""
    flat = c.reshape(-1, 3)
    d = torch.cdist(srgb_to_lab(flat), srgb_to_lab(pal))
    i = d.argmin(1)
    return pal[i].reshape(c.shape), i.reshape(c.shape[:-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--grid", default="60x30", help="GWxGH character grid")
    ap.add_argument("--cell", default="18x36", help="CWxCH pixel cell")
    ap.add_argument("--ramp", type=float, default=0.35, help="ink softness, as a fraction of cluster separation")
    ap.add_argument("--jnd", type=float, default=4.0, help="Lab separation below which a cell is flat")
    ap.add_argument("--ratio", type=float, default=1.5, help="min separation/within-spread to count as structure")
    ap.add_argument("--palette", type=int, default=0, help="quantize fg/bg to N colors (0 = off)")
    ap.add_argument("--out", default=None, help="output dir (default: alongside the image)")
    args = ap.parse_args()

    GW, GH = (int(t) for t in args.grid.lower().split("x"))
    CW, CH = (int(t) for t in args.cell.lower().split("x"))
    out = args.out or os.path.dirname(os.path.abspath(args.image))
    os.makedirs(out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    im = Image.open(args.image).convert("RGB").resize((GW * CW, GH * CH), Image.LANCZOS)
    img = torch.from_numpy(np.asarray(im, np.float32) / 255.0)

    d = decompose(img, GH, GW, CH, CW, ramp=args.ramp, jnd=args.jnd, ratio=args.ratio)
    nom = nomination_target(d)
    fg, bg = d["fg"], d["bg"]
    if args.palette:
        pal = build_palette(torch.cat([fg, bg], 0), args.palette)
        fg, _ = snap_to_palette(fg, pal)
        bg, _ = snap_to_palette(bg, pal)

    n_flat = int((d["gate"] < 1e-3).sum())
    M = GH * GW
    print(f"{stem}: {M} cells, {n_flat} flat ({100 * n_flat / M:.0f}%), "
          f"mean Lab separation {float(d['sep'].mean()):.1f}, "
          f"mean ink {float(d['ink'].mean()):.3f}")

    def save(t, path, gray=False):
        a = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(a, "L" if gray else "RGB").save(path)

    save(nom, os.path.join(out, f"{stem}_cink.png"), gray=True)
    # two-color reconstruction: what the decomposition claims the image is
    rec = (bg[:, None, :] + (fg - bg)[:, None, :] * d["ink"][:, :, None]) \
        .view(GH, GW, CH, CW, 3).permute(0, 2, 1, 3, 4).reshape(GH * CH, GW * CW, 3)
    save(rec, os.path.join(out, f"{stem}_2col.png"))
    # residual: where the two-color assumption breaks (3+ color cells)
    res = (rec - img).abs().mean(-1)
    save((res / res.max().clamp_min(1e-6)), os.path.join(out, f"{stem}_resid.png"), gray=True)
    print(f"  two-color residual: mean {float(res.mean()):.4f}, p99 {float(res.flatten().quantile(0.99)):.4f}")

    panel = torch.cat([img, nom[:, :, None].expand(-1, -1, 3), rec], dim=1)
    save(panel, os.path.join(out, f"{stem}_panel.png"))
    np.savez(os.path.join(out, f"{stem}_pal.npz"),
             fg=fg.numpy(), bg=bg.numpy(), gate=d["gate"].numpy(), sep=d["sep"].numpy(),
             gh=GH, gw=GW, ch=CH, cw=CW)
    print(f"  wrote {stem}_cink.png / _2col.png / _resid.png / _panel.png / _pal.npz -> {out}")


if __name__ == "__main__":
    main()
