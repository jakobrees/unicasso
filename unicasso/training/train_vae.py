"""Train the glyph-VAE.

Every loss component is an additive, flag-gated term, so objectives are A/B-able:
multiscale reconstruction (the default), KL (--beta), ink-axis supervision,
invariance contrastive, structure affinity, uniformity, denoising reconstruction,
symmetry axes, and a cross-modal line-cell classification term.

After training it always runs the viz harness and the eval baseline, and saves a
checkpoint + the mu codebook, into --outdir (default runs/<name>).
"""
import argparse
import json
import os

import numpy as np
import glob

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from unicasso.substrate import glyphs as G
from unicasso.training import eval as gv_eval
from unicasso.training import viz
from unicasso.training import augment as gv_aug
from unicasso.substrate import orientation as O
from unicasso.substrate.model import (GlyphVAE, multiscale_recon_loss, kl_divergence, nt_xent,
                    alignment_loss, teacher_neg_bias, structure_affinity_loss, uniformity_loss)


def parse_args():
    p = argparse.ArgumentParser(description="Train the glyph-VAE (flag-gated loss terms)")
    p.add_argument("--name", default="vae_run", help="run name -> runs/<name>/")
    p.add_argument("--outdir", default=None, help="override output dir")
    p.add_argument("--latent-dim", type=int, default=12)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    # LR schedule: linear warmup then cosine annealing with `lr-cycles` restarts
    # (1 = a single cosine decay; 2 = two cosine drops with a warm restart between).
    p.add_argument("--warmup", type=int, default=100, help="linear-warmup epochs")
    p.add_argument("--lr-cycles", type=int, default=1, help="cosine cycles after warmup (warm restarts)")
    p.add_argument("--min-lr-ratio", type=float, default=0.02, help="cosine floor as fraction of --lr")
    # Fixed-scale latent noise (denoising-AE regularizer; works even at beta=0).
    p.add_argument("--latent-noise", type=float, default=0.0, help="std of fixed isotropic latent noise")
    p.add_argument("--latent-l2", type=float, default=0.0,
                   help="penalty on mean mu^2 -- bounds latent magnitude at beta=0 "
                        "(needed if using --latent-noise without KL, else mu runs away). try 1e-4")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (L2 on weights)")
    # Loss terms (defaults: recon only).
    p.add_argument("--multiscale", action="store_true", default=True,
                   help="multiscale reconstruction (on by default)")
    p.add_argument("--no-multiscale", dest="multiscale", action="store_false")
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL weight (0 = deterministic AE; >0 = VAE)")
    p.add_argument("--sample", action="store_true",
                   help="reparameterize (sample z) during training; implied useful with --beta>0")
    # Ink axis: reserve one latent axis for ink density (tone). Supervises mu[:,axis] to the
    # standardized ink density. NOTE: this aligns the *encoder*; whether the *decoder* routes
    # tone through that axis is a separate question -- check the tone-sweep in ink_axis.png.
    p.add_argument("--ink-weight", type=float, default=0.0, help="weight for the ink-axis term")
    p.add_argument("--ink-axis", type=int, default=0, help="which latent coordinate is the ink axis")
    # Invariance contrastive (InfoNCE on mu over augmented views).
    p.add_argument("--contrastive-weight", type=float, default=0.0, help="weight for the invariance term")
    p.add_argument("--contrastive-temp", type=float, default=0.2, help="NT-Xent temperature")
    p.add_argument("--contrastive-mode", choices=["nce", "align"], default="nce",
                   help="nce=InfoNCE (instance discrimination, has hard negatives); "
                        "align=pull views together only, no negatives (keeps look-alikes clustered)")
    p.add_argument("--teacher-ckpt", default=None,
                   help="frozen model.pt from an earlier run, used for SOFT negatives: its glyph "
                        "similarity down-weights look-alike negatives in nce mode. Must share "
                        "the current charset + field (train it at the same --pad).")
    p.add_argument("--teacher-neg-strength", type=float, default=1.0,
                   help="0 = plain NT-Xent, 1 = fully spare teacher-identical glyphs as negatives")
    # Structure affinity: sample a few random glyphs each step and push/pull their
    # latents toward matching orientation and/or quality (cleanliness). See structure_affinity_loss.
    p.add_argument("--struct-orient-weight", type=float, default=0.0,
                   help="pull same-orientation glyphs together (quality-gated so M isn't pulled to |)")
    p.add_argument("--struct-quality-weight", type=float, default=0.0,
                   help="pull similar-quality glyphs together -> builds the clean-vs-messy axis")
    p.add_argument("--struct-batch", type=int, default=64,
                   help="random glyphs sampled per step for the structure-affinity term")
    p.add_argument("--struct-lin-weight", type=float, default=0.5,
                   help="quality = coh^(1-w)*lin^w used as the cleanliness target (w>0.5 = linearity-led)")
    p.add_argument("--struct-sigma", type=float, default=0.0, help="structure-tensor pre-blur for theta/coh")
    p.add_argument("--struct-gate-pow", type=float, default=1.0,
                   help="exponent on the (q_i*q_j) orientation gate; higher = only the cleanest glyphs steer")
    p.add_argument("--struct-ink-weight", type=float, default=0.0,
                   help="relational TONE+PLACEMENT term: pull glyphs with similar [density, center-of-mass] together (replaces the reserved ink-axis)")
    p.add_argument("--struct-cat-weight", type=float, default=0.0,
                   help="relational CATEGORY term: pull glyphs sharing a category together (needs --quality-labels)")
    p.add_argument("--quality-labels", default=None,
                   help="glyph_labels.json from glyph_annotator: manual quality overrides + category memberships")
    # Symmetry axes: LEARNED directions (never reserved coordinates) made meaningful by
    # self-generated transform views + heads. 4 unit vectors u_rh/u_rv/u_tx/u_ty:
    #   reflection: antisymmetric projection under flips, s(E(flip x)) = -s(E(x))
    #   translation: projection REGRESSES the view's true ink centroid (a position spectrum,
    #                so real codepoint families like ▔¯─_▁ string out along the line)
    #   residual (all 4 axes projected out) must be transform-invariant
    #   linear classifier on z: every view must still classify as its source glyph -- the
    #   anti-collapse force that pushes chiral twins (/ \) to opposite ends
    p.add_argument("--sym-weight", type=float, default=0.0,
                   help="weight on the symmetry geometry (antisym + centroid regression + "
                        "residual invariance). Weak, like the other struct terms; try 0.02")
    p.add_argument("--sym-class-weight", type=float, default=0.0,
                   help="weight on the glyph-identity CE over all transform views; try 0.05")
    p.add_argument("--sym-chars", default=None,
                   help="charset-format file (unicasso.curation.sym_selector output) restricting ALL "
                        "symmetry views/losses to the listed glyphs. Without it, every inked "
                        "glyph participates; blanks are always excluded (a blank's centroid "
                        "target is meaningless and would drag the space latent off-manifold)")
    p.add_argument("--sym-ext-max", type=float, default=0.85,
                   help="glyphs whose ink spans more than this fraction of the CELL along an "
                        "axis are not shifted along it (a full-height '│' has no meaningful "
                        "vertical position -- shifting only clips ink and corrupts the view)")
    # Uniformity (Wang-Isola): the decoupled 'spread' force; pairs with --contrastive-mode align.
    p.add_argument("--uniformity-weight", type=float, default=0.0, help="weight for the uniformity (global spread) term")
    p.add_argument("--uniformity-t", type=float, default=2.0, help="uniformity Gaussian-potential temperature")
    p.add_argument("--uniformity-ramp-frac", type=float, default=0.0,
                   help="linearly ramp the uniformity weight 0 -> full, COMPLETING at this "
                        "fraction of epochs (0 = on from step 0). Lets recon/denoise arrange "
                        "the latent topology first, then spreads it; complete while LR is high.")
    p.add_argument("--uniformity-ramp-start-frac", type=float, default=0.0,
                   help="fraction of epochs at which the uniformity ramp BEGINS (weight 0 "
                        "before). Pair with --lr-cycles: start at a warm restart so the term "
                        "arrives with fresh LR.")
    p.add_argument("--nce-ramp-frac", type=float, default=0.0,
                   help="ramp in the NCE NEGATIVES: contrastive = (1-r)*align + r*nt_xent, "
                        "r reaching 1 at this fraction of epochs. Positive pull (invariance) "
                        "is full-strength from step 0; only instance-discrimination repulsion "
                        "fades in. nce mode only (0 = plain nt_xent).")
    p.add_argument("--nce-ramp-start-frac", type=float, default=0.0,
                   help="fraction of epochs at which the negatives ramp BEGINS (pure align before)")
    p.add_argument("--beta-ramp-frac", type=float, default=0.0,
                   help="linearly ramp beta (KL) 0 -> full, completing at this fraction of "
                        "epochs (0 = full from step 0). Classic KL warm-up: maximum latent "
                        "freedom first, prior pressure after.")
    p.add_argument("--beta-ramp-start-frac", type=float, default=0.0,
                   help="fraction of epochs at which the beta ramp begins")
    p.add_argument("--denoise-weight", type=float, default=0.0,
                   help="denoising recon: reconstruct the CLEAN glyph from an AUGMENTED view (collapse-proof invariance + look-alike preservation; the working replacement for align). Try ~1.0 (recon-scale)")
    # CROSS-MODAL: the SAME encoder also classifies real ascii-line cells against the codebook
    # (cdist, NOT detached) so the real-line task SHAPES the codebook geometry. Cells carry a real-pixel
    # margin (continuation context). Train/test split is BY IMAGE.
    p.add_argument("--crossmodal-weight", type=float, default=0.0, help="weight for the line-cell -> glyph term (0 = off)")
    p.add_argument("--crossmodal-temp", type=float, default=0.5, help="temperature on -cdist(cell, codebook) logits")
    p.add_argument("--crossmodal-blank-weight", type=float, default=0.2,
                   help="CE weight on the blank/space class (<1 so the term shapes LINE glyphs, not just blank->space); ignored if --crossmodal-balance")
    p.add_argument("--crossmodal-balance", action="store_true",
                   help="inverse-frequency per-class CE weights (w_c proportional to 1/count^pow) so over-represented glyphs (block, _, -) don't dominate the codebook reshaping; subsumes --crossmodal-blank-weight")
    p.add_argument("--crossmodal-balance-pow", type=float, default=0.5,
                   help="exponent for --crossmodal-balance: 0.5 = inverse-sqrt (mild), 1.0 = full inverse-freq (aggressive), 0 = uniform")
    p.add_argument("--crossmodal-batch", type=int, default=256, help="line cells sampled per step")
    p.add_argument("--line-data-dir", default=os.path.join(os.path.dirname(__file__), "dataset"),
                   help="curated dataset/ (<stem>.npz indices + <stem>_line.png)")
    p.add_argument("--test-frac", type=float, default=0.15, help="fraction of IMAGES held out for cross-modal test")
    p.add_argument("--pad", type=int, default=0,
                   help="white margin (px) per side; needed so augs don't clip. Use 4 with any augmentation-based term")
    p.add_argument("--pad-w", type=int, default=None,
                   help="horizontal margin override (default = --pad). The conv stack needs "
                        "H+2*pad and W+2*pad_w both div-by-4; sfmono 36x18 cells -> --pad 4 --pad-w 3 (44x24)")
    p.add_argument("--aug-shift", type=float, default=2.0, help="max sub-pixel shift (px)")
    p.add_argument("--aug-scale", type=float, default=0.1, help="scale jitter (range 1-+/-this)")
    p.add_argument("--aug-rot", type=float, default=8.0, help="max rotation (deg)")
    p.add_argument("--aug-blur-sigma", type=float, default=0.8, help="gaussian blur sigma (kept small)")
    p.add_argument("--aug-blur-alpha", type=float, default=0.7, help="max blur mix-in (0..1)")
    p.add_argument("--aug-blur-p", type=float, default=0.5, help="prob a view gets blurred")
    # Pixel-noise on the DENOISE input (encode noisy -> decode CLEAN): makes the encoder robust to
    # non-clean inputs -> better warm-start when asciify encodes real (gray/speckled) image cells.
    # Smooth low-freq field + sparse sharp spikes (see augment.pixel_noise).
    p.add_argument("--aug-pixel-smooth-amp", type=float, default=0.0, help="smooth-field noise std on the denoise input (0 = pixel-noise off); ~0.08 good")
    p.add_argument("--aug-pixel-smooth-sigma", type=float, default=2.0, help="smooth-field blur sigma (px)")
    p.add_argument("--aug-pixel-sharp-frac", type=float, default=0.0, help="fraction of pixels getting a sharp +/- spike (~0.02)")
    p.add_argument("--aug-pixel-sharp-amp", type=float, default=0.75, help="sharp spike magnitude")
    p.add_argument("--aug-pixel-clean-frac", type=float, default=0.1, help="fraction of glyphs kept pixel-noise-FREE each iter (anchors the clean-glyph encoding)")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.contrastive_weight > 0 and args.pad == 0:
        print("WARNING: --contrastive-weight > 0 with --pad 0; augmentations will clip glyphs. "
              "Use --pad 4.")
    return args


def make_lr_lambda(args):
    """Linear warmup -> cosine annealing with `lr-cycles` warm restarts.

    Each restart (cycles > 1) gets a short LINEAR re-warmup instead of jumping straight
    from the cosine floor back to peak -- the abrupt ~peak/floor jump can kick the net into
    decoder saturation and collapse it. Returns a multiplier in [min_lr_ratio, 1].
    """
    warmup = max(0, args.warmup)
    cycles = max(1, args.lr_cycles)
    floor = args.min_lr_ratio
    total_after = max(1, args.epochs - warmup)
    cycle_len = total_after / cycles
    restart_warmup = 0.05 * cycle_len if cycles > 1 else 0.0

    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        t = epoch - warmup
        cycle_idx = int(t // cycle_len)
        local = t - cycle_idx * cycle_len  # epochs into current cycle
        if cycle_idx > 0 and local < restart_warmup:
            return floor + (1.0 - floor) * (local / restart_warmup)  # re-warmup ramp
        denom = cycle_len - restart_warmup if (cycle_idx > 0) else cycle_len
        phase = min(max((local - (restart_warmup if cycle_idx > 0 else 0.0)) / denom, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * phase))

    return fn


def load_line_cells(data_dir, pad, chars, H, W, device, test_frac, seed):
    """Real ascii-line cells (cell + pad-px REAL neighbor margin, ink=1) -> (Xtr,ytr,Xte,yte) split
    BY IMAGE (no leakage). Each <stem>.npz gives the per-cell glyph-index labels; <stem>_line.png the
    pixels. Cell patch = (CH+2pad, CW+2pad) == the glyph raster size, so it feeds the SAME encoder."""
    npzs = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    per_img = []
    for nz in npzs:
        d = np.load(nz, allow_pickle=True)
        if str(d["chars"]) != chars:
            continue
        idx = d["indices"].astype(np.int64)
        GH, GW, CW, CH = int(d["GH"]), int(d["GW"]), int(d["CW"]), int(d["CH"])
        stem = os.path.splitext(os.path.basename(nz))[0]
        line = os.path.join(data_dir, stem + "_line.png")
        if not os.path.exists(line):
            continue
        img = Image.open(line).convert("L").resize((GW * CW, GH * CH), Image.LANCZOS)
        ink = 1.0 - np.asarray(img, np.float32) / 255.0                 # ink=1, white=0 (encoder convention)
        ink = np.pad(ink, pad, constant_values=0.0)                     # off-frame = white(0)
        cells = np.stack([ink[i * CH:i * CH + CH + 2 * pad, j * CW:j * CW + CW + 2 * pad]
                          for i in range(GH) for j in range(GW)])       # (M, CH+2p, CW+2p)
        assert cells.shape[1:] == (H, W), f"{stem}: cell {cells.shape[1:]} != glyph raster {(H, W)} (pad mismatch)"
        per_img.append((torch.from_numpy(cells)[:, None], torch.from_numpy(idx.reshape(-1))))
    assert per_img, f"no usable line data in {data_dir} (need <stem>.npz + <stem>_line.png, matching charset)"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(per_img))
    n_te = max(1, int(round(test_frac * len(per_img))))
    te, tr = perm[:n_te], perm[n_te:]
    cat = lambda ids, k: torch.cat([per_img[i][k] for i in ids]).to(device)
    return cat(tr, 0), cat(tr, 1), cat(te, 0), cat(te, 1), len(perm) - n_te, n_te


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = args.outdir or os.path.join(G.REPO_ROOT, "runs", args.name)
    os.makedirs(outdir, exist_ok=True)

    pad_hw = (args.pad, args.pad if args.pad_w is None else args.pad_w)
    ink, chars = G.load_glyphs(device=args.device, pad=pad_hw)
    device = ink.device
    N, _, H, W = ink.shape
    print(f"Loaded {N} glyphs ({H}x{W}, pad={pad_hw}) on {device}")
    if H % 4 or W % 4:
        print(f"WARNING: padded raster {H}x{W} not divisible by 4 -- the conv stack will "
              f"mis-tile; adjust --pad/--pad-w")

    aug_kw = dict(shift=args.aug_shift, scale=args.aug_scale, rot_deg=args.aug_rot,
                  blur_sigma=args.aug_blur_sigma, blur_alpha=args.aug_blur_alpha,
                  blur_p=args.aug_blur_p)

    model = GlyphVAE(latent_dim=args.latent_dim, char_h=H, char_w=W).to(device)
    sym_on = args.sym_weight > 0 or args.sym_class_weight > 0
    sym_params = []
    if sym_on:
        # participant subset: picked file intersected with "has ink" (blanks never in)
        has_ink = ink.sum(dim=(1, 2, 3)) >= 2.0
        if args.sym_chars:
            with open(args.sym_chars, encoding="utf-8") as f:
                picked = {l.split("\t")[0] for l in f if l.rstrip("\n")}
            in_file = torch.tensor([c in picked for c in chars], device=device)
            missing = picked - set(chars)
            if missing:
                print(f"sym-chars: {len(missing)} listed glyphs not in charset (ignored)")
            sym_idx = torch.nonzero(in_file & has_ink).squeeze(1)
        else:
            sym_idx = torch.nonzero(has_ink).squeeze(1)
        assert sym_idx.numel() > 0, "--sym-weight active but no participating glyphs"
        # per-axis shift participation: full-span glyphs don't translate along that axis
        cell_ink = ink[:, 0, pad_hw[0]:H - pad_hw[0], pad_hw[1]:W - pad_hw[1]] > 0.5

        def span_frac(a):                                       # (N, L) bool -> span fraction
            L_ = a.shape[1]
            first = a.float().argmax(1)
            last = L_ - 1 - a.flip(1).float().argmax(1)
            return torch.where(a.any(1), (last - first + 1).float() / L_,
                               torch.zeros(a.shape[0], device=a.device))
        ext_y = span_frac(cell_ink.any(dim=2))
        ext_x = span_frac(cell_ink.any(dim=1))
        sym_wy = (ext_y < args.sym_ext_max).float()[sym_idx]    # (B,) 0 = y-shift locked
        sym_wx = (ext_x < args.sym_ext_max).float()[sym_idx]
        print(f"symmetry term: {sym_idx.numel()}/{N} glyphs participate"
              + (f" (from {args.sym_chars})" if args.sym_chars else " (all inked)")
              + f"; shift-locked (ext>{args.sym_ext_max}): "
              f"{int((sym_wy == 0).sum())} tall (y), {int((sym_wx == 0).sum())} wide (x)")
        # 4 learned directions (rows: reflect_h, reflect_v, translate_x, translate_y),
        # a scalar affine per translation axis, a linear identity head on z.
        sym_U = nn.Parameter(0.1 * torch.randn(4, args.latent_dim, device=device))
        sym_ab = nn.Parameter(torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device))
        sym_cls = nn.Linear(args.latent_dim, N).to(device)
        sym_params = [sym_U, sym_ab] + list(sym_cls.parameters())
        # per-pixel coordinate grids + per-view ink centroid targets in [-0.5, 0.5]
        ygr = ((torch.arange(H, device=device) + 0.5) / H)[None, None, :, None]
        xgr = ((torch.arange(W, device=device) + 0.5) / W)[None, None, None, :]

        def sym_centroid(x):
            """Targets (B,2) + supervision weight (B,): views with ~no ink (a blank, or a
            shift that pushed the glyph out of frame) have UNDEFINED position -- weight 0,
            never a garbage corner target."""
            m = x.sum(dim=(1, 2, 3))
            w = (m >= 2.0).float()
            m = m.clamp_min(1.0)
            cx = (x * xgr).sum(dim=(1, 2, 3)) / m - 0.5
            cy = (x * ygr).sum(dim=(1, 2, 3)) / m - 0.5
            return torch.stack([cx, cy], dim=1).detach(), w.detach()

        def sym_shift(x, max_dy, max_dx, wy=None, wx=None):
            B = x.shape[0]
            dy = torch.empty(B, device=device).uniform_(-max_dy, max_dy)
            dx = torch.empty(B, device=device).uniform_(-max_dx, max_dx)
            if wy is not None:
                dy = dy * wy
            if wx is not None:
                dx = dx * wx
            th = torch.zeros(B, 2, 3, device=device)
            th[:, 0, 0] = th[:, 1, 1] = 1.0
            th[:, 0, 2] = -2.0 * dx / max(W - 1, 1)             # content moves +dx px
            th[:, 1, 2] = -2.0 * dy / max(H - 1, 1)
            grid = F.affine_grid(th, x.shape, align_corners=False)
            return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    opt = optim.AdamW(list(model.parameters()) + sym_params,
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=make_lr_lambda(args))

    # Ink-axis target: per-glyph ink density, standardized to ~N(0,1) so it's scale-compatible
    # with the latent prior (KL) rather than fighting it.
    dens = G.ink_density(ink).detach()
    dens_std = (dens - dens.mean()) / (dens.std() + 1e-8)

    # Structure-affinity targets: per-glyph orientation (mod pi) + quality (cleanliness), computed
    # ONCE on the (padded) glyph rasters -- fixed labels the structure-affinity term steers the latent to.
    struct_on = (args.struct_orient_weight > 0 or args.struct_quality_weight > 0
                 or args.struct_ink_weight > 0 or args.struct_cat_weight > 0)
    cat_mat = None
    if struct_on:
        g_theta, g_coh, _ = O.glyph_orientations(ink, sigma=args.struct_sigma)
        g_elong, _, _ = O.glyph_linearity(ink)
        cc = g_coh.clamp(0, 1); ll = g_elong.clamp(0, 1)
        g_qual = cc.pow(1.0 - args.struct_lin_weight) * ll.pow(args.struct_lin_weight)
        g_theta = g_theta.to(device); g_qual = g_qual.to(device)
        # relational tone+placement target: [density, com_x, com_y], each in [0,1]. CoM (where the ink
        # sits) disambiguates same-density opposites (_ vs overline, . vs `) that density alone conflates.
        ig = ink[:, 0]                                                       # (N,H,W) 1=ink
        Hh, Ww = ig.shape[-2:]
        ys = torch.linspace(0, 1, Hh, device=ig.device).view(1, Hh, 1)
        xs = torch.linspace(0, 1, Ww, device=ig.device).view(1, 1, Ww)
        mass = ig.sum(dim=(1, 2))
        m = mass.clamp_min(1e-6)
        comx = (ig * xs).sum(dim=(1, 2)) / m
        comy = (ig * ys).sum(dim=(1, 2)) / m
        blank = mass < 1e-6
        comx[blank] = 0.5; comy[blank] = 0.5                                # blanks -> centered
        d01 = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
        g_dens_feat = torch.stack([d01.to(device), comx.to(device), comy.to(device)], dim=1)  # (N,3)
        # Annotator labels: override computed quality where manual, build the category matrix.
        if args.quality_labels:
            with open(args.quality_labels) as f:
                lab = json.load(f)
            cat_names = lab.get("categories", [])
            idxc = {c: i for i, c in enumerate(chars)}
            cat_mat = torch.zeros(N, max(1, len(cat_names)), device=device)
            n_over = 0
            for ch, rec in lab.get("glyphs", {}).items():
                if ch not in idxc:
                    continue
                i = idxc[ch]
                if rec.get("manual"):
                    g_qual[i] = float(rec.get("quality", g_qual[i])); n_over += 1
                for cn in rec.get("categories", []):
                    if cn in cat_names:
                        cat_mat[i, cat_names.index(cn)] = 1.0
            print(f"Loaded {args.quality_labels}: {n_over} manual-quality overrides, "
                  f"{int((cat_mat.sum(1) > 0).sum())} categorized ({len(cat_names)} categories)")
        print(f"Structure-affinity on: orient_w={args.struct_orient_weight} "
              f"quality_w={args.struct_quality_weight} ink_w={args.struct_ink_weight} "
              f"cat_w={args.struct_cat_weight} batch={args.struct_batch}")

    # Optional frozen teacher -> soft-negative bias (nce mode only).
    neg_bias = None
    if args.teacher_ckpt and args.contrastive_weight > 0 and args.contrastive_mode == "nce":
        ck = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
        if (ck["char_h"], ck["char_w"]) != (H, W) or len(ck["chars"]) != N:
            raise ValueError(f"teacher field/charset {ck['char_h']}x{ck['char_w']}/{len(ck['chars'])} "
                             f"!= current {H}x{W}/{N}; retrain the teacher at --pad {args.pad}")
        teacher = GlyphVAE(latent_dim=ck["latent_dim"], char_h=H, char_w=W).to(device)
        teacher.load_state_dict(ck["state_dict"]); teacher.eval()
        with torch.no_grad():
            mu_t, _ = teacher.encode(ink)
        neg_bias = teacher_neg_bias(mu_t, strength=args.teacher_neg_strength)
        print(f"Loaded teacher {args.teacher_ckpt}: soft negatives, strength={args.teacher_neg_strength}")

    # Cross-modal: load real ascii-line cells (train/test split by image) + blank-downweighted CE.
    crossmodal_on = args.crossmodal_weight > 0
    if crossmodal_on:
        Xc_tr, yc_tr, Xc_te, yc_te, n_img_tr, n_img_te = load_line_cells(
            args.line_data_dir, args.pad, chars, H, W, device, args.test_frac, args.seed)
        blank_idx = chars.index(" ") if " " in chars else -1
        if args.crossmodal_balance:
            counts = torch.bincount(yc_tr, minlength=N).float()                      # train-split freq
            ce_w = torch.where(counts > 0, counts.clamp(min=1).pow(-args.crossmodal_balance_pow),
                               torch.zeros_like(counts))                             # absent classes -> 0 (never targets)
            med = ce_w[ce_w > 0].median()
            blk = (ce_w[blank_idx] / med).item() if blank_idx >= 0 else float("nan")
            print(f"Cross-modal: inverse-freq class balance (pow={args.crossmodal_balance_pow}); "
                  f"{int((counts > 0).sum())}/{N} classes present; blank weight = {blk:.3f}x median")
        else:
            ce_w = torch.ones(N, device=device)
            if blank_idx >= 0:
                ce_w[blank_idx] = args.crossmodal_blank_weight
        print(f"Cross-modal on: {n_img_tr} train / {n_img_te} test images -> {len(yc_tr)} train cells, "
              f"{len(yc_te)} test cells; blank-frac(test)={float((yc_te == blank_idx).float().mean()):.3f}")

        @torch.no_grad()
        def xm_eval():
            model.eval()
            mu_g, _ = model.encode(ink)                                  # codebook
            preds = torch.cat([(-torch.cdist(model.encode(Xc_te[i:i + 2048])[0], mu_g)).argmax(1)
                               for i in range(0, len(Xc_te), 2048)])
            top1 = (preds == yc_te).float().mean().item()
            nb = yc_te != blank_idx
            nb1 = (preds[nb] == yc_te[nb]).float().mean().item() if nb.any() else float("nan")
            model.train()
            return top1, nb1

    sample = args.sample or args.beta > 0
    hist = {k: [] for k in ("loss", "recon_full", "kl", "contrastive", "struct", "uniform", "denoise", "crossmodal", "lr", "m_unif", "m_nce", "m_beta", "sym", "symcls")}
    pbar = tqdm(range(args.epochs))
    for ep in pbar:
        opt.zero_grad()
        x_hat, mu, logvar, _ = model(ink, sample=sample, noise_std=args.latent_noise)

        if args.multiscale:
            recon, full, coarse = multiscale_recon_loss(x_hat, ink)
        else:
            recon = ((x_hat - ink) ** 2).mean()
            full = recon.detach(); coarse = torch.tensor(0.0)

        kl = kl_divergence(mu, logvar) if args.beta > 0 else torch.tensor(0.0, device=device)
        lat_l2 = mu.pow(2).mean() if args.latent_l2 > 0 else torch.tensor(0.0, device=device)
        ink_loss = ((mu[:, args.ink_axis] - dens_std) ** 2).mean() if args.ink_weight > 0 \
            else torch.tensor(0.0, device=device)

        # Ramp multipliers: 0 before start_frac, linear to 1 at end_frac (1.0 when end == 0).
        def ramp(start, end):
            if end <= 0:
                return 1.0
            f = (ep + 1) / args.epochs
            if end <= start:
                return 1.0 if f >= end else 0.0            # step at end_frac
            return min(1.0, max(0.0, (f - start) / (end - start)))
        m_unif = ramp(args.uniformity_ramp_start_frac, args.uniformity_ramp_frac)
        m_nce = ramp(args.nce_ramp_start_frac, args.nce_ramp_frac)
        m_beta = ramp(args.beta_ramp_start_frac, args.beta_ramp_frac)

        # Invariance contrastive: InfoNCE on mu over two augmented views (recon/KL stay on the clean glyphs;
        # contrastive shapes the encoder's snapping metric to be augmentation-invariant).
        if args.contrastive_weight > 0:
            mu1, _ = model.encode(gv_aug.augment(ink, **aug_kw))
            mu2, _ = model.encode(gv_aug.augment(ink, **aug_kw))
            if args.contrastive_mode == "align":
                contrastive = alignment_loss(mu1, mu2)
            elif m_nce < 1.0:
                # negatives ramp: full-strength positive pull from step 0, repulsion fades in
                contrastive = (1.0 - m_nce) * alignment_loss(mu1, mu2) \
                    + m_nce * nt_xent(mu1, mu2, temp=args.contrastive_temp, neg_bias=neg_bias)
            else:
                contrastive = nt_xent(mu1, mu2, temp=args.contrastive_temp, neg_bias=neg_bias)
        else:
            contrastive = torch.tensor(0.0, device=device)

        # Structure affinity on a random sub-batch of glyphs (reuse the clean mu we
        # already encoded above -- just index it).
        if struct_on:
            b = min(args.struct_batch, N)
            idx = torch.randperm(N, device=device)[:b]
            struct = structure_affinity_loss(
                mu[idx], theta=g_theta[idx], qual=g_qual[idx], dens=g_dens_feat[idx],
                cats=(cat_mat[idx] if cat_mat is not None else None),
                orient_w=args.struct_orient_weight, quality_w=args.struct_quality_weight,
                ink_w=args.struct_ink_weight,
                cat_w=(args.struct_cat_weight if cat_mat is not None else 0.0),
                gate_pow=args.struct_gate_pow)
        else:
            struct = torch.tensor(0.0, device=device)

        uniform = uniformity_loss(mu, t=args.uniformity_t) \
            if args.uniformity_weight * m_unif > 0 else torch.tensor(0.0, device=device)

        # Cross-modal: classify real line cells against the LIVE codebook (mu, NOT detached) so
        # the real-line task reshapes the glyph embeddings. One encoder, double duty.
        if crossmodal_on:
            bi = torch.randint(0, len(Xc_tr), (min(args.crossmodal_batch, len(Xc_tr)),), device=device)
            mu_cell, _ = model.encode(Xc_tr[bi])                         # (B, L)
            logits = -torch.cdist(mu_cell, mu) / args.crossmodal_temp    # (B, N) vs codebook, mu live
            crossmodal = F.cross_entropy(logits, yc_tr[bi], weight=ce_w)
        else:
            crossmodal = torch.tensor(0.0, device=device)

        # Denoising recon: reconstruct the CLEAN glyph from an AUGMENTED view's latent. The decoder
        # must produce the SPECIFIC clean glyph, so the augmented mapping can't collapse to a constant
        # (unlike align), and clean & augmented are pulled to the same decodable region (true
        # invariance, look-alikes preserved -- no negatives). The collapse-proof replacement for align.
        if args.denoise_weight > 0:
            din = gv_aug.augment(ink, **aug_kw)                          # geometric augment
            if args.aug_pixel_smooth_amp > 0 or args.aug_pixel_sharp_frac > 0:   # + pixel noise (encoder robustness)
                noised = gv_aug.pixel_noise(din, smooth_sigma=args.aug_pixel_smooth_sigma,
                                            smooth_amp=args.aug_pixel_smooth_amp,
                                            sharp_frac=args.aug_pixel_sharp_frac, sharp_amp=args.aug_pixel_sharp_amp)
                keep = (torch.rand(din.shape[0], device=device) < args.aug_pixel_clean_frac)  # random glyphs stay clean
                din = torch.where(keep[:, None, None, None], din, noised)
            x_hat_aug = model(din, sample=sample, noise_std=args.latent_noise)[0]
            if args.multiscale:
                denoise, _, _ = multiscale_recon_loss(x_hat_aug, ink)    # target = CLEAN ink
            else:
                denoise = ((x_hat_aug - ink) ** 2).mean()
        else:
            denoise = torch.tensor(0.0, device=device)

        # Symmetry axes: self-generated transform views; geometry on learned directions,
        # identity via the linear head (see the --sym-weight help for the full contract).
        sym = symcls = torch.tensor(0.0, device=device)
        if sym_on:
            xb = ink[sym_idx]                                    # participants only
            views = [(xb, mu[sym_idx])]
            for xv in (torch.flip(xb, [-1]),                     # flip_h
                       torch.flip(xb, [-2]),                     # flip_v
                       sym_shift(xb, pad_hw[0] * 0.9, pad_hw[1] * 0.9,
                                 wy=sym_wy, wx=sym_wx)):
                views.append((xv, model.encode(xv)[0]))
            Uh = F.normalize(sym_U, dim=1)                       # (4,L) rh/rv/tx/ty
            S = [z @ Uh.t() for _, z in views]                   # each (B,4)
            anti = ((S[1][:, 0] + S[0][:, 0]) ** 2).mean() \
                + ((S[2][:, 1] + S[0][:, 1]) ** 2).mean()
            reg = torch.tensor(0.0, device=device)
            for (xv, _), s in zip(views, S):
                tc, wm = sym_centroid(xv)
                reg = reg + (wm * ((sym_ab[0, 0] * s[:, 2] + sym_ab[0, 1] - tc[:, 0]) ** 2
                                   + (sym_ab[1, 0] * s[:, 3] + sym_ab[1, 1] - tc[:, 1]) ** 2)
                             ).sum() / wm.sum().clamp_min(1.0)
            reg = reg / len(views)
            # residual (all 4 axes projected out) must be transform-invariant. Modified
            # Gram-Schmidt basis (MPS-safe, differentiable); near-dependent directions
            # (e.g. u_rv converging onto u_ty for bars) drop out instead of ill-conditioning.
            basis = []
            for i in range(4):
                v = Uh[i]
                for q in basis:
                    v = v - (v @ q) * q
                nv = v.norm()
                if float(nv) > 0.1:
                    basis.append(v / nv)

            def resid(z):
                for q in basis:
                    z = z - (z @ q)[:, None] * q[None]
                return z
            r0 = resid(views[0][1])
            res = sum(((resid(z) - r0) ** 2).sum(1).mean() for _, z in views[1:]) / 3.0
            # soft orthogonality, tx-ty ONLY: the two translation axes are genuinely
            # independent position parameters (unlike rv/ty, which may legitimately
            # converge for bars); without this they collapse onto one direction
            ortho = (Uh[2] @ Uh[3]) ** 2
            sym = anti + reg + res + 0.1 * ortho
            symcls = sum(F.cross_entropy(sym_cls(z), sym_idx) for _, z in views) / len(views)

        loss = (recon + args.beta * m_beta * kl + args.latent_l2 * lat_l2
                + args.ink_weight * ink_loss + args.contrastive_weight * contrastive
                + struct + args.uniformity_weight * m_unif * uniform
                + args.crossmodal_weight * crossmodal
                + args.denoise_weight * denoise
                + args.sym_weight * sym + args.sym_class_weight * symcls)

        loss.backward()
        opt.step()
        scheduler.step()
        post = dict(loss=f"{loss.item():.5f}", full=f"{float(full):.5f}",
                    kl=f"{float(kl):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        if args.ink_weight > 0:
            post["ink"] = f"{float(ink_loss):.4f}"
        if args.contrastive_weight > 0:
            post["nce"] = f"{float(contrastive):.4f}"
            if args.nce_ramp_frac > 0:
                post["r_nce"] = f"{m_nce:.2f}"
        if struct_on:
            post["str"] = f"{float(struct):.4f}"
        if args.uniformity_weight > 0:
            post["unif"] = f"{float(uniform):.3f}"
            if args.uniformity_ramp_frac > 0:
                post["r_uni"] = f"{m_unif:.2f}"
        if args.beta_ramp_frac > 0:
            post["r_kl"] = f"{m_beta:.2f}"
        if args.denoise_weight > 0:
            post["den"] = f"{float(denoise):.4f}"
        if sym_on:
            post["sym"] = f"{float(sym):.4f}"
            post["scls"] = f"{float(symcls):.3f}"
        if crossmodal_on:
            post["xm"] = f"{float(crossmodal):.3f}"
        pbar.set_postfix(post)
        for k, v in (("loss", loss), ("recon_full", full), ("kl", kl), ("contrastive", contrastive),
                     ("struct", struct), ("uniform", uniform), ("denoise", denoise), ("crossmodal", crossmodal),
                     ("sym", sym), ("symcls", symcls)):
            hist[k].append(float(v))
        hist["lr"].append(scheduler.get_last_lr()[0])
        hist["m_unif"].append(m_unif)
        hist["m_nce"].append(m_nce)
        hist["m_beta"].append(m_beta)

        if crossmodal_on and (ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1):
            t1, nb1 = xm_eval()
            tqdm.write(f"  [xm] ep {ep:4d}  test top1 {t1:.3f}  non-blank top1 {nb1:.3f}")

    # Diagnostics: viz PNGs + eval metrics.
    model.eval()
    print("\nRunning viz harness...")
    viz.run_all(model, ink, chars, outdir, ink_axis=args.ink_axis, aug_kw=aug_kw)

    print("Running eval baseline...")
    metrics = gv_eval.evaluate(model, ink, aug_kw=aug_kw)
    metrics["final_loss"] = float(loss.item())
    metrics["recon_full_mse"] = float(full)
    if crossmodal_on:
        t1, nb1 = xm_eval()
        metrics["xm_test_top1"] = t1
        metrics["xm_test_nonblank_top1"] = nb1
        print(f"  cross-modal TEST: top1 {t1:.3f}  non-blank top1 {nb1:.3f}")
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("  metrics:", json.dumps(metrics, indent=2))

    if sym_on:
        # Axis-coherence probe: how well did the LEARNED directions organize the real
        # codepoint families? (mirror pairs should be antisymmetric along u_rh/u_rv with
        # matching residuals; s_ty should correlate with actual glyph ink height)
        with torch.no_grad():
            mu_f = model.encode(ink)[0]
            Uh = F.normalize(sym_U, dim=1)
            Sf = mu_f @ Uh.t()
            cyx = torch.stack(
                [(ink * xgr).sum((1, 2, 3)) / ink.sum((1, 2, 3)).clamp_min(1.0) - 0.5,
                 (ink * ygr).sum((1, 2, 3)) / ink.sum((1, 2, 3)).clamp_min(1.0) - 0.5], 1)
            corr = lambda a, b: float(torch.corrcoef(torch.stack([a, b]))[0, 1])
            print(f"[sym] s_tx~centroid_x corr {corr(Sf[:, 2], cyx[:, 0]):+.3f}   "
                  f"s_ty~centroid_y corr {corr(Sf[:, 3], cyx[:, 1]):+.3f}")
            for lbl, pairs, ax in (("h", [("/", "\\"), ("(", ")"), ("[", "]"), ("<", ">"),
                                          ("╱", "╲"), ("▏", "▕"), ("╴", "╶")], 0),
                                   ("v", [("▔", "▁"), ("▀", "▄"), ("¯", "_"),
                                          ("╵", "╷"), ("b", "p"), ("▴", "▾")], 1)):
                ss = [(float(Sf[chars.index(a), ax]), float(Sf[chars.index(b), ax]))
                      for a, b in pairs if a in chars and b in chars]
                if ss:
                    antis = [sa + sb for sa, sb in ss]
                    mags = [abs(sa) + abs(sb) for sa, sb in ss]
                    print(f"[sym] reflect_{lbl}: |s_a+s_b| median "
                          f"{np.median(np.abs(antis)):.3f} vs pair magnitude median "
                          f"{np.median(mags):.3f} ({len(ss)} pairs; small/large = good)")
            angs = (Uh @ Uh.t()).abs()
            print(f"[sym] |cos| between axes: rh-tx {angs[0, 2]:.2f}  rv-ty {angs[1, 3]:.2f}  "
                  f"rh-rv {angs[0, 1]:.2f}  tx-ty {angs[2, 3]:.2f}")

    # Checkpoint + codebook.
    torch.save({"state_dict": model.state_dict(), "chars": chars,
                "latent_dim": args.latent_dim, "char_h": H, "char_w": W,
                "sym_U": (sym_U.detach().cpu() if sym_on else None),
                "config": vars(args)}, os.path.join(outdir, "model.pt"))
    np.save(os.path.join(outdir, "codebook_mu.npy"), viz.codebook(model, ink))
    np.savez(os.path.join(outdir, "loss_hist.npz"), **{k: np.array(v) for k, v in hist.items()})
    print(f"\nSaved model.pt, codebook_mu.npy, metrics.json, loss_hist.npz -> {outdir}")


if __name__ == "__main__":
    main()
