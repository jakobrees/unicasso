"""Image -> ASCII/Unicode art optimizer (CLI).

Renders an image as a grid of monospace glyphs by direct optimization against the target.
Modes: swarm (canonical -- K traveling latent slots per cell with learned arbitration
weights, see unicasso/engine/swarm.py), knn-smooth (single-latent soft-blend baseline
ablation), and pool (free-logit candidate pools per cell). The loss combines pixel/multiscale
reconstruction, a CLIP perceptual stack (random crops, dense conv sweep, snap/consistency
planes, optional domain adapters), and structural terms (orientation,
affinity, emptiness/blank rent). The discrete search runs as periodic
nomination rounds: proposal channels pitch glyphs per cell, optionally verified by
measured live probing of hard swaps on the current render. Optional color mode fits
per-cell fg/bg colors and emits ANSI truecolor output.
"""
import argparse
import json
import math
import os
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from unicasso.substrate import glyphs as G
from unicasso.substrate import orientation as O
from unicasso.substrate.model import GlyphVAE

from unicasso.substrate import raster as train


def load_vae(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    H, W = ck["char_h"], ck["char_w"]
    model = GlyphVAE(latent_dim=ck["latent_dim"], char_h=H, char_w=W).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Training pad (white margin around the cell) from the ckpt config; (pad_h, pad_w) may
    # differ (sfmono 36x18 trains at 44x24 = pad 4/3). Legacy fallback derives from the cell
    # constants -- only correct for ckpts that predate config saving, on the default font.
    cfg = ck.get("config") or {}
    ph = cfg.get("pad")
    if ph is None:
        ph = (H - train.CHAR_HEIGHT) // 2
    pw = cfg.get("pad_w")
    if pw is None:
        pw = ph
    model.sym_U = ck.get("sym_U")     # learned symmetry axes (None on pre-sym ckpts)
    return model, ck["chars"], ck["latent_dim"], (ph, pw)


def target_line_stats(target, GH, GW, CH, CW, row_gap, gate_pow=1.0):
    """Per-cell line evidence from the TARGET, computed once: ink centroid (yc, xc in [0,1]),
    unit normal (ny, nx) to the stroke direction (centroid errors are projected on it -- the
    along-line offset is irrelevant), and the coherence x ink-band gate. Shared by the swarm
    height prior / join centroid gate and the coordinate loss."""
    with torch.no_grad():
        M = GH * GW
        tgt_ink = 1.0 - cells_flat(target, GH, GW, CH, CW, row_gap)             # (M, P)
        dev = tgt_ink.device
        yg = ((torch.arange(CH, device=dev) + 0.5) / CH)[:, None].expand(CH, CW).reshape(-1)
        xg = ((torch.arange(CW, device=dev) + 0.5) / CW)[None, :].expand(CH, CW).reshape(-1)
        tm = tgt_ink.sum(1)
        tyc = (tgt_ink * yg).sum(1) / tm.clamp_min(1e-6)
        txc = (tgt_ink * xg).sum(1) / tm.clamp_min(1e-6)
        tth, tcoh, _ = O.glyph_orientations(tgt_ink.view(M, 1, CH, CW), sigma=0.0)
        tth, tcoh = tth.to(dev), tcoh.to(dev).clamp(0, 1)
        tink = tgt_ink.mean(1)
        gate = tcoh.pow(gate_pow) * ((tink > 0.05) & (tink < 0.5)).float()
        cent = torch.stack([tyc, txc], dim=1)
        norm = torch.stack([torch.cos(tth), -torch.sin(tth)], dim=1)
    return cent, norm, gate


_PERSIST = {}      # heavy-model cache for persistent E-step workers
# (armed only under ASCIIFY_PERSIST=1: one process runs many refinement jobs
# back-to-back and VAE/CLIP reloads would dominate short runs; normal CLI
# invocations construct fresh models exactly as before)


def cycled_temp(it, iters, v0, v1, cycles, decay, end_frac, shape="geo", lin_end=0.0, lin_n=0):
    """Phased-hardening temperature: [0, end_frac*iters) split into `cycles` cosine anneals, each
    from a decaying reheat peak down to v1; holds v1 after. Each cycle is a full election + a
    softer re-arbitration window (SGDR-style warm restarts, applied to the discretization).
    Peak sequence: geo = v1 + (v0-v1)*decay^i (front-loads the cooling); lingeo = max(linear
    descent over the cycles, geo) -- linear through the early/middle generations, geometric
    tail once decay^i overtakes (late cycles = refinement, never a zero-reheat)."""
    end = max(1.0, end_frac * iters)
    if it >= end:
        return v1
    seg = end / max(1, cycles)
    i = min(int(it // seg), cycles - 1)
    frac = decay ** i
    if shape == "lingeo" and cycles > 1:
        tf = max(0.0, (lin_end - v1) / max(1e-9, v0 - v1))   # linear ramp's final peak frac
        if lin_n and lin_n > 1:
            # piecewise: linear v0 -> lin_end over the first lin_n cycles, then geometric decay
            # CONTINUING from lin_end for the remaining cycles
            if i < lin_n:
                frac = 1.0 - (1.0 - tf) * i / (lin_n - 1)
            else:
                frac = tf * decay ** (i - (lin_n - 1))
        else:
            frac = max(frac, 1.0 - (1.0 - tf) * i / (cycles - 1))
    peak = v1 + (v0 - v1) * frac
    p = (it - i * seg) / seg
    return v1 + (peak - v1) * 0.5 * (1.0 + math.cos(math.pi * min(max(p, 0.0), 1.0)))


def schedule_value(it, iters, v0, v1, kind="linear", warmup_frac=0.0, warmup_start=None, end_frac=1.0):
    """Anneal v0 -> v1 over `iters` steps. If warmup_frac > 0, the first fraction linearly ramps
    warmup_start -> v0, then `kind` (linear|cosine) anneals v0 -> v1, COMPLETING by end_frac of
    training and holding v1 afterward (end_frac<1 = reach the final value early -> settle time).
    v1=None => const v0. (warmup_frac=0, end_frac=1, kind=linear is a plain linear anneal.)"""
    if v1 is None:
        return v0
    warm = int(warmup_frac * iters)
    if warm > 0 and it < warm:
        ws = v0 if warmup_start is None else warmup_start
        return ws + (v0 - ws) * (it / warm)
    end = max(warm + 1.0, end_frac * (iters - 1))
    p = min(max((it - warm) / (end - warm), 0.0), 1.0)
    if kind == "cosine":
        return v1 + 0.5 * (v0 - v1) * (1.0 + math.cos(math.pi * p))
    return v0 + (v1 - v0) * p


def clip_recon_grad_maps(g_clip, g_recon, GH, GW, CH, CW):
    """Per-cell maps from the render-space gradients (row_gap 0). g_clip, g_recon: (H,W).
    Under --color both gradients are (H,W,3) and the cosine runs over the flattened cell INCLUDING
    the color axis, so the conflict map also reports chromatic disagreement -- CLIP wanting a hue
    recon does not, which the gray version had no way to express.
    Returns (conflict, clipmag): conflict = -cos(per-cell grads) in [-1,1] (>0 = CLIP & recon fight),
    clipmag = per-cell CLIP-gradient norm. Both EMA'd across steps by the caller (training dynamics)."""
    def cells(g):  # (H,W) -> (GH,GW,CH*CW);  (H,W,3) -> (GH,GW,CH*CW*3)
        if g.dim() == 3:
            return g[:GH * CH, :GW * CW].reshape(GH, CH, GW, CW, 3) \
                .permute(0, 2, 1, 3, 4).reshape(GH, GW, -1)
        return g[:GH * CH, :GW * CW].reshape(GH, CH, GW, CW).permute(0, 2, 1, 3).reshape(GH, GW, -1)
    gc, gr = cells(g_clip), cells(g_recon)
    conflict = -(gc * gr).sum(-1) / (gc.norm(dim=-1) * gr.norm(dim=-1) + 1e-9)
    return conflict, gc.norm(dim=-1)


def save_clip_diag(soft, conflict, clipmag, resid, curve, it, total_iters, path,
                   inj_status=None, inj_margin=None, w_space=None, vit_agree=None, vit_gcos=None):
    """Top row of 4 map panels over a bottom row of (extra small maps + a log loss plot).
    Default top: soft render | EMA CLIP<->recon conflict | EMA CLIP push | snap residual.
    With injection maps: panels 1&2 become injection STATUS + survival-MARGIN, and CLIP push drops to a
    small bottom-left panel. With w_space (--space-candidate): a 'space weight' map (which cells are
    going blank) joins the bottom row. The loss plot squishes to fill whatever's left."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np_ = lambda t: t.detach().cpu().numpy()
    soft, resid = np_(soft), np_(resid)
    clipmag = np_(clipmag); clipmag = clipmag / (clipmag.max() + 1e-9)
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85], hspace=0.28, wspace=0.35)
    axp = [fig.add_subplot(gs[0, i]) for i in range(4)]
    if soft.ndim == 3:                      # color render: imshow takes RGB directly
        axp[0].imshow(np.clip(soft, 0, 1)); axp[0].set_title("soft COLOR render (what CLIP sees)", fontsize=10)
    else:
        axp[0].imshow(soft, cmap="gray"); axp[0].set_title("soft render (what CLIP sees)", fontsize=10)
    bottom_maps = []   # (data, cmap, vmin, vmax, title) panels placed left-of-loss in the bottom row
    if inj_status is not None:
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch
        status = np_(inj_status)
        cmap = ListedColormap(["0.88", "#f4c430", "#2ca02c", "#d62728"])   # idle | competing | winning | failed
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        axp[1].imshow(status, cmap=cmap, norm=norm, interpolation="nearest")
        n_comp = int((status == 1).sum()); n_win = int((status == 2).sum()); n_fail = int((status == 3).sum())
        axp[1].set_title("injection status", fontsize=10)
        axp[1].legend(handles=[Patch(color="#2ca02c", label=f"winning {n_win}"),
                               Patch(color="#f4c430", label=f"competing {n_comp}"),
                               Patch(color="#d62728", label=f"failed {n_fail}")],
                      fontsize=7, loc="upper right", framealpha=0.9)
        margin = np.ma.masked_invalid(np_(inj_margin))
        cmm = plt.get_cmap("RdBu").copy(); cmm.set_bad("0.92")
        mmax = max(1e-3, float(np.abs(margin).max()) if margin.count() else 1e-3)
        im = axp[2].imshow(margin, cmap=cmm, vmin=-mmax, vmax=mmax, interpolation="nearest")
        axp[2].set_title("injection survival margin (blue=safe, red=will prune)", fontsize=10)
        fig.colorbar(im, ax=axp[2], fraction=0.046, pad=0.02)
        bottom_maps.append((clipmag, "magma", 0, 1, "CLIP push EMA (bright=CLIP pushing)"))  # bumped from top
    else:
        conflict = np_(conflict)
        for a, dat, cm, lo, hi, ttl in (
                (axp[1], conflict, "coolwarm", -1, 1, "CLIP<->recon conflict EMA (red=fighting)"),
                (axp[2], clipmag, "magma", 0, 1, "CLIP push EMA (bright=CLIP pushing)")):
            im = a.imshow(dat, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest"); a.set_title(ttl, fontsize=10)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    im = axp[3].imshow(resid, cmap="viridis", interpolation="nearest")    # snap residual: always panel 3
    axp[3].set_title("snap residual (bright=no glyph fits)", fontsize=10)
    fig.colorbar(im, ax=axp[3], fraction=0.046, pad=0.02)
    if w_space is not None:   # --space-candidate: which cells the optimizer is emptying
        ws = np_(w_space)
        bottom_maps.append((ws, "Greens", 0.0, 1.0, "space weight (green=cell going blank)"))
    if vit_agree is not None:   # read-only ViT render-vs-target patch agreement (green=match, red=off)
        bottom_maps.append((np_(vit_agree), "RdYlGn", -1.0, 1.0, f"ViT patch agreement (global {vit_gcos:.2f})"))
    for a in axp:
        a.axis("off")
    for i, (dat, cm, lo, hi, ttl) in enumerate(bottom_maps):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(dat, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(ttl, fontsize=10); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axl = fig.add_subplot(gs[1, len(bottom_maps):])   # loss plot fills the rest of the bottom row
    its = curve.get("it", [])
    for k in ("loss", "recon", "clip", "commit", "sym", "cons", "affin", "neighbor", "nudge", "orient", "div"):
        if curve.get(k):
            axl.plot(its[:len(curve[k])], curve[k], lw=1, label=k)
    if curve.get("hard_recon"):
        axl.plot(curve["hard_it"], curve["hard_recon"], lw=1.3, ls="--", color="crimson", label="hard recon")
    if curve.get("vit_sim"):   # ViT render-vs-target global cosine (validation, higher=better; positive so log-safe)
        axl.plot(curve["vit_sim_it"], curve["vit_sim"], lw=1.3, ls=":", color="teal", label="vit_sim ↑")
    axl.set_yscale("log"); axl.set_xlim(0, total_iters); axl.axvline(it, color="0.6", lw=0.8)
    if bottom_maps:   # y-axis on the RIGHT so its label/ticks don't collide with the left panel's colorbar
        axl.yaxis.set_label_position("right"); axl.yaxis.tick_right()
    axl.set_xlabel("iter"); axl.set_ylabel("loss (log)"); axl.legend(fontsize=8, ncol=5, loc="upper right")
    fig.suptitle(f"iter {it}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def assemble(cells, gh, gw, ch, cw, gap, img_h, img_w):
    """(B,GH,GW,CH,CW) white=1 cells -> (B,img_h,img_w) image (mirrors train.render_ascii)."""
    B = cells.shape[0]
    grid = cells.permute(0, 1, 3, 2, 4).reshape(B, gh, ch, gw * cw)
    if gap > 0:
        out = torch.ones(B, img_h, img_w, device=cells.device)
        for i in range(gh):
            y = i * (ch + gap)
            out[:, y:y + ch, :] = grid[:, i]
        return out
    return grid.reshape(B, img_h, img_w)


def assemble_rgb(cells, gh, gw, ch, cw, gap, img_h, img_w):
    """(B,GH,GW,CH,CW,3) RGB cells -> (B,img_h,img_w,3). Color twin of `assemble`."""
    B = cells.shape[0]
    grid = cells.permute(0, 1, 3, 2, 4, 5).reshape(B, gh, ch, gw * cw, 3)
    if gap > 0:
        out = torch.ones(B, img_h, img_w, 3, device=cells.device)
        for i in range(gh):
            y = i * (ch + gap)
            out[:, y:y + ch, :, :] = grid[:, i]
        return out
    return grid.reshape(B, img_h, img_w, 3)


def save_loss_curve(curve, path):
    """Plot the recorded training series -> PNG (+ raw .npz). Three panels:
      1. loss components (log y) -- total + each active term.
      2. soft-vs-hard recon gap -- the discretization gap (does the soft proxy commit?).
      3. anneal schedules (knn-temp, z-noise) -- context for why the curves move.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = curve["it"]
    comp = [k for k in ("loss", "recon", "commit", "clip", "orient", "sym", "nudge", "neighbor", "empty", "rent", "div")
            if k in curve]
    sched = [k for k in ("knn_temp", "z_noise", "pool_noise", "w_temp") if k in curve]
    n = 2 + (1 if sched else 0)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), sharex=True)

    for k in comp:
        axes[0].plot(its, curve[k], lw=1, label=k)
    axes[0].set_yscale("log"); axes[0].set_ylabel("loss (log)")
    axes[0].legend(fontsize=7, ncol=4); axes[0].set_title("loss components")

    axes[1].plot(its, curve["recon"], lw=1, label="recon (soft render)")
    if "hard_recon" in curve:
        axes[1].plot(curve["hard_it"], curve["hard_recon"], lw=1.4, color="crimson",
                     label="recon (hard snap)")
    axes[1].set_ylabel("recon"); axes[1].legend(fontsize=8)
    axes[1].set_title("soft vs hard render -- discretization gap (want it to close)")

    if sched:
        for k in sched:
            axes[2].plot(its, curve[k], lw=1, label=k)
        axes[2].set_ylabel("schedule"); axes[2].legend(fontsize=8)
        axes[2].set_title("anneal schedules")
    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    npz = os.path.splitext(path)[0] + ".npz"
    np.savez(npz, **{k: np.asarray(v, dtype=np.float32) for k, v in curve.items()})
    return npz


def cells_flat(tgt, gh, gw, ch, cw, gap):
    """(img_h, img_w) target -> (GH*GW, CH*CW) flattened per-cell patches (inverse of assemble)."""
    if gap == 0:
        return tgt.view(gh, ch, gw, cw).permute(0, 2, 1, 3).reshape(gh * gw, ch * cw)
    return torch.stack([tgt[i * (ch + gap):i * (ch + gap) + ch, j * cw:j * cw + cw].reshape(-1)
                        for i in range(gh) for j in range(gw)])


def apply_profile_recipe(p):
    """Fold the active font kit's profile.json "recipe" block into the parser defaults.

    Most of the canonical recipe is font-independent and lives in the add_argument
    defaults. A few knobs are not: the coordinate prior and the join centroid shaping
    are both measured in units of a CELL, and kits differ in cell aspect (sfmono 18x36
    = 2.00, dejavu 16x34 = 2.125), so the values that are right for one kit are
    slightly wrong for the other. Those live in the kit next to the geometry that
    causes them, rather than in a README footnote nobody re-reads.

    These are DEFAULTS -- an explicit command-line flag still wins.
    """
    path = G.active_profile_path()
    if not path or not os.path.exists(path):
        return
    try:
        with open(path) as f:
            rec = json.load(f).get("recipe") or {}
    except (OSError, ValueError) as e:
        print(f"warning: could not read recipe from {path}: {e}")
        return
    if not rec:
        return
    valid = {a.dest for a in p._actions}
    unknown = sorted(set(rec) - valid)
    if unknown:
        print(f"warning: profile recipe keys not recognised, ignored: {unknown}")
    # Recipe path values name files the REPO owns, so they resolve against the repo
    # root, not the caller's cwd -- unlike the same flag typed on the command line,
    # which is the user's path and stays theirs.
    PATH_KEYS = {"vae_ckpt", "clip_adapter", "clip_steer", "init_vae_ckpt"}
    use = {k: (G.repo_path(v) if k in PATH_KEYS and isinstance(v, str) else v)
           for k, v in rec.items() if k in valid}
    if use:
        p.set_defaults(**use)
        print(f"kit recipe ({os.path.basename(os.path.dirname(path))}): "
              + ", ".join(f"{k}={v!r}" for k, v in sorted(use.items())))


def parse_args():
    p = argparse.ArgumentParser(description="Asciify: image -> ASCII/Unicode art optimizer")
    p.add_argument("input_image")
    p.add_argument("--mode", choices=["swarm", "knn-smooth", "pool"], default="swarm",
                   help="swarm=traveling-slot particle pool: K learned latent "
                        "positions + learned weights per cell (unicasso/engine/swarm.py; reuses "
                        "the --pool-* nomination/probe/record machinery); knn-smooth=the k-nearest "
                        "blend IS the forward render (soft interp, anneal --knn-temp to sharpen); "
                        "pool=free-logit candidate pool per cell (VAE only seeds/dedups; see "
                        "unicasso/engine/pool.py)")
    p.add_argument("--vae-ckpt", default=None, help="required (all modes)")
    p.add_argument("--knn-k", type=int, default=10, help="neighbors for the knn-smooth blend")
    p.add_argument("--knn-temp", type=float, default=0.8, help="softmax temperature over latent distances (knn-smooth); start value if annealing")
    p.add_argument("--knn-temp-end", type=float, default=0.02,
                   help="if set, anneal knn-temp -> this over training (soft->sharp). default: constant")
    p.add_argument("--knn-temp-schedule", choices=["linear", "cosine"], default="cosine",
                   help="(knn-smooth) annealing curve for knn-temp after warmup")
    p.add_argument("--knn-temp-warmup-frac", type=float, default=0.0,
                   help="fraction of iters to linearly warm up knn-temp before annealing (0 = none)")
    p.add_argument("--knn-temp-warmup-start", type=float, default=None,
                   help="temp to warm up FROM (default: knn-temp-end, i.e. sharp -> soft peak -> sharp)")
    # Pool mode: per-cell candidate pool + FREE logits (see unicasso/engine/pool.py). tau reuses the
    # --knn-temp* schedule (+ per-cell --empty-temp-scale sharpen); Gumbel logit noise replaces
    # z-noise; the blend-diversity penalty replaces knn-smooth's geometric look-alike guarantee.
    p.add_argument("--pool-k", type=int, default=12, help="(pool) candidates per cell")
    p.add_argument("--pool-init-single", action="store_true",
                   help="(pool) start each pool as ONE glyph (the warm-start/init-scan/init-text "
                        "choice) duplicated across all K slots -- renders pure, and every further "
                        "member must be nominated (and, with --pool-probe, MEASURED) into a copy's "
                        "slot: membership strictly earned from a single seed. Pair with "
                        "--pool-evict-below 1.1 (duplicate copies sit exactly at the 1/K share)")
    p.add_argument("--pool-noise", type=float, default=1.0,
                   help="(pool) start scale of annealed Gumbel noise on the logits (logit units; the "
                        "z-noise successor -- stochastic auditions for buried candidates)")
    p.add_argument("--pool-noise-end", type=float, default=0.0, help="(pool) anneal Gumbel noise -> this")
    p.add_argument("--pool-noise-schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--pool-noise-end-frac", type=float, default=1.0,
                   help="(pool) fraction of training by which the Gumbel noise reaches its end value, "
                        "then holds (e.g. 0.85 = a quiet consolidation phase before the temperature "
                        "bottoms out, countering late-run snap oscillation)")
    p.add_argument("--pool-logit-cap", type=float, default=0.0,
                   help="(pool) per-cell logit-spread ceiling: after each step, center the cell's logits "
                        "and clamp their std to this (one-sided -- weaker opinions untouched); 0 = off. "
                        "Audition-reach guard: a Gumbel flip needs sigma*(noise gap) > LOGIT gap and tau "
                        "cancels, so once demotion grows spreads well past sigma, auditions die early. "
                        "~0.6-0.8 keeps rivals within 1-1.5 sigma all window; commitment still arrives "
                        "via the tau anneal")
    p.add_argument("--pool-diversity-weight", type=float, default=1e-2,
                   help="(pool) weight on the blend-diversity penalty D(c) = sum_k w_k*||B_k||^2 - ||render_c||^2 "
                        "(per-pixel weighted variance of the blend: charges for incoherent composites, the CLIP "
                        "blur-exploit guard). START value; set 0 + --pool-diversity-weight-end to ramp in late")
    p.add_argument("--pool-diversity-weight-end", type=float, default=1e-2,
                   help="(pool) ramp the diversity weight to this by --pool-diversity-end-frac")
    p.add_argument("--pool-diversity-weight-start-frac", type=float, default=0.15,
                   help="(pool) hold the diversity weight at its start value until this fraction of training "
                        "(exploration needs shape-different arrivals; D(c) is the COMMIT-phase guard)")
    p.add_argument("--pool-diversity-weight-end-frac", type=float, default=1.0,
                   help="(pool) fraction of training by which the diversity weight reaches its end value")
    # Pool nomination: score the WHOLE codebook per cell with the render-gradient EMA
    # (S = g_ema @ bm_flat^T -- the same signal that drives the logit gradient, unmasked by w) and
    # rotate wanted outsiders into the pool: evict the weakest post-grace slot, tabu the evictee.
    p.add_argument("--pool-nominate-rate", type=int, default=25,
                   help="(pool) iters between nomination rounds; 0 = off (static warm-start pool)")
    p.add_argument("--pool-latent-h", type=float, default=1.0,
                   help="(pool) latent-channel adjacency bandwidth, x median codebook NN spacing "
                        "(kernel P = exp(-d^2/2h^2)); the frontier's step size through latent space")
    p.add_argument("--pool-latent-floor", type=float, default=0.1,
                   help="(pool) latent channel abstains in cells whose best adjacency score "
                        "sum_k w_ema_k*P(cand_k, g) is below this (isolated pools propose nothing)")
    p.add_argument("--pool-latent-margin", type=float, default=-0.25,
                   help="(pool) latent-channel score edge over the best member (sigma units)")
    p.add_argument("--pool-logit-decay", type=float, default=0.01,
                   help="(pool) AdamW weight decay on the logits. Decay-toward-zero = decay toward "
                        "the UNIFORM blend, i.e. an always-on entropy regularizer. Higher = softer/"
                        "longer-contested blends, 0 = pure gradient dynamics")
    p.add_argument("--pool-grad-every", type=int, default=1,
                   help="(pool) grad channel participates only every Nth nomination round (random "
                        "exploration as a FRACTION of proposals, not the diet); 1 = every round")
    # Ports channel: tile-continuity nominations (unicasso/engine/tile_ports.py). Proposals only
    # from the tile whitelist; score = target port match + live-neighbor compatibility; NO gradient
    # gate (ports exist because the linear score can't see structure) -- it puts line-glyph
    # families on the ballot in cells where no other channel would propose them.
    p.add_argument("--pool-ports-chars", default="default",
                   help="(pool) tile whitelist for the ports channel ('default' = box+blocks+slashes)")
    p.add_argument("--pool-ports-overrides", default=None,  # None -> <active kit>/port_overrides.json
                   help="(pool) JSON of hand-curated glyph-pair bans (built with `python -m "
                        "unicasso.engine.tile_ports --interactive`; default: the active kit's "
                        "port_overrides.json)")
    p.add_argument("--pool-ports-floor", type=float, default=1.0,
                   help="(pool) ports channel abstains below this combined score (target match + "
                        "neighbor compat; raw sim units -- line joins score ~1.4-3.4)")
    p.add_argument("--pool-ports-nb-weight", type=float, default=1.0,
                   help="(pool) weight of live-neighbor compatibility vs target port match")
    p.add_argument("--pool-ports-gamma", type=float, default=1.0, help="(pool) port sim mismatch penalty")
    p.add_argument("--pool-blend-thresh", type=float, default=0.25,
                   help="(swarm 'blend' channel) blur-ink presence threshold: the channel matches "
                        "glyphs against the BINARIZED clean soft render (raw MSE prefers a mixture's "
                        "dominant component over the union glyph -- binarizing is what lets `═│` "
                        "propose `╪`)")
    p.add_argument("--pool-blend-margin", type=float, default=0.75,
                   help="(swarm 'blend') propose only if the best fit < this x the incumbent snap "
                        "glyph's fit (committed cells whose blur is already realized stay silent)")
    p.add_argument("--pool-blend-nb-weight", type=float, default=0.02,
                   help="(swarm 'blend') weight of the live-neighbor port-join score as a tie-break "
                        "among near-equal blur fits")
    p.add_argument("--pool-join-floor", type=float, default=1.0,
                   help="(swarm 'join' channel) pure tile-connection completion, blur-blind "
                        "(`═▄`/`║` -> `╗`): propose the argmax live-neighbor port score if it "
                        "clears this floor AND beats the incumbent's score")
    p.add_argument("--pool-blend-resid", type=float, default=0.02,
                   help="(swarm 'blend') abstain unless the blur contains at least this fraction "
                        "of cell pixels the INCUMBENT glyph doesn't cover (an `L`+`_` union is "
                        "basically `L` -- nothing to realize)")
    p.add_argument("--pool-join-dangling", type=float, default=3.0,
                   help="(swarm join/blend tie-break) penalty weight on candidate ports that point "
                        "at neighbors with BLANK facing edges: prefer the corner over the T-cross "
                        "when there's no third stroke to meet (`╕` over `╤`). 0 = off")
    p.add_argument("--pool-join-centroid", type=float, default=0.0,
                   help="(swarm 'join') HARD centroid veto: in coherent-line cells, block join "
                        "proposals whose glyph ink centroid, PROJECTED ON THE LINE NORMAL, is "
                        "farther than this from the target cell's (cell units). Superseded by "
                        "the softer --pool-join-coord-sigma. 0 = off; try 0.25")
    p.add_argument("--pool-pixel-margin", type=float, default=0.85,
                   help="(swarm 'pixel' channel) propose only if the candidate's full-res "
                        "pixel fit beats the incumbent's by this factor")
    p.add_argument("--pool-pixel-gate", type=float, default=0.25,
                   help="(swarm 'pixel') coherence split: cells above propose best pixel-match "
                        "STRUCTURE; cells at/below (isotropic texture) propose the density-"
                        "matched TONE glyph (shades/blocks) when dark enough")
    p.add_argument("--pool-pixel-tone-ink", type=float, default=0.35,
                   help="(swarm 'pixel') min target-cell ink for a tone proposal in "
                        "incoherent cells (the region-C grill signature)")
    p.add_argument("--pool-pixel-grating", type=int, default=0,
                   help="(swarm 'pixel') 0=off. Sub-Nyquist stripe gate: dark cells whose "
                        "target shows >= this many distinct ink runs within +-1.5 cells "
                        "(measured along the cell orientation AND its normal) are forced "
                        "into the TONE regime regardless of coherence. Grilles/fences are "
                        "COHERENT texture -- oriented right but with a pitch unresolvable at "
                        "cell scale -- so the coherence gate alone can never admit them to "
                        "tone proposals. Try 8")
    p.add_argument("--pool-pixel-grating-area", type=int, default=15,
                   help="(swarm 'pixel') min 4-connected component size (cells, no "
                        "diagonals) for grating cells to switch regime: only sustained "
                        "fields (a grill) abstract to tone; isolated dense cells inside "
                        "linework stay structure")
    p.add_argument("--pool-join-coord-sigma", type=float, default=0.0,
                   help="(swarm 'join') coord-agreement SHAPING: subtract gate*(e/sigma)^2 "
                        "from every candidate's join score before the argmax (e = normal-"
                        "projected centroid error). Join redirects to right-position "
                        "completions instead of getting vetoed; incumbent shaped too. "
                        "0 = off; try 0.2 (e=0.1 costs 0.25 score, e=0.3 costs 2.25 vs floor 1)")
    p.add_argument("--pool-blend-rel-thresh", type=float, default=0.0,
                   help="(swarm 'blend') relative presence threshold: binarize at "
                        "min(--pool-blend-thresh, this x the cell's max soft ink), floor 0.02. "
                        "Gives FAINT-line cells a presence map (at abs 0.25 they produce an empty "
                        "map and the channel never advocates for them). 0 = off; try 0.5")
    p.add_argument("--swarm-height-weight", type=float, default=0.0,
                   help="(swarm) height prior: detached W bias -w x |slot blend ink centroid - "
                        "target cell centroid| on coherent-line cells (structure-tensor coherence "
                        "x ink band [0.05, 0.5] gate). Fixed W-unit bias / annealing tau_W = "
                        "negligible in exploration, decisive at tie time: breaks the collective "
                        "wrong-height line equilibrium CLIP is indifferent to. 0 = off; try 0.5")
    p.add_argument("--swarm-height-gate-pow", type=float, default=1.0,
                   help="(swarm) exponent on the coherence gate of the height prior")
    p.add_argument("--clip-dense-weight", type=float, default=0.25,
                   help="dense geometric CLIP term: ONE fully-conv forward of the whole render "
                        "(stem+layers) vs the target's cached maps, mean (1-cos) over feature "
                        "locations. Even coverage at fixed scale -- no crop lottery, no center "
                        "bias -- the mid-scale slot between pixel recon and global CLIP. "
                        "Needs an RN backbone clipper. 0 = off; try 0.5")
    p.add_argument("--clip-dense-weight-end", type=float, default=None)
    p.add_argument("--clip-dense-weight-schedule", default="cosine")
    p.add_argument("--clip-dense-weight-end-frac", type=float, default=1.0)
    p.add_argument("--clip-dense-layers", default="1,2",
                   help="conv stages for the dense term (1=fine/8px ... 4=coarse); '1,2' = "
                        "the cheap geometric pair (forward early-exits after the deepest)")
    p.add_argument("--clip-dense-res", type=int, default=544,
                   help="long-side working resolution of the dense sweep")
    p.add_argument("--swarm-tone-weight", type=float, default=0.0,
                   help="(swarm) tone W-bias: -w x |slot blend ink - target ink| on the slot "
                        "race in TONE-regime cells (coherence <= --pool-pixel-gate AND ink >= "
                        "--pool-pixel-tone-ink). Self-annealing via /tau_W: nudge early, "
                        "decisive at commitment -- the retention counterweight to CLIP's "
                        "tendency to out-race admitted tone glyphs with thin bars. "
                        "Drift-proof (reads current blend density). 0 = off; try 0.5")
    p.add_argument("--clip-snap-weight", type=float, default=0.2,
                   help="(swarm) weight on d_CLIP(SNAP render, target): the shipping image scored "
                        "every step, straight-through (hard forward, soft-blend backward). Breaks "
                        "the space-vs-ink tie equilibria the soft loss actively maintains. Each "
                        "active crop encodes 3 images instead of 2 (+50%% CLIP cost while on)")
    p.add_argument("--clip-snap-start-frac", type=float, default=0.2,
                   help="(swarm) the snap CLIP term switches on at this fraction of training "
                        "(early soft exploration stays unpenalized)")
    p.add_argument("--clip-consist-weight", type=float, default=0.15,
                   help="(swarm) CONSOLIDATION: weight on d_CLIP(soft render, snap render) -- "
                        "charges the discretization gap directly, pulls mixtures toward REALIZABLE "
                        "composites. Incumbent-reinforcing, so it ramps in late")
    p.add_argument("--clip-consist-start-frac", type=float, default=0.45,
                   help="(swarm) consolidation term switches on at this fraction")
    p.add_argument("--clip-snap-mode", choices=["winner", "glyph"], default="winner",
                   help="(swarm) what the snap CLIP term's 'hard' component is. winner = the ARGMAX "
                        "PARTICLE's blend render: fully differentiable through that slot's z (real "
                        "gradients -- the winner can NAVIGATE, e.g. travel to space), near-hard "
                        "late as tau_z anneals. glyph = the discrete snap bitmap with straight-"
                        "through backward (measured: a flat tax with little leverage)")
    p.add_argument("--clip-snap-gate", choices=["none", "entropy"], default="entropy",
                   help="(swarm) entropy: composite the snap-term image per cell as m*hard + "
                        "(1-m)*soft with m = H(v)/logK DETACHED -- uncommitted cells are scored on "
                        "their hard reality (no hiding in the mixture), committed cells pay "
                        "nothing, and the term self-anneals as the run commits. none = ungated "
                        "(measured: a flat tax concentrated on nothing)")
    p.add_argument("--pool-allow-blank-noms", action="store_true",
                   help="(swarm) let channels nominate blank glyphs. Default: BLOCKED for every "
                        "channel (births/boosts can never point at space/NBSP) -- blank stays "
                        "reachable only via continuous latent drift, so whitespace can't be "
                        "proposed into ink regions")
    p.add_argument("--swarm-adam-reset", action="store_true",
                   help="(swarm) at births/purge-reseeds, zero Adam's FIRST moment for the "
                        "transplanted slot (kills the evictee's ghost momentum); the second moment "
                        "is kept (zeroing it mid-run yields near-unbounded steps). Off by default: "
                        "the un-reset behavior gives newborns oversized early steps")
    p.add_argument("--swarm-purge-cycles", action=argparse.BooleanOptionalAction, default=True,
                   help="(swarm) ELITE PURGE at every tau_W reheat seam: keep only each open "
                        "cell's argmax slot, free the rest (respawn budget), and re-seed one "
                        "RANDOM allowed glyph per cell at W = elite - 1. Each cycle restarts as "
                        "incumbent + wildcard + open slot under a fresh reheat -- structural "
                        "exploration instead of noise")
    p.add_argument("--pool-join-every", type=int, default=2,
                   help="(pool/swarm) join channel participates only every Nth nomination round "
                        "(grad-every semantics; join's high accept rate lets it dominate the "
                        "measurement queue and over-extend line continuations)")
    p.add_argument("--swarm-epilogue-frac", type=float, default=0.0,
                   help="(swarm) KNN-SMOOTH EPILOGUE: for the last fraction of the run (after "
                        "arbitration has reduced cells to one dominant particle), reheat tau_z and "
                        "z-noise to the epilogue values and cosine-anneal both to their ends -- a "
                        "mini knn-smooth refinement pass per cell under full gradient legibility. "
                        "0 = off")
    p.add_argument("--swarm-epilogue-cycles", action=argparse.BooleanOptionalAction, default=True,
                   help="(swarm) run the knn-smooth epilogue in the tail of EVERY consolidation "
                        "cycle (last --swarm-epilogue-frac of each tau_W segment), not just the "
                        "run end: each cycle's elected winners get a within-family refinement "
                        "pass before the purge/reheat. The bump decays back onto the global "
                        "schedules by the seam (no whiplash)")
    p.add_argument("--swarm-epilogue-temp", type=float, default=0.5,
                   help="(swarm) epilogue tau_z reheat peak (anneals to --knn-temp-end)")
    p.add_argument("--swarm-epilogue-noise", type=float, default=0.4,
                   help="(swarm) epilogue z-noise reheat peak (anneals to 0)")
    p.add_argument("--swarm-epilogue-skip-final", action=argparse.BooleanOptionalAction, default=True,
                   help="(swarm) skip the RUN-END epilogue window AND the last cycle's tail "
                        "(both fire after nomination/probe policing ends -- measured to "
                        "inject fringe confetti / scramble winners under adapters). Earlier "
                        "cycle tails keep running: their purge+reheat heals any churn")
    p.add_argument("--swarm-epilogue-cycles-max", type=int, default=2,
                   help="(swarm) only the first N cycles get epilogue tails (0 = all). "
                        "E.g. 2 with 3 cycles: tails after cycles 1-2, none in the endgame")
    p.add_argument("--swarm-phase-gates", action=argparse.BooleanOptionalAction, default=True,
                   help="(swarm) CYCLE-PHASE DISCIPLINE: within each tau_W cycle, the soft phase "
                        "explores (nominations + W-noise on, consolidator off) and the hardening "
                        "tail crystallizes (consolidator ON at --clip-consist-weight, nomination "
                        "rounds PAUSED, W-noise silenced). The final post-window hold counts as "
                        "hardening. Overrides --clip-consist-start-frac")
    p.add_argument("--swarm-phase-frac", type=float, default=0.75,
                   help="(swarm) segment progress at which a cycle's hardening phase begins")
    p.add_argument("--swarm-purge-keep", type=int, default=2,
                   help="(swarm) elites kept per cell at each purge (top-N by W)")
    p.add_argument("--swarm-purge-reseed", type=int, default=1,
                   help="(swarm) wildcards re-seeded per cell at each purge (random allowed "
                        "glyphs; the wildcard slot's value is as a fresh mobile vehicle -- the "
                        "random seed itself almost never wins directly)")
    p.add_argument("--swarm-empty-pin", type=float, default=0.6,
                   help="(swarm) W floor (capped units) re-asserted each step on the space "
                        "slot of confidently-empty cells, whose z/W noise is also fully "
                        "zeroed (otherwise AdamW weight decay erodes the seeded lead and "
                        "noise walks empty cells into confetti). Not a freeze: gradients "
                        "still flow, a challenger can out-earn the floor. 0 = off; max "
                        "sustainable lead at K=3 is ~0.71")
    p.add_argument("--swarm-boost-exclude", default="join",
                   help="comma-separated channels whose proposals may only take the measured "
                        "probe path, never the free boost path (join reinforces the neighbor "
                        "consensus, a grid-bias risk). '' = all channels may boost")
    p.add_argument("--swarm-boost-ink", type=float, default=0.04,
                   help="(swarm) boosts/nudges only fire in cells with target ink >= this (0=off). "
                        "Boosts are FREE (unmeasured) -- in near-white/shaded-background cells the "
                        "join/blend echo chamber can cascade phantom structure that end-state "
                        "CLIP disowns")
    p.add_argument("--swarm-draw-guard", type=float, default=0.0,
                   help="(swarm) cells whose gradient-EMA draw-push mean(-g_ema) exceeds this are "
                        "exempt from blank-closure (CLIP is consistently asking for ink there). "
                        "0 = log-only: wp_hist lands in --pool-record so the scale can be read "
                        "off a real run before choosing a threshold")
    p.add_argument("--pool-blend-allow-letters", action="store_true",
                   help="(swarm 'blend') let the blend channel propose Unicode letters (excluded "
                        "by default: accent dots pixel-match blur specks, and accented letters "
                        "chain into false structure)")
    p.add_argument("--pool-nominate-file", default=None,
                   help="(pool) npz of externally certified proposals (per-cell before/after "
                        "snap indices + chars) -- one glyph per cell, proposed by the 'file' "
                        "channel until adopted or tabu'd; the audition adjudicates")
    p.add_argument("--pool-channels", default="latent,blend,affinity,join,pixel,ports",
                   help="(pool) nomination proposal channels, comma-list in PRIORITY order (first "
                        "passing gate wins the cell's audition slot): ports = tile-continuity match "
                        "(availability of line families), file = probe-certified proposals (needs "
                        "--pool-nominate-file), affinity/neighbor = context copying, latent = codebook "
                        "family adjacency, grad = blind linear reach (retired from the default: "
                        "near-zero conversion). Channels only PROPOSE; the audition adjudicates")
    p.add_argument("--pool-nominate-density-band", type=float, default=0.25,
                   help="(pool) nominees must have glyph ink density within this of the cell's TARGET "
                        "ink (fraction of cell area). Counters the linear score's overshoot -- it "
                        "credits direction not destination, so 'wants darker' nominates solid "
                        "blocks. 0 = off")
    p.add_argument("--pool-neighbor-margin", type=float, default=0.0,
                   help="(pool) neighbor-channel score edge over the best member (sigma units); low, "
                        "because the per-cell gradient can't see the coordination benefit")
    p.add_argument("--pool-affinity-margin", type=float, default=-0.25,
                   help="(pool) affinity-channel score edge over the best member (sigma units)")
    p.add_argument("--pool-empty-block", type=float, default=0.5,
                   help="(pool) cells with emptiness confidence above this are CLOSED to search: "
                        "no nominations, exempt from --pool-logit-cap. The empty-cell stack owns "
                        "them by design; ungated, the grad channel injects thin glyphs into "
                        "settled white and destabilizes CLIP's reference. >1 = off")
    p.add_argument("--pool-nominate-start-frac", type=float, default=0.05,
                   help="(pool) fraction of training before nomination starts (let the warm start settle)")
    p.add_argument("--pool-nominate-end-frac", type=float, default=0.85,
                   help="(pool) fraction of training when nomination STOPS -- must end before the sharp "
                        "phase (softmax saturation makes arrivals gradient-dead at low temp)")
    p.add_argument("--pool-nominate-margin", type=float, default=1.0,
                   help="(pool) required score edge of the nominee over the pool's BEST member, in units "
                        "of the per-cell member-score STD (dimensionless; nominee must beat everything "
                        "currently rendered, not just the weakest slot -- that would churn the pool)")
    p.add_argument("--pool-evict-below", type=float, default=1.0,
                   help="(pool) a slot is evictable only if its weight EMA < this/K (below a uniform "
                        "share = abandoned by the optimizer); higher = more evictable")
    p.add_argument("--pool-grace", type=int, default=50,
                   help="(pool) iters an arrival is eviction-protected (one fair audition cycle)")
    p.add_argument("--pool-tabu", type=int, default=500,
                   help="(pool) iters an evicted glyph is barred from re-nomination in that cell "
                        "(the tangent that nominated it still likes it; the bar prevents thrash)")
    p.add_argument("--pool-arrival-rank-lo", type=int, default=2,
                   help="(pool) arrival logit inits to a random current logit of rank >= this (mid-pack)")
    p.add_argument("--pool-arrival-rank-hi", type=int, default=5, help="(pool) ...and rank < this")
    p.add_argument("--pool-dedup-eps", type=float, default=0.5,
                   help="(pool) min latent distance of a nominee from every pool member, as a fraction "
                        "of the median codebook NN spacing (don't spend a slot on 0 when O is in)")
    p.add_argument("--pool-grad-ema-window", type=int, default=50,
                   help="(pool) EMA window (steps) for the render-gradient nomination signal")
    p.add_argument("--pool-w-ema", type=float, default=0.95,
                   help="(pool) EMA decay on the clean blend weights (the eviction evidence)")
    p.add_argument("--pool-probe", action=argparse.BooleanOptionalAction, default=True,
                   help="(pool) LIVE PROBING: each nomination round, hard-swap a spaced batch of the "
                        "round's proposals onto the current snap render, measure them with two dense "
                        "CLIP forwards, APPLY only measured improvements and TABU measured failures. "
                        "~2 extra forwards per round; needs an RN conv backbone. Unprobed proposals "
                        "(spacing overflow) simply wait for a later round")
    p.add_argument("--pool-probe-margin", type=float, default=2e-5,
                   help="(pool) accept a probed swap only if its measured loss delta < -this "
                        "(floor above MPS conv nondeterminism noise)")
    p.add_argument("--pool-probe-spacing", type=int, default=9,
                   help="(pool) min chebyshev distance (cells) between simultaneously probed swaps "
                        "(measured: ~91%% of a swap's feature ripple lies within 5 cells)")
    p.add_argument("--pool-probe-window", type=int, default=5,
                   help="(pool) attribution window half-width (cells) for the probe delta")
    p.add_argument("--pool-probe-per-chan", type=int, default=2,
                   help="(pool) runner-up nominations per channel for the probe: 2 = each "
                        "score-matrix channel (grad/blend/join/pixel/latent/ports) also "
                        "pitches its second-best, queued AFTER the full first pass over all "
                        "channels (raise --pool-probe-batches to actually measure them)")
    p.add_argument("--pool-probe-rate", type=int, default=1,
                   help="(pool) probe cadence in ITERS (0 = probe only on nomination rounds). "
                        "1 = probe every iteration: with memoization most rounds only measure "
                        "what's new; merges/boosts/blank-close keep the nominate-rate cadence")
    p.add_argument("--pool-probe-memo-ttl", type=int, default=200,
                   help="(pool) memoized (cell,glyph) probe scores stay valid this many iters "
                        "(image drift bound); cached candidates skip re-measurement and their "
                        "scores join each round's comparison; cache cleared for admitted cells. "
                        "0 = no memoization")
    p.add_argument("--pool-probe-batches", type=int, default=4,
                   help="(pool) disjoint spaced batches measured per round (B+1 dense forwards for "
                        "~B*15 measurements; raises probe throughput when proposals outnumber slots)")
    p.add_argument("--pool-record", default=None,
                   help="(pool) npz path: swap log (it, cell, in, out) + interval-sampled snap/entropy "
                        "trajectories + final pool composition (churn diagnostics)")
    p.add_argument("--pool-record-interval", type=int, default=25,
                   help="(pool) iters between snap/entropy history samples for --pool-record")

    # Swarm mode: K traveling slots per cell (learned latent position z_k + learned weight logit
    # W_k), two-level render v=softmax(W/tau_W) over per-slot knn-smooth blends. See unicasso/
    # engine/swarm.py. REUSES: --knn-temp* (per-slot blend temp tau_z), --z-noise/-commit/-adapt (per-slot
    # exploration), and the whole --pool-* nomination stack (channels/margins/probe/grace/tabu/
    # evict-below/density-band/empty-block/record/grad-ema/w-ema/diversity).
    p.add_argument("--swarm-k", type=int, default=3, help="(swarm) slots per cell")
    p.add_argument("--swarm-knn-k", type=int, default=4,
                   help="(swarm) per-slot blend support (nearest codebook glyphs); small by design: "
                        "measured, a settled cell renders 1-2 and an undecided one 4-6 -- the "
                        "across-slot mixture carries the rest of knn-smooth's early softness")
    p.add_argument("--swarm-w-temp", type=float, default=1.0,
                   help="(swarm) across-slot weight temperature tau_W start")
    p.add_argument("--swarm-w-temp-end", type=float, default=0.0, help="(swarm) tau_W end value")
    p.add_argument("--swarm-w-temp-schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--swarm-w-temp-cycles", type=int, default=2,
                   help="(swarm) PHASED HARDENING: split the tau_W anneal into this many "
                        "harden->soften cycles (cosine with decaying reheat peaks, SGDR-style): "
                        "each cycle elects, then re-softens so losers whose neighbors have since "
                        "moved can re-arbitrate; the last cycle lands at --swarm-w-temp-end. "
                        "0/1 = single monotone anneal")
    p.add_argument("--swarm-w-temp-cycle-decay", type=float, default=0.66,
                   help="(swarm) reheat peak decay per cycle: peak_i = end + (start-end)*decay^i")
    p.add_argument("--swarm-w-temp-cycle-shape", choices=["geo", "lingeo"], default="geo",
                   help="(swarm) reheat peak sequence: geo = pure geometric decay; lingeo = "
                        "max(linear descent, geometric) -- linear through the early/middle "
                        "generations, geometric tail for the last ones")
    p.add_argument("--swarm-w-temp-cycle-lin-end", type=float, default=0.0,
                   help="(swarm, lingeo) the linear peak component descends from the start temp "
                        "to THIS temp at the final cycle (geometric takes over wherever it is "
                        "higher). 0 = descend toward the floor (original lingeo)")
    p.add_argument("--swarm-w-temp-cycle-lin-n", type=int, default=0,
                   help="(swarm, lingeo) number of cycles in the linear ramp (start temp -> "
                        "lin-end); the remaining cycles decay GEOMETRICALLY from lin-end. "
                        "0 = the max(linear, geometric) blend over all cycles")
    p.add_argument("--swarm-blank-close", type=int, default=10,
                   help="(swarm) close a cell to search after its snap has been a blank glyph for "
                        "this many consecutive nomination rounds (blank-destined cells otherwise "
                        "soak up a large share of the probe budget). 0 = off")
    p.add_argument("--swarm-blank-close-ink", type=float, default=0.04,
                   help="(swarm) cells with target ink >= this are EXEMPT from blank-closure: a "
                        "contested faint-content cell idling blank still needs its search support "
                        "(closure can otherwise starve faint-but-real content into space-vs-ink "
                        "coin flips)")
    p.add_argument("--swarm-init-single", action="store_true",
                   help="(swarm) start with ONLY slot 0 live (the warm start); other slots begin "
                        "FREE = pure birth budget, membership earned through measured births "
                        "(also kills the init-duplicate merge wave)")
    p.add_argument("--swarm-w-temp-end-frac", type=float, default=0.9,
                   help="(swarm) tau_W reaches its end BY this fraction -- EARLIER than tau_z by "
                        "default: the expensive cross-family mixture discretizes while the winning "
                        "slot can still slide continuously to compensate; the cheap within-family "
                        "blend (look-alikes) commits last. Two small commitment installments "
                        "instead of one large tau-collapse shock")
    p.add_argument("--swarm-w-cap", type=float, default=0.5,
                   help="(swarm) per-cell W-logit spread cap over LIVE slots (0=off): every slot "
                        "stays render-visible and gradient-alive (grad on both W_k and z_k scales "
                        "with v_k -- an invisible slot can neither argue nor navigate)")
    p.add_argument("--swarm-w-decay", type=float, default=0.02,
                   help="(swarm) AdamW weight decay on W (pull toward uniform arbitration). "
                        "Decay = the W race's MEMORY HORIZON (~1/(lr*wd) steps): high = leads "
                        "must be re-earned from current evidence (challenger-friendly), low = "
                        "incumbency persists")
    p.add_argument("--swarm-w-decay-end", type=float, default=None,
                   help="(swarm) schedule the decay toward this value: high early (challenger "
                        "churn through the cycles) -> low late, so leads earned near hardening "
                        "STICK instead of being flattened into last-moment drift lotteries. "
                        "Try 0.002")
    p.add_argument("--swarm-w-decay-schedule", default="cosine")
    p.add_argument("--swarm-w-decay-end-frac", type=float, default=0.9,
                   help="reach the end decay by this fraction (default 0.9 = at the final anneal)")
    p.add_argument("--swarm-w-noise", type=float, default=0.1,
                   help="(swarm) Gumbel audition noise on W (logit units, annealed on the "
                        "--pool-noise schedule); 0 puts the whole exploration budget in per-slot "
                        "z-noise and probe-measured births instead")
    p.add_argument("--swarm-temp-adapt", choices=["off", "nn1"], default="nn1",
                   help="(swarm) scale each slot's blend temp by its glyph's local codebook density "
                        "(nn1 wall distance / median, clamped by --z-noise-adapt-clip): slots in "
                        "sparse regions blend wider, dense regions sharper")
    p.add_argument("--swarm-boost-radius", type=float, default=0.75,
                   help="(swarm) nominee within this (x median codebook NN spacing) of a live slot "
                        "-> dedup-by-boost instead of a birth: bump that slot's W + nudge its z "
                        "toward the nominee (free, unmeasured)")
    p.add_argument("--swarm-boost", type=float, default=0.15,
                   help="(swarm) W bump per boost (logit units; cap re-projects next step)")
    p.add_argument("--swarm-boost-nudge", type=float, default=0.25,
                   help="(swarm) fraction of the way the boosted slot's z steps toward the nominee")
    p.add_argument("--swarm-boost-cooldown", type=int, default=50,
                   help="(swarm) min iters between boosts of the same slot (anti-runaway)")
    p.add_argument("--swarm-merge-eps", type=float, default=0.5,
                   help="(swarm) live slots closer than this (x median NN spacing) merge: W folds "
                        "via logsumexp (earned mass kept, no snap vote-splitting), loser parked as "
                        "FREE respawn budget for births")
    p.add_argument("--swarm-strong-gap", type=float, default=0.1,
                   help="(swarm) strong (measured clearly-better) birth lands this far below the "
                        "live W max -- an immediate contender")
    p.add_argument("--swarm-lottery-margin", type=float, default=2e-5,
                   help="(swarm) GRADED admission: probe delta < -pool-probe-margin -> strong "
                        "birth; delta < +this -> minority-weight lottery birth (ahead-of-its-time "
                        "hypotheses exist and wait instead of being tabu'd); else reject+tabu. "
                        "0 = strict pool-style gate (no lottery)")
    p.add_argument("--swarm-final-elect", type=int, default=0,
                   help="(swarm) after training, run this many measured RUNNER-UP ELECTION sweeps: "
                        "every open cell whose 2nd-best live slot snaps to a different glyph gets "
                        "that finalist hard-probed IN THE FINAL CONTEXT; accepted swaps flip the "
                        "winner. Restricted to the cell's own learned slots (not an open-codebook "
                        "polish -- that was measured to hack the objective); multiple sweeps let "
                        "flipped neighbors re-judge. 0 = off")
    p.add_argument("--swarm-affinity-winner-frac", type=float, default=1.0,
                   help="(swarm) after this fraction of training, the affinity pull retargets from "
                        "the cell blend embedding to the ARGMAX slot's z directly (knn-smooth "
                        "semantics restored once arbitration has decided); 1.0 = never")
    # Empty-space handling: target-empty BACKGROUND cells (LOCALLY gated so interior negative space is
    # spared) get a sharpened per-cell temp (#2) so they CAN reach pure space, plus an ink penalty (#1)
    # so they DO. Both driven by the SAME annealed gate -> no high-temp loss fight. Default = off.
    p.add_argument("--empty-weight", type=float, default=0.18,
                   help="#1: penalty on rendered ink in confidently-empty cells (cleans the white bg CLIP reads as 'messy'); 0 = off")
    p.add_argument("--empty-temp-scale", type=float, default=0.3,
                   help="#2: min knn-temp factor for fully-empty cells (e.g. 0.1 sharpens toward pure space); 1.0 = off")
    p.add_argument("--empty-noise-scale", type=float, default=0.3,
                   help="per-cell z-noise multiplier for confident-empty cells (e.g. 0.1): quiets their exploration so they settle to space instead of being kicked back out; 1.0 = off")
    p.add_argument("--empty-snap-init", action="store_true",
                   help="bypass the encoder on confidently-empty cells and SEED their latent directly at the blank/space glyph (for images where the VAE hallucinates ink from near-white patches). Uses the same emptiness gate as #1/#2")
    p.add_argument("--empty-snap-thresh", type=float, default=0.5,
                   help="emptiness-gate cutoff (s_c in [0,1]) at/above which a cell is snap-initialized to the blank glyph (--empty-snap-init)")
    p.add_argument("--empty-thresh", type=float, default=0.95,
                   help="cell mean whiteness (white=1) above which it counts as (graded) empty")
    p.add_argument("--empty-window", type=int, default=5,
                   help="KxK neighborhood for the local emptiness-agreement gate (stray white in dense ink is not 'safe')")
    p.add_argument("--empty-gamma", type=float, default=2.0,
                   help="sharpening exponent on neighborhood emptiness agreement (higher = require a stronger empty surround)")
    p.add_argument("--empty-anneal", type=float, default=1.0,
                   help="emptiness enforcement strength at the START (gates both #1 and #2); ramps -> --empty-anneal-end")
    p.add_argument("--empty-anneal-end", type=float, default=1.0, help="emptiness enforcement strength at the END")
    p.add_argument("--empty-anneal-schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--empty-anneal-end-frac", type=float, default=0.7,
                   help="fraction of training by which emptiness enforcement reaches full, then holds")
    p.add_argument("--empty-clean-target", action=argparse.BooleanOptionalAction, default=True,
                   help="whiten the TARGET (feathered) in confident-empty regions so CLIP/recon don't chase background artifacts that corrupt the crop-level signal")
    p.add_argument("--empty-clean-sigma", type=float, default=4.0,
                   help="gaussian feather (px) for --empty-clean-target; softens cell-boundary edges so whitening doesn't create new artifacts")
    p.add_argument("--clip-content-crop", action=argparse.BooleanOptionalAction, default=True,
                   help="restrict CLIP to the content bounding box (cell-aligned, from non-empty cells) so crops aren't wasted on empty margin and the global semantic isn't diluted by whitespace")
    p.add_argument("--clip-content-pad", type=int, default=2,
                   help="cells of context kept around the content bbox for --clip-content-crop")
    # Non-local semantic affinity: Laplacian latent pull where edges = each cell's top-k
    # CORRESPONDING cells anywhere (CNN visual sim GATED by DINO semantics), not a spatial window.
    # Constrains the solution space so semantically-similar regions adopt consistent glyphs (the
    # regularization alternative to candidate injection). See unicasso/engine/affinity.py.
    p.add_argument("--affinity-weight", type=float, default=0.0, help="weight on the non-local semantic-affinity latent pull (the START value if --affinity-weight-end is set)")
    p.add_argument("--affinity-weight-end", type=float, default=5e-3, help="ramp the affinity weight to this by --affinity-weight-end-frac (set start≈0 so early exploration isn't frozen, end=full)")
    p.add_argument("--affinity-weight-schedule", choices=["linear", "cosine"], default="cosine", help="affinity-weight anneal shape (cosine = the 1-cos ramp)")
    p.add_argument("--affinity-weight-start-frac", type=float, default=0.4, help="hold the affinity weight at its START value until this fraction of training, THEN ramp up (delays the pull so early exploration runs unconstrained)")
    p.add_argument("--affinity-weight-end-frac", type=float, default=1.0, help="fraction of training by which the affinity weight reaches its end value (then holds)")
    p.add_argument("--affinity-layers", default="1,2,3", help="RN101 conv layers for the CNN visual ingredient (1 fine..4 coarse)")
    p.add_argument("--affinity-beta", type=float, default=2.0, help="DINO gate sharpness: sim = stretch(CNN * DINO^beta) (0=pure CNN, higher=stricter semantic gate)")
    p.add_argument("--affinity-topk", type=int, default=2, help="edges per cell = its k most-similar cells (keep small to avoid over-smoothing)")
    p.add_argument("--affinity-gamma", type=float, default=1.0, help="extra sharpening on the kept edge weights (sim^gamma)")
    p.add_argument("--affinity-min-ink", type=float, default=0.02, help="drop edges touching cells with ink fraction below this (don't tie blank regions)")
    p.add_argument("--affinity-max-side", type=int, default=448, help="long-side px the image runs through the CLIP conv trunk at")
    p.add_argument("--affinity-feat-image", default=None, help="run CLIP/DINO on THIS image instead of the line art (e.g. the original photo); resized to the grid")
    # Pixel-grounded latent nudge: pull z toward the embedding weighted by each
    # char's PIXEL fit, over the top-k pixel-fitters (global signal -> can escape local basins).
    # Implemented as a soft-attraction loss so Adam balances its magnitude.
    p.add_argument("--pixel-nudge-weight", type=float, default=0.0, help="weight for the pixel-fit latent nudge (knn-smooth)")
    p.add_argument("--pixel-nudge-k", type=int, default=10, help="top-k pixel-fitting chars to pull toward (N = all)")
    p.add_argument("--pixel-nudge-temp", type=float, default=0.05, help="softmax temp over per-char pixel MSE (start)")
    p.add_argument("--pixel-nudge-temp-end", type=float, default=None,
                   help="if set, anneal pixel-nudge-temp -> this (soft->sharp)")
    p.add_argument("--pixel-nudge-weight-end", type=float, default=None,
                   help="if set, anneal pixel-nudge-weight -> this (fade the nudge as things settle so it stops fighting the low-noise/low-temp endgame; set 0 + cosine to track z-noise)")
    p.add_argument("--pixel-nudge-weight-schedule", choices=["linear", "cosine"], default="cosine",
                   help="annealing curve for pixel-nudge-weight (cosine holds it higher early, like z-noise)")
    p.add_argument("--pixel-nudge-weight-end-frac", type=float, default=1.0,
                   help="fraction of training by which pixel-nudge-weight reaches its end value, then holds")
    # Orientation field: per-cell structure-tensor orientation match (render vs target), gated
    # by target coherence. Steers contour cells toward orientation-matching glyphs; abstains on
    # flat/shading cells (let recon/density pick). sigma is the structure-tensor window in px.
    p.add_argument("--orient-weight", type=float, default=1e-2, help="weight for the orientation-match term")
    # Coordinate loss: the VAE's LEARNED position axes (sym_U, --sym-weight training) make
    # sub-cell position a linear readout of z; anchor it to the target's measured centroid.
    # SCAFFOLDING like the recon ramp: strong early to structure placement, anneal low so
    # CLIP's position tolerance (needed for global arrangement) takes over. Mode-agnostic.
    p.add_argument("--coord-weight", type=float, default=0.1,
                   help="weight on the latent-unit position error (readout vs target centroid, "
                        "normal- and reliability-gated, mean over gated cells) -- commit-scale "
                        "semantics; needs a --sym-weight-trained ckpt. Try 0.1 -> 0.01")
    p.add_argument("--coord-weight-end", type=float, default=None)
    p.add_argument("--coord-weight-schedule", default="cosine")
    p.add_argument("--coord-weight-end-frac", type=float, default=1.0)
    p.add_argument("--orient-sigma", type=float, default=4.0, help="structure-tensor window (px)")
    # CLIPasso-style perceptual loss: match the render's CLIP embedding to the target's.
    # Lazy-loads CLIP RN101 only when weight > 0. Slow (CLIP fwd per iter) -> lower --iters.
    p.add_argument("--clip-weight", type=float, default=1.2, help="weight for the CLIPasso perceptual loss")
    p.add_argument("--clip-aug", type=int, default=16, help="random-resized-crop augmentations per step")
    p.add_argument("--clip-crop-scale", type=float, nargs=2, default=[0.4, 0.9], metavar=("MIN", "MAX"),
                   help="area fraction range for the CLIP random-resized crops (CLIPasso default 0.7 1.0; "
                        "lower MIN = more aggressive zoom-in -> more local detail + robustness, weaker global semantics)")
    p.add_argument("--clip-scale-alpha", type=float, default=0.8,
                   help="crops sampled log-uniform (small oversampled), weighted s^alpha. 1=unbiased uniform-across-scales (var-reduced); <1=small-scale emphasis; 0=raw small-scale dominance")
    p.add_argument("--clip-aspect-jitter", type=float, nargs=2, default=[0.75, 1.3333],
                   help="CLIP crops: SQUARE base + aspect-ratio jitter in [lo,hi] (CLIPasso default 3:4-4:3), stretched to res^2; caps the resize stretch so crops stay in-distribution. Use 1.0 1.0 for pure squares")
    p.add_argument("--clip-edge-frac", type=float, default=0.0, help="fraction of CLIP crops that place their top-left from Beta(a,a) (U-shaped) to over-cover the EDGES, counteracting uniform placement's center bias (0=off; the bias matters more with --clip-content-crop since edges = subject extremities)")
    p.add_argument("--clip-edge-beta", type=float, default=0.5, help="Beta(a,a) shape for edge crops: 1=uniform(off), 0.5=arcsine(gentle U), ->0 flush-corner (harder edge push)")
    p.add_argument("--clip-edge-auto", action="store_true", help="auto-solve --clip-edge-frac to flatten center/edge coverage (f*=-Cov(u,d)/Var(d) via one-time MC at first iter); overrides --clip-edge-frac")
    p.add_argument("--clip-microbatch", type=int, default=0,
                   help="encode CLIP crops in chunks of N with per-chunk gradient accumulation "
                        "into a detached render leaf (exact gradients via surrogate); caps peak "
                        "encoder-activation memory at N crop triples instead of --clip-aug. "
                        "0 = off. Try 2-4 on MPS / low-VRAM GPUs.")
    p.add_argument("--clip-batch-aug", action="store_true",
                   help="encode ALL clip-aug crops in ONE forward instead of n_aug small "
                        "ones. Big win on CUDA (kernel-launch bound); memory scales with "
                        "--clip-aug, so leave off on MPS unless headroom is known")
    p.add_argument("--clip-adapter", default=None, metavar="CKPT",
                   help="domain-adaptation adapters from unicasso.adapter.clip_adapt (adapters_best.pt): "
                        "render/snap passes run the ascii-ADAPTED tower, target passes the frozen "
                        "base -- the dual-path wiring the adapters were trained under. RN models only")
    p.add_argument("--clip-steer", default=None, metavar="VEC",
                   help="steering vector from unicasso.engine.steer (text.pt / data.pt): a fixed "
                        "unit direction added to the normalized TARGET embedding before the "
                        "semantic cosine, so the render is asked for 'the target, but "
                        "ascii-flavoured'. Acts on the fc/semantic term only (the one term in a "
                        "space text and image share) -- pair with --clip-steer-weight. RN models only")
    p.add_argument("--clip-steer-weight", type=float, default=0.5, metavar="LAMBDA",
                   help="lambda in normalize(normalize(e_target) + lambda*delta). The cosine "
                        "renormalizes, so the loss stays bounded and only its DIRECTION rotates: "
                        "lambda=1 puts the steered target halfway between the true target and the "
                        "pure steering direction. No effect without --clip-steer")
    p.add_argument("--clip-semantic-weight", type=float, default=0.1, help="weight on the CLIPasso SEMANTIC term (1 - cosine of the final CLS/fc embeddings); 0.1 = CLIPasso default; the geometric conv-layer L2 has weight 1")
    p.add_argument("--clip-model", default="RN101", help="open_clip model name (RN101=CLIPasso conv; ViT-B-16=CLIPascene-style global-attention ViT; convnext_base_w=LAION conv)")
    p.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag; RN101/ViT-* use 'openai', convnext_base_w needs a LAION tag e.g. 'laion2b_s13b_b82k'")
    p.add_argument("--clip-vit-layers", default="7", help="ViT geometric term: transformer resblock(s) to L2, as 'idx' or 'idx:weight' comma-list (e.g. '7' or '4:1,7:1,11:0.5'); shallow=geometry, deep=semantics (ViT models only)")
    p.add_argument("--clip-vit-drop-cls", action="store_true", help="ViT geometric term: exclude the CLS token from the token L2 -> purely-spatial patch match (default keeps CLS, matching CLIPasso)")
    p.add_argument("--clip-reg-frac", type=float, default=0.0,
                   help="CROP REGISTRATION: before each CLIP comparison, nudge the TARGET crop by the "
                        "small shift (+/- this fraction of a cell) that best aligns its ink field with "
                        "the render crop's (NCC on half-res maps, no grad through the search). "
                        "Per-crop grid-phase tolerance: CLIP forgives placement error within the "
                        "bound, punishes beyond it. 0.5 = half a cell; 0 = off")
    p.add_argument("--clip-fp16", action="store_true", help="run the (frozen) CLIP visual encoder in half precision (~2x faster on MPS; the render gradient still flows). Helps most for the heavy ViT models")
    # Diagnostic snapshots: where/why the perceptual loss struggles (needs --row-gap 0).
    p.add_argument("--clip-diagnostic", default=None, help="path prefix -> save 4-panel diagnostic PNGs (soft render | clip<->recon conflict | clip confusion | snap residual)")
    p.add_argument("--clip-diag-interval", type=int, default=200, help="iters between diagnostic snapshots")
    p.add_argument("--clip-diag-ema-window", type=int, default=20, help="EMA window (steps) for the CLIP<->recon conflict / push maps (training dynamics, not per-aug)")
    # Opt-in CLIPasso-style warp augmentation (rotation + shear) on TOP of the known-good crop.
    # Default 0 = crop only (the proven path). MPS-safe manual bilinear warp (grid_sample backward
    # is unimplemented on MPS). clip backbone only.
    p.add_argument("--clip-rotate", type=float, default=6.0, help="(clip) max rotation (deg) in the warp aug; 0 = crop only")
    p.add_argument("--clip-shear", type=float, default=5.0, help="(clip) max shear (deg) in the warp aug; 0 = crop only")
    p.add_argument("--clip-invert-frac", type=float, default=0.0,
                   help="(clip) fraction of crop pairs judged in INVERTED polarity (render AND "
                        "target flipped pre-normalization): a decorrelated feature view of the same "
                        "matching problem -- light-on-dark monospace text is abundant in CLIP's "
                        "training data")
    # Perceptual backbone: CLIP (CLIPasso) vs DINOv3 self-similarity structure loss. Both gated by
    # --clip-weight and share --clip-aug / --clip-crop-scale (the augmentation knobs).
    p.add_argument("--perceptual", choices=["clip", "dino"], default="clip",
                   help="perceptual loss backbone: clip=CLIPasso (RN101); dino=DINOv3 self-similarity")
    p.add_argument("--dino-model", default="vit_base_patch16_dinov3.lvd1689m",
                   help="timm DINOv3 model. convnext_small.dinov3_lvd1689m = distilled CNN, much "
                        "faster (run at --dino-res 448 for ViT-like granularity)")
    p.add_argument("--dino-res", type=int, default=224,
                   help="DINO input resolution (ConvNeXt: bump to 448 for a finer feature map)")
    p.add_argument("--dino-struct-weight", type=float, default=1.0,
                   help="weight on the DINO patch self-similarity (structure) term")
    p.add_argument("--dino-global-weight", type=float, default=0.1,
                   help="weight on the DINO CLS cosine (global layout) term")
    # Latent exploration noise: add annealed isotropic noise to z BEFORE snapping/blending so the
    # optimizer explores different codebook neighbors early, then commits as it anneals to 0. Scale
    # to codebook NEIGHBOR spacing (>> the VAE's training latent-noise) to actually hop glyphs.
    # Complements --knn-temp annealing (that softens the blend; this moves where z sits).
    p.add_argument("--z-noise", type=float, default=0.7,
                   help="(knn-smooth) start std of latent exploration noise added to z")
    p.add_argument("--z-noise-end", type=float, default=0.2,
                   help="anneal z-noise -> this over iters (0 = fully commit by the end)")
    p.add_argument("--z-noise-schedule", choices=["linear", "cosine"], default="cosine",
                   help="annealing curve for z-noise (cosine holds the noise higher early)")
    p.add_argument("--z-noise-commit", type=float, default=0.05,
                   help="SGLD: after each opt step, PERSIST this fraction of the (random, pre-drift) exploration noise into z itself so z actually random-walks (crosses barriers) instead of only jittering the eval point. 0 = today's eval-only jitter. ~0.05-0.15 to start; anneals with z-noise -> 0")
    p.add_argument("--z-noise-adapt", choices=["off", "nn1", "clusterk"], default="nn1",
                   help="density-EQUALIZED per-cell z-noise: scale each cell's sigma by its local codebook wall distance so the ESCAPE probability (in sigmas) is constant everywhere -- dense basins calmed (stop mush), sparse regions energized (stop freezing). nn1 = nearest-neighbor wall (flip-to-adjacent); clusterk = knn-k-th neighbor radius (escape the whole local cluster). off = today's global sigma")
    p.add_argument("--z-noise-adapt-clip", type=float, default=3.0,
                   help="clamp the per-cell adaptive multiplier to [1/clip, clip] so near-duplicate glyphs (wall~0) don't freeze fully and outliers don't explode; the overall noise LEVEL stays set by --z-noise (multiplier is mean-normalized)")
    p.add_argument("--z-noise-quench-frac", type=float, default=0.75,
                   help="after this fraction of training, scale the z-noise linearly to 0 by the end "
                        "(a QUENCH on top of the schedule). Lets --z-noise-end hold a floor through "
                        "the exploration/election phases without leaving the final soft render "
                        "noise-jittered. 1.0 = off")
    p.add_argument("--base-width", "--width", "--cols", type=int, default=60, help="grid width in chars (auto height)")
    p.add_argument("--glyph-budget", type=int, default=0,
                   help="size the grid by TOTAL CELLS instead of by column count: solve the "
                        "width so GW x GH ~= this, at the image's own aspect. Fixed columns "
                        "hold constant the thing that matters least (a display width) and let "
                        "the actual resource swing with aspect -- at 60 columns a 3:4 portrait "
                        "gets 2280 cells and a 16:9 frame gets 960, a 2.4x gap for the same "
                        "label. 0 = off, use --base-width")
    p.add_argument("--budget-max-width", type=int, default=100,
                   help="(--glyph-budget) column ceiling, so a panoramic frame spends its "
                        "budget on rows instead of running off the terminal. Set 80 to keep "
                        "every render inside a classic terminal width")
    p.add_argument("--row-gap", type=int, default=0)
    # Banning is inference-time: drop these glyphs from the codebook so snapping never picks
    # them. No retrain needed (the VAE still embeds all glyphs; we just exclude them).
    p.add_argument("--ban-chars", type=str, default="", help="characters to exclude from snapping")
    p.add_argument("--ban-blocks", action="store_true", help="also ban block chars: ░▒▓█▄▌▐▀■")
    p.add_argument("--ban-letters", action="store_true", help="also ban all Unicode letters: A-Z a-z + accented/diacritic (é à ñ ü ö ç …)")
    p.add_argument("--iters", type=int, default=3500)
    p.add_argument("--schedule-stretch", type=float, default=1.0,
                   help="slow ALL anneal schedules uniformly by this factor (the run still uses --iters); >1 stretches every schedule with shape/margins preserved, so a run that converges early can fill its whole length instead")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--commit", type=float, default=0.1, help="commitment weight")
    p.add_argument("--multiscale-weight", type=float, default=0.5)
    # ---------------- COLOR fork ----------------
    p.add_argument("--color", action="store_true",
                   help="COLOR mode: per-candidate fg/bg leaves, RGB render, color-aware CLIP+recon. "
                        "Nominations still run on the extracted monochrome structure "
                        "(unicasso/engine/color.py)")
    p.add_argument("--color-pin", action="store_true",
                   help="control arm: pin fg=black/bg=white (reproduces the grayscale render exactly), "
                        "so a quality delta can be attributed to the color code vs the lost adapter")
    p.add_argument("--color-jnd", type=float, default=4.0,
                   help="Lab separation below which a cell is flat (no structure to nominate on)")
    p.add_argument("--color-ratio", type=float, default=1.5,
                   help="min separation/within-spread for a cell to count as structure (a smooth blur "
                        "ramp scores ~3.5, so raise this to reject bokeh)")
    p.add_argument("--color-ramp", type=float, default=0.35,
                   help="ink softness as a fraction of cluster separation")
    p.add_argument("--color-lr", type=float, default=None,
                   help="learning rate for the fg/bg leaves (default: --lr)")
    p.add_argument("--color-div-weight", type=float, default=1e-3,
                   help="weight on the COLOR blend-variance term: charges cells whose slots disagree "
                        "on color, since a soft red/blue mix renders purple -- a color no candidate "
                        "can ship after the snap")
    p.add_argument("--color-palette", type=int, default=0,
                   help="quantize fg/bg to N k-means colors (straight-through). 0 = continuous; "
                        "emission snaps to ANSI regardless")
    p.add_argument("--output-ans", default=None, help="(color) write an ANSI-colored .ans")
    # NB the --color-lite-* defaults are what `unicasso.lite` ships for the v2
    # models (blend read, ridge 1.0), so `--color-lite CKPT` alone colours the
    # run exactly as the lite model would; the trainer passes them explicitly.
    p.add_argument("--color-lite", default=None, metavar="CKPT",
                   help="NOTE: the learned per-cell contrast (--color-contrast-learn) "
                        "is NOT applied in this mode; the colours are exactly the "
                        "model's. COLOR FROM THE LITE MODEL: fg/bg come from the distilled "
                        "mask decoder instead of the closed-form MSE fit, so the run "
                        "optimizes toward colors the lite model can actually reproduce "
                        "and the refined grid is color-aware by construction. fg/bg "
                        "stay out of the optimizer (detached lookups, never leaves)")
    p.add_argument("--color-lite-mode", default="birth",
                   choices=["emit", "birth", "forward"],
                   help="how far the lite coloring reaches: 'emit' = probe renders + "
                        "emission only (candidates MEASURED in the model's colors); "
                        "'birth' = + per-slot colors set at init/birth/reseed, so the "
                        "soft render, the probe and emission all agree (default); "
                        "'forward' = + re-queried every forward from each slot's "
                        "current nearest glyph, so colors track slots as they travel")
    p.add_argument("--color-lite-ink", default="render",
                   choices=["render", "center"],
                   help="what goes in the ink channel of the color query: 'render' = "
                        "the whole 5x3 window rendered from the current grid (what "
                        "lite.recolor does and what the v2 models' colour pools were "
                        "coloured with in training, but a cell's colors then depend on its "
                        "neighbours); 'center' = the photo's own ink everywhere with "
                        "only the center cell replaced by the candidate glyph, making "
                        "colors a pure function of (cell, glyph) -- permanently "
                        "cacheable, one cell requeried per probe, but no current "
                        "checkpoint has been trained on it")
    p.add_argument("--color-lite-font", default=None, choices=["dejavu", "sfmono"])
    p.add_argument("--color-lite-path", default="blend",
                   choices=["topk", "sample", "blend"],
                   help="mask read for the color query. 'blend' = every pixel "
                        "weighted by its probability (mask_fit), which is what "
                        "the trainer uses under --read-path blend; keep them "
                        "matched or the pool is refined through a read the "
                        "model was not trained under")
    p.add_argument("--color-lite-frac", type=float, default=0.25)
    p.add_argument("--color-lite-count", default="argmax",
                   choices=["fixed", "argmax"])
    p.add_argument("--color-lite-temp", type=float, default=1.0,
                   help="(--color-lite-path sample) draw temperature")
    p.add_argument("--color-lite-ridge", type=float, default=1.0)
    p.add_argument("--color-lite-chunk", type=int, default=512,
                   help="windows per lite-color forward. The full (n,4,116,96) "
                        "batch is never materialized -- this caps the transient "
                        "allocation (512 ~ 90MB; the whole M*K batch at w60 "
                        "would be 1.3GB, reallocated every step in forward mode)")
    p.add_argument("--color-lite-k-scale", type=float, default=0.0,
                   help="(v1 lite models only; v2 checkpoints have no k head) "
                        "learned per-cell contrast in the lite coloring: "
                        "k = 1 + k_scale*(k_hat - 1). DEFAULT 0 = the k head is "
                        "not used at all -- it was trained against the soft "
                        "mask-weighted mean, is uncorrelated with fit separation "
                        "under the selection read, and can satisfy separation "
                        "constraints by itself. 1.0 applies it as trained; the "
                        "knob exists so its absence can be A/B'd, not assumed")
    p.add_argument("--color-mask", default=None,
                   help="(color, swarm) FROZEN-MASK coloring: npz with per-cell "
                        "fg/bg pixel masks (masks (M,>=2,CH,CW), gh, gw -- e.g. "
                        "lite --dump-masks). fg/bg are computed ONCE as the "
                        "mask-weighted fit on the target cells and held constant: "
                        "colors are glyph-independent, the run arbitrates shape "
                        "only. Overrides --color-fit")
    p.add_argument("--color-mask-mode", default="frozen",
                   choices=["frozen", "init"],
                   help="frozen = mask-fit colors held constant (pure shape "
                        "arbitration). init = colors WARM-STARTED from the "
                        "mask fit then left FREE under CLIP/recon -- the "
                        "engine optimizes colors directly (no softmax in the "
                        "path)")
    p.add_argument("--color-mask-ridge", type=float, default=1.0,
                   help="pixel-mass ridge toward the cell mean for low-mass "
                        "layers in the frozen-mask fit")
    p.add_argument("--color-init", default=None,
                   help="(color) NPZ with per-cell fg/bg (M,3) to seed the free color "
                        "leaves, instead of the engine's own per-slot closed-form re-fit. "
                        "Use it to start a refine at the colors the LITE model actually "
                        "shipped. Overridden by --color-pin / --color-mask")
    p.add_argument("--color-anchor", type=float, default=0.0,
                   help="cosine-ramped L2 pulling the FREE fg/bg leaves toward the closed-form "
                        "distance-weighted MSE fit of the placed glyphs; regularizes CLIP-"
                        "degenerate colors on cells CLIP ignores. This is the END weight (the "
                        "value at --color-anchor-end-frac); >0 also arms the schedule")
    p.add_argument("--color-anchor-start", type=float, default=0.0,
                   help="anchor weight at --color-anchor-start-frac (the START value; default 0). "
                        "The weight cosine-ramps from here to --color-anchor over the fraction "
                        "window, holding flat outside it")
    p.add_argument("--color-anchor-start-frac", type=float, default=0.0,
                   help="run fraction (of the schedule length) where the ramp BEGINS; before "
                        "this the weight is pinned at --color-anchor-start (default 0.0)")
    p.add_argument("--color-anchor-end-frac", type=float, default=1.0,
                   help="run fraction where the ramp REACHES --color-anchor; after this the "
                        "weight is pinned at --color-anchor (default 1.0)")
    p.add_argument("--color-anchor-mc", type=float, default=0.12,
                   help="min-contrast luminance floor applied to the color-anchor target fit")
    p.add_argument("--recolor", action="store_true", default=True,
                   help="(color, DEFAULT ON) post-step: refit fg/bg by closed-form MSE against the "
                        "original at the placed glyphs, and ship that as *_mserefit.png/.ans")
    p.add_argument("--no-recolor", dest="recolor", action="store_false")
    p.add_argument("--recolor-min-contrast", type=float, default=0.12,
                   help="minimum fg/bg luminance gap in the refit (0 = pure MSE, glyphs vanish "
                        "in smooth regions)")
    p.add_argument("--color-fit", action=argparse.BooleanOptionalAction, default=True,
                   help="(color, DEFAULT ON) drive fg/bg by the CLOSED-FORM MSE fit against the "
                        "target every step instead of learning them: colors become a function of "
                        "each slot's own ink, so CLIP sees the colors that actually ship and "
                        "tinting a full block to fake a tone stops working. Also what makes live "
                        "probing possible under --color (a counterfactual swap's colors are "
                        "exactly computable). --no-color-fit reverts to learned (free) colors AND "
                        "silently gives up the measured probe tier")
    p.add_argument("--color-fit-min-contrast", type=float, default=None,
                   help="(--color-fit) in-loop legibility floor (default: --recolor-min-contrast, "
                        "so the optimized render matches the shipped refit)")
    p.add_argument("--color-fit-detach", action="store_true",
                   help="(--color-fit) colors follow the shape but pass NO gradient back through "
                        "the fit -- the ablation for whether the fit's gradient path does work")
    p.add_argument("--color-source", default="blend", choices=["fit", "cluster", "blend"],
                   help="(--color-fit) shipped fg/bg: 'fit' = pure closed-form MSE at the ink, "
                        "'cluster' = the decomposition's own two cluster colors (glyph-independent, "
                        "preserves each cell's bimodal contrast), 'blend' (default) = "
                        "--color-cluster-alpha mix of both -- punchier than the plain fit without "
                        "losing the rendered mean")
    p.add_argument("--color-cluster-alpha", type=float, default=0.5,
                   help="cluster share of the shipped colors under --color-source blend")
    p.add_argument("--color-bg-dist-pow", type=float, default=1.0,
                   help="bg-fit votes weighted by (pixel distance to the glyph's ink)^pow: "
                        "stroke-fringe pixels (antialiasing/misalignment contamination) barely "
                        "count, so the background color stops smearing across strokes. 0 = off "
                        "(plain fit). Glyph-dependent: applied per slot in-loop and at the placed "
                        "glyph at emission")
    p.add_argument("--color-contrast-learn", action=argparse.BooleanOptionalAction, default=True,
                   help="(--color-fit, DEFAULT ON) let CLIP learn a PER-CELL contrast multiplier k on top of "
                        "the fitted colors, over the last --color-contrast-iters steps. k=1 is "
                        "exactly the current colors; the cell's MEAN color is invariant in k, so "
                        "it can only trade pixel error for legibility -- never hue or brightness")
    p.add_argument("--color-contrast-iters", type=int, default=500,
                   help="how many FINAL iterations k is learnable for (0 = the whole run). Gating "
                        "it to the tail means the discrete search is over, so k cannot steer "
                        "nominations -- it only tunes glyphs that are already chosen")
    p.add_argument("--color-contrast-lr", type=float, default=0.02)
    p.add_argument("--color-contrast-tv", type=float, default=0.05,
                   help="total-variation smoothness on the k field (an unconstrained per-cell k "
                        "has previously come out salt-and-pepper; this is the counter)")
    p.add_argument("--color-contrast-max", type=float, default=4.0,
                   help="clamp on k (0 = flat cell, 1 = the fitted colors)")
    p.add_argument("--output-color-png", default=None, help="(color) write the RGB render")
    p.add_argument("--recon-weight", type=float, default=0.25,
                   help="weight on pixel/multiscale recon; set 0 for a pure CLIPasso (CLIP-only) test")
    p.add_argument("--recon-jolt", type=float, default=1.0,
                   help="recon-weight multiplier at the START, decaying to 1 (jolt cells to fit content before consistency/blend dominate; e.g. 3)")
    p.add_argument("--recon-jolt-frac", type=float, default=0.3,
                   help="fraction of training over which the recon jolt decays to 1")
    p.add_argument("--recon-weight-end", type=float, default=0.07,
                   help="if set, anneal recon-weight -> this (independent of recon-jolt, which multiplies on top)")
    p.add_argument("--recon-weight-schedule", choices=["linear", "cosine"], default="cosine",
                   help="annealing curve for recon-weight")
    p.add_argument("--recon-weight-end-frac", type=float, default=0.4,
                   help="fraction of training by which recon-weight reaches its end value, then holds (e.g. 0.7)")
    p.add_argument("--knn-temp-end-frac", type=float, default=1.0,
                   help="fraction of training by which knn-temp reaches its end value, then holds (e.g. 0.8 = sharpen early -> settle time)")
    p.add_argument("--random-init", action="store_true", help="random z init instead of warm start")
    p.add_argument("--context-init", action="store_true",
                   help="warm-start: encode each target cell WITH its real neighbor-pixel margin (matches a multimodal/context-trained VAE); default pads the margin white")
    p.add_argument("--init-text", default=None,
                   help="warm-start from a previous run's --output-text grid instead of the encoder: "
                        "z0 = codebook[glyph] per cell. E.g. seed pool mode with a knn-smooth result "
                        "(the two-stage pipeline: coherent global solution -> local repair). Grid must "
                        "match --base-width/height and the charset (incl. bans)")
    p.add_argument("--init-w-lead", type=float, default=0.0,
                   help="(swarm) extra W handed to slot 0 -- the warm-start hypothesis -- at init, "
                        "before the spread cap projects the gaps. With --swarm-w-cap 0.5 and K=3, "
                        "~1.0 saturates the cap: the strongest expressible lead a rival can still "
                        "contest. Pairs with --init-text to TRUST a previous result (e.g. a lite-"
                        "model grid) instead of merely starting from it")
    p.add_argument("--init-scan", action="store_true",
                   help="warm-start each cell at its ORACLE glyph: dense integer-shift sweep of the "
                        "whole image (+/- --init-scan-range px), ink-preserving views only, blanks "
                        "excluded for ink cells; z0 = codebook[best-locking glyph]. Fixes the "
                        "large share of warm-start misfit that is pure grid phase (choice-side "
                        "only -- the render stays grid-locked)")
    p.add_argument("--init-scan-range", type=int, default=4, help="shift sweep radius (px) for --init-scan")
    p.add_argument("--init-vae-ckpt", default=None,
                   help="separate VAE used ONLY for the warm-start: encode+snap each cell with it (e.g. a pixel-noise-robust VAE), then start z at the MAIN codebook's latent for that glyph. Bridges the two (incompatible) latent spaces via the shared glyph choice. Charset must match the main VAE")
    # Target preprocessing, opt-in. Orthogonal to the mode choice.
    p.add_argument("--blank-weight", type=float, default=0.2,
                   help="RENT on rendered ink in target-unsupported cells (anti-confetti). "
                        "Per-cell (unlike the windowed empty gate): fringe cells pay full "
                        "rent if their own target region is blank. Soft: supported lone "
                        "accents pay one small fee and survive; colonies get evicted. Try 0.1-0.3")
    p.add_argument("--blank-tau", type=float, default=0.03,
                   help="target cell-ink above this = fully rent-free; g=(1-t/tau)^2 below")
    p.add_argument("--blank-start-frac", type=float, default=0.3,
                   help="rent activates after this fraction (exploration stays free)")
    p.add_argument("--target-white", type=float, default=0.9,
                   help="white-point levels on the target: values above this clip to pure "
                        "white (rest rescaled). Kills paper-texture smudges that adapted "
                        "CLIP faithfully renders as marks. Try 0.85-0.92 on dirty scans")
    p.add_argument("--seed", type=int, default=None,
                   help="seed torch/numpy/random for a reproducible run (default: unseeded)")
    p.add_argument("--progress-every", type=int, default=0,
                   help="save a hard-render snapshot of the CURRENT optimization state every N "
                        "iterations to <output stem>_progress/ (0 = off). Cheap: one glyph "
                        "assembly per snapshot, no extra optimization work")
    # Early stop: once the run has stopped moving, the remaining iterations cannot change
    # the emitted grid, so they are pure cost. TWO independent signals must agree, because
    # either alone is fooled: the hard snap can sit still through a soft phase that is
    # still re-arbitrating underneath it, and the diversity term can flatten while the
    # argmax is still flipping between near-tied slots.
    p.add_argument("--early-stop-patience", type=int, default=6,
                   help="stop once BOTH (a) the hard snap has been unchanged for this many "
                        "consecutive checks and (b) the blend-diversity term has plateaued "
                        "(0 = off, the default). A check runs every --early-stop-every iters, "
                        "so patience x every = the stable window required. Try 6 with "
                        "--early-stop-every 50 (a 300-iter stable snap)")
    p.add_argument("--early-stop-every", type=int, default=50,
                   help="iters between hard-snap stability checks (one cdist; negligible)")
    p.add_argument("--early-stop-tol", type=float, default=0.0,
                   help="fraction of cells allowed to change and still count as stable "
                        "(0 = exact). Small values ride out a single late churning cell; "
                        "0.001 of a 2200-cell grid is ~2 cells")
    p.add_argument("--early-stop-div-window", type=int, default=50,
                   help="window (iters) for the diversity-plateau test: the mean of the last "
                        "window is compared with the mean of the window before it. 0 = drop "
                        "the diversity condition and stop on snap stability alone")
    p.add_argument("--early-stop-div-tol", type=float, default=0.02,
                   help="plateau threshold: diversity counts as flat when it fell by less "
                        "than this FRACTION of its own level over one window. 0.02 = still "
                        "shedding 2%% of the blend variance per window still counts as moving")
    p.add_argument("--early-stop-frac", type=float, default=0.86,
                   help="floor: never stop before this fraction of the schedule, whatever the "
                        "signals say. A guard against a quiet warm-up being read as convergence, "
                        "not a convergence estimate -- the two signals decide")
    p.add_argument("--output", "--out", default=None,
                   help="output PNG (default: <input stem>_ascii.png in the current dir)")
    p.add_argument("--output-text", default=None,
                   help="output .txt grid (default: <--output stem>.txt)")
    p.add_argument("--output-indices", default=None,
                   help="npz path: per-cell glyph indices (GH,GW) + charset/dims, for VAE/model training data")
    # QR artcode: the finished piece packed into a scannable code (needs `segno`).
    p.add_argument("--qr", action="store_true",
                   help="print a QR code of the finished piece to the terminal. The artwork "
                        "itself travels in the URL fragment -- no server, no stored record")
    p.add_argument("--qr-image", default=None, metavar="PATH",
                   help="write the QR code as a PNG")
    p.add_argument("--qr-url", default=None,
                   help="URL prefix the payload hangs off (default: artcode.BASE_URL, the "
                        "live decoder). Keep the HOST uppercase (alphanumeric mode = 5.5 "
                        "bits/char); the path must match the deployed one exactly, since URL "
                        "paths are case-sensitive")
    p.add_argument("--qr-error", default="m", choices=list("lmqh"),
                   help="QR error-correction level; higher survives more damage but holds less")
    p.add_argument("--qr-scale", type=int, default=8, help="PNG pixels per QR module")
    p.add_argument("--qr-no-ansi", action="store_true",
                   help="draw the terminal QR without black-on-white ANSI (light terminals)")
    p.add_argument("--overlay", default=None,
                   help="PNG path: overlay the final ASCII (red ink) on the original target in the "
                        "SAME frame to check spatial alignment")
    p.add_argument("--loss-curve", default=None,
                   help="PNG path: plot loss components (log) + anneal schedules + the soft-vs-hard "
                        "recon discretization gap over iters. Also writes the raw series to .npz")
    # Animation of the optimization (snapped render at intervals -> GIF).
    p.add_argument("--anim", default=None, help="path to write an optimization GIF (e.g. run.gif)")
    p.add_argument("--anim-interval", type=int, default=50, help="capture a frame every N iters")
    p.add_argument("--anim-fps", type=int, default=15, help="GIF playback fps")
    # Per-kit recipe overrides BEFORE parsing, so explicit CLI flags still win.
    apply_profile_recipe(p)
    args = p.parse_args()
    if not args.vae_ckpt:
        p.error(f"--mode {args.mode} requires --vae-ckpt")
    if args.orient_weight > 0 and args.row_gap > 0:
        p.error("--orient-weight needs --row-gap 0 (per-cell pooling assumes a contiguous grid)")
    if args.mode in ("pool", "swarm") and args.row_gap > 0:
        p.error(f"--mode {args.mode} needs --row-gap 0 (per-cell gradient slicing assumes a contiguous grid)")
    if args.clip_steer and args.clip_semantic_weight <= 0:
        p.error("--clip-steer acts on the SEMANTIC term only; --clip-semantic-weight 0 makes it a no-op")
    if args.clip_steer and args.clip_steer_weight == 0:
        p.error("--clip-steer with --clip-steer-weight 0 is the unsteered baseline; drop --clip-steer to say so")
    return args


def main():
    args = parse_args()
    if args.output is None:
        args.output = os.path.splitext(os.path.basename(args.input_image))[0] + "_ascii.png"
    if args.output_text is None:
        args.output_text = os.path.splitext(args.output)[0] + ".txt"

    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = train.DEVICE

    uses_vae = args.mode in ("knn-smooth", "pool", "swarm")
    pool_mode = args.mode == "pool"
    swarm_mode = args.mode == "swarm"
    if pool_mode:
        # z-based machinery has no substrate in pool mode (there is no latent position): the Gumbel
        # noise anneal replaces z-noise, nomination replaces injection, and commit/pixel-nudge are moot.
        inactive = [nm for nm, on in (
            ("--z-noise", args.z_noise > 0), ("--z-noise-commit", args.z_noise_commit > 0),
            ("--z-noise-adapt", args.z_noise_adapt != "off"),
            ("--pixel-nudge-weight", args.pixel_nudge_weight > 0),
        ) if on]
        if inactive:
            print(f"pool mode: z-based flags ignored: {', '.join(inactive)}")
    pad, codebook, model, L = (0, 0), None, None, None
    if uses_vae:
        if os.environ.get("ASCIIFY_PERSIST"):     # persistent E-step workers
            key = ("vae", args.vae_ckpt, str(device))
            if key not in _PERSIST:
                _PERSIST[key] = load_vae(args.vae_ckpt, device)
            model, vae_chars, L, pad = _PERSIST[key]
        else:
            model, vae_chars, L, pad = load_vae(args.vae_ckpt, device)

    ink, chars = G.load_glyphs(device=device, pad=pad)  # sets train.CHARS + fonts
    if uses_vae:
        if chars != vae_chars:
            raise ValueError("charset mismatch between current glyph render and the VAE checkpoint")
        with torch.no_grad():
            codebook, _ = model.encode(ink)            # (N, L)
    N = len(chars)

    train.ROW_GAP = args.row_gap
    char_bitmaps = train.create_char_bitmaps().to(device)  # (N,24,12) white=1
    CH, CW = train.CHAR_HEIGHT, train.CHAR_WIDTH

    # Inference-time banning: filter codebook/bitmaps/charset so snapping can't pick banned glyphs.
    banned = set(args.ban_chars) | (set("░▒▓█▄▌▐▀■") if args.ban_blocks else set())
    if args.ban_letters:
        # any Unicode letter (category L*) -> catches A-Z a-z AND accented/diacritic letters (é à ñ ü ö ç …)
        banned |= {c for c in chars if unicodedata.category(c).startswith("L")}
    keep_t = None                                           # (hoisted) glyph-index filter for --init-vae-ckpt
    if banned:
        keep = [i for i, c in enumerate(chars) if c not in banned]
        keep_t = torch.tensor(keep, device=device)
        chars = "".join(chars[i] for i in keep)
        char_bitmaps = char_bitmaps[keep_t]
        if codebook is not None:
            codebook = codebook[keep_t]
        N = len(chars)
        print(f"Banned {len(banned)} char(s); snapping over {N} glyphs")
    bm_flat = char_bitmaps.view(N, CH * CW)
    bm_sq = bm_flat.pow(2).sum(1)   # (N,) per-glyph squared pixel mass, for the pool diversity term

    # Grid from image aspect (reuses the rasterizer helpers).
    img_w, img_h = train.native_size(args.input_image)
    GW = args.base_width
    if args.glyph_budget > 0:
        # Constant glyph budget. With GH chosen to preserve aspect,
        #   GH = GW * CW * img_h / ((CH + gap) * img_w)   =>   M = GW^2 * CW * img_h / ((CH+gap) * img_w)
        # so the width that spends exactly the budget is
        #   GW = sqrt( M * (CH + gap) * img_w / (CW * img_h) ).
        GW = int(round(math.sqrt(args.glyph_budget * (CH + args.row_gap) * img_w
                                 / max(CW * img_h, 1e-9))))
        GW = max(8, min(GW, args.budget_max_width))
    GH = train.grid_height_for_aspect(img_w, img_h, GW, CW, CH, args.row_gap)
    if args.glyph_budget > 0:
        print(f"glyph budget {args.glyph_budget}: aspect {img_w / max(img_h, 1):.3f} -> "
              f"{GW}x{GH} = {GW * GH} cells"
              + (f" (width clamped to --budget-max-width {args.budget_max_width})"
                 if GW == args.budget_max_width else ""))
    train.GRID_WIDTH, train.GRID_HEIGHT = GW, GH
    train.IMAGE_WIDTH = CW * GW
    train.IMAGE_HEIGHT = CH * GH + args.row_gap * (GH - 1)
    IMG_H, IMG_W, M = train.IMAGE_HEIGHT, train.IMAGE_WIDTH, GH * GW
    print(f"mode={args.mode} grid {GW}x{GH} -> {IMG_W}x{IMG_H}")

    target = train.load_target_image(args.input_image)  # (IMG_H, IMG_W) white=1
    if args.target_white < 1.0:
        # white-point levels: paper texture / smudges above the point clip to clean white.
        # Base CLIP was blind to coherent faint blotches; the adapted metric faithfully
        # renders them as marks -- suppress at the SOURCE, not in the metric.
        target = (target / args.target_white).clamp(0.0, 1.0)

    # COLOR: split the target in two. `tgt_rgb` is what CLIP and recon judge; `target` becomes the
    # monochrome STRUCTURE map from the per-cell two-color decomposition, which every nomination-side
    # consumer below (empty_safe, pix_fit, cell_ink, tone_gate, tgt_cent, blur_T) then reads unchanged
    # -- that is the whole point of the split: the discrete search never learns about color.
    tgt_rgb = color_dec = None
    if args.color and args.color_pin:
        # PLUMBING CONTROL. Pinning fg/bg alone is not a control: the decomposition target and the
        # RGB recon target would both still differ, so a delta could come from anywhere. Here the
        # gray target is replicated to 3 channels and the nomination target is left ALONE, so the
        # color code path is fed exactly the grayscale problem and must reproduce it bit for bit.
        # Any difference vs the same run without --color is a plumbing bug, full stop.
        from unicasso.engine import color as CO
        tgt_rgb = target[:, :, None].expand(-1, -1, 3).contiguous()
        color_dec = CO.decompose(tgt_rgb[:GH * CH, :GW * CW], GH, GW, CH, CW,
                                 ramp=args.color_ramp, jnd=args.color_jnd, ratio=args.color_ratio)
        print("color: --color-pin -> gray target replicated to RGB, nomination target untouched "
              "(plumbing control: must match the same run without --color)")
    elif args.color:
        from unicasso.engine import color as CO
        tgt_rgb = train.load_target_image(args.input_image, keep_rgb=True).to(device).float()
        color_dec = CO.decompose(tgt_rgb[:GH * CH, :GW * CW], GH, GW, CH, CW,
                                 ramp=args.color_ramp, jnd=args.color_jnd, ratio=args.color_ratio)
        nom = CO.nomination_target(color_dec)                              # (GH*CH, GW*CW) white=1
        target = torch.ones_like(target)
        target[:GH * CH, :GW * CW] = nom
        if args.target_white < 1.0:
            # re-apply the white point to the STRUCTURE map: it was applied to the luminance
            # target above, which the decomposition then replaced -- without this it is a no-op
            # under --color, silently dropping the anti-smudge half of the anti-confetti stack
            target = (target / args.target_white).clamp(0.0, 1.0)
        n_flat = int((color_dec["gate"] < 1e-3).sum())
        print(f"color: {M} cells, {n_flat} flat ({100 * n_flat / M:.0f}%), "
              f"mean Lab sep {float(color_dec['sep'].mean()):.1f}, "
              f"mean ink {float(color_dec['ink'].mean()):.3f} "
              f"(jnd {args.color_jnd}, ratio {args.color_ratio})")

    # Empty-space confidence (precompute once; target is fixed): graded whiteness, LOCALLY gated so a
    # stray white cell amid ink (interior negative space) is NOT treated as background. s_c in [0,1]
    # drives the annealed per-cell temp sharpen (#2) + ink penalty (#1) -> clean white bg for CLIP.
    empty_safe = None
    if (args.empty_weight > 0 or args.empty_temp_scale < 1.0 or args.empty_clean_target
            or args.clip_content_crop or args.empty_snap_init):
        white_cell = cells_flat(target, GH, GW, CH, CW, args.row_gap).mean(1)            # (M,) white=1
        e = ((white_cell - args.empty_thresh) / (1.0 - args.empty_thresh)).clamp(0, 1)   # graded emptiness
        K = args.empty_window
        eg = F.pad(e.view(1, 1, GH, GW), (K // 2,) * 4, mode="replicate")                # edge = same, not ink
        nbr = F.conv2d(eg, torch.ones(1, 1, K, K, device=device) / (K * K))[0, 0]        # local mean emptiness
        empty_safe = (e.view(GH, GW) * nbr.clamp(0, 1) ** args.empty_gamma).reshape(M)   # (M,) s_c in [0,1]
        print(f"emptiness: {int((empty_safe > 0.5).sum())}/{M} confidently-empty cells "
              f"(thresh {args.empty_thresh}, window {K}, gamma {args.empty_gamma})")

    # Blank-cell rent weights (anti-confetti term): per-cell target ink -> g = (1-t/tau)^2,
    # full rent on truly blank cells, tapering through the smudge band, zero on real ink.
    blank_g = None
    if args.blank_weight > 0:
        t_ink = 1.0 - cells_flat(target, GH, GW, CH, CW, args.row_gap).mean(1)       # (M,)
        blank_g = (1.0 - (t_ink / max(args.blank_tau, 1e-6)).clamp(max=1.0)).pow(2)  # (M,)
        print(f"blank rent: weight {args.blank_weight}, tau {args.blank_tau}, "
              f"{int((blank_g > 0.5).sum())}/{M} cells above half rent, "
              f"active from frac {args.blank_start_frac}")

    # Whiten the TARGET in confident-empty regions (feathered) so CLIP/recon don't chase background
    # artifacts that corrupt the crop-level CLIP reference. Cleans the shared reference for ALL losses
    # + the warm-start (done before they read `target`). Feather avoids hard cell-edge artifacts.
    if args.empty_clean_target and empty_safe is not None:
        Himg, Wimg = target.shape
        mask = F.interpolate(empty_safe.view(1, 1, GH, GW), size=(Himg, Wimg), mode="nearest")   # per-cell -> pixels
        sg = max(0.5, args.empty_clean_sigma); r = max(1, int(round(3 * sg)))
        xs = torch.arange(-r, r + 1, device=device, dtype=target.dtype)
        ker = torch.exp(-(xs ** 2) / (2 * sg * sg)); ker = ker / ker.sum()
        mask = F.conv2d(mask, ker.view(1, 1, 1, -1), padding=(0, r))                              # separable gaussian
        mask = F.conv2d(mask, ker.view(1, 1, -1, 1), padding=(r, 0))[0, 0].clamp(0, 1)
        target = target + (1.0 - target) * mask                                                   # empties -> white(1)
        print(f"empty-clean-target: feathered-whitened the reference in empty regions (sigma {sg})")

    # CLIP content crop: restrict CLIP to the cell-aligned bbox of NON-empty cells (+pad), so crops
    # aren't wasted on empty margin and the global semantic isn't diluted by whitespace. CLIP-only
    # (recon/orient stay full-frame so the off-content background still gets kept white).
    content_bbox = None
    if args.clip_content_crop and empty_safe is not None:
        occ = (empty_safe.view(GH, GW) < 0.5)                                # content (non-empty) cells
        if occ.any():
            rows = occ.any(1).nonzero().flatten(); cols = occ.any(0).nonzero().flatten()
            pd = args.clip_content_pad
            r0 = max(0, int(rows.min()) - pd); r1 = min(GH, int(rows.max()) + 1 + pd)
            c0 = max(0, int(cols.min()) - pd); c1 = min(GW, int(cols.max()) + 1 + pd)
            sy = CH + args.row_gap
            content_bbox = (r0 * sy, (r1 - 1) * sy + CH, c0 * CW, c1 * CW)    # (y0,y1,x0,x1), cell-aligned
            print(f"clip-content-crop: cells[{r0}:{r1},{c0}:{c1}] -> px{content_bbox} of {tuple(target.shape)}")

    # --- parameters ---
    param_groups = []
    pool = None
    swarm = None
    sw_temp_lut = None     # (N,) per-glyph density temp multiplier (swarm per-slot blend temps)
    z = None
    knn_z_init = knn_z_hist = knn_hist_it = None
    if uses_vae:
        if args.random_init:
            z0 = torch.randn(M, L, device=device) * 0.1
        elif args.init_text:       # previous run's text grid -> z0 = codebook[glyph] per cell
            with open(args.init_text, encoding="utf-8") as f:
                rows = [line.rstrip("\n") for line in f if line.rstrip("\n") != ""]
            if len(rows) != GH or any(len(r) != GW for r in rows):
                raise ValueError(f"--init-text grid is {len(rows)}x{sorted(set(len(r) for r in rows))}, "
                                 f"need {GH}x{GW} (same --base-width as the source run)")
            missing = sorted({c for r in rows for c in r if c not in chars})
            if missing:
                raise ValueError(f"--init-text contains glyphs outside the current charset "
                                 f"(check --ban-* flags match the source run): {''.join(missing)}")
            gidx = torch.tensor([chars.index(c) for r in rows for c in r], device=device)
            z0 = codebook[gidx].clone()
            print(f"init-text: warm-start from {args.init_text}")
        elif args.init_scan:       # ORACLE warm start: best-locking glyph over a dense shift sweep
            with torch.no_grad():
                R_ = args.init_scan_range
                blank_ids = torch.tensor([i for i, c in enumerate(chars) if c.isspace() or c == "\xa0"],
                                         device=device)
                ink0 = (1.0 - cells_flat(target, GH, GW, CH, CW, args.row_gap)).mean(1)   # (M,)
                band = torch.maximum(0.5 * ink0, torch.full_like(ink0, 0.02))
                best_d = torch.full((M,), float("inf"), device=device)
                best_g = torch.zeros(M, dtype=torch.long, device=device)
                Ht, Wt = target.shape
                for dy in range(-R_, R_ + 1):
                    for dx in range(-R_, R_ + 1):
                        s = torch.ones_like(target)
                        ys, ye = max(0, dy), min(Ht, Ht + dy)
                        xs, xe = max(0, dx), min(Wt, Wt + dx)
                        s[ys:ye, xs:xe] = target[ys - dy:ye - dy, xs - dx:xe - dx]
                        cf = 1.0 - cells_flat(s, GH, GW, CH, CW, args.row_gap)            # ink flat
                        zs, _ = model.encode(F.pad(cf.view(M, 1, CH, CW), (pad[1], pad[1], pad[0], pad[0]), value=0.0))
                        dsh = torch.cdist(zs, codebook)
                        dsh[:, blank_ids] = float("inf")                  # blanks never a detection
                        dmin, gmin = dsh.min(dim=1)
                        ok = (cf.mean(1) - ink0).abs() <= band            # ink-preservation guard
                        upd = ok & (dmin < best_d)
                        best_d = torch.where(upd, dmin, best_d)
                        best_g = torch.where(upd, gmin, best_g)
                if " " in chars:                                          # near-empty cells seed blank
                    best_g[ink0 <= 0.02] = chars.index(" ")
                z0 = codebook[best_g].clone()
            print(f"init-scan: oracle warm start, +/-{R_}px sweep ({(2 * R_ + 1) ** 2} views)")
        elif args.init_vae_ckpt:   # separate init VAE: encode+snap with IT, start z at MAIN codebook[glyph]
            with torch.no_grad():
                iv_model, iv_chars, _, iv_pad = load_vae(args.init_vae_ckpt, device)
                if iv_chars != vae_chars:
                    raise ValueError("--init-vae-ckpt charset must match the main VAE")
                iv_cb, _ = iv_model.encode(G.load_glyphs(device=device, pad=iv_pad)[0])   # init VAE codebook
                if keep_t is not None:
                    iv_cb = iv_cb[keep_t]                # match the main (banned) glyph set
                cells = [target[i * (CH + args.row_gap):i * (CH + args.row_gap) + CH, j * CW:j * CW + CW]
                         for i in range(GH) for j in range(GW)]
                iv_in = F.pad((1.0 - torch.stack(cells)).unsqueeze(1),
                              (iv_pad[1], iv_pad[1], iv_pad[0], iv_pad[0]), value=0.0)
                iv_z, _ = iv_model.encode(iv_in)
                snap = torch.cdist(iv_z, iv_cb).argmin(dim=1)     # robust glyph choice from the init VAE
                z0 = codebook[snap].clone()                       # -> MAIN VAE latent for that glyph
            print(f"init-vae: warm-start via {args.init_vae_ckpt} (encode+snap -> main codebook)")
        else:  # warm start: encode each target patch (tone-aware init)
            with torch.no_grad():
                if args.context_init:
                    # real neighbor-pixel margin (matches multimodal/context training): pad the WHOLE
                    # image white, then slice cell + pad-px of real surroundings. Border -> white.
                    ink_pad = F.pad((1.0 - target)[None, None], (pad[1], pad[1], pad[0], pad[0]), value=0.0)[0, 0]
                    sy = CH + args.row_gap
                    cell_ink = torch.stack([ink_pad[i * sy:i * sy + CH + 2 * pad[0], j * CW:j * CW + CW + 2 * pad[1]]
                                            for i in range(GH) for j in range(GW)]).unsqueeze(1)
                else:                                   # white margin (matches glyph-only training)
                    cells = [target[i * (CH + args.row_gap):i * (CH + args.row_gap) + CH, j * CW:j * CW + CW]
                             for i in range(GH) for j in range(GW)]
                    cell_ink = F.pad((1.0 - torch.stack(cells)).unsqueeze(1), (pad[1], pad[1], pad[0], pad[0]), value=0.0)
                z0, _ = model.encode(cell_ink)
        # Emptiness snap-init: for confidently-empty cells, discard the (possibly unreliable) encoder
        # latent and seed z directly at the blank/space glyph, so those cells START clean instead of at
        # whatever ink the VAE hallucinated from a near-white patch. Applies over any init above. The
        # cells are still free to move afterward; this only fixes the seed.
        if args.empty_snap_init:
            if " " not in chars:
                print("  (--empty-snap-init needs a blank glyph in the charset; disabled)")
            elif empty_safe is None:
                print("  (--empty-snap-init: emptiness gate unavailable; disabled)")
            else:
                sidx = chars.index(" ")
                snap_mask = empty_safe > args.empty_snap_thresh          # (M,) bool
                z0[snap_mask] = codebook[sidx].to(z0.dtype)
                print(f"empty-snap-init: seeded {int(snap_mask.sum())}/{M} empty cells "
                      f"(s_c > {args.empty_snap_thresh}) at the blank glyph")
        if pool_mode or swarm_mode:
            # z0 seeds the pool/swarm then is discarded as a single position. --empty-snap-init
            # (above) still composes: it moves z0 to the blank glyph first. The channel/nomination
            # state (search mask, density band, latent/ports/file channels) is SHARED machinery --
            # both hosts expose the same attribute names.
            space_i = chars.index(" ") if " " in chars else None
            if pool_mode:
                from unicasso.engine.pool import CandidatePool
                pool = CandidatePool(GH, GW, args.pool_k, device)
                n_seed = pool.init_from_z0(z0, codebook, empty_safe=empty_safe, space_idx=space_i)
                if args.pool_init_single:                  # one-glyph start: top choice fills all slots
                    pool.cand.copy_(pool.cand[:, 0:1].expand(-1, pool.K).contiguous())
                    pool.logits.data.copy_(
                        pool.logits.data[:, 0:1].expand(-1, pool.K).contiguous())
                    if space_i is not None and empty_safe is not None:   # empties still seed space
                        em = empty_safe > 0.5
                        pool.cand[em] = space_i
                    print(f"pool: single-glyph init (membership earned from one seed per cell)")
                host = pool
            else:
                from unicasso.engine.swarm import ParticleSwarm
                swarm = ParticleSwarm(GH, GW, args.swarm_k, L, device)
                n_seed = swarm.init_from_z0(z0, codebook, empty_safe=empty_safe, space_idx=space_i,
                                            single=args.swarm_init_single)
                if args.swarm_init_single:
                    print("swarm: single-slot init (slots 1+ start FREE; membership earned by births)")
                if args.init_w_lead:
                    # Trust the warm start: slot 0 enters with a lead in raw W units. The
                    # spread-cap projection below keeps it expressible-but-contestable;
                    # empty-seeded cells already hold a decisive +3 and are closed anyway.
                    swarm.W.data[:, 0] += args.init_w_lead
                    print(f"init-w-lead: slot 0 starts +{args.init_w_lead:g} W ahead")
                host = swarm
            host.space_idx = space_i                       # never evicted (the escape hatch)
            if not pool_mode and args.swarm_empty_pin > 0 and empty_safe is not None \
                    and space_i is not None:
                swarm.empty_pin_mask = empty_safe > 0.5    # the seeded population
                print(f"swarm empty-pin: {int(swarm.empty_pin_mask.sum())} cells held at "
                      f"W floor {args.swarm_empty_pin} (noise fully zeroed; decay countered)")
            if empty_safe is not None and args.pool_empty_block <= 1.0:
                host.search_mask = empty_safe < args.pool_empty_block   # empty cells closed to search
                print(f"{args.mode}: {int((~host.search_mask).sum())}/{M} confidently-empty cells closed "
                      f"to nomination/cap (empty_safe >= {args.pool_empty_block})")
            if args.pool_nominate_density_band > 0:        # target-ink band on nominees (anti-overshoot)
                ink_c = (1.0 - cells_flat(target, GH, GW, CH, CW, args.row_gap)).mean(1)   # (M,)
                ink_g = (1.0 - bm_flat).mean(1)                                            # (N,)
                host.density_block = ((ink_c[:, None] - ink_g[None, :]).abs()
                                      > args.pool_nominate_density_band)
                print(f"{args.mode} density band {args.pool_nominate_density_band}: "
                      f"{float(host.density_block.float().mean()):.0%} of (cell,glyph) nominations blocked")
            sw_height = None
            if not pool_mode and (args.swarm_height_weight > 0 or args.pool_join_centroid > 0
                                  or args.pool_join_coord_sigma > 0):
                # target-side centroid evidence, computed once: where a cell holds a coherent
                # line, its ink centroid is the height/position the render should match
                with torch.no_grad():
                    yg_ = ((torch.arange(CH, device=device) + 0.5) / CH)[:, None] \
                        .expand(CH, CW).reshape(-1)
                    xg_ = ((torch.arange(CW, device=device) + 0.5) / CW)[None, :] \
                        .expand(CH, CW).reshape(-1)
                    swarm.tgt_cent, swarm.tgt_norm, cgate = target_line_stats(
                        target, GH, GW, CH, CW, args.row_gap,
                        gate_pow=args.swarm_height_gate_pow)
                    swarm.cent_gate = cgate
                    gink = 1.0 - bm_flat                                            # (N,P)
                    gm = gink.sum(1)
                    swarm.glyph_cent = torch.stack(
                        [(gink * yg_).sum(1) / gm.clamp_min(1e-6),
                         (gink * xg_).sum(1) / gm.clamp_min(1e-6)], dim=1)
                    if args.swarm_height_weight > 0:
                        sw_height = (args.swarm_height_weight, yg_, xg_)
                    print(f"swarm height prior: weight {args.swarm_height_weight}, join gate "
                          f"{args.pool_join_centroid}, {int((cgate > 0.5).sum())}/{M} cells "
                          f"strongly gated (coherence^{args.swarm_height_gate_pow} x ink band)")
            if not pool_mode and args.swarm_tone_weight > 0 \
                    and "pixel" not in args.pool_channels:
                # tone bias without the pixel channel: compute the shared regime tensors here
                with torch.no_grad():
                    ti0 = 1.0 - cells_flat(target, GH, GW, CH, CW, args.row_gap)
                    _, pc0, _ = O.glyph_orientations(ti0.view(M, 1, CH, CW), sigma=0.0)
                    swarm.cell_coh = pc0.to(device).clamp(0, 1)
                    swarm.cell_ink = ti0.mean(1)
            if not pool_mode and "pixel" in args.pool_channels:
                # pixel channel tables, computed once: full-res target-vs-glyph MSE (the
                # structure evidence), raw cell coherence + inks (the regime split), and
                # the tone family (shades + block elements) for incoherent-dark proposals
                with torch.no_grad():
                    ti_ = 1.0 - cells_flat(target, GH, GW, CH, CW, args.row_gap)   # (M,P)
                    gi_ = 1.0 - bm_flat                                            # (N,P)
                    swarm.pix_fit = (ti_.pow(2).sum(1, keepdim=True)
                                     - 2.0 * ti_ @ gi_.t()
                                     + gi_.pow(2).sum(1)[None]) / float(CH * CW)
                    _, pcoh_, _ = O.glyph_orientations(ti_.view(M, 1, CH, CW), sigma=0.0)
                    swarm.cell_coh = pcoh_.to(device).clamp(0, 1)
                    swarm.cell_ink = ti_.mean(1)
                    swarm.glyph_ink = gi_.mean(1)
                    swarm.tone_mask = torch.tensor(
                        [0x2580 <= ord(c) <= 0x259F for c in chars], device=device)
                    n_tone = int(((swarm.cell_coh <= args.pool_pixel_gate)
                                  & (swarm.cell_ink >= args.pool_pixel_tone_ink)).sum())
                    n_struct = int((swarm.cell_coh > args.pool_pixel_gate).sum())
                    print(f"pixel channel: {int(swarm.tone_mask.sum())} tone-family glyphs; "
                          f"{n_struct} structure-regime cells, {n_tone} tone-regime cells "
                          f"(coh<={args.pool_pixel_gate} & ink>={args.pool_pixel_tone_ink})")
            if (not pool_mode and args.pool_pixel_grating > 0
                    and getattr(swarm, "cell_coh", None) is not None):
                # Sub-Nyquist stripe gate: dense oriented texture (grilles/fences) reads
                # as HIGH coherence -- it IS oriented -- so the coh<=gate split can never
                # hand it to the tone family. Resolvability is stroke COUNT: >= N distinct
                # ink runs within +-1.5 cells (max over 4 axes; grille texture shows
                # ~8-11 runs vs drawable structure's 2-6). Zeroing the EFFECTIVE
                # coherence of these cells routes every downstream consumer (pixel tone
                # proposals, tone W-bias defense) with no further changes.
                with torch.no_grad():
                    ink_full = (1.0 - target).float().cpu()
                    Hf, Wf = ink_full.shape
                    pitchy = CH + args.row_gap
                    cand = (swarm.cell_ink.cpu().view(GH, GW)
                            >= args.pool_pixel_tone_ink).nonzero()
                    grat = torch.zeros(GH, GW, dtype=torch.bool)
                    ts = torch.arange(-1.5 * CH, 1.5 * CH + 1, 1.0)
                    for y, x in cand.tolist():
                        cy, cx = y * pitchy + CH // 2, x * CW + CW // 2
                        best = 0
                        for a in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
                            yy = (cy + ts * float(np.sin(a))).round().long().clamp(0, Hf - 1)
                            xx = (cx + ts * float(np.cos(a))).round().long().clamp(0, Wf - 1)
                            pr = ink_full[yy, xx] > 0.35
                            runs = int((pr[1:] & ~pr[:-1]).sum()) + int(pr[0])
                            best = max(best, runs)
                        if best >= args.pool_pixel_grating:
                            grat[y, x] = True
                    # area filter: only SUSTAINED grating fields switch regime -- a lone
                    # dense cell inside linework is detail, not a texture region
                    n_raw = int(grat.sum())
                    if args.pool_pixel_grating_area > 1 and n_raw:
                        seen = torch.zeros_like(grat)
                        keep = torch.zeros_like(grat)
                        for y0, x0 in grat.nonzero().tolist():
                            if seen[y0, x0]:
                                continue
                            comp, stack = [], [(y0, x0)]
                            seen[y0, x0] = True
                            while stack:
                                y, x = stack.pop()
                                comp.append((y, x))
                                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                    ny, nx = y + dy, x + dx
                                    if (0 <= ny < GH and 0 <= nx < GW
                                            and grat[ny, nx] and not seen[ny, nx]):
                                        seen[ny, nx] = True
                                        stack.append((ny, nx))
                            if len(comp) >= args.pool_pixel_grating_area:
                                for y, x in comp:
                                    keep[y, x] = True
                        grat = keep
                    if bool(grat.any()):
                        swarm.cell_coh = swarm.cell_coh.masked_fill(
                            grat.view(-1).to(swarm.cell_coh.device), 0.0)
                    print(f"pixel grating gate: {int(grat.sum())} dense-stripe cells -> "
                          f"tone regime (>= {args.pool_pixel_grating} runs / 3 cells, "
                          f"area >= {args.pool_pixel_grating_area}; "
                          f"{n_raw - int(grat.sum())} isolated dropped)")
            if not pool_mode and args.swarm_tone_weight > 0 and swarm.cell_coh is not None:
                swarm.tone_gate = ((swarm.cell_coh <= args.pool_pixel_gate)
                                   & (swarm.cell_ink >= args.pool_pixel_tone_ink)).float()
                print(f"swarm tone bias: weight {args.swarm_tone_weight}, "
                      f"{int(swarm.tone_gate.sum())} tone-regime cells defended")
            cbd_ = torch.cdist(codebook, codebook)
            cbd_.fill_diagonal_(float("inf"))
            nn_med = float(cbd_.min(dim=1).values.median())   # median codebook NN spacing
            if pool_mode and args.pool_dedup_eps > 0:      # nominee dedup radius, in absolute latent units
                pool.dedup_eps = args.pool_dedup_eps * nn_med   # (swarm: dedup = boost-radius instead)
            if "latent" in args.pool_channels:             # adjacency kernel for the latent channel
                h = max(1e-6, args.pool_latent_h * nn_med)
                if pool_mode:
                    cbd_.fill_diagonal_(0.0)
                    pool.cb_prox = torch.exp(-cbd_.pow(2) / (2 * h * h))
                else:
                    swarm.latent_h = h                     # swarm computes prox from LIVE slot z
                print(f"{args.mode} latent channel: adjacency bandwidth h={h:.2f} "
                      f"({args.pool_latent_h} x median NN {nn_med:.2f})")
            if any(k in args.pool_channels for k in ("ports", "join", "blend")):
                # tile-continuity state (tile_ports.py): the ports channel's target scores + the
                # compat matrices that ports/join/blend all read for the live-neighbor term
                from unicasso.engine import tile_ports as TP
                tp_chars = TP.DEFAULT_TILE_CHARS if args.pool_ports_chars == "default" \
                    else args.pool_ports_chars
                glyph_ink = (1.0 - char_bitmaps).unsqueeze(1)            # (N,1,CH,CW) unpadded
                prof_g = TP.edge_profiles(glyph_ink)
                Ch_, Cv_ = TP.compat_matrices(prof_g, gamma=args.pool_ports_gamma)
                ports_ovr = args.pool_ports_overrides or os.path.join(G.kit_dir(), "port_overrides.json")
                TP.apply_overrides(Ch_, Cv_, TP.load_overrides(ports_ovr), chars)
                wl = torch.tensor([c in set(tp_chars) for c in chars], device=device)
                T_ = TP.target_port_scores(target, GH, GW, CH, CW, prof_g,
                                           gamma=args.pool_ports_gamma)
                T_[:, ~wl] = float("-inf")                               # proposals: whitelist only
                # live-neighbor compat: no opinion (0) when either side is off-whitelist; clamp the
                # -1e9 override bans to a strong-but-finite discouragement for the additive term
                for C_ in (Ch_, Cv_):
                    C_.clamp_(-10.0, 10.0)
                    C_[~wl, :] = 0.0
                    C_[:, ~wl] = 0.0
                host.ports_T, host.ports_Ch, host.ports_Cv = T_, Ch_, Cv_
                print(f"{args.mode} ports channel: {int(wl.sum())}/{N} tile glyphs, "
                      f"floor {args.pool_ports_floor}, overrides {ports_ovr}")
                if swarm_mode:   # per-glyph L/R/U/D edge-band ink (3px) for the dangling penalty
                    gi3 = 1.0 - char_bitmaps                             # (N,CH,CW) ink
                    host.port_mass = torch.stack(
                        [gi3[:, :, :3].mean((1, 2)), gi3[:, :, -3:].mean((1, 2)),
                         gi3[:, :3, :].mean((1, 2)), gi3[:, -3:, :].mean((1, 2))], dim=1)
                    if not args.pool_blend_allow_letters:
                        host.blend_block = torch.tensor(
                            [unicodedata.category(c).startswith("L") for c in chars], device=device)
                        print(f"blend channel: {int(host.blend_block.sum())} letter glyphs excluded "
                              f"(--pool-blend-allow-letters to re-admit)")
            if not args.pool_allow_blank_noms:
                host.nom_block = torch.tensor(
                    [c.isspace() or c == "\xa0" for c in chars], device=device)
                print(f"nominations: {int(host.nom_block.sum())} blank glyphs blocked for ALL "
                      f"channels (blank reachable via latent drift only)")
            if swarm_mode and args.swarm_boost_ink > 0:
                ink_bc = (1.0 - cells_flat(target, GH, GW, CH, CW, args.row_gap)).mean(1)
                swarm.boost_ink_ok = ink_bc >= args.swarm_boost_ink
                print(f"boosts: gated to {int(swarm.boost_ink_ok.sum())}/{M} cells with target "
                      f"ink >= {args.swarm_boost_ink}")
            if args.pool_nominate_file:                    # probe-certified proposals ('file' channel)
                pf = np.load(args.pool_nominate_file)
                f_chars = "".join(pf["chars"])
                fb, fa = pf["before_snap"], pf["after_snap"]
                fprop = torch.full((M,), -1, dtype=torch.long, device=device)
                n_map = n_skip = 0
                for c in range(min(M, len(fa))):
                    if fa[c] != fb[c]:
                        ch_ = f_chars[fa[c]]
                        if ch_ in chars:
                            fprop[c] = chars.index(ch_)
                            n_map += 1
                        else:
                            n_skip += 1
                host.file_prop = fprop
                if "file" not in args.pool_channels:
                    args.pool_channels = "file," + args.pool_channels
                print(f"{args.mode} file channel: {n_map} certified proposals loaded "
                      f"({n_skip} outside charset) from {args.pool_nominate_file}")
            if pool_mode:
                param_groups.append({"params": [pool.logits], "lr": args.lr,
                                     "weight_decay": args.pool_logit_decay})
                print(f"pool: K={args.pool_k} per cell (warm-start k-NN support), "
                      f"space seeded into {n_seed} empty cells, dedup eps {pool.dedup_eps:.3f}")
            else:
                if args.color:
                    # per-candidate colors, closed-form MSE-initialized on each seeded slot's
                    # OWN glyph -- so W arbitrates (shape, color) jointly from step 0
                    swarm.color_on = True
                    swarm.bm_flat = bm_flat
                    swarm.tgt_cell_rgb = color_dec["cell_rgb"].to(device)
                    if args.color_palette:
                        from unicasso.engine import color as CO
                        pal = CO.build_palette(
                            torch.cat([color_dec["fg"], color_dec["bg"]], 0).to(device),
                            args.color_palette)
                        swarm.set_palette(pal)
                        print(f"color: quantizing fg/bg to {pal.shape[0]} palette entries (STE)")
                    sg0, _ = swarm.slot_glyphs(codebook)                  # (M,K) seeded glyphs
                    swarm.fit_slot_colors(1.0 - bm_flat[sg0])
                    if args.color_init:
                        # start the free leaves at the colors the LITE model actually
                        # SHIPPED (whatever that lite version's recipe is: v1 cluster
                        # blend + distance-weighted fit + k-hat, v2 the blend-read
                        # mask fit), not the engine's plainer per-slot re-fit. Makes
                        # --refine a true polish of the lite output instead of a
                        # re-derivation, and puts the anchor's drift metric on a
                        # meaningful zero.
                        zi = np.load(args.color_init)
                        ifg = torch.from_numpy(np.asarray(zi["fg"], np.float32)) \
                            .to(device).view(-1, 3)
                        ibg = torch.from_numpy(np.asarray(zi["bg"], np.float32)) \
                            .to(device).view(-1, 3)
                        if ifg.shape[0] != M or ibg.shape[0] != M:
                            raise ValueError(
                                f"--color-init has {ifg.shape[0]} cells, grid has "
                                f"M={M} (same width/font as the lite render?)")
                        swarm.fg.data.copy_(ifg[:, None, :].expand(M, swarm.K, 3))
                        swarm.bg.data.copy_(ibg[:, None, :].expand(M, swarm.K, 3))
                        print(f"color: --color-init -> fg/bg seeded from {args.color_init} "
                              f"(lite's shipped colors, all {swarm.K} slots)")
                    color_fg0 = swarm.fg.data.clone()                     # closed-form init, for
                    color_bg0 = swarm.bg.data.clone()                     # the end-of-run drift report
                    if args.color_lite:
                        from unicasso.engine.lite_color import LiteColorer
                        swarm.lite_colorer = LiteColorer(
                            args.color_lite, args.input_image, GH, GW, device=device,
                            font=args.color_lite_font, ink=args.color_lite_ink,
                            read=dict(color_path=args.color_lite_path,
                                      frac=args.color_lite_frac,
                                      count=args.color_lite_count,
                                      temp=args.color_lite_temp,
                                      ridge=args.color_lite_ridge),
                            k_scale=args.color_lite_k_scale,
                            chunk=args.color_lite_chunk)
                        swarm.lite_mode = args.color_lite_mode
                        if args.color_contrast_learn:
                            print("color: NOTE --color-lite -> the learned per-cell "
                                  "contrast k (--color-contrast-learn) is not used; "
                                  "fg/bg are the lite model's, unmodified")
                        swarm.lite_grid = swarm.snap(codebook)
                        if args.color_lite_mode in ("birth", "forward"):
                            swarm.fit_slot_colors(1.0 - bm_flat[sg0], glyphs=sg0)
                        print(f"color: --color-lite {args.color_lite} "
                              f"(mode {args.color_lite_mode}, ink "
                              f"{args.color_lite_ink}, read "
                              f"{args.color_lite_path} frac {args.color_lite_frac:g} "
                              f"count {args.color_lite_count}) -> fg/bg from the "
                              f"distilled mask decoder, NOT optimized")
                    elif args.color_pin:
                        swarm.fg.data.zero_(); swarm.bg.data.fill_(1.0)
                        print("color: --color-pin -> fg=black/bg=white (grayscale control arm)")
                    elif args.color_mask:
                        # FROZEN-MASK coloring: fg/bg = the mask-
                        # weighted fit on the target cells, computed once and
                        # held constant across the whole run. Colors belong to
                        # the content, not the glyph: every slot in a cell
                        # carries the same pair, W arbitrates pure shape, and
                        # fg/bg stay OUT of param_groups.
                        zm = np.load(args.color_mask)
                        pm = torch.from_numpy(np.asarray(zm["masks"],
                                                         np.float32)).to(device)
                        if pm.shape[0] != M or pm.shape[-2:] != (CH, CW):
                            raise ValueError(
                                f"--color-mask masks {tuple(pm.shape)} do not match "
                                f"grid M={M} cell {CH}x{CW} (same width/font?)")
                        tgt_c = swarm.tgt_cell_rgb                     # (M,P,3)
                        meanc = tgt_c.mean(1)
                        rid = args.color_mask_ridge
                        pf = pm[:, 0].reshape(M, -1)
                        pb = pm[:, 1].reshape(M, -1)
                        fgm = ((pf[..., None] * tgt_c).sum(1) + rid * meanc) \
                            / (pf.sum(1, keepdim=True) + rid)
                        bgm = ((pb[..., None] * tgt_c).sum(1) + rid * meanc) \
                            / (pb.sum(1, keepdim=True) + rid)
                        if "k" in zm.files:            # lite k rides the masks
                            kv = torch.from_numpy(np.asarray(zm["k"], np.float32)) \
                                .to(device).view(M, 1)
                            midm = 0.5 * (fgm + bgm)
                            fgm = (midm + kv * (fgm - midm)).clamp(0, 1)
                            bgm = (midm + kv * (bgm - midm)).clamp(0, 1)
                        K_ = swarm.fg.shape[1]
                        swarm.fg.data.copy_(fgm[:, None, :].expand(M, K_, 3))
                        swarm.bg.data.copy_(bgm[:, None, :].expand(M, K_, 3))
                        if args.color_mask_mode == "init":
                            _clr = args.lr if args.color_lr is None else args.color_lr
                            param_groups.append({"params": [swarm.fg, swarm.bg],
                                                 "lr": _clr})
                            print(f"color: --color-mask-mode init -> colors warm-"
                                  f"started from {args.color_mask}, FREE under "
                                  f"CLIP/recon (lr {_clr:g})")
                            if args.color_anchor > 0:
                                from unicasso.engine.color import glyph_bg_dist
                                swarm._anchor_bgdist = glyph_bg_dist(1.0 - bm_flat, CH, CW)
                                swarm._anchor_w = args.color_anchor
                                swarm._anchor_mc = args.color_anchor_mc
                                swarm._anchor_fg = swarm._anchor_bg = None
                                print(f"color: --color-anchor {args.color_anchor:g} -> cosine-"
                                      f"ramped L2 to closed-form distwt fit "
                                      f"(mc {args.color_anchor_mc:g})")
                        else:
                            print(f"color: --color-mask -> frozen mask-fit colors from "
                                  f"{args.color_mask} (ridge {rid:g}"
                                  + (", k applied" if "k" in zm.files else "") + ")")
                    elif args.color_fit:
                        # colors are a FUNCTION of the ink now: there are no leaves left to
                        # optimize, so fg/bg deliberately stay OUT of param_groups
                        swarm.color_fit = True
                        swarm.color_fit_mc = (args.recolor_min_contrast
                                              if args.color_fit_min_contrast is None
                                              else args.color_fit_min_contrast)
                        swarm.color_fit_detach = args.color_fit_detach
                        if args.color_bg_dist_pow > 0:
                            from unicasso.engine.color import glyph_bg_dist
                            swarm.color_bg_dist = glyph_bg_dist(1.0 - bm_flat, CH, CW)
                            swarm.color_bg_dist_pow = args.color_bg_dist_pow
                        if args.color_source != "fit" and color_dec is not None:
                            swarm.cluster_fg = color_dec["fg"].detach()
                            swarm.cluster_bg = color_dec["bg"].detach()
                            swarm.cluster_alpha = (1.0 if args.color_source == "cluster"
                                                   else args.color_cluster_alpha)
                        if args.color_contrast_learn:
                            swarm.k_contrast_max = args.color_contrast_max
                            param_groups.append({"params": [swarm.k_contrast],
                                                 "lr": args.color_contrast_lr})
                            print(f"color: --color-contrast-learn -> per-cell k over the last "
                                  f"{args.color_contrast_iters} iters (lr {args.color_contrast_lr}, "
                                  f"tv {args.color_contrast_tv}, max {args.color_contrast_max})")
                        print("color: --color-fit -> fg/bg = closed-form MSE optimum every step "
                              f"(min-contrast {swarm.color_fit_mc:g}"
                              + (", detached" if args.color_fit_detach else "")
                              + (f", bg-dist pow {args.color_bg_dist_pow:g}"
                                 if swarm.color_bg_dist is not None else "")
                              + (f", cluster alpha {swarm.cluster_alpha:g}"
                                 if swarm.cluster_fg is not None else "") + ")")
                    else:
                        _clr = args.lr if args.color_lr is None else args.color_lr
                        param_groups.append({"params": [swarm.fg, swarm.bg], "lr": _clr})
                        if args.color_anchor > 0:
                            # free CLIP colors, but cosine-ramp them back toward the
                            # closed-form fit of the placed glyphs late in the run.
                            # Mask-independent: works on the k-hat (no-mask) model too.
                            from unicasso.engine.color import glyph_bg_dist
                            swarm._anchor_bgdist = glyph_bg_dist(1.0 - bm_flat, CH, CW)
                            swarm._anchor_w = args.color_anchor
                            swarm._anchor_mc = args.color_anchor_mc
                            swarm._anchor_fg = swarm._anchor_bg = None
                            print(f"color: --no-color-fit + --color-anchor "
                                  f"{args.color_anchor:g} -> free fg/bg, cosine-ramped "
                                  f"L2 to the closed-form fit (mc {args.color_anchor_mc:g})")
                    if args.color_anchor > 0 and not hasattr(swarm, "_anchor_w"):
                        _mode = ("--color-pin" if args.color_pin else
                                 f"--color-mask (mode {args.color_mask_mode})"
                                 if args.color_mask else "--color-fit (the DEFAULT)")
                        raise SystemExit(
                            f"--color-anchor {args.color_anchor:g} is a NO-OP under "
                            f"{_mode}: the anchor pulls the FREE fg/bg leaves toward the "
                            f"closed-form fit, but in this mode fg/bg are not optimizer "
                            f"parameters at all (under --color-fit the shipped colors ARE "
                            f"already the closed-form fit, so there is nothing to anchor).\n"
                            f"Use one of:  --no-color-fit  (free colors, fit-anchored)\n"
                            f"             --color-mask FILE --color-mask-mode init")
                param_groups.append({"params": [swarm.z], "lr": args.lr})
                param_groups.append({"params": [swarm.W], "lr": args.lr,
                                     "weight_decay": args.swarm_w_decay,
                                     "swarm_w_group": True})   # tag: decay scheduled per-iter
                if args.swarm_w_cap > 0:
                    swarm.clamp_w_spread(args.swarm_w_cap)     # project the init gaps into capped units
                if args.swarm_temp_adapt != "off":             # per-slot blend-temp density scaling
                    wall_ = cbd_.clone()
                    wall_.fill_diagonal_(float("inf"))
                    wall_ = wall_.min(dim=1).values
                    sw_temp_lut = (wall_ / (wall_.median() + 1e-8)).clamp(
                        1.0 / args.z_noise_adapt_clip, args.z_noise_adapt_clip)
                print(f"swarm: K={args.swarm_k} slots x {args.swarm_knn_k}-NN blends per cell, "
                      f"W cap {args.swarm_w_cap}, space seeded into {n_seed} empty cells, "
                      f"boost radius {args.swarm_boost_radius * nn_med:.2f} / "
                      f"merge eps {args.swarm_merge_eps * nn_med:.2f} (abs)")
                # absolute-unit knobs derived from codebook geometry
                args._swarm_boost_radius_abs = args.swarm_boost_radius * nn_med
                args._swarm_merge_eps_abs = args.swarm_merge_eps * nn_med
        else:
            z = nn.Parameter(z0.clone())
            param_groups.append({"params": [z], "lr": args.lr})
            # displacement diagnostics (--pool-record in knn-smooth mode): init latents
            # + sampled z trajectory, so late-run mobility can be measured after the run
            knn_z_init = z0.detach().clone() if args.pool_record else None
            knn_z_hist, knn_hist_it = ([], []) if args.pool_record else (None, None)

    opt = torch.optim.AdamW(param_groups)

    def adam_reset_slots(c_, k_):
        """Zero Adam's first moment for freshly (re)born swarm slots (--swarm-adam-reset): the
        evictee's ghost momentum dies with it. exp_avg_sq is kept -- see the flag help."""
        if swarm is None or not args.swarm_adam_reset or c_.numel() == 0:
            return
        for prm in (swarm.z, swarm.W, swarm.fg, swarm.bg):
            st = opt.state.get(prm)
            if st and "exp_avg" in st:
                st["exp_avg"][c_, k_] = 0.0
    crop = lambda d: d[:, 0, pad[0]:pad[0] + CH, pad[1]:pad[1] + CW]

    coord = None
    if args.coord_weight > 0 and uses_vae and args.row_gap == 0:
        U_ = getattr(model, "sym_U", None)
        if U_ is None:
            print("--coord-weight: checkpoint has no sym_U (train the VAE with --sym-weight); disabled")
        else:
            Uh_ = F.normalize(U_.to(device).float(), dim=1)                  # rows rh/rv/tx/ty
            gink_ = 1.0 - bm_flat
            ygc = ((torch.arange(CH, device=device) + 0.5) / CH)[:, None].expand(CH, CW).reshape(-1)
            xgc = ((torch.arange(CW, device=device) + 0.5) / CW)[None, :].expand(CH, CW).reshape(-1)
            gm_ = gink_.sum(1).clamp_min(1.0)
            gcy = (gink_ * ygc).sum(1) / gm_ - 0.5
            gcx = (gink_ * xgc).sum(1) / gm_ - 0.5
            S_ = (codebook @ Uh_.t()).cpu()                                  # (N,4); fit on cpu

            def _fit(s, t):
                A_ = torch.stack([s, torch.ones_like(s)], 1)
                ab = torch.linalg.lstsq(A_, t[:, None]).solution.squeeze()
                r = float(torch.corrcoef(torch.stack([A_ @ ab, t]))[0, 1])
                return float(ab[0]), float(ab[1]), r
            ay_, by_, ry_ = _fit(S_[:, 3], gcy.cpu())
            ax_, bx_, rx_ = _fit(S_[:, 2], gcx.cpu())
            cc_, nn_, gg_ = target_line_stats(target, GH, GW, CH, CW, args.row_gap)
            # LATENT-unit targets (inverse affine): centroid-unit error
            # attenuates the z-gradient by the readout slope a (~0.1) and a mean over ALL
            # cells dilutes another ~4x => per-cell force ~1e-5, invisible. In s-units with
            # a gated mean the term competes at --commit scale. Axis terms weighted by the
            # readout's r^2 so the weak x-axis can't inject noise.
            tt_ = cc_ - 0.5
            coord = dict(uy=Uh_[3], ux=Uh_[2],
                         ty_s=(tt_[:, 0] - by_) / (ay_ if abs(ay_) > 1e-3 else 1.0),
                         tx_s=(tt_[:, 1] - bx_) / (ax_ if abs(ax_) > 1e-3 else 1.0),
                         wy=max(ry_, 0.0) ** 2 * (abs(ay_) > 1e-3),
                         wx=max(rx_, 0.0) ** 2 * (abs(ax_) > 1e-3),
                         n=nn_, g=gg_, gsum=float(gg_.sum().clamp_min(1.0)))
            print(f"coord loss: axis readout corr y {ry_:+.2f} / x {rx_:+.2f}; "
                  f"{int((gg_ > 0.25).sum())}/{M} cells gated; weight {args.coord_weight}"
                  + (f"->{args.coord_weight_end}" if args.coord_weight_end is not None else " (const)"))

    clipper = None
    if args.clip_weight > 0:
        if args.perceptual == "dino":
            from unicasso.engine.dino_loss import DINOPerceptualLoss
            print(f"Loading DINOv3 ({args.dino_model}) for self-similarity perceptual loss...")
            clipper = DINOPerceptualLoss(device, model_name=args.dino_model, n_aug=args.clip_aug,
                                         crop_scale=tuple(args.clip_crop_scale), res=args.dino_res,
                                         struct_weight=args.dino_struct_weight,
                                         global_weight=args.dino_global_weight,
                                         cache_target=True,
                                         scale_alpha=args.clip_scale_alpha)
        else:
            from unicasso.engine.clip_loss import CLIPPerceptualLoss
            print(f"Loading CLIP ({args.clip_model}) for CLIPasso perceptual loss...")
            vit_layers = tuple((int(t.split(":")[0]), float(t.split(":")[1]) if ":" in t else 1.0)
                               for t in args.clip_vit_layers.split(","))   # 'idx' or 'idx:weight' pairs
            ck_ = ("clipper", args.clip_model, args.clip_pretrained, args.clip_aug,
                   args.clip_semantic_weight, tuple(args.clip_crop_scale),
                   args.clip_rotate, args.clip_shear, args.clip_invert_frac,
                   args.clip_scale_alpha, tuple(args.clip_aspect_jitter),
                   args.clip_edge_frac, args.clip_edge_beta, args.clip_edge_auto,
                   vit_layers, args.clip_vit_drop_cls, args.clip_fp16,
                   args.clip_adapter, args.clip_batch_aug, args.clip_reg_frac,
                   CH, CW, args.clip_microbatch, args.clip_steer,
                   args.clip_steer_weight, str(device))
            if os.environ.get("ASCIIFY_PERSIST") and ck_ in _PERSIST:
                clipper = _PERSIST[ck_]
                # target-derived caches are PER-IMAGE: a reused worker instance
                # must not carry the previous job's dense target maps. The
                # recompute guard is _dense_key -- clear THAT (clearing only
                # _dense_tgt leaves the key matched and crashes on None)
                clipper._dense_key = None
            else:
                clipper = CLIPPerceptualLoss(device, model_name=args.clip_model, pretrained=args.clip_pretrained,
                                         n_aug=args.clip_aug, fc_weight=args.clip_semantic_weight,
                                         crop_scale=tuple(args.clip_crop_scale),
                                         rotate_deg=args.clip_rotate, shear_deg=args.clip_shear,
                                         invert_frac=args.clip_invert_frac,
                                         scale_alpha=args.clip_scale_alpha,
                                         aspect_jitter=tuple(args.clip_aspect_jitter),
                                         edge_frac=args.clip_edge_frac, edge_beta=args.clip_edge_beta,
                                         edge_auto=args.clip_edge_auto,
                                         vit_layers=vit_layers, vit_drop_cls=args.clip_vit_drop_cls,
                                         fp16=args.clip_fp16, adapter=args.clip_adapter,
                                         batch_aug=args.clip_batch_aug,
                                         reg_frac=args.clip_reg_frac, cell_h=CH, cell_w=CW, microbatch=args.clip_microbatch,
                                         steer=args.clip_steer, steer_weight=args.clip_steer_weight)
                if os.environ.get("ASCIIFY_PERSIST"):
                    _PERSIST[ck_] = clipper
            if args.clip_reg_frac > 0:
                print(f"  crop registration ON: target crops nudged up to +/-{args.clip_reg_frac} cell "
                      f"({int(round(args.clip_reg_frac * CW))}x{int(round(args.clip_reg_frac * CH))}px)")

    from PIL import Image

    # Non-local semantic affinity: per-cell top-k corresponding cells (CNN visual sim gated by DINO
    # semantics) -> edges for the same graph-Laplacian latent pull as consistency, but NON-local.
    affin_idx = None
    if (args.affinity_weight > 0 or (args.affinity_weight_end or 0) > 0) and uses_vae:
        try:
            from unicasso.engine.affinity import gated_matrix, affinity_edges
            if args.affinity_feat_image:
                fi = Image.open(args.affinity_feat_image).convert("RGB").resize((IMG_W, IMG_H), Image.LANCZOS)
                feat_t = torch.from_numpy(np.asarray(fi, np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            else:
                feat_t = target[None, None]                  # the line art itself (white=1)
            layers = [int(v) for v in args.affinity_layers.split(",")]
            reuse = clipper if (type(clipper).__name__ == "CLIPPerceptualLoss" and hasattr(getattr(clipper, "visual", None), "conv1")
                                and args.affinity_feat_image is None) else None  # reuse only RN101 (has conv1 for dense_features); ViT/ConvNeXt -> affinity loads its own RN101
            print(f"Loading DINOv3 ({args.dino_model}) + CLIP-conv "
                  f"({'reusing RN101' if reuse is not None else 'fresh RN101'}) for non-local affinity...")
            A = gated_matrix(feat_t, GH, GW, device, layers=layers, beta=args.affinity_beta,
                             dino_model=args.dino_model, max_side=args.affinity_max_side, clip=reuse)
            tcells = torch.stack([target[i * (CH + args.row_gap):i * (CH + args.row_gap) + CH, j * CW:j * CW + CW]
                                  for i in range(GH) for j in range(GW)])
            ink = (1.0 - tcells).mean(dim=(-1, -2)).cpu().numpy()   # per-cell mean darkness, for the blank gate
            si, dj, w = affinity_edges(A, args.affinity_topk, gamma=args.affinity_gamma,
                                       ink=ink, min_ink=args.affinity_min_ink, device=device)
            affin_idx = (si, dj, w)
            print(f"affinity: {len(si)} non-local edges (layers {layers}, beta {args.affinity_beta}, "
                  f"topk {args.affinity_topk}, min-ink {args.affinity_min_ink})")
            if pool is not None or swarm is not None:   # strongest partner per cell -> nomination channel
                part = torch.full((M,), -1, dtype=torch.long, device=device)
                order = torch.argsort(w)            # ascending: the highest-weight edge is written last
                part[si[order]] = dj[order]
                (pool if pool is not None else swarm).affinity_partner = part
                print(f"{args.mode} affinity channel: partners for {int((part >= 0).sum())}/{M} cells")
        except Exception as e:
            # The canonical recipe turns affinity on by default, so a failed DINOv3
            # fetch is the FIRST thing that breaks on a fresh install and it would brick
            # the advertised one-line quickstart. Fail loudly, drop the non-local term,
            # and finish the render: everything else is unchanged.
            print("\n" + "!" * 78)
            print(f"AFFINITY DISABLED -- could not load DINOv3 ({args.dino_model})")
            print(f"  {type(e).__name__}: {e}")
            print("  The timm weights normally download without a HuggingFace account,")
            print("  so this is usually network or a stale timm. Try:")
            print("    pip install -U timm     # older versions lack the dinov3 entries")
            print("  Meta's own facebook/dinov3-* repos ARE licence-gated; if you pointed")
            print("  --dino-model at one of those, accept it there and huggingface-cli login.")
            print("  Continuing WITHOUT affinity. Pass --affinity-weight-end 0 to silence this.")
            print("!" * 78 + "\n")
            args.affinity_weight = 0.0
            args.affinity_weight_end = 0.0
            affin_idx = None

    def snap_indices():
        """Current hard per-cell glyph choice, any mode. pool: argmax clean logit (consistent with
        the blend by construction -- no cdist snap that could drop a render-visible winner)."""
        if pool_mode:
            return pool.snap()
        if swarm_mode:
            return swarm.snap(codebook)
        return torch.cdist(z, codebook).argmin(dim=1)

    def render_hard():
        """Snap to discrete chars and assemble the hard render -> (idx, uint8 image)."""
        i = snap_indices()
        o = assemble(char_bitmaps[i].view(1, GH, GW, CH, CW), GH, GW, CH, CW, args.row_gap, IMG_H, IMG_W)[0]
        return i, (o.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def render_hard_color():
        """Hard render with each cell's WINNING slot's own fg/bg -> uint8 (H,W,3).
        This is the shipping image: the same glyph AND the same colors the .ans will carry."""
        with torch.no_grad():
            i = snap_indices()
            k1 = swarm.W.data.masked_fill(swarm.free, float("-inf")).argmax(1)
            if getattr(swarm, "color_fit", False):
                fgv, bgv = swarm.fit_emit_colors(1.0 - bm_flat[i], gidx=i)
            else:
                fgv, bgv = swarm._colors()
                fgv = fgv[swarm._ar, k1].clamp(0, 1)
                bgv = bgv[swarm._ar, k1].clamp(0, 1)
            cc = bgv[:, None, :] + (fgv - bgv)[:, None, :] * (1.0 - bm_flat[i])[:, :, None]
            o = assemble_rgb(cc.view(1, GH, GW, CH, CW, 3), GH, GW, CH, CW, args.row_gap, IMG_H, IMG_W)[0]
        return i, (o.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # defined here, AFTER the swarm setup block above, so it reflects the final state
    color_anim = bool(args.color) and getattr(swarm, "color_on", False) if swarm is not None else False

    # Density-equalized noise LUT. Per codebook glyph, the local wall distance -> a per-cell
    # multiplier m = wall/median(wall), mean-normalized so --z-noise still sets the overall level while
    # sigma_c/wall_c is constant (uniform escape probability). Indexed each step by the cell's current snap.
    adapt_lut = None
    if args.z_noise_adapt != "off" and uses_vae and not pool_mode:
        with torch.no_grad():
            cbd = torch.cdist(codebook, codebook)
            cbd.fill_diagonal_(float("inf"))
            if args.z_noise_adapt == "nn1":
                wall = cbd.min(dim=1).values                                  # nearest-neighbor distance
            else:                                                             # clusterk: k-th neighbor radius
                kk = min(args.knn_k, codebook.shape[0] - 1)
                wall = cbd.topk(kk, dim=1, largest=False).values[:, -1]
            adapt_lut = (wall / (wall.median() + 1e-8)).clamp(
                1.0 / args.z_noise_adapt_clip, args.z_noise_adapt_clip)       # (N,) per-glyph multiplier
        print(f"z-noise-adapt: {args.z_noise_adapt} wall median {float(wall.median()):.3f} "
              f"mult range [{float(adapt_lut.min()):.2f}, {float(adapt_lut.max()):.2f}]")

    frames = []
    from collections import defaultdict
    curve = defaultdict(list)
    hard_every = max(1, args.iters // 100)  # ~100 sampled points for the hard-snap recon
    dense_layers = tuple(int(x) for x in args.clip_dense_layers.split(",") if x.strip())
    dense_on = (args.clip_dense_weight > 0 and clipper is not None
                and args.perceptual != "dino" and hasattr(clipper, "dense_loss")
                and getattr(clipper, "visual", None) is not None
                and hasattr(clipper.visual, "layer1"))
    if args.clip_dense_weight > 0 and not dense_on:
        print("--clip-dense-weight needs an active RN-backbone clipper; disabled")
    if dense_on:
        print(f"clip-dense: layers {dense_layers} @ res {args.clip_dense_res}, "
              f"weight {args.clip_dense_weight}"
              + (f"->{args.clip_dense_weight_end}" if args.clip_dense_weight_end is not None else ""))
    diag_on = bool(args.clip_diagnostic) and clipper is not None and args.row_gap == 0
    if args.clip_diagnostic and not diag_on:
        print("  (--clip-diagnostic needs a perceptual loss + --row-gap 0; disabled)")
    ema_decay = 1.0 - 1.0 / max(1, args.clip_diag_ema_window)
    ema_conflict = ema_clipmag = None
    # All anneal schedules read sched_iters (not args.iters), so --schedule-stretch >1 SLOWS every
    # schedule uniformly (shape/margins preserved): a trajectory that would flatten early instead
    # fills the whole run. The loop still runs args.iters real steps.
    sched_iters = max(2, int(round(args.iters * args.schedule_stretch)))
    # Pool nomination: window in sched_iters units (consistent with every other schedule);
    # must END before the sharp phase -- at low temp the softmax saturates and arrivals are
    # gradient-dead. grad-EMA decay from the window flag, like the diag EMA.
    nom_on = (pool_mode or swarm_mode) and args.pool_nominate_rate > 0
    nom_lo = int(args.pool_nominate_start_frac * sched_iters)
    nom_hi = int(args.pool_nominate_end_frac * sched_iters)
    pool_grad_decay = 1.0 - 1.0 / max(1, args.pool_grad_ema_window)
    pool_chans = [s.strip() for s in args.pool_channels.split(",") if s.strip()]   # ORDER = priority
    # LIVE PROBING: measured verification of nominations on the current hard render. Needs the
    # RN101 conv trunk (dense_features). Target features cached once.
    live_probe = ((pool_mode or swarm_mode) and args.pool_probe and clipper is not None
                  and hasattr(getattr(clipper, "visual", None), "conv1"))
    # COLOR PROBING. With FREE color leaves the loss is
    # driven by learned per-slot fg/bg, and a counterfactual swap drops a different glyph into a
    # slot whose colors were fit to the OLD one, so the measured delta would confound shape with a
    # stale palette. Under --color-fit that confound is gone -- colors are a deterministic function
    # of the placed glyph, so any counterfactual's hard render is exactly computable and the probe
    # measures in the same RGB metric the loss actually uses.
    probe_color = bool(args.color) and getattr(swarm, "color_fit", False)
    if args.color and live_probe and not probe_color:
        print("--pool-probe under --color needs --color-fit (with free color leaves a swap's colors "
              "are stale, so the measured delta confounds shape with palette) -- live probing disabled")
        live_probe = False
    if (pool_mode or swarm_mode) and args.pool_probe and not live_probe:
        print("  (--pool-probe needs the RN101-style conv CLIP trunk; disabled)")
    F_tgt_probe = probe_norm = None
    # Both the comparison target and the recon normalizer follow the space the LOSS lives in:
    # color recon is a mean over H*W*3, gray over H*W. Get this wrong and the probe margin silently
    # means something different in color than the value that was tuned in gray.
    probe_tgt = tgt_rgb if probe_color else target
    probe_recon_div = (IMG_H * IMG_W * 3) if probe_color else (IMG_H * IMG_W)
    if live_probe:
        with torch.no_grad():
            F_tgt_probe = clipper.dense_features(probe_tgt, layers=(2, 3), max_side=768,
                                                 base=True)   # target = frozen base path
            probe_norm = {l: float(F_tgt_probe[l].numel()) for l in (2, 3)}
        print(f"{args.mode} live-probe ON: nominations verified on the hard render "
              f"(margin {args.pool_probe_margin}, spacing {args.pool_probe_spacing}"
              + (f", lottery margin {args.swarm_lottery_margin}" if swarm_mode else "") + ")")

    def probe_err_maps(img):
        Fi = clipper.dense_features(img, layers=(2, 3), max_side=768)
        return {l: ((Fi[l] - F_tgt_probe[l]) ** 2).sum(dim=0) for l in (2, 3)}

    def probe_delta_at(E1, E0, img1, img0, cell, tgt_now, rw_now):
        """Measured loss delta of one hard swap: windowed layer2+3 feature-error delta (training-
        scale normalized) + the cell's recon delta, in run-loss units."""
        r0, c0 = divmod(int(cell), GW)
        y0 = max(0, r0 - args.pool_probe_window) * CH
        y1 = min(GH, r0 + args.pool_probe_window + 1) * CH
        x0 = max(0, c0 - args.pool_probe_window) * CW
        x1 = min(GW, c0 + args.pool_probe_window + 1) * CW
        dclip = 0.0
        for l in (2, 3):
            h, w = E1[l].shape
            fy0, fy1 = int(y0 * h / IMG_H), int(math.ceil(y1 * h / IMG_H))
            fx0, fx1 = int(x0 * w / IMG_W), int(math.ceil(x1 * w / IMG_W))
            dclip += float((E1[l][fy0:fy1, fx0:fx1] - E0[l][fy0:fy1, fx0:fx1]).sum()) / probe_norm[l]
        cy = slice(r0 * CH, (r0 + 1) * CH)
        cx = slice(c0 * CW, (c0 + 1) * CW)
        drecon = float(((img1[cy, cx] - tgt_now[cy, cx]) ** 2).sum()
                       - ((img0[cy, cx] - tgt_now[cy, cx]) ** 2).sum()) / probe_recon_div
        return args.clip_weight * dclip + rw_now * drecon

    def probe_render(gidx):
        """Hard render of a glyph-index grid (M,), in whatever space the loss lives in.

        Under --color-fit the colors are refit closed-form at the PLACED glyphs, which is exactly
        what emission ships -- so a counterfactual's measured delta is the delta of a realizable
        image, not of a render nothing could produce."""
        if probe_color:
            ink_ = 1.0 - bm_flat[gidx]                                   # (M,P), 1 = glyph ink
            fgp, bgp = swarm.fit_emit_colors(ink_, gidx=gidx)
            cc = bgp[:, None, :] + (fgp - bgp)[:, None, :] * ink_[:, :, None]
            return assemble_rgb(cc.view(1, GH, GW, CH, CW, 3), GH, GW, CH, CW,
                                args.row_gap, IMG_H, IMG_W)[0]
        return assemble(char_bitmaps[gidx].view(1, GH, GW, CH, CW),
                        GH, GW, CH, CW, args.row_gap, IMG_H, IMG_W)[0]

    n_probe_acc = n_probe_rej = 0
    probe_cache = {}                       # cell -> {glyph: (delta, it_measured)} memoization
    probe_last = torch.full((M,), -1, dtype=torch.long, device=device)   # LRU sweep order
    n_swaps_total = 0
    n_swaps_ch = torch.zeros(9, dtype=torch.long)     # grad/neighbor/affinity/latent/ports/file/blend/join/pixel
    prec = None
    if args.pool_record and (pool_mode or swarm_mode):
        prec = dict(swap_it=[], swap_cell=[], swap_in=[], swap_out=[], swap_ch=[],
                    round_it=[], gate_pass=[], hist_it=[], snap_hist=[], ent_hist=[],
                    probe_it=[], probe_cell=[], probe_in=[], probe_delta=[], probe_accept=[],
                    probe_ch=[])
        if swarm_mode:   # swarm extras: birth kind, boost/merge logs, per-slot trajectories,
            # round economics (econ row: [cells-with-proposal, birth-ready, grace-cooling,
            # free-slot-avail, boost-near, winner-nudged, boost-cool-blocked, boost-applied])
            prec.update(swap_kind=[], swap_slot=[],
                        boost_it=[], boost_cell=[], boost_slot=[], boost_g=[],
                        merge_it=[], merge_cell=[], merge_d=[], merge_vl=[],
                        econ_it=[], econ=[],
                        slot_g_hist=[], slot_v_hist=[], slot_d_hist=[], slot_gn_hist=[],
                        slot_step_hist=[], wp_hist=[])
        print(f"{args.mode}-record: churn diagnostics -> {args.pool_record}")
    if nom_on:
        print(f"{args.mode} nomination: every {args.pool_nominate_rate} iters in [{nom_lo}, {nom_hi}) "
              f"(grace {args.pool_grace}, tabu {args.pool_tabu}, priority {pool_chans})")
    n_boosts = n_merges = n_lottery = 0
    sw_seams = set()
    if swarm_mode and args.swarm_purge_cycles and args.swarm_w_temp_cycles > 1:
        seg_ = args.swarm_w_temp_end_frac * sched_iters / args.swarm_w_temp_cycles
        sw_seams = {int(round(k * seg_)) for k in range(1, args.swarm_w_temp_cycles)}
        print(f"swarm elite purge at reheat seams: iters {sorted(sw_seams)}")
    # ---- early stop (see --early-stop-patience) -------------------------------------
    # Two signals, ANDed, because each alone has a blind spot: the hard SNAP can sit still
    # through a soft phase that is still re-arbitrating underneath it (the argmax has not
    # moved, but the weights behind it have), and the DIVERSITY term can flatten while the
    # argmax still flips between near-tied slots. Together: nothing is moving, above or
    # below the discretization. --early-stop-frac is only a floor against reading a quiet
    # warm-up as convergence.
    es_on = args.early_stop_patience > 0
    es_floor, es_tol = int(args.early_stop_frac * sched_iters), int(round(args.early_stop_tol * M))
    es_prev, es_stable, es_break = None, 0, False
    es_div_hist, es_div_flat = [], True
    # COLOUR REFINE PHASE. Placement (glyph choice) and colour (per-cell contrast k) converge
    # on different signals: k never moves the snap, so the snap/diversity plateau below cannot
    # see the k-phase working. So instead of disabling early-stop under --color-contrast-learn,
    # the placement plateau ARMS a fixed colour-contrast budget: k learning is forced on, and
    # the run then goes exactly --color-contrast-iters more steps before stopping (the same
    # window it would have gotten at the run's tail, just started as soon as placement is done
    # rather than at a fixed offset). es_refine_start = the iter it was armed (None = placement).
    es_refine = args.color and args.color_contrast_learn and args.color_contrast_iters > 0
    es_refine_start = None
    # the diversity term's own activation condition, evaluated here rather than in the loop
    es_div_avail = (pool_mode or swarm_mode) and (
        args.pool_diversity_weight > 0 or (args.pool_diversity_weight_end or 0) > 0)
    es_div_on = es_on and args.early_stop_div_window > 0 and es_div_avail
    if es_on:
        if args.early_stop_div_window > 0 and not es_div_avail:
            print("early-stop: diversity term inactive, so the plateau condition is dropped "
                  "-- stopping on hard-snap stability alone")
    if es_on:
        print(f"early stop: floor iter {es_floor} ({args.early_stop_frac:.2f} of schedule), "
              f"then stop when the snap is unchanged for {args.early_stop_patience} checks "
              f"x {args.early_stop_every} iters "
              f"({args.early_stop_patience * args.early_stop_every} iters"
              + (f", tolerance {es_tol} cells" if es_tol else ", exact") + ")"
              + (f" AND diversity has fallen < {args.early_stop_div_tol:.0%} over "
                 f"{args.early_stop_div_window} iters" if es_div_on else ""))
    sw_blank_ids = (torch.tensor([i for i, c in enumerate(chars) if c.isspace() or c == "\xa0"],
                                 device=device) if swarm_mode else None)
    sw_blank_streak = (torch.zeros(M, dtype=torch.long, device=device) if swarm_mode else None)
    sw_static_closed = None      # the empty_safe-closed base set (dynamic closure never reopens it)
    pbar = tqdm(range(args.iters))
    for it in pbar:
        opt.zero_grad()
        commit_vec = None                      # SGLD committed-noise slice (set in the z-noise block)

        tgt = target
        tgt_recon = tgt

        # emptiness gate, annealed: the SAME g_emp drives the per-cell temp sharpen (#2) and the ink
        # penalty (#1), so #1 only ever pushes where #2 has already made pure space reachable.
        g_emp = None
        if empty_safe is not None:
            a_emp = schedule_value(it, sched_iters, args.empty_anneal, args.empty_anneal_end,
                                   kind=args.empty_anneal_schedule, end_frac=args.empty_anneal_end_frac)
            g_emp = a_emp * empty_safe                                          # (M,) in [0,1]

        # render cells
        cell_embed = None  # per-cell codebook embedding (blend embed; affinity/consistency substrate)
        w_pool = None      # (M,K) pool blend weights (pool mode only; diversity term reads it)
        sw = None          # swarm render dict (swarm mode only; diversity/records read it)
        cell_rgb = None    # (M,P,3) color cells (color mode only)
        if pool_mode:
            # free-logit pool blend: w = softmax((l + sigma*gumbel)/tau_c). tau reuses the knn-temp
            # schedule (+ the per-cell empty sharpen); Gumbel noise is the z-noise successor (jitters
            # the DECISION variable each step, anneals to 0 to commit). Snap/anim use clean logits.
            kt = schedule_value(it, sched_iters, args.knn_temp, args.knn_temp_end,
                                kind=args.knn_temp_schedule, warmup_frac=args.knn_temp_warmup_frac,
                                warmup_start=(args.knn_temp_warmup_start if args.knn_temp_warmup_start
                                              is not None else args.knn_temp_end),
                                end_frac=args.knn_temp_end_frac)
            pnz = schedule_value(it, sched_iters, args.pool_noise, args.pool_noise_end,
                                 kind=args.pool_noise_schedule, end_frac=args.pool_noise_end_frac)
            if g_emp is not None and args.empty_temp_scale < 1.0:
                tau_c = kt * (1.0 - g_emp * (1.0 - args.empty_temp_scale))   # (M,) per-cell temp (#2)
            else:
                tau_c = kt
            gate = None
            if g_emp is not None and args.empty_noise_scale < 1.0:
                gate = 1.0 - g_emp * (1.0 - args.empty_noise_scale)          # empties audition quietly
            w_pool = pool.weights(tau_c, pnz, gate)                          # (M,K)
            cell_img = (w_pool[:, :, None, None] * char_bitmaps[pool.cand]).sum(dim=1)   # (M,CH,CW)
            cell_embed = (w_pool[:, :, None] * codebook[pool.cand]).sum(dim=1)           # (M,L) blend embed
            commit = torch.tensor(0.0, device=device)                        # no z -> no commit anchor
            if nom_on:
                pool.update_stats(tau_c, args.pool_w_ema)                    # clean-w EMA + slot ages
        elif swarm_mode:
            # Two-level swarm render: v = softmax(W/tau_W) over per-slot knn-smooth blends.
            # tau_z (per-slot blend) reuses the knn-temp schedule; tau_W has its own EARLIER-
            # collapsing schedule (slot election mid-run while the winner can still slide; the
            # cheap look-alike blend commits last). The empty stack sharpens/quiets BOTH levels.
            kt = schedule_value(it, sched_iters, args.knn_temp, args.knn_temp_end,
                                kind=args.knn_temp_schedule, warmup_frac=args.knn_temp_warmup_frac,
                                warmup_start=(args.knn_temp_warmup_start if args.knn_temp_warmup_start
                                              is not None else args.knn_temp_end),
                                end_frac=args.knn_temp_end_frac)
            if args.swarm_w_temp_cycles > 1:   # phased hardening: elect / re-soften / re-elect
                wt = cycled_temp(it, sched_iters, args.swarm_w_temp, args.swarm_w_temp_end,
                                 args.swarm_w_temp_cycles, args.swarm_w_temp_cycle_decay,
                                 args.swarm_w_temp_end_frac, shape=args.swarm_w_temp_cycle_shape,
                                 lin_end=args.swarm_w_temp_cycle_lin_end,
                                 lin_n=args.swarm_w_temp_cycle_lin_n)
            else:
                wt = schedule_value(it, sched_iters, args.swarm_w_temp, args.swarm_w_temp_end,
                                    kind=args.swarm_w_temp_schedule, end_frac=args.swarm_w_temp_end_frac)
            # cycle-phase state: hardening = the tail of the current tau_W segment, or the final
            # post-window hold. Under --swarm-phase-gates: soft phase explores (nominations +
            # W-noise on, consolidator off), hardening crystallizes (consolidator on, rounds
            # paused, W-noise silenced -- Gumbel jitter during a collapse is just visual noise).
            sw_hardening = False
            if args.swarm_w_temp_cycles > 1:
                wend_ = args.swarm_w_temp_end_frac * sched_iters
                if it >= wend_:
                    sw_hardening = True
                else:
                    seg_ = wend_ / args.swarm_w_temp_cycles
                    sw_hardening = (it % seg_) / seg_ >= args.swarm_phase_frac
            else:
                sw_hardening = it >= args.swarm_w_temp_end_frac * sched_iters * args.swarm_phase_frac
            # knn-smooth EPILOGUE: arbitration is over (one dominant particle per cell); give the
            # winner a mini knn-smooth pass -- tau_z and z-noise reheat, then cosine-anneal to
            # their ends, so every cell finishes with within-family refinement under full
            # gradient legibility.
            ep_cos = None
            ep_n = int(args.swarm_epilogue_frac * sched_iters)
            if ep_n > 0:
                p_ep = None
                if it >= sched_iters - ep_n:                             # run-end window
                    if not args.swarm_epilogue_skip_final:               # (adapter runs: the
                        p_ep = (it - (sched_iters - ep_n)) / max(1, ep_n)   # unpoliced reheat
                elif args.swarm_epilogue_cycles and args.swarm_w_temp_cycles > 1:   # injects confetti)
                    wend_ = args.swarm_w_temp_end_frac * sched_iters     # per-segment tail windows
                    seg_ = wend_ / args.swarm_w_temp_cycles
                    ep_seg = args.swarm_epilogue_frac * seg_
                    pos_ = it % seg_
                    cyc_ = int(it // seg_)
                    cyc_ok = (args.swarm_epilogue_cycles_max <= 0
                              or cyc_ < args.swarm_epilogue_cycles_max)
                    # skip-final also covers the LAST cycle's tail: it abuts the tau_W
                    # end (hard W, nominations over) so it IS the unpoliced endgame
                    # reheat -- and colliding with the run-end mask would truncate the
                    # bump mid-flight (full-amplitude reheat, then a snap back --
                    # scrambles winners with nothing left to heal them).
                    if args.swarm_epilogue_skip_final:
                        cyc_ok = cyc_ok and cyc_ < args.swarm_w_temp_cycles - 1
                    if it < wend_ and pos_ >= seg_ - ep_seg and cyc_ok:
                        p_ep = (pos_ - (seg_ - ep_seg)) / max(1.0, ep_seg)
                if p_ep is not None:
                    ep_cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p_ep)))
                    # bump ABOVE the global track only; decays back onto it by the window end
                    kt = kt + max(0.0, args.swarm_epilogue_temp - kt) * ep_cos
            if g_emp is not None and args.empty_temp_scale < 1.0:
                sharp = 1.0 - g_emp * (1.0 - args.empty_temp_scale)          # (M,)
                tau_c, tau_w_c = kt * sharp, wt * sharp
            else:
                tau_c, tau_w_c = kt, wt
            if args.color_contrast_learn and getattr(swarm, "color_fit", False):
                swarm.k_contrast_on = (args.color_contrast_iters <= 0
                                       or it >= sched_iters - args.color_contrast_iters
                                       or es_refine_start is not None)   # armed early by plateau
            gate = None
            if g_emp is not None and args.empty_noise_scale < 1.0:
                gate = 1.0 - g_emp * (1.0 - args.empty_noise_scale)          # empties audition quietly
            if getattr(swarm, "empty_pin_mask", None) is not None:
                if gate is None:
                    gate = torch.ones(M, device=device)
                gate = gate * (~swarm.empty_pin_mask).float()  # pinned: NO z/W noise at all
            # per-slot exploration noise: knn-smooth's machinery applied per slot (adaptive sigma
            # by the slot's local codebook density, SGLD commit slice; bias/blur are single-z-only)
            znz = schedule_value(it, sched_iters, args.z_noise, args.z_noise_end,
                                 kind=args.z_noise_schedule)
            if args.z_noise_quench_frac < 1.0:            # release the floor for the final stretch
                q0 = args.z_noise_quench_frac * sched_iters
                if it >= q0:
                    znz *= max(0.0, 1.0 - (it - q0) / max(1.0, sched_iters - q0))
            if ep_cos is not None:                        # epilogue bump above the (quenched) track
                znz = znz + max(0.0, args.swarm_epilogue_noise - znz) * ep_cos
            if znz > 0:
                noise = torch.randn_like(swarm.z) * znz
                if adapt_lut is not None:
                    sg_, _ = swarm.slot_glyphs(codebook)                     # pre-noise basins (M,K)
                    noise = noise * adapt_lut[sg_][:, :, None]
                if gate is not None:
                    noise = noise * gate[:, None, None]
                commit_vec = noise if args.z_noise_commit > 0 else None
                z_eff_sw = swarm.z + noise
            else:
                z_eff_sw = swarm.z
            if args.swarm_w_decay_end is not None:
                wd_now = schedule_value(it, sched_iters, args.swarm_w_decay,
                                        args.swarm_w_decay_end,
                                        kind=args.swarm_w_decay_schedule,
                                        end_frac=args.swarm_w_decay_end_frac)
                for pg_ in opt.param_groups:
                    if pg_.get("swarm_w_group"):
                        pg_["weight_decay"] = wd_now
            w_sig = schedule_value(it, sched_iters, args.swarm_w_noise, 0.0,
                                   kind=args.pool_noise_schedule,
                                   end_frac=args.pool_noise_end_frac) if args.swarm_w_noise > 0 else 0.0
            if args.swarm_phase_gates and sw_hardening:
                w_sig = 0.0                                  # no auditions while crystallizing
            sw = swarm.render(z_eff_sw, codebook, bm_flat, bm_sq, args.swarm_knn_k, tau_c, tau_w_c,
                              temp_lut=sw_temp_lut, w_noise=w_sig, w_gate=gate, height=sw_height,
                              tone_w=args.swarm_tone_weight)
            cell_img = sw["cell"].view(M, CH, CW)
            cell_rgb = sw["cell_rgb"]                       # (M,P,3) or None
            cell_embed = sw["embed"]
            # per-slot commit anchor, v-weighted (dominant slot anchors hardest; equals knn-smooth's
            # commit exactly when v is one-hot). Anchors the REAL param, refs from the noisy sample.
            commit = (((swarm.z - sw["commit_ref"]) ** 2).mean(-1) * sw["v"].detach()).sum(1).mean()
            # VIP/consolidation planes: straight-through snap render for the CLIP loss (forward =
            # hard shipping image, backward = the soft blend's Jacobian to every W and z)
            sw_snap_w = (args.clip_snap_weight
                         if it >= args.clip_snap_start_frac * sched_iters else 0.0)
            if args.swarm_phase_gates:
                sw_cons_w = args.clip_consist_weight if sw_hardening else 0.0
            else:
                sw_cons_w = (args.clip_consist_weight
                             if it >= args.clip_consist_start_frac * sched_iters else 0.0)
            if sw_snap_w > 0 or sw_cons_w > 0:
                if args.clip_snap_mode == "winner":
                    # 'hard' = the ARGMAX PARTICLE's blend render -- fully
                    # differentiable through the winner slot's z (selection detached), so the
                    # winner can NAVIGATE under this term (e.g. travel to the space basin)
                    kwin = swarm.W.detach().masked_fill(swarm.free, float("-inf")).argmax(dim=1)
                    hard_c = sw["slot_img"][torch.arange(M, device=device), kwin].view(M, CH, CW)
                else:                                                    # discrete glyph + STE backward
                    with torch.no_grad():
                        hard_c = char_bitmaps[swarm.snap(codebook)]      # (M,CH,CW)
                    hard_c = hard_c + (cell_img - cell_img.detach())
                if args.clip_snap_gate == "entropy":
                    # entropy-gated compositing: forward = m*hard + (1-m)*soft per cell (m DETACHED
                    # -- gradient through m would reward committing toward a BAD argmax).
                    # Uncommitted cells can't hide in the mixture / proxy-space gray.
                    with torch.no_grad():
                        vcl = sw["v"]
                        m_ = (-(vcl * (vcl + 1e-9).log()).sum(1)
                              / math.log(max(2, args.swarm_k))).clamp(0, 1)[:, None, None]
                    vip_cell = m_ * hard_c + (1.0 - m_) * cell_img
                else:
                    vip_cell = hard_c
            else:
                vip_cell = None
            if nom_on:
                swarm.update_stats(tau_w_c, args.pool_w_ema)
        elif uses_vae:
            # annealed latent exploration noise: perturb z so snapping/blend explores different
            # codebook neighbors early; anneals to z-noise-end (default 0 -> commits by the end).
            znz = schedule_value(it, sched_iters, args.z_noise, args.z_noise_end,
                                 kind=args.z_noise_schedule)
            if args.z_noise_quench_frac < 1.0:            # release the floor for the final stretch
                q0 = args.z_noise_quench_frac * sched_iters
                if it >= q0:
                    znz *= max(0.0, 1.0 - (it - q0) / max(1.0, sched_iters - q0))
            if znz > 0:
                noise = torch.randn_like(z)
                noise = znz * noise
                if adapt_lut is not None:                               # density-equalized per-cell sigma
                    snap_c = torch.cdist(z, codebook).argmin(dim=1)      # current basin (pre-noise z)
                    noise = noise * adapt_lut[snap_c][:, None]           # scales eval AND committed slice
                commit_vec = noise if args.z_noise_commit > 0 else None  # random slice for SGLD commit
                if g_emp is not None and args.empty_noise_scale < 1.0:
                    nsc = (1.0 - g_emp * (1.0 - args.empty_noise_scale))   # (M,) empty cells -> quieter
                    noise = noise * nsc[:, None]
                    if commit_vec is not None:
                        commit_vec = commit_vec * nsc[:, None]            # empties don't walk either
                z_eff = z + noise
            else:
                z_eff = z
            dist = torch.cdist(z_eff, codebook)           # (M, N)
            # softmax blend of the k nearest REAL bitmaps, selected by the VAE metric so the
            # blend is among look-alikes (decoder-free). knn-smooth: the blend IS the forward
            # render (soft interp); anneal --knn-temp to sharpen toward one glyph and close the
            # soft->hard gap before the final snap. idx[:,0] is the nearest (commit + final snap).
            kt = schedule_value(it, sched_iters, args.knn_temp, args.knn_temp_end,
                                kind=args.knn_temp_schedule, warmup_frac=args.knn_temp_warmup_frac,
                                warmup_start=(args.knn_temp_warmup_start if args.knn_temp_warmup_start
                                              is not None else args.knn_temp_end),
                                end_frac=args.knn_temp_end_frac)
            vals, idxk = torch.topk(dist, args.knn_k, dim=1, largest=False)   # (M,k)
            if g_emp is not None and args.empty_temp_scale < 1.0:
                tau_c = (kt * (1.0 - g_emp * (1.0 - args.empty_temp_scale))).clamp_min(1e-6)
                blend_logits = -vals / tau_c[:, None]                          # (M,k) per-cell temp (#2)
            else:
                blend_logits = -vals / kt                                      # (M,k)
            cand_idx, cand_logits = idxk, blend_logits
            w = torch.softmax(cand_logits, dim=1)                             # (M, k)
            idx = cand_idx[torch.arange(M, device=device), w.argmax(1)]       # snap = argmax WEIGHT
            cell_embed = (w[:, :, None] * codebook[cand_idx]).sum(dim=1)      # (M,L) soft blend embed
            surr = (w[:, :, None, None] * char_bitmaps[cand_idx]).sum(dim=1)   # (M,CH,CW) white
            cell_img = surr                                                   # soft forward
            commit = ((z - codebook[idx].detach()) ** 2).mean()  # anchor the real param, not the noisy sample

        img = assemble(cell_img.view(1, GH, GW, CH, CW), GH, GW, CH, CW, args.row_gap, IMG_H, IMG_W)
        # `img` stays the monochrome STRUCTURE render: orient and the diagnostics keep reading it,
        # and it stays in the graph through z/W. `img_rgb` is the shipping image -- what recon and
        # CLIP actually judge once color is on.
        img_rgb = None
        if cell_rgb is not None:
            img_rgb = assemble_rgb(cell_rgb.view(1, GH, GW, CH, CW, 3), GH, GW, CH, CW,
                                   args.row_gap, IMG_H, IMG_W)                    # (1,H,W,3)
        if img_rgb is not None:
            recon = F.mse_loss(img_rgb[0], tgt_rgb)
            if args.multiscale_weight:
                r = F.avg_pool2d(img_rgb.permute(0, 3, 1, 2), 4, 2)
                t = F.avg_pool2d(tgt_rgb.permute(2, 0, 1)[None], 4, 2)
                recon = recon + args.multiscale_weight * F.mse_loss(r, t)
        else:
            recon = F.mse_loss(img[0], tgt_recon)
            if args.multiscale_weight:
                r = F.avg_pool2d(img, 4, 2)
                t = F.avg_pool2d(tgt_recon.unsqueeze(0).unsqueeze(0), 4, 2).squeeze(1)
                recon = recon + args.multiscale_weight * F.mse_loss(r, t)

        recon_mult = 1.0 if args.recon_jolt == 1.0 else schedule_value(
            it, sched_iters, args.recon_jolt, 1.0, end_frac=args.recon_jolt_frac)
        rw = args.recon_weight if args.recon_weight_end is None else schedule_value(
            it, sched_iters, args.recon_weight, args.recon_weight_end,
            kind=args.recon_weight_schedule, end_frac=args.recon_weight_end_frac)
        loss = rw * recon_mult * recon + args.commit * commit
        coord_l = None
        if coord is not None and cell_embed is not None:
            # coordinate loss: the learned position readout of the RENDERED embedding must
            # match the target centroid -- in LATENT units, per-axis gated by the line
            # normal and readout reliability, averaged over GATED cells (commit-scale)
            dy_c = (cell_embed @ coord["uy"]) - coord["ty_s"]
            dx_c = (cell_embed @ coord["ux"]) - coord["tx_s"]
            e2_c = coord["wy"] * coord["n"][:, 0].pow(2) * dy_c.pow(2) \
                + coord["wx"] * coord["n"][:, 1].pow(2) * dx_c.pow(2)
            coord_l = (coord["g"] * e2_c).sum() / coord["gsum"]
            cw_ = args.coord_weight if args.coord_weight_end is None else schedule_value(
                it, sched_iters, args.coord_weight, args.coord_weight_end,
                kind=args.coord_weight_schedule, end_frac=args.coord_weight_end_frac)
            loss = loss + cw_ * coord_l

        # Emptiness ink penalty (#1): push rendered ink -> 0 in confidently-empty cells, gated + scaled
        # by the SAME annealed g_emp as the per-cell temp, normalized so its magnitude is cell-count-stable.
        if g_emp is not None and args.empty_weight > 0:
            cell_ink = (1.0 - cell_img).mean(dim=(-1, -2))                         # (M,) rendered ink, white=1
            empty_pen = (g_emp * cell_ink).sum() / (g_emp.sum() + 1e-9)
            loss = loss + args.empty_weight * empty_pen

        # Blank-cell ink RENT (the anti-confetti term): per-cell price on rendered ink in
        # target-unsupported cells, g = (1 - t/tau)^2. Unlike the WINDOWED empty gate, this
        # is per-cell -- fringe cells next to structure (where confetti actually lives) pay
        # full rent if THEIR target region is blank. Soft economics: a supported lone accent
        # pays one small fee and survives; a colony pays n fees with nothing bidding for it.
        # Ramped in (start-frac) so exploration's transient blank-cell use stays free.
        rent = None
        if args.blank_weight > 0 and it >= args.blank_start_frac * sched_iters:
            r_ink = (1.0 - cell_img).mean(dim=(-1, -2))                            # (M,)
            rent = (blank_g * r_ink).sum() / (blank_g.sum() + 1e-9)
            loss = loss + args.blank_weight * rent

        # Pool blend-diversity penalty: D(c) = sum_k w_k*||B_k||^2 - ||render_c||^2 = per-pixel
        # weighted VARIANCE of the blend summed over the cell (0 iff one-hot or identical bitmaps).
        # Charges for incoherent composites -- the anti-blur guard that replaces knn-smooth's
        # geometric look-alike support. Ramped in LATE (start-frac) so shape-different arrivals can
        # audition during exploration; at commit time it forces one-glyph (or one-family) blends.
        div = torch.tensor(0.0, device=device)
        div_on = (w_pool is not None or sw is not None) and \
            (args.pool_diversity_weight > 0 or (args.pool_diversity_weight_end or 0) > 0)
        if div_on:
            if sw is not None:      # two-level variance from the swarm render (same units)
                D = sw["div"]
            else:
                D = (w_pool * bm_sq[pool.cand]).sum(1) - cell_img.reshape(M, -1).pow(2).sum(1)
            div = D.clamp_min(0).mean() / (CH * CW)
            dw = args.pool_diversity_weight if args.pool_diversity_weight_end is None else schedule_value(
                it, sched_iters, args.pool_diversity_weight, args.pool_diversity_weight_end,
                kind="cosine", warmup_frac=args.pool_diversity_weight_start_frac, warmup_start=0.0,
                end_frac=args.pool_diversity_weight_end_frac)
            loss = loss + dw * div
        if args.color_div_weight > 0 and sw is not None and sw.get("div_c") is not None:
            loss = loss + args.color_div_weight * sw["div_c"].mean() / (CH * CW)
        if (args.color_contrast_tv > 0 and swarm is not None
                and getattr(swarm, "k_contrast_on", False)):
            kf_ = swarm.k_contrast.view(GH, GW)
            loss = loss + args.color_contrast_tv * (
                (kf_[:, 1:] - kf_[:, :-1]).pow(2).mean() + (kf_[1:] - kf_[:-1]).pow(2).mean())

        # Orientation field: align the render's per-cell contour orientation to the target's,
        # gated by target coherence (abstains on flat/shading cells). Grad flows through the STE
        # surrogate to z. Uses the assembled images so the structure tensor crosses cell seams.
        orient = torch.tensor(0.0, device=device)
        if args.orient_weight > 0:
            orient, _ = O.orientation_match_loss(img[0], tgt, CH, CW, sigma=args.orient_sigma)
            loss = loss + args.orient_weight * orient

        # CLIPasso perceptual loss: match render's CLIP embedding to the target's.
        clip_l = torch.tensor(0.0, device=device)
        if clipper is not None:
            img_vip = None
            if swarm_mode and vip_cell is not None and args.perceptual != "dino":
                img_vip = assemble(vip_cell.view(1, GH, GW, CH, CW), GH, GW, CH, CW,
                                   args.row_gap, IMG_H, IMG_W)
            kw_c = {}
            clip_parts = None
            if img_vip is not None:
                kw_c = dict(alt_w=sw_snap_w, cons_w=sw_cons_w, return_parts=True)
            if content_bbox is not None:
                y0, y1, x0, x1 = content_bbox
                if img_vip is not None:
                    kw_c["alt"] = img_vip[0][y0:y1, x0:x1]
                if img_rgb is not None:
                    clip_l = clipper(img_rgb[0][y0:y1, x0:x1], tgt_rgb[y0:y1, x0:x1], **kw_c)
                else:
                    clip_l = clipper(img[0][y0:y1, x0:x1], tgt[y0:y1, x0:x1], **kw_c)
            else:
                if img_vip is not None:
                    kw_c["alt"] = img_vip[0]
                clip_l = clipper(img_rgb[0] if img_rgb is not None else img[0],
                                 tgt_rgb if img_rgb is not None else tgt, **kw_c)
            if isinstance(clip_l, tuple):
                clip_l, clip_parts = clip_l
            loss = loss + args.clip_weight * clip_l

        dense_l = None
        if dense_on:
            # dense geometric term: whole-image fully-conv sweep at fixed scale -- even
            # coverage (no crop lottery / center bias), conv-natural position tolerance
            if content_bbox is not None:
                y0, y1, x0, x1 = content_bbox
                _di = img_rgb[0] if img_rgb is not None else img[0]
                _dt = tgt_rgb if img_rgb is not None else tgt
                dense_l = clipper.dense_loss(_di[y0:y1, x0:x1], _dt[y0:y1, x0:x1],
                                             layers=dense_layers, max_side=args.clip_dense_res)
            else:
                dense_l = clipper.dense_loss(img_rgb[0] if img_rgb is not None else img[0],
                                             tgt_rgb if img_rgb is not None else tgt,
                                             layers=dense_layers, max_side=args.clip_dense_res)
            dw_ = args.clip_dense_weight if args.clip_dense_weight_end is None else \
                schedule_value(it, sched_iters, args.clip_dense_weight,
                               args.clip_dense_weight_end,
                               kind=args.clip_dense_weight_schedule,
                               end_frac=args.clip_dense_weight_end_frac)
            loss = loss + dw_ * dense_l

        # Pixel-grounded latent nudge: pull z toward the embedding of the top-k pixel-fitters,
        # softmax-weighted by per-char pixel MSE (annealable temp). Global -> escapes local basins.
        # (pool mode: no z, and its job -- global pixel-fit escape -- is nomination's; inactive.)
        nudge = torch.tensor(0.0, device=device)
        if args.pixel_nudge_weight > 0 and uses_vae and not pool_mode:
            tcells = cells_flat(tgt, GH, GW, CH, CW, args.row_gap)             # (M, P)
            pmse = (torch.cdist(tcells, bm_flat) ** 2) / (CH * CW)            # (M, N)
            kp = min(args.pixel_nudge_k, N)
            vals_p, idxp = torch.topk(pmse, kp, dim=1, largest=False)         # (M, kp)
            pt = args.pixel_nudge_temp if args.pixel_nudge_temp_end is None else \
                args.pixel_nudge_temp + (args.pixel_nudge_temp_end - args.pixel_nudge_temp) * (it / max(1, args.iters - 1))
            wpix = torch.softmax(-vals_p / pt, dim=1)                         # (M, kp)
            target_embed = (wpix[:, :, None] * codebook[idxp]).sum(dim=1)     # (M, L)
            if swarm_mode:   # per-slot pull, v-weighted: mainly steers the winner (losers barely)
                nudge = (((swarm.z - target_embed[:, None, :].detach()) ** 2).mean(-1)
                         * sw["v"].detach()).sum(1).mean()
            else:
                nudge = ((z - target_embed.detach()) ** 2).mean()
            pnw = args.pixel_nudge_weight if args.pixel_nudge_weight_end is None else schedule_value(
                it, sched_iters, args.pixel_nudge_weight, args.pixel_nudge_weight_end,
                kind=args.pixel_nudge_weight_schedule, end_frac=args.pixel_nudge_weight_end_frac)
            loss = loss + pnw * nudge

        # pool/swarm have no single z; the blend embedding e_c = sum_k w_k*e_k substitutes
        # (grad flows to logits / to W and every slot's z).
        zl = cell_embed if (pool_mode or swarm_mode) else z

        # Non-local semantic affinity: same Laplacian pull over the precomputed CNN-gated-by-DINO edges.
        # Weight ramps up (1-cos) so the early phase explores freely before consistency is enforced.
        affin = torch.tensor(0.0, device=device)
        if affin_idx is not None:
            sa, da, wa = affin_idx
            za = zl
            if swarm_mode and args.swarm_affinity_winner_frac < 1.0 \
                    and it >= args.swarm_affinity_winner_frac * sched_iters:
                # arbitration has decided: retarget the pull to the ARGMAX slot's z directly
                # (knn-smooth affinity semantics restored for the refinement phase)
                kwin = swarm.W.detach().masked_fill(swarm.free, float("-inf")).argmax(dim=1)
                za = swarm.z[torch.arange(M, device=device), kwin]
            edge = wa * ((za[sa] - za[da]) ** 2).sum(1)
            affin = edge.sum()
            aw = args.affinity_weight if args.affinity_weight_end is None else schedule_value(
                it, sched_iters, args.affinity_weight, args.affinity_weight_end,
                kind=args.affinity_weight_schedule, warmup_frac=args.affinity_weight_start_frac,
                end_frac=args.affinity_weight_end_frac)
            loss = loss + aw * affin

        # Diagnostic: keep the render's gradient from the REAL backward (free). recon grad is cheap;
        # CLIP grad = total@img - recon - orient (consistency/sym/commit act on z/cell_img, not img).
        if nom_on:
            img.retain_grad()   # the nomination signal is dL/d(render), free from the real backward
            if img_rgb is not None:
                img_rgb.retain_grad()   # color: the real signal lives here (img carries no recon/clip)
        if diag_on:
            img.retain_grad()
            _dtarget = img_rgb if img_rgb is not None else img
            if img_rgb is not None:
                img_rgb.retain_grad()
            g_recon_unit = torch.autograd.grad(recon, _dtarget, retain_graph=True)[0][0]
            # orient acts on the GRAY structure render, a sibling branch of img_rgb -- so under
            # color it contributes nothing to img_rgb.grad and must NOT be subtracted out
            g_orient_unit = (torch.autograd.grad(orient, img, retain_graph=True)[0][0]
                             if (args.orient_weight > 0 and img_rgb is None) else None)

        if getattr(swarm, "_anchor_w", 0.0) > 0:
            import math as _math
            # scheduled anchor weight: cosine-ramp v0 -> v1 over [f0, f1] of the run,
            # held flat outside the window. v1 = --color-anchor, v0 = --color-anchor-start.
            _v1 = swarm._anchor_w
            _v0 = args.color_anchor_start
            _f0 = args.color_anchor_start_frac
            _f1 = args.color_anchor_end_frac
            _prog = min(1.0, max(0.0, (it / max(1, sched_iters) - _f0) / max(1e-6, _f1 - _f0)))
            _ease = 0.5 * (1.0 - _math.cos(_math.pi * _prog))
            _aw = _v0 + (_v1 - _v0) * _ease
            if _aw > 0 and (it % 10 == 0 or swarm._anchor_fg is None):
                with torch.no_grad():
                    _gi = swarm.snap(codebook)
                    _mask = (1.0 - char_bitmaps[_gi]).reshape(M, -1)
                    _bw = swarm._anchor_bgdist[_gi]
                    from unicasso.engine.color import fit_fg_bg
                    _af, _ab = fit_fg_bg(swarm.tgt_cell_rgb, _mask, 1e-3, bg_w=_bw)
                    _mc = swarm._anchor_mc
                    if _mc > 0:
                        _lf = 0.299 * _af[:, 0] + 0.587 * _af[:, 1] + 0.114 * _af[:, 2]
                        _lb = 0.299 * _ab[:, 0] + 0.587 * _ab[:, 1] + 0.114 * _ab[:, 2]
                        _gap = _lf - _lb
                        _push = torch.where(_gap <= 0, -1.0, 1.0) * (_mc - _gap.abs()).clamp_min(0)
                        _af = _af + _push[:, None]
                    swarm._anchor_fg, swarm._anchor_bg = _af, _ab
            if _aw > 0 and swarm._anchor_fg is not None:
                _la = (swarm.fg - swarm._anchor_fg[:, None, :]).pow(2).mean() \
                    + (swarm.bg - swarm._anchor_bg[:, None, :]).pow(2).mean()
                loss = loss + _aw * _la
                if args.progress_every and it % args.progress_every == 0:
                    # observability: is the pull actually closing the gap?
                    with torch.no_grad():
                        # per-CELL drift; degeneracy is a TAIL phenomenon, so report
                        # the tail (p95/max) alongside the mean, not the mean alone
                        _pc = 0.5 * ((swarm.fg.mean(1) - swarm._anchor_fg).abs().mean(-1)
                                     + (swarm.bg.mean(1) - swarm._anchor_bg).abs().mean(-1))
                        _d, _p95, _mx = _pc.mean(), _pc.quantile(0.95), _pc.max()
                        _nbad = int((_pc > 0.25).sum())
                    print(f"  anchor: w={_aw:.4g} (v0={_v0:g} -> v1={_v1:g} over "
                          f"[{_f0:g},{_f1:g}])  |fg/bg - fit| mean={float(_d):.4f} "
                          f"p95={float(_p95):.4f} max={float(_mx):.4f} "
                          f"n>0.25={_nbad}  L2={float(_la):.5f}", flush=True)
        loss.backward()
        opt.step()
        if swarm_mode and args.color and not args.color_pin:
            swarm.fg.data.clamp_(0.0, 1.0)      # no dead zone outside the STE clamp
            swarm.bg.data.clamp_(0.0, 1.0)
        if pool_mode and args.pool_logit_cap > 0:
            pool.clamp_spread(args.pool_logit_cap)   # before nominate, so arrivals init in capped units
        if swarm_mode and args.swarm_w_cap > 0:
            swarm.clamp_w_spread(args.swarm_w_cap)   # every step: all live slots stay visible/alive
            if getattr(swarm, "empty_pin_mask", None) is not None:
                # re-assert the space slot's floor AFTER the spread projection (decay counter);
                # noise is zeroed for these cells, so the space slot never leaves the seed point
                pm_ = swarm.empty_pin_mask
                idx_ = pm_.nonzero().squeeze(1)
                ksp_ = (swarm.z.data[idx_] - codebook[swarm.space_idx][None, None, :]) \
                    .norm(dim=-1).argmin(dim=1)
                cur_ = swarm.W.data[idx_, ksp_]
                swarm.W.data[idx_, ksp_] = torch.maximum(
                    cur_, torch.full_like(cur_, args.swarm_empty_pin))
        if swarm_mode and it in sw_seams:            # elite purge at the reheat seam
            n_p, ps_c, ps_k = swarm.purge_to_winner(codebook, it, avoid=swarm.nom_block,
                                                    keep=args.swarm_purge_keep,
                                                    reseed=args.swarm_purge_reseed)
            n_s = int(ps_c.numel())
            adam_reset_slots(ps_c, ps_k)
            print(f"\n  elite purge @ it {it}: {n_p} slots freed, {n_s} wildcards seeded")
            if prec is not None:
                prec["merge_it"].append(it)          # logged in the merge stream, d=-1 marks purge
                prec["merge_cell"].append(-1)
                prec["merge_d"].append(-1.0)
                prec["merge_vl"].append(float(n_p))
        if commit_vec is not None:             # SGLD: persist the random slice so z actually random-walks
            (swarm.z if swarm_mode else z).data.add_(commit_vec, alpha=args.z_noise_commit)
        if swarm_mode and prec is not None:
            swarm.track_step(codebook)         # per-slot flips + step-length EMA (one cdist)
        if nom_on and (img.grad is not None or (img_rgb is not None and img_rgb.grad is not None)):
            # per-cell render gradient (row_gap 0, enforced at parse) -> EMA; nomination rounds swap
            # gradient-wanted outsiders into the pool (evict weakest post-grace slot, tabu the loser).
            if img_rgb is not None and img_rgb.grad is not None:
                # COLOR chain rule (mandatory -- without it S = g_ema @ bm_flat.t() is meaningless):
                # d(cell_rgb)/d(slot_img) = v_k * (bg_k - fg_k), so the RGB render gradient must be
                # PROJECTED onto the dominant slot's color-contrast direction to become a gradient
                # w.r.t. ink, which is the space bm_flat lives in. Sign falls out for free, so
                # light-on-dark cells nominate correctly -- something the gray pipeline cannot do.
                gr = img_rgb.grad[0][:GH * CH, :GW * CW].reshape(GH, CH, GW, CW, 3) \
                    .permute(0, 2, 1, 3, 4).reshape(M, CH * CW, 3)
                nh = pool if pool_mode else swarm
                k1_ = nh.W.data.masked_fill(nh.free, float("-inf")).argmax(1) if swarm_mode else None
                if k1_ is not None:
                    dcol_ = (nh.bg.data - nh.fg.data)[nh._ar, k1_]            # (M,3)
                else:
                    dcol_ = torch.tensor([-1.0, -1.0, -1.0], device=device).expand(M, 3)
                g_cells = (gr * dcol_[:, None, :]).sum(-1)                     # (M,P)
            else:
                g_cells = img.grad[0][:GH * CH, :GW * CW].reshape(GH, CH, GW, CW) \
                    .permute(0, 2, 1, 3).reshape(M, CH * CW)
            nom_host = pool if pool_mode else swarm
            nom_host.update_grad_ema(g_cells, pool_grad_decay)
            if swarm_mode:
                swarm.update_zgrad_ema(pool_grad_decay)   # per-slot gradient legibility diagnostic
            nom_round = it % args.pool_nominate_rate == 0
            probe_round = (live_probe and args.pool_probe_rate > 0
                           and it % args.pool_probe_rate == 0)
            if nom_lo <= it < nom_hi and (nom_round or probe_round) \
                    and not (swarm_mode and args.swarm_phase_gates and sw_hardening):
                round_i = it // args.pool_nominate_rate    # grad/join join only every Nth round
                chans_now = tuple(c for c in pool_chans
                                  if (c != "grad" or round_i % max(1, args.pool_grad_every) == 0)
                                  and (c != "join" or round_i % max(1, args.pool_join_every) == 0))
                kw = dict(margin=args.pool_nominate_margin, grace=args.pool_grace,
                          evict_below=args.pool_evict_below, channels=chans_now,
                          neighbor_margin=args.pool_neighbor_margin,
                          affinity_margin=args.pool_affinity_margin,
                          latent_margin=args.pool_latent_margin,
                          latent_floor=args.pool_latent_floor,
                          ports_floor=args.pool_ports_floor,
                          ports_nb_weight=args.pool_ports_nb_weight)
                if swarm_mode:
                    # SWARM round: merge co-located slots (frees respawn budget), then channels
                    # propose; near-slot nominees become BOOSTS (free, unmeasured dedup-by-boost);
                    # far nominees get probe-measured with GRADED admission -- clearly better ->
                    # strong birth near the W top, measured ~neutral -> minority-weight lottery
                    # ticket (may exist and wait for its neighbors), clearly worse -> tabu.
                    if args.swarm_blank_close > 0:
                        # dynamic closure: cells stably snapped to a blank stop consuming the
                        # search/measurement budget. REVERSIBLE: an ink snap resets the streak,
                        # so a cell the gradient pulls back to ink regains its nomination path
                        # next round. INK-GATED: cells with real target ink are never closed --
                        # an idling-blank contested cell is exactly who needs search support.
                        if sw_static_closed is None:
                            sw_static_closed = (~swarm.search_mask if swarm.search_mask is not None
                                                else torch.zeros(M, dtype=torch.bool, device=device))
                            sw_ink_c = 1.0 - cells_flat(target, GH, GW, CH, CW, args.row_gap).mean(1)
                            sw_closable = sw_ink_c < args.swarm_blank_close_ink
                        if nom_round:
                            sb_ = torch.isin(snap_indices(), sw_blank_ids)
                            sw_blank_streak = torch.where(sb_, sw_blank_streak + 1,
                                                          torch.zeros_like(sw_blank_streak))
                        closable_now = sw_closable
                        if args.swarm_draw_guard > 0 and swarm.g_ema is not None:
                            # draw-guard: CLIP's gradient EMA consistently asks for ink here
                            # (mean g_ema > 0 = darker-wanted) -> never close, whatever the streak
                            closable_now = sw_closable & (swarm.g_ema.mean(1) <= args.swarm_draw_guard)
                        swarm.search_mask = ~(sw_static_closed
                                              | ((sw_blank_streak >= args.swarm_blank_close)
                                                 & closable_now))
                    if nom_round:
                        mc_, md_, mv_ = swarm.merge(args._swarm_merge_eps_abs, it)
                    else:
                        mc_ = md_ = mv_ = torch.zeros(0, device=device)
                    n_merges_r = int(mc_.numel())
                    n_merges += n_merges_r
                    blur_T = None
                    if "blend" in chans_now:   # the cell's self-drawn request: clean soft render,
                        with torch.no_grad():  # binarized to a presence map (see --pool-blend-thresh)
                            sw_cl = swarm.render(swarm.z, codebook, bm_flat, bm_sq,
                                                 args.swarm_knn_k, tau_c, tau_w_c,
                                                 temp_lut=sw_temp_lut, height=sw_height,
                                                 tone_w=args.swarm_tone_weight)
                            inkr_ = 1.0 - sw_cl["cell"]
                            thr_ = torch.full_like(inkr_[:, :1], args.pool_blend_thresh)
                            if args.pool_blend_rel_thresh > 0:   # faint cells: relative presence
                                thr_ = torch.minimum(
                                    thr_, (args.pool_blend_rel_thresh
                                           * inkr_.max(dim=1, keepdim=True).values).clamp_min(0.02))
                            blur_T = (inkr_ > thr_).float()
                    P = swarm.propose_full(bm_flat, codebook, it, blur_T=blur_T,
                                           per_chan=args.pool_probe_per_chan,
                                           blend_margin=args.pool_blend_margin,
                                           blend_nb_weight=args.pool_blend_nb_weight,
                                           join_floor=args.pool_join_floor,
                                           blend_resid=args.pool_blend_resid,
                                           join_dangling=args.pool_join_dangling,
                                           join_centroid=args.pool_join_centroid,
                                           join_coord_sigma=args.pool_join_coord_sigma,
                                           pixel_margin=args.pool_pixel_margin,
                                           pixel_gate=args.pool_pixel_gate,
                                           pixel_tone_ink=args.pool_pixel_tone_ink, **kw)
                    sc = s_in = s_out = s_ch = torch.zeros(0, dtype=torch.long, device=device)
                    s_kind = torch.zeros(0, dtype=torch.bool, device=device)
                    gate_ct = (P["gate_counts"] if P is not None
                               else torch.zeros(9, dtype=torch.long))
                    if prec is not None and n_merges_r:
                        prec["merge_it"].extend([it] * n_merges_r)
                        prec["merge_cell"].extend(mc_.cpu().tolist())
                        prec["merge_d"].extend(md_.cpu().tolist())
                        prec["merge_vl"].extend(mv_.cpu().tolist())
                    if P is not None and nom_round:
                        bc_, bk_, bg_, bstat = swarm.boost_from_proposals(
                            P, codebook, it, radius=args._swarm_boost_radius_abs,
                            amount=args.swarm_boost, nudge=args.swarm_boost_nudge,
                            cooldown=args.swarm_boost_cooldown,
                            exclude_chans=tuple(
                                c for c in args.swarm_boost_exclude.split(",") if c))
                        n_boosts += int(bc_.numel())
                        if prec is not None and bc_.numel():
                            prec["boost_it"].extend([it] * int(bc_.numel()))
                            prec["boost_cell"].extend(bc_.cpu().tolist())
                            prec["boost_slot"].extend(bk_.cpu().tolist())
                            prec["boost_g"].extend(bg_.cpu().tolist())
                        if prec is not None:   # round economics (audition throughput diagnostics)
                            open_m = (swarm.search_mask if swarm.search_mask is not None
                                      else torch.ones(M, dtype=torch.bool, device=device))
                            prec["econ_it"].append(it)
                            prec["econ"].append([
                                int((P["valid"].any(dim=1) & open_m).sum()),
                                int(P["ok_cell"].sum()),
                                int((open_m & (swarm.age.min(dim=1).values
                                               < args.pool_grace)).sum()),
                                int((open_m & swarm.free.any(dim=1)).sum()),
                                bstat["near"], bstat["nudge_win"], bstat["cool"],
                                int(bc_.numel())])
                    if P is not None:
                        if live_probe:
                            cand_cells = (P["ok_cell"] & P["valid"].any(dim=1)).nonzero().flatten()
                            # LEAST-RECENTLY-PROBED first (deterministic rolling sweep; random
                            # tiebreak) -> every active cell is guaranteed a check within
                            # ~active/(cells-per-batch) rounds instead of coupon-collector luck
                            ordr = torch.argsort(probe_last[cand_cells] * 1024
                                                 + torch.randint(0, 1024, (cand_cells.numel(),),
                                                                 device=device))
                            occ, sel_cells = [], []
                            for cell in cand_cells[ordr].tolist():
                                r0, c0 = divmod(cell, GW)
                                if all(max(abs(r0 - r1), abs(c0 - c1)) >= args.pool_probe_spacing
                                       for r1, c1 in occ):
                                    sel_cells.append(cell)
                                    occ.append((r0, c0))
                            depth = max(1, args.pool_probe_batches)
                            ttl = args.pool_probe_memo_ttl
                            best, rejected = {}, []   # best[cell] = (rank, dlt, g, chn)
                            def _rank(dlt):
                                return (0 if dlt < -args.pool_probe_margin else
                                        1 if (args.swarm_lottery_margin > 0
                                              and dlt < args.swarm_lottery_margin) else 2)
                            cellprops = {}
                            for cell in sel_cells:
                                lst, seen = [], set()
                                cache_c = probe_cache.get(cell, {}) if ttl > 0 else {}
                                for j in range(P["props"].shape[0]):
                                    if P["valid"][cell, j]:
                                        g = int(P["props"][j, cell])
                                        if g in seen:
                                            continue
                                        seen.add(g)
                                        ent = cache_c.get(g)
                                        if ent is not None and it - ent[1] <= ttl:
                                            rk = _rank(ent[0])   # MEMOIZED score joins for free
                                            if rk < 2 and (cell not in best
                                                           or (rk, ent[0]) < best[cell][:2]):
                                                best[cell] = (rk, ent[0], g, int(P["chans"][j]))
                                        else:
                                            lst.append((g, int(P["chans"][j])))
                                cellprops[cell] = lst[:depth]
                                probe_last[cell] = it
                            if any(cellprops.values()):   # all-cached rounds skip the forwards
                                with torch.no_grad():
                                    base = snap_indices()
                                    img0 = probe_render(base)
                                    E0 = probe_err_maps(img0)
                                for b in range(depth):
                                    swaps = [(cell, cellprops[cell][b]) for cell in sel_cells
                                             if len(cellprops[cell]) > b]
                                    if not swaps:
                                        break
                                    with torch.no_grad():
                                        s2 = base.clone()
                                        for cell, (g, _) in swaps:
                                            s2[cell] = g
                                        img1 = probe_render(s2)
                                        E1 = probe_err_maps(img1)
                                        for cell, (g, chn) in swaps:
                                            dlt = probe_delta_at(E1, E0, img1, img0, cell,
                                                                 probe_tgt, rw)
                                            if prec is not None:
                                                prec["probe_it"].append(it)
                                                prec["probe_cell"].append(cell)
                                                prec["probe_in"].append(g)
                                                prec["probe_delta"].append(dlt)
                                                prec["probe_ch"].append(chn)
                                            if args.pool_probe_memo_ttl > 0:
                                                probe_cache.setdefault(cell, {})[g] = (dlt, it)
                                            rank = _rank(dlt)
                                            if prec is not None:
                                                prec["probe_accept"].append(2 - rank)  # 2=strong 1=lottery 0=reject
                                            if rank < 2:
                                                if cell not in best or (rank, dlt) < best[cell][:2]:
                                                    best[cell] = (rank, dlt, g, chn)
                                            else:
                                                rejected.append((cell, g))
                            n_probe_acc += len(best)
                            n_probe_rej += len(rejected)
                            if best:
                                bcs = torch.tensor(sorted(best.keys()), device=device)
                                bgs = torch.tensor([best[int(c_)][2] for c_ in bcs], device=device)
                                bchs = torch.tensor([best[int(c_)][3] for c_ in bcs], device=device)
                                strong_ = torch.tensor([best[int(c_)][0] == 0 for c_ in bcs],
                                                       device=device)
                                swarm.birth(bcs, P["slot"][bcs], bgs, P["out_g"][bcs], strong_,
                                            codebook, it, tabu=args.pool_tabu,
                                            strong_gap=args.swarm_strong_gap)
                                adam_reset_slots(bcs, P["slot"][bcs])
                                for c_ in best:
                                    probe_cache.pop(int(c_), None)
                                n_lottery += int((~strong_).sum())
                                sc, s_in, s_ch, s_kind = bcs, bgs, bchs, strong_
                                s_out = P["out_g"][bcs]
                                if prec is not None:
                                    prec["swap_slot"].extend(P["slot"][bcs].cpu().tolist())
                            if rejected:
                                rc = torch.tensor([c_ for c_, _ in rejected], device=device)
                                rg = torch.tensor([g_ for _, g_ in rejected], device=device)
                                swarm.tabu_add(rc, rg, it, tabu=args.pool_tabu)
                        else:
                            # unmeasured swarm round: first valid channel's proposal -> strong birth
                            has_prop = P["valid"].any(dim=1)
                            prio = torch.arange(P["props"].shape[0], 0, -1, device=device,
                                                dtype=torch.float32)
                            pick = (P["valid"].float() * prio).argmax(dim=1)
                            nom_g = P["props"][pick, torch.arange(M, device=device)].clamp_min(0)
                            okc = has_prop & P["ok_cell"]
                            if okc.any():
                                c_ = okc.nonzero().flatten()
                                strong_ = torch.ones(c_.numel(), dtype=torch.bool, device=device)
                                swarm.birth(c_, P["slot"][c_], nom_g[c_], P["out_g"][c_], strong_,
                                            codebook, it, tabu=args.pool_tabu,
                                            strong_gap=args.swarm_strong_gap)
                                adam_reset_slots(c_, P["slot"][c_])
                                sc, s_in, s_out = c_, nom_g[c_], P["out_g"][c_]
                                s_ch, s_kind = P["chans"][pick[c_]], strong_
                                if prec is not None:
                                    prec["swap_slot"].extend(P["slot"][c_].cpu().tolist())
                    if prec is not None and sc.numel():
                        prec["swap_kind"].extend([int(k_) for k_ in s_kind.cpu()])
                elif live_probe:
                    # MEASURED MULTI-CHANNEL round: every channel's proposal for every cell is on
                    # the table; a spaced random cell subset gets up to --pool-probe-batches of its
                    # (distinct, priority-ordered) channel proposals hard-swap measured -- batch b
                    # carries each cell's b-th proposal -- and the BEST measured improvement per
                    # cell is admitted. Channel priority = queue order only, never preemption.
                    # Measured failures -> tabu (channels stop re-pitching them).
                    P = pool.propose_full(bm_flat, codebook, it,
                                          per_chan=args.pool_probe_per_chan, **kw)
                    sc = s_in = s_out = s_ch = torch.zeros(0, dtype=torch.long, device=device)
                    gate_ct = (P["gate_counts"] if P is not None
                               else torch.zeros(6, dtype=torch.long))
                    if P is not None:
                        cand_cells = (P["ok_cell"] & P["valid"].any(dim=1)).nonzero().flatten()
                        occ, sel_cells = [], []
                        for cell in cand_cells[torch.randperm(cand_cells.numel())].tolist():
                            r0, c0 = divmod(cell, GW)
                            if all(max(abs(r0 - r1), abs(c0 - c1)) >= args.pool_probe_spacing
                                   for r1, c1 in occ):
                                sel_cells.append(cell)
                                occ.append((r0, c0))
                        depth = max(1, args.pool_probe_batches)
                        cellprops = {}
                        for cell in sel_cells:
                            lst, seen = [], set()
                            for j in range(P["props"].shape[0]):
                                if P["valid"][cell, j]:
                                    g = int(P["props"][j, cell])
                                    if g not in seen:
                                        seen.add(g)
                                        lst.append((g, int(P["chans"][j])))
                            cellprops[cell] = lst[:depth]
                        best, rejected = {}, []
                        if sel_cells:
                            with torch.no_grad():
                                base = snap_indices()
                                img0 = probe_render(base)
                                E0 = probe_err_maps(img0)
                            for b in range(depth):
                                swaps = [(cell, cellprops[cell][b]) for cell in sel_cells
                                         if len(cellprops[cell]) > b]
                                if not swaps:
                                    break
                                with torch.no_grad():
                                    s2 = base.clone()
                                    for cell, (g, _) in swaps:
                                        s2[cell] = g
                                    img1 = probe_render(s2)
                                    E1 = probe_err_maps(img1)
                                    for cell, (g, chn) in swaps:
                                        dlt = probe_delta_at(E1, E0, img1, img0, cell,
                                                             probe_tgt, rw)
                                        if prec is not None:
                                            prec["probe_it"].append(it)
                                            prec["probe_cell"].append(cell)
                                            prec["probe_in"].append(g)
                                            prec["probe_delta"].append(dlt)
                                            prec["probe_accept"].append(
                                                int(dlt < -args.pool_probe_margin))
                                            prec["probe_ch"].append(chn)
                                        if dlt < -args.pool_probe_margin:
                                            if cell not in best or dlt < best[cell][0]:
                                                best[cell] = (dlt, g, chn)
                                        else:
                                            rejected.append((cell, g))
                        n_probe_acc += len(best)
                        n_probe_rej += len(rejected)
                        if best:
                            bc = torch.tensor(sorted(best.keys()), device=device)
                            bg = torch.tensor([best[int(c_)][1] for c_ in bc], device=device)
                            bch = torch.tensor([best[int(c_)][2] for c_ in bc], device=device)
                            pool.apply_swaps(bc, P["slot"][bc], bg, P["out_g"][bc], it,
                                             tabu=args.pool_tabu,
                                             rank_lo=args.pool_arrival_rank_lo,
                                             rank_hi=args.pool_arrival_rank_hi)
                            sc, s_in, s_ch = bc, bg, bch
                            s_out = P["out_g"][bc]
                        if rejected:
                            rc = torch.tensor([c_ for c_, _ in rejected], device=device)
                            rg = torch.tensor([g_ for _, g_ in rejected], device=device)
                            pool.tabu_add(rc, rg, it, tabu=args.pool_tabu)
                else:
                    sc, s_in, s_out, s_ch, gate_ct, s_slot = pool.nominate(
                        bm_flat, codebook, it, tabu=args.pool_tabu,
                        rank_lo=args.pool_arrival_rank_lo, rank_hi=args.pool_arrival_rank_hi,
                        **kw)
                n_swaps_total += int(sc.numel())
                if sc.numel():
                    n_swaps_ch += torch.bincount(s_ch, minlength=9).cpu()
                if prec is not None:
                    prec["round_it"].append(it)
                    prec["gate_pass"].append(gate_ct.cpu().numpy())      # (3,) by channel code
                if prec is not None and sc.numel():
                    prec["swap_it"].extend([it] * int(sc.numel()))
                    prec["swap_cell"].extend(sc.cpu().tolist())
                    prec["swap_in"].extend(s_in.cpu().tolist())
                    prec["swap_out"].extend(s_out.cpu().tolist())
                    prec["swap_ch"].extend(s_ch.cpu().tolist())

        # Early-stop probe. Measured HERE (after opt.step + the nomination round, so the
        # snap is this iteration's final state) but the loop is broken at the BOTTOM, so
        # the curve/progress/record blocks still capture the iteration we stop on.
        if es_div_on:
            es_div_hist.append(float(div))          # already synced for the postfix below
        # PLACEMENT-phase plateau detection -- only runs until refine is armed (k does not
        # move the snap, so this signal is meaningless during the colour phase).
        if es_on and es_refine_start is None and it >= es_floor \
                and it % args.early_stop_every == 0:
            with torch.no_grad():
                es_now = snap_indices()
            if es_prev is not None:
                es_stable = es_stable + 1 if int((es_now != es_prev).sum()) <= es_tol else 0
            es_prev = es_now
            if es_div_on:
                # plateau = the last window shed less than div-tol of its own level versus
                # the window before it. Relative, so it reads the same at any diversity
                # weight; a run still committing sheds far more than a few percent.
                wdv = args.early_stop_div_window
                if len(es_div_hist) >= 2 * wdv:
                    cur_d = sum(es_div_hist[-wdv:]) / wdv
                    prev_d = sum(es_div_hist[-2 * wdv:-wdv]) / wdv
                    es_div_flat = (prev_d - cur_d) <= args.early_stop_div_tol * max(prev_d, 1e-12)
                else:
                    es_div_flat = False
            if es_stable >= args.early_stop_patience and es_div_flat:
                if es_refine:
                    es_refine_start = it     # placement done -> arm the colour-contrast budget
                    print(f"\nearly stop: placement converged at iter {it}; arming "
                          f"colour-contrast refinement for {args.color_contrast_iters} iters")
                else:
                    es_break = True
        # REFINE phase: once armed, run exactly --color-contrast-iters more, then stop
        if es_refine_start is not None and it >= es_refine_start + args.color_contrast_iters:
            es_break = True

        if diag_on:
            _gsrc = img_rgb if img_rgb is not None else img
            g_clip = _gsrc.grad[0] - rw * recon_mult * g_recon_unit
            if g_orient_unit is not None:
                g_clip = g_clip - args.orient_weight * g_orient_unit
            conflict, clipmag = clip_recon_grad_maps(g_clip, g_recon_unit, GH, GW, CH, CW)
            ema_conflict = conflict if ema_conflict is None else ema_decay * ema_conflict + (1 - ema_decay) * conflict
            ema_clipmag = clipmag if ema_clipmag is None else ema_decay * ema_clipmag + (1 - ema_decay) * clipmag
        post = dict(recon=f"{recon.item():.4f}")
        if coord_l is not None:
            post["coord"] = f"{float(coord_l):.4f}"
        if uses_vae and not pool_mode:
            post["commit"] = f"{commit.item():.3f}"
        if pool_mode:
            post["pnz"] = f"{pnz:.3f}"
            if div_on:
                post["div"] = f"{float(div):.4f}"
            if nom_on:
                post["nom"] = str(n_swaps_total)
        if swarm_mode:
            post["wt"] = f"{wt:.3f}"
            if div_on:
                post["div"] = f"{float(div):.4f}"
            if nom_on:
                post["nom"] = str(n_swaps_total)
                post["boost"] = str(n_boosts)
                post["merge"] = str(n_merges)
        if args.pixel_nudge_weight > 0:
            post["nudge"] = f"{float(nudge):.3f}"
        if args.empty_weight > 0:
            post["empty"] = f"{float(empty_pen):.4f}"
        if rent is not None:
            post["rent"] = f"{float(rent):.4f}"
        if args.orient_weight > 0:
            post["orient"] = f"{float(orient):.3f}"
        if affin_idx is not None:
            post["affin"] = f"{float(affin):.4f}"
        if clipper is not None:
            post["clip"] = f"{float(clip_l):.3f}"
        if dense_l is not None:
            post["cdns"] = f"{float(dense_l):.3f}"
        if uses_vae and not pool_mode and args.z_noise > 0:
            post["znz"] = f"{znz:.3f}"
        pbar.set_postfix(post)

        if args.progress_every and (it % args.progress_every == 0 or it == args.iters - 1):
            pdir = os.path.splitext(args.output)[0] + "_progress"
            os.makedirs(pdir, exist_ok=True)
            if args.color and swarm_mode and getattr(swarm, "color_on", False):
                frame = Image.fromarray(render_hard_color()[1], mode="RGB")
            else:
                with torch.no_grad():
                    ip = snap_indices()
                    op = assemble(char_bitmaps[ip].view(1, GH, GW, CH, CW), GH, GW, CH, CW,
                                  args.row_gap, IMG_H, IMG_W)[0]
                frame = Image.fromarray((op.clamp(0, 1) * 255).byte().cpu().numpy(), mode="L")
            frame.save(os.path.join(pdir, f"iter_{it:05d}.png"))
        if args.loss_curve or diag_on:
            curve["it"].append(it)
            curve["loss"].append(float(loss)); curve["recon"].append(float(recon))
            if uses_vae and not pool_mode:
                curve["commit"].append(float(commit))
            if pool_mode:
                curve["pool_noise"].append(float(pnz))
                if div_on:
                    curve["div"].append(float(div))
            if swarm_mode:
                curve["w_temp"].append(float(wt))
                if args.swarm_w_decay_end is not None:
                    curve["w_decay"].append(wd_now)
                if div_on:
                    curve["div"].append(float(div))
            if clipper is not None:
                curve["clip"].append(float(clip_l))
            if dense_l is not None:
                curve["clip_dense"].append(float(dense_l))
                if clip_parts is not None:   # decomposed: match / VIP(snap-vs-tgt) / consolidation
                    curve["clip_main"].append(clip_parts[0])
                    curve["clip_snap"].append(clip_parts[1])
                    curve["clip_cons"].append(clip_parts[2])
            if args.orient_weight > 0:
                curve["orient"].append(float(orient))
            if affin_idx is not None:
                curve["affin"].append(float(affin))
            if args.pixel_nudge_weight > 0:
                curve["nudge"].append(float(nudge))
            if coord_l is not None:
                curve["coord"].append(float(coord_l))
            if args.empty_weight > 0:
                curve["empty"].append(float(empty_pen))
            if args.blank_weight > 0:
                curve["rent"].append(0.0 if rent is None else float(rent))
            if uses_vae:
                curve["knn_temp"].append(float(kt))
            if uses_vae and not pool_mode and args.z_noise > 0:
                curve["z_noise"].append(float(znz))
            if it % hard_every == 0 or it == args.iters - 1:
                with torch.no_grad():
                    ih = snap_indices()
                    oh = assemble(char_bitmaps[ih].view(1, GH, GW, CH, CW), GH, GW, CH, CW,
                                  args.row_gap, IMG_H, IMG_W)[0]
                    hr = F.mse_loss(oh, tgt)
                    if args.multiscale_weight:
                        hr = hr + args.multiscale_weight * F.mse_loss(
                            F.avg_pool2d(oh[None, None], 4, 2), F.avg_pool2d(tgt[None, None], 4, 2))
                curve["hard_it"].append(it); curve["hard_recon"].append(float(hr))

        if diag_on and it % args.clip_diag_interval == 0 and ema_conflict is not None:
            if pool_mode:   # no cdist snap; show UNCOMMITMENT (1 - clean top-1 weight) instead
                with torch.no_grad():
                    resid = (1.0 - pool.weights(tau_c).max(dim=1).values).view(GH, GW)
            elif swarm_mode:   # across-slot uncommitment (1 - clean top-1 v)
                with torch.no_grad():
                    resid = (1.0 - swarm.weights(tau_w_c, bias=swarm.h_bias).max(dim=1).values).view(GH, GW)
            else:
                resid = torch.cdist(z, codebook).min(1).values.view(GH, GW)
            save_clip_diag((img_rgb[0] if img_rgb is not None else img[0]).detach(),
                           ema_conflict, ema_clipmag, resid, curve, it,
                           args.iters, f"{args.clip_diagnostic}_{it:05d}.png")

        if args.anim and it % args.anim_interval == 0:
            with torch.no_grad():
                frames.append(Image.fromarray(render_hard_color()[1], mode="RGB") if color_anim
                              else Image.fromarray(render_hard()[1], mode="L"))

        if prec is not None and it % args.pool_record_interval == 0:   # sampled snap/entropy trajectories
            with torch.no_grad():
                prec["hist_it"].append(it)
                if pool_mode:
                    wcl = pool.weights(tau_c)
                    prec["snap_hist"].append(pool.snap().cpu().numpy().astype(np.int16))
                    prec["ent_hist"].append((-(wcl * (wcl + 1e-9).log()).sum(1)).cpu().numpy().astype(np.float32))
                else:   # swarm: across-slot entropy + per-slot glyph / clean weight / dist / grad-norm
                    vcl = swarm.weights(tau_w_c, bias=swarm.h_bias)
                    sg_h, sd_h = swarm.slot_glyphs(codebook)
                    prec["snap_hist"].append(swarm.snap(codebook).cpu().numpy().astype(np.int16))
                    prec["ent_hist"].append((-(vcl * (vcl + 1e-9).log()).sum(1)).cpu().numpy().astype(np.float32))
                    prec["slot_g_hist"].append(sg_h.cpu().numpy().astype(np.int16))
                    prec["slot_v_hist"].append(vcl.cpu().numpy().astype(np.float32))
                    prec["slot_d_hist"].append(sd_h.cpu().numpy().astype(np.float32))
                    prec["slot_gn_hist"].append(
                        (swarm.gz_ema if swarm.gz_ema is not None
                         else torch.zeros_like(vcl)).cpu().numpy().astype(np.float32))
                    prec["slot_step_hist"].append(swarm.step_ema.cpu().numpy().astype(np.float32))
                    prec["wp_hist"].append(                # per-cell draw-push (>0 = CLIP wants ink)
                        (swarm.g_ema.mean(1) if swarm.g_ema is not None
                         else torch.zeros(M, device=device)).cpu().numpy().astype(np.float32))

        if knn_z_hist is not None and it % args.pool_record_interval == 0:
            with torch.no_grad():
                knn_hist_it.append(it)
                knn_z_hist.append(z.detach().cpu().numpy().astype(np.float32))

        if es_break:
            skipped_es = args.iters - it - 1
            reason = (f"colour-contrast budget of {args.color_contrast_iters} iters done"
                      if es_refine_start is not None
                      else (f"hard snap unchanged for {es_stable} checks "
                            f"({es_stable * args.early_stop_every} iters)"
                            + (f" within {es_tol} cells" if es_tol else "")
                            + (f" and diversity flat over {args.early_stop_div_window} iters"
                               if es_div_on else "")))
            print(f"\nearly stop at iter {it}/{args.iters}: {reason} -- skipping {skipped_es} "
                  f"remaining ({100 * skipped_es / max(1, args.iters):.0f}% of the run)")
            break

    # --- final measured runner-up election (swarm): each open cell's 2nd-best slot finalist is
    # hard-probed in the FINAL context (all neighbors settled, so mid-run rejections can become
    # clear accepts). Slot-finalists only; no open-codebook search.
    if swarm_mode and args.swarm_final_elect > 0 and live_probe:
        with torch.no_grad():
            n_fe = 0
            for sweep in range(args.swarm_final_elect):
                Wm = swarm.W.data.masked_fill(swarm.free, float("-inf"))
                t2 = Wm.topk(2, dim=1)
                k1, k2 = t2.indices[:, 0], t2.indices[:, 1]
                sgl, _ = swarm.slot_glyphs(codebook)
                g1 = sgl[swarm._ar, k1]
                g2 = sgl[swarm._ar, k2]
                cand = (g1 != g2) & torch.isfinite(t2.values[:, 1])
                if swarm.search_mask is not None:
                    cand &= swarm.search_mask
                todo = cand.nonzero().flatten().tolist()
                acc_sweep = 0
                base = snap_indices()
                img0 = assemble(char_bitmaps[base].view(1, GH, GW, CH, CW),
                                GH, GW, CH, CW, args.row_gap, IMG_H, IMG_W)[0]
                E0 = probe_err_maps(img0)
                while todo:
                    occ, batch, rest = [], [], []
                    for cell in todo:               # greedy spaced partition (full coverage)
                        r0, c0 = divmod(cell, GW)
                        if all(max(abs(r0 - r1), abs(c0 - c1)) >= args.pool_probe_spacing
                               for r1, c1 in occ):
                            batch.append(cell); occ.append((r0, c0))
                        else:
                            rest.append(cell)
                    todo = rest
                    s2 = base.clone()
                    for cell in batch:
                        s2[cell] = g2[cell]
                    img1 = assemble(char_bitmaps[s2].view(1, GH, GW, CH, CW),
                                    GH, GW, CH, CW, args.row_gap, IMG_H, IMG_W)[0]
                    E1 = probe_err_maps(img1)
                    for cell in batch:
                        dlt = probe_delta_at(E1, E0, img1, img0, cell, tgt, rw)
                        if prec is not None:
                            prec["probe_it"].append(args.iters + sweep)   # tagged post-run
                            prec["probe_cell"].append(cell)
                            prec["probe_in"].append(int(g2[cell]))
                            prec["probe_delta"].append(dlt)
                            prec["probe_accept"].append(int(dlt < -args.pool_probe_margin) * 2)
                            prec["probe_ch"].append(5)                    # 'file' code = election
                        if dlt < -args.pool_probe_margin:
                            swarm.W.data[cell, k2[cell]] = swarm.W.data[cell, k1[cell]] + 0.05
                            acc_sweep += 1
                n_fe += acc_sweep
                print(f"final-elect sweep {sweep + 1}: {int(cand.sum())} finalists probed, "
                      f"{acc_sweep} flipped")
                if acc_sweep == 0:
                    break
            print(f"final measured election: {n_fe} cells flipped to their runner-up slot")

    # --- final hard render + text ---
    with torch.no_grad():
        idx, arr = render_hard()
    Image.fromarray(arr, mode="L").save(args.output)
    idx_grid = idx.view(GH, GW).cpu()
    art_text = "".join("".join(chars[idx_grid[i, j].item()] for j in range(GW)) + "\n"
                       for i in range(GH))
    with open(args.output_text, "w", encoding="utf-8") as f:
        f.write(art_text)
    print(f"\nSaved {args.output} and {args.output_text}")

    if args.qr or args.qr_image:
        # The QR *is* the artwork: the whole grid rides in the URL fragment, so the
        # code needs no server and no stored record. Encoded from `art_text`, the
        # same string just written to disk, so the code and the .txt cannot disagree.
        try:
            from unicasso import artcode
            from unicasso.artcode import qr as qrmod
            payload = artcode.encode(art_text)
            if artcode.decode(payload) != art_text:
                raise RuntimeError("artcode round-trip mismatch -- refusing to emit")
            url = (args.qr_url or artcode.BASE_URL) + artcode.to_base32(payload)
            code = qrmod.make(url, error=args.qr_error)
            print(f"artcode: {GW}x{GH} -> {len(payload)} B "
                  f"({8 * len(payload) / max(GW * GH, 1):.2f} bits/cell) | "
                  f"{qrmod.describe(code, url)}")
            if args.qr:
                print(qrmod.to_terminal(code, ansi=not args.qr_no_ansi))
            print(url)
            if args.qr_image:
                qrmod.to_png(code, args.qr_image, scale=args.qr_scale)
                print(f"Saved QR -> {args.qr_image}")
        except Exception as e:
            # A failed QR must never lose a finished render.
            print(f"QR generation failed ({type(e).__name__}): {e}")

    if args.color and swarm_mode and not args.color_pin and not args.color_fit \
            and not args.color_mask:
        # How far did CLIP/recon actually MOVE the colors off their closed-form MSE init? If this
        # is ~0 the color leaves are inert and the run is just a post-hoc colorizer with extra steps.
        with torch.no_grad():
            live_ = ~swarm.free
            dfg = (swarm.fg.data - color_fg0).abs().mean(-1)[live_]
            dbg = (swarm.bg.data - color_bg0).abs().mean(-1)[live_]
            print(f"color drift off the closed-form init: fg mean {float(dfg.mean()):.4f} "
                  f"max {float(dfg.max()):.4f} | bg mean {float(dbg.mean()):.4f} "
                  f"max {float(dbg.max()):.4f}  (units: RGB in [0,1])")

    if args.color_contrast_learn and swarm_mode and getattr(swarm, "color_fit", False):
        with torch.no_grad():
            kv_ = swarm.k_contrast.data.clamp(0, swarm.k_contrast_max).cpu()
            q_ = torch.quantile(kv_, torch.tensor([0.05, 0.5, 0.95]))
            print(f"learned contrast k: mean {float(kv_.mean()):.3f}  p05 {float(q_[0]):.2f} "
                  f"med {float(q_[1]):.2f} p95 {float(q_[2]):.2f}  "
                  f"({int((kv_ < 0.9).sum())} softened, {int((kv_ > 1.1).sum())} exaggerated)")

    if args.color and swarm_mode:
        # ship the WINNING slot's glyph together with ITS OWN fitted colors
        with torch.no_grad():
            k1e = swarm.W.data.masked_fill(swarm.free, float("-inf")).argmax(1)
            if getattr(swarm, "color_fit", False):
                # refit at the PLACED glyph, matching what the loop optimized
                fge, bge = swarm.fit_emit_colors(1.0 - bm_flat[idx.view(M)], gidx=idx.view(M))
            else:
                fge, bge = swarm._colors()
                fge = fge[swarm._ar, k1e].clamp(0, 1)      # (M,3)
                bge = bge[swarm._ar, k1e].clamp(0, 1)
            gi = idx.view(M)
            cell_c = bge[:, None, :] + (fge - bge)[:, None, :] * (1.0 - bm_flat[gi])[:, :, None]
            rgb_out = assemble_rgb(cell_c.view(1, GH, GW, CH, CW, 3), GH, GW, CH, CW,
                                   args.row_gap, IMG_H, IMG_W)[0]
        cpng = args.output_color_png or (os.path.splitext(args.output)[0] + "_color.png")
        Image.fromarray((rgb_out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                        mode="RGB").save(cpng)
        print(f"Saved color render -> {cpng}")
        ans = args.output_ans or (os.path.splitext(args.output_text)[0] + ".ans")
        f8 = (fge.view(GH, GW, 3).cpu().numpy() * 255).astype(np.uint8)
        b8 = (bge.view(GH, GW, 3).cpu().numpy() * 255).astype(np.uint8)
        ig = idx.view(GH, GW)
        with open(ans, "w", encoding="utf-8") as fh:
            for i in range(GH):
                row = []
                for j in range(GW):
                    fr, fgn, fb = f8[i, j]
                    br, bg_, bb = b8[i, j]
                    row.append(f"\x1b[38;2;{fr};{fgn};{fb}m\x1b[48;2;{br};{bg_};{bb}m"
                               + chars[ig[i, j].item()])
                fh.write("".join(row) + "\x1b[0m\n")
        print(f"Saved ANSI truecolor -> {ans}")

        if args.recolor:
            # STANDARD POST-STEP: refit fg/bg closed-form against the ORIGINAL image at the
            # placed glyphs. CLIP optimizes a perceptual objective and the colors it likes are
            # measurably not the colors that minimize pixel error, so the shipping colors come
            # from the fit, not from the run.
            from unicasso.engine.color import fit_fg_bg
            with torch.no_grad():
                mk = 1.0 - bm_flat[gi]                                    # (M,P) ink=1
                tc = tgt_rgb[:GH * CH, :GW * CW].reshape(GH, CH, GW, CW, 3) \
                    .permute(0, 2, 1, 3, 4).reshape(M, CH * CW, 3)
                rf, rb = fit_fg_bg(tc, mk)
                if args.recolor_min_contrast > 0:
                    lum = lambda t: 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]
                    gap = lum(rf) - lum(rb)
                    push = torch.where(gap <= 0, -1.0, 1.0) * (args.recolor_min_contrast - gap.abs()).clamp_min(0)
                    rf = (rf + push[:, None]).clamp(0, 1)
                if args.color_contrast_learn and getattr(swarm, "k_contrast_on", False):
                    # the refit must carry the learned contrast too, or *_mserefit silently throws
                    # away the thing the tail of the run was spent learning
                    kk_ = swarm.k_contrast.data.clamp(0, swarm.k_contrast_max)[:, None]
                    mid_ = 0.5 * (rf + rb)
                    rf = (mid_ + kk_ * (rf - mid_)).clamp(0, 1)
                    rb = (mid_ + kk_ * (rb - mid_)).clamp(0, 1)
                rc = rb[:, None, :] + (rf - rb)[:, None, :] * mk[:, :, None]
                rout = assemble_rgb(rc.view(1, GH, GW, CH, CW, 3), GH, GW, CH, CW,
                                    args.row_gap, IMG_H, IMG_W)[0]
                res_run = float((rgb_out - tgt_rgb).abs().mean())
                res_fit = float((rout - tgt_rgb).abs().mean())
            rp = os.path.splitext(args.output)[0] + "_mserefit.png"
            Image.fromarray((rout.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), "RGB").save(rp)
            rf8 = (rf.clamp(0, 1).view(GH, GW, 3).cpu().numpy() * 255).astype(np.uint8)
            rb8 = (rb.clamp(0, 1).view(GH, GW, 3).cpu().numpy() * 255).astype(np.uint8)
            rans = os.path.splitext(args.output_text)[0] + "_mserefit.ans"
            with open(rans, "w", encoding="utf-8") as fh:
                for i in range(GH):
                    fh.write("".join(
                        f"\x1b[38;2;{rf8[i,j,0]};{rf8[i,j,1]};{rf8[i,j,2]}m"
                        f"\x1b[48;2;{rb8[i,j,0]};{rb8[i,j,1]};{rb8[i,j,2]}m" + chars[ig[i, j].item()]
                        for j in range(GW)) + "\x1b[0m\n")
            print(f"MSE refit (min-contrast {args.recolor_min_contrast}): "
                  f"residual {res_run:.4f} (run colors) -> {res_fit:.4f} (refit)")
            print(f"Saved {rp} and {rans}")
    if prec is not None:   # churn record: swap log + trajectories + final composition (pool/swarm)
        import os as _os
        _os.makedirs(_os.path.dirname(args.pool_record) or ".", exist_ok=True)
        rec = dict(GH=GH, GW=GW, chars=np.array(list(chars)), mode=np.array(args.mode),
                   swap_it=np.array(prec["swap_it"], dtype=np.int32),
                   swap_cell=np.array(prec["swap_cell"], dtype=np.int32),
                   swap_in=np.array(prec["swap_in"], dtype=np.int16),
                   swap_out=np.array(prec["swap_out"], dtype=np.int16),
                   swap_ch=np.array(prec["swap_ch"], dtype=np.int8),
                   round_it=np.array(prec["round_it"], dtype=np.int32),
                   gate_pass=np.array(prec["gate_pass"], dtype=np.int32),   # (R,6) gate passes/round
                   probe_it=np.array(prec["probe_it"], dtype=np.int32),
                   probe_cell=np.array(prec["probe_cell"], dtype=np.int32),
                   probe_in=np.array(prec["probe_in"], dtype=np.int16),
                   probe_delta=np.array(prec["probe_delta"], dtype=np.float32),
                   probe_accept=np.array(prec["probe_accept"], dtype=np.int8),
                   probe_ch=np.array(prec["probe_ch"], dtype=np.int8),
                   hist_it=np.array(prec["hist_it"], dtype=np.int32),
                   snap_hist=np.array(prec["snap_hist"]),                    # (T,M)
                   ent_hist=np.array(prec["ent_hist"]),                      # (T,M)
                   final_snap=idx.cpu().numpy())
        if pool_mode:
            rec.update(final_cand=pool.cand.cpu().numpy().astype(np.int16),
                       final_logits=pool.logits.detach().cpu().numpy().astype(np.float32),
                       final_w_ema=pool.w_ema.cpu().numpy().astype(np.float32),
                       final_age=pool.age.cpu().numpy().astype(np.int32))
        else:   # swarm: birth kinds, boost/merge logs, per-slot trajectories + final slot state
            sg_f, sd_f = swarm.slot_glyphs(codebook)
            rec.update(swap_kind=np.array(prec["swap_kind"], dtype=np.int8),  # 1=strong 0=lottery
                       swap_slot=np.array(prec["swap_slot"], dtype=np.int8),
                       boost_it=np.array(prec["boost_it"], dtype=np.int32),
                       boost_cell=np.array(prec["boost_cell"], dtype=np.int32),
                       boost_slot=np.array(prec["boost_slot"], dtype=np.int8),
                       boost_g=np.array(prec["boost_g"], dtype=np.int16),
                       merge_it=np.array(prec["merge_it"], dtype=np.int32),
                       merge_cell=np.array(prec["merge_cell"], dtype=np.int32),
                       merge_d=np.array(prec["merge_d"], dtype=np.float32),
                       merge_vl=np.array(prec["merge_vl"], dtype=np.float32), # folded v_ema (eps diag)
                       econ_it=np.array(prec["econ_it"], dtype=np.int32),
                       econ=np.array(prec["econ"], dtype=np.int32),           # (R,8) round economics
                       slot_g_hist=np.array(prec["slot_g_hist"]),             # (T,M,K)
                       slot_v_hist=np.array(prec["slot_v_hist"]),             # (T,M,K) clean v
                       slot_d_hist=np.array(prec["slot_d_hist"]),             # (T,M,K) dist to glyph
                       slot_gn_hist=np.array(prec["slot_gn_hist"]),           # (T,M,K) z-grad EMA
                       slot_step_hist=np.array(prec["slot_step_hist"]),       # (T,M,K) ||dz||/step EMA
                       wp_hist=np.array(prec["wp_hist"]),                     # (T,M) draw-push (>0 = ink wanted)
                       final_flips=swarm.flips.cpu().numpy().astype(np.int32),
                       final_z=swarm.z.detach().cpu().numpy().astype(np.float32),
                       final_W=swarm.W.detach().cpu().numpy().astype(np.float32),
                       final_free=swarm.free.cpu().numpy(),
                       final_slot_g=sg_f.cpu().numpy().astype(np.int16),
                       final_slot_d=sd_f.cpu().numpy().astype(np.float32),
                       final_v_ema=swarm.v_ema.cpu().numpy().astype(np.float32),
                       final_age=swarm.age.cpu().numpy().astype(np.int32))
        np.savez(args.pool_record, **rec)
        print(f"wrote {args.mode}-record -> {args.pool_record} ({n_swaps_total} swaps: "
              f"grad {int(n_swaps_ch[0])}, neighbor {int(n_swaps_ch[1])}, "
              f"affinity {int(n_swaps_ch[2])}, latent {int(n_swaps_ch[3])}, "
              f"ports {int(n_swaps_ch[4])}, file {int(n_swaps_ch[5])}, "
              f"blend {int(n_swaps_ch[6])}, join {int(n_swaps_ch[7])}, "
              f"pixel {int(n_swaps_ch[8])})"
              + (f" | live-probe: {n_probe_acc} accepted, {n_probe_rej} rejected+tabu'd"
                 if live_probe else "")
              + (f" | boosts {n_boosts}, merges {n_merges}, lottery births {n_lottery}"
                 if swarm_mode else ""))
    elif args.pool_record and args.mode == "knn-smooth":
        # slim displacement record: init/final latents + codebook, for the
        # staleness figure (how far does a single-latent cell travel from init?)
        import os as _os
        _os.makedirs(_os.path.dirname(args.pool_record) or ".", exist_ok=True)
        np.savez(args.pool_record, GH=GH, GW=GW, chars=np.array(list(chars)),
                 mode=np.array(args.mode),
                 z_init=knn_z_init.cpu().numpy().astype(np.float32),
                 z_final=z.detach().cpu().numpy().astype(np.float32),
                 z_hist=np.array(knn_z_hist, dtype=np.float32),   # (T, M, L)
                 hist_it=np.array(knn_hist_it, dtype=np.int32),
                 codebook=codebook.cpu().numpy().astype(np.float32),
                 final_snap=idx.cpu().numpy())
        print(f"wrote knn-smooth displacement record -> {args.pool_record}")

    if args.output_indices:
        # self-describing training target: indices index into `chars` (codebook order, after banning)
        os.makedirs(os.path.dirname(args.output_indices) or ".", exist_ok=True)
        np.savez(args.output_indices, indices=idx_grid.numpy().astype(np.int16),
                 chars=np.array(chars), GW=GW, GH=GH, CW=CW, CH=CH, row_gap=args.row_gap,
                 input_image=np.array(args.input_image))
        print(f"Saved glyph indices -> {args.output_indices}")

    if args.loss_curve and curve["it"]:
        npz = save_loss_curve(curve, args.loss_curve)
        print(f"Saved loss curve -> {args.loss_curve} (+ {npz})")

    if args.overlay:
        # ASCII ink (red) over the original (the SAME-frame target), original
        # shown faint so misalignment between the two reads as red-not-on-the-lines.
        og = (tgt.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # white=255
        ascii_ink = 1.0 - (arr.astype(np.float32) / 255.0)                    # 1 where ASCII ink is
        base = 255 - (255 - og.astype(np.float32)) * 0.35                     # fade original to 35%
        rgb = np.stack([base, base, base], axis=-1)
        rgb[..., 1] *= (1.0 - ascii_ink)   # subtract green+blue where ASCII ink -> red shows through
        rgb[..., 2] *= (1.0 - ascii_ink)
        Image.fromarray(rgb.clip(0, 255).astype(np.uint8), mode="RGB").save(args.overlay)
        print(f"Saved overlay -> {args.overlay}")

    if args.anim:
        frames.append(Image.fromarray(render_hard_color()[1], mode="RGB") if color_anim
                      else Image.fromarray(arr, mode="L"))  # final frame
        anim_path = args.anim if args.anim.lower().endswith(".gif") else args.anim + ".gif"
        frames[0].save(anim_path, format="GIF", save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.anim_fps), loop=0)
        print(f"Saved {len(frames)}-frame animation -> {anim_path}")


if __name__ == "__main__":
    main()
