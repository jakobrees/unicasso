"""Joint trainer: three objectives, summed, one backward.

    L = w_line * CE_line  +  w_pool * CE_pool  +  lam * CLIP_mask

  CE_line   dense CE on the static lineart cache. The ANCHOR -- present in every
            step, not sampled, because a regulariser that is absent from most
            steps is not regularising. Its weight is fixed rather than grad-norm
            balanced on purpose: if the colour objective starts degrading line
            art, this term's gradient must be free to GROW and pull back. That
            self-correcting feedback is the whole point, and normalising it away
            would break it.
  CE_pool   dense CE on engine-refined glyph grids from the rolling pool. The
            pool's teacher colours with the model itself (--color-lite), so the
            targets are reachable rather than aspirational.
  CLIP_mask the only term that trains the mask decoder: crop a photo, sample
            glyphs straight-through, colourise them through the mask, render,
            and judge with the same CLIPasso objective the engine uses. With
            probability --flip-frac the fg/bg colours are swapped -- explicit
            exploration of the one degree of freedom (which cluster is
            foreground) that the objective otherwise pins only weakly.

lam is grad-norm calibrated against the CE terms IN THE SAME BACKWARD, so the
ratio is measured on one graph instead of EMA'd across steps that saw different
data and different weights.

Deleted relative to cellclf_color_train: every per-pixel mask target, the k head
and all k terms, the decompose-derived priors (sep floor, structure, abstain,
blur anchor, colour target/blend), the coltext / ansT / colorart arms, and all
corruption paths.

    ./train_campaign.sh sfmono|dejavu            # from scratch, the full curriculum
    python -m unicasso.training.joint_train --init runs/joint/<run>/model.pt ...   # continue one
"""

import argparse
import glob
import gc
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile, ImageOps

Image.MAX_IMAGE_PIXELS = None      # the photo set has >100 MP scans;
# PIL's bomb guard would warn on every sample and eventually refuse them
ImageFile.LOAD_TRUNCATED_IMAGES = True   # a few files are short a scanline;
# decode what is there rather than killing an unattended run

from unicasso.substrate import glyphs as G
from unicasso.lite import _conf_fit
from unicasso.engine.color import (decompose, nomination_target, smoothstep,
                                    srgb_to_lab)
from unicasso.training.cellclf_color_train import (cell_feats, grid_windows,
                                                   grid_windows_t, mask_fit,
                                                   token_maps)
from unicasso.training.pool_manager import RollingPool, photo_groups
from unicasso.training.train_cell_classifier import (CellCache, MaskDecoder,
                                                     MaskAttnDecoder,
                                                     identity_block,
                                                     MuonWithAdamW,
                                                     TokenTransformer,
                                                     binomial_kernel, evaluate,
                                                     grad_norm_of, kernel_ce,
                                                     split_muon_params)

ROWS, COLS = 3, 5

# ---------------------------------------------------------------- schedule
# The campaign curriculum: one launch runs the whole thing. Each row is
# (from_step, line_ce, line_color, pool_color, pool_line, clip).
#
# Weights sum to 1.0 in every phase ON PURPOSE. Under per-arm gradient
# normalisation each arm contributes exactly w_i in gradient norm, so the total
# norm IS the sum -- a constant sum keeps the effective learning rate stable
# across hand-overs instead of silently changing it at every phase boundary.
CAMPAIGN_PHASES = [
    # step  line_ce line_col pool_col pool_line clip   mask_tgt
    (0,    0.60,    0.40,    0.00, 0.00,    0.00,  0.00),   # lineart only
    (500,  1 / 3,   1 / 3,   0.00, 0.00,    1 / 3, 0.00),   # CLIP on
    (700,  0.50,    0.00,    0.00, 0.00,    0.50,  0.00),   # line_color off
    (1000, 1 / 3,   0.00,    0.00, 1 / 3,   1 / 3, 0.00),   # line pool on
    (1500, 0.15,    0.00,    0.25, 0.20,    0.40,  0.00),   # colour pool on
]
# mask_tgt = decompose's own fg/bg partition on PHOTOS as a per-pixel mask
# target, as an ARM (the --mask-target regulariser rides the 5% channel and
# cannot bootstrap anything). It is NOT in the phase table: it is injected by
# train_campaign.sh over [1500, 2000) decaying to 0 -- the recipe that took
# campaign_sfmono from inv 40% to 6.6% in 100 steps. Putting it in phase 0
# (campaign_dejavu v1, 0.20 over 0-500) HARDENED the mask before CLIP arrived:
# ent 0.000 from step 500, held-out reach 0.03-0.17, clip stuck at .43 where
# sfmono was at .40. The mask must be shaped by the ink CE and CLIP first;
# the polarity convention comes later, when there is something to orient.
ARMS = ("line_ce", "line_color", "pool_color", "pool_line", "clip", "mask_tgt")
CAMPAIGN_LR_STEP = 700        # from-scratch rates before, settled rates after
CAMPAIGN_LR_FACTOR = 0.667    # 3e-4 -> 2e-4, 0.02 -> 0.0133 (see --lr-late)
CAMPAIGN_POOL_LINE_AT = 1000
CAMPAIGN_POOL_COLOR_AT = 1500
CAMPAIGN_REG_AT = 2500        # space/density ramp in here
CAMPAIGN_MASK_TGT_OFF = 500   # phase-0 mask_tgt decays LINEARLY to 0 here


def phase_weights(step, phases):
    """The active weight vector at `step` -- the last row whose from_step <= step."""
    w = phases[0][1:]
    for row in phases:
        if step >= row[0]:
            w = row[1:]
    return dict(zip(ARMS, w))


def dropblock(x, drop_prob, block):
    """Zero contiguous BLOCKS of a feature map, not scattered pixels.

    Elementwise dropout on a conv input is close to a no-op for copying: the
    next 3x3 conv reconstructs a missing pixel from its neighbours. Removing a
    whole block leaves nothing local to reconstruct from, so the decoder has to
    get that region's evidence from somewhere else -- which is the entire point
    of putting it on the skip."""
    if drop_prob <= 0:
        return x
    N, C, H, W = x.shape
    b = max(1, block | 1)                      # odd, so padding is symmetric
    denom = max(1, (H - b + 1) * (W - b + 1))
    gamma = drop_prob / (b * b) * (H * W) / denom
    seed = (torch.rand(N, C, H, W, device=x.device) < gamma).float()
    blocked = torch.nn.functional.max_pool2d(seed, b, stride=1, padding=b // 2)
    keep = 1.0 - blocked
    return x * keep * keep.numel() / keep.sum().clamp_min(1.0)


def build_random(chars, cfg, device):
    """A fresh TokenTransformer -- no checkpoint. build_model can only LOAD, so
    a from-scratch campaign had no entry point at all. Every surgery the
    architecture needs is applied here unconditionally; the caller then adds
    mask_ctx / glyph_ctx / bcast through the same paths a resumed run uses."""
    m = TokenTransformer(ROWS, COLS, cfg["cell_h"], cfg["cell_w"], cfg["pad_h"],
                         cfg["pad_w"], len(chars), dim=cfg["dim"],
                         heads=cfg["heads"], n_blocks=cfg["blocks"],
                         in_ch=cfg.get("in_ch", 4))
    m.color_proj = nn.Linear(cfg["feat_dim"], cfg["dim"])
    # the optional modules the surgery block probes for. build_model creates
    # them conditionally from a checkpoint; a fresh module has no such
    # attributes at all, and the probes are a mix of getattr and direct access.
    for _n in ("mask_dec", "mask_blocks", "aux_block", "rgb_trunk",
               "mask_ctx_trunk", "mask_ctx_proj", "glyph_ctx_proj"):
        setattr(m, _n, None)
    return m.to(device), chars, cfg


def build_model(path, device):
    ck = torch.load(G.repo_path(path), map_location="cpu", weights_only=False)
    cfg, sd = ck["config"], ck["state_dict"]
    m = TokenTransformer(ROWS, COLS, cfg["cell_h"], cfg["cell_w"], cfg["pad_h"],
                         cfg["pad_w"], len(ck["chars"]), dim=cfg["dim"],
                         heads=cfg["heads"], n_blocks=cfg["blocks"],
                         in_ch=cfg.get("in_ch", 1))
    if cfg.get("feat_dim"):
        m.color_proj = nn.Linear(cfg["feat_dim"], cfg["dim"])
    if "mode_emb" in sd:
        m.mode_emb = nn.Parameter(torch.zeros(sd["mode_emb"].shape[0], cfg["dim"]))
    if "k_head.weight" in sd:                 # loaded so the ckpt matches, then
        m.k_head = nn.Linear(cfg["dim"], 1)   # never called -- k is out
    if "col_head.weight" in sd:
        m.col_head = nn.Linear(cfg["dim"], 6)
    if "mask_dec.out.weight" in sd:
        m.mask_dec = MaskDecoder(cfg["dim"],
                                 flip=bool(cfg.get("mask_flip"))
                                 or "mask_dec.flip.weight" in sd)
    if cfg.get("mask_attn") or "mask_dec.q.weight" in sd:
        m.mask_dec = MaskAttnDecoder(cfg["dim"], d=cfg.get("mask_attn_dim", 32),
                                     depth=cfg.get("mask_attn_depth", 2),
                                     mid=cfg.get("mask_attn_mid"))
        npx = cfg.get("mask_attn_extra", 0) or sum(
            1 for k in sd if k.startswith("mask_dec.px_res.")
            and k.endswith(".c1.weight"))
        if npx:
            m.mask_dec.add_px_blocks(npx, cfg.get("mask_attn_mid"))
        # broadcast conditioning widens conv0, so it MUST be rebuilt before the
        # load or the shapes disagree. Inferred from the weight when cfg is
        # silent: n_glyph off gemb, and the mean channels are whatever width is
        # left over once rgb and the embedding are accounted for.
        ng = cfg.get("mask_bcast_glyph")
        if ng is None:
            ng = sd["mask_dec.gemb.weight"].shape[0] \
                if "mask_dec.gemb.weight" in sd else 0
        bm = cfg.get("mask_bcast_mean")
        if bm is None:
            bm = ("mask_dec.px.0.weight" in sd
                  and sd["mask_dec.px.0.weight"].shape[1] - 3 - ng == 3)
        if ng or bm:
            m.mask_dec.add_bcast(len(ck["chars"]), int(ng or 0), bool(bm),
                                 cfg["pad_h"], cfg["pad_w"])
    if cfg.get("glyph_ctx_aux"):
        m.glyph_ctx_aux = True
    if cfg.get("mask_ctx") or "mask_ctx_proj.weight" in sd:
        m.add_mask_ctx(cfg["dim"])
        m.mask_ctx_inject = cfg.get("mask_ctx_inject", 0)
    if cfg.get("glyph_ctx") or "glyph_ctx_proj.weight" in sd:
        m.add_glyph_ctx(cfg["dim"], len(ck["chars"]))
        m.glyph_ctx_inject = cfg.get("glyph_ctx_inject", 1)
    if cfg.get("aux_block") or "aux_block.ln1.weight" in sd:
        from unicasso.training.train_cell_classifier import Block
        m.aux_block = Block(cfg["dim"], cfg["heads"])
    if cfg.get("rgb_trunk") or any(k.startswith("rgb_trunk.") for k in sd):
        m.rgb_trunk = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU())
    nmb = cfg.get("mask_blocks", 0) or sum(
        1 for k in sd if k.startswith("mask_blocks.") and k.endswith("ln1.weight"))
    if nmb:
        m.mask_blocks = nn.ModuleList(identity_block(cfg["dim"], cfg["heads"])
                                      for _ in range(nmb))
    m.load_state_dict(sd, strict=(nmb == 0 or "mask_blocks.0.ln1.weight" in sd))
    return m.to(device), ck["chars"], cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", default=None,
                   help="checkpoint to continue from (a joint_train or cellclf "
                        "model). Omit with --random-init; train_campaign.sh does")
    p.add_argument("--cache", default="runs/cellclf/cache_sfmono")
    p.add_argument("--photos",
                   default="data/color_images_plus,data/dataset_v1/photos")
    p.add_argument("--profile", default="sfmono")
    p.add_argument("--vae-ckpt", default="weights/vae_sfmono/model.pt")
    p.add_argument("--name", default="joint01")
    # ---- objective weights
    p.add_argument("--w-line", type=float, default=1.0,
                   help="lineart CE weight -- FIXED, never recalibrated. The cache "
                        "is static, so this term's gradient moves only when the "
                        "model does; renormalising it would cancel the anchor's "
                        "feedback (line art degrades -> gradient grows -> pull back)")
    p.add_argument("--pool-ratio", type=float, default=1.0,
                   help="target ||g_pool|| / ||g_line||. w_pool is re-derived every "
                        "--recal-every steps to hold it: the pool's entries are "
                        "replaced wholesale each refresh, so its loss scale really "
                        "is non-stationary and does need renormalising. 1.0 = the "
                        "pool carries the same gradient as the anchor")
    p.add_argument("--clip-weight-target", type=float, default=0.2,
                   help="target ||g_clip|| / ||g_ce||; lam is derived, not set")
    p.add_argument("--balance", default="ce", choices=["ce", "photo"],
                   help="which arm is the fixed reference the others are scaled "
                        "to. 'ce' (inherited): w_line is pinned at 1 and lam = "
                        "clip_weight_target * ||g_ce||/||g_clip||, so the photo "
                        "arm's absolute push is proportional to the LINEART "
                        "gradient -- it inherits lineart's crop-sampling noise, "
                        "and it quiets down as lineart gets fit, which is "
                        "backwards. 'photo': lam is pinned at 1 and the CE arms "
                        "are scaled to ||g_clip|| instead, so the photo arm gets "
                        "a stable absolute budget. TRADE-OFF: under 'photo' the "
                        "lineart contribution is normalised to constant "
                        "magnitude, which removes the anchor's self-correcting "
                        "feedback (degrade lineart -> its gradient grows -> pull "
                        "back). Raise --line-ratio to compensate")
    p.add_argument("--line-ratio", type=float, default=1.0,
                   help="(--balance photo) target ||w_line*g_line|| / ||g_clip||")
    p.add_argument("--line-regions", type=int, default=1,
                   help="lineart crops averaged into ONE anchor CE per step. 1 "
                        "is a batch of one region out of 254 runs and `line` "
                        "swings 5x step to step; a few costs ~26 ms each")
    p.add_argument("--render-ensemble", default="center",
                   choices=["center", "mean"],
                   help="how the render picks a glyph. 'center' = each cell "
                        "from the window centred on it -- the readout "
                        "--centre-only actually trains, and what `lite "
                        "--ensemble center` ships. 'mean' = the old binomial "
                        "ensemble over all 15 covering windows, which put 81.2%% "
                        "of the weight on positions no CE supervises")
    p.add_argument("--reg-share", type=float, default=0.05,
                   help="regulariser channel target share of the total gradient "
                        "norm. Replaces reg_scale, which was ema_clip (~0.042) "
                        "and silently multiplied every regulariser weight")
    p.add_argument("--reg-at", type=int, default=2500,
                   help="step at which space/density begin ramping in")
    p.add_argument("--schedule", default="campaign",
                   choices=["campaign", "flat"],
                   help="'campaign' runs the built-in curriculum end to end in "
                        "one launch; 'flat' holds --w-* constant")
    p.add_argument("--w-line-ce", type=float, default=0.5)
    p.add_argument("--w-line-color", type=float, default=0.0)
    p.add_argument("--w-pool-color", type=float, default=0.0)
    p.add_argument("--w-pool-line", type=float, default=0.0)
    p.add_argument("--w-clip", type=float, default=0.5)
    p.add_argument("--mask-tgt-from", type=int, default=None,
                   help="with --mask-tgt-until/-w: inject the mask_tgt arm at "
                        "weight W over [from, until) ON TOP of the schedule -- the "
                        "other arms are scaled by (1-W) so the sum stays 1. For "
                        "resuming a run that never had the bootstrap")
    p.add_argument("--mask-tgt-until", type=int, default=None)
    p.add_argument("--mask-tgt-w", type=float, default=0.0)
    p.add_argument("--w-mask-tgt", type=float, default=0.0,
                   help="(--schedule flat) weight of the decompose polarity "
                        "mask-target arm on photos (the campaign injects it via "
                        "--mask-tgt-from/-until/-w instead)")
    p.add_argument("--lr-late", type=float, default=0.667,
                   help="multiply every LR group by this from --lr-late-step on. "
                        "One step-down, no cosine: 3e-4 -> 2e-4 at the default")
    p.add_argument("--lr-late-step", type=int, default=700)
    p.add_argument("--muon-lr-late", type=float, default=None,
                   help="late factor for the Muon group only (default: --lr-late). "
                        "The campaign's settled rates are 2e-4 / 0.01 / 4e-3 from "
                        "3e-4 / 0.02 / 6e-3 -- 2/3 for AdamW and mask, 1/2 for Muon")
    p.add_argument("--mask-lr-late", type=float, default=None,
                   help="late factor for the mask-branch group only (default: --lr-late)")
    p.add_argument("--pool-line-at", type=int, default=CAMPAIGN_POOL_LINE_AT,
                   help="step at which the lineart pool is prefilled")
    p.add_argument("--pool-color-at", type=int, default=CAMPAIGN_POOL_COLOR_AT,
                   help="step at which the colour pool is prefilled")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="continue a run from one of its ckpt_stepNNNNN.pt: weights, "
                        "step counter (phases/LR/prefill timing pick up where "
                        "they were), both pools reloaded from their dirs, and "
                        "optimizer/EMA state when the checkpoint carries it "
                        "(checkpoints written after 2026-08-28 do). Overrides "
                        "--init and --random-init")
    p.add_argument("--resume-step", type=int, default=None,
                   help="override the step recorded in the --resume checkpoint "
                        "(pool_teacher.pt snapshots carry none)")
    p.add_argument("--resume-fresh-opt", action="store_true",
                   help="with --resume: DISCARD the checkpoint's optimizer/EMA "
                        "state and re-warm (--resume-warmup). A newly injected "
                        "arm moves a fresh optimizer far harder than one carrying "
                        "1500 steps of other arms' moments: the sfmono polarity "
                        "injection (fresh, no state) took inv 40%->6.6% in 100 "
                        "steps; dejavu's (state carried) took 400 steps to 22%")
    p.add_argument("--resume-warmup", type=int, default=50,
                   help="LR re-warm steps after --resume when the checkpoint has "
                        "NO optimizer state (fresh Adam/Muon moments take a few "
                        "over-sized steps otherwise). 0 when state is present")
    p.add_argument("--random-init", action="store_true",
                   help="build the model from scratch instead of loading "
                        "--init. Uses --rand-* for geometry")
    p.add_argument("--rand-dim", type=int, default=96)
    p.add_argument("--rand-heads", type=int, default=4)
    p.add_argument("--rand-blocks", type=int, default=3)
    p.add_argument("--val-photos", default=None,
                   help="explicit newline/comma list of held-out photos (or a "
                        "file of them). Without it the set is derived from a "
                        "DIRECTORY LISTING and silently changes when the photo "
                        "dir does -- local and the box already disagreed")
    p.add_argument("--pool-line", default=None,
                   help="lineart pool dir (default runs/<name>/pool_line)")
    p.add_argument("--pool-line-images", default="data/corpus/images",
                   help="source images for the lineart pool")
    p.add_argument("--pool-line-exclude", default=None,
                   help="cache meta.json whose val parents are EXCLUDED from "
                        "the lineart pool, or [val@] leaks")
    p.add_argument("--recal-every", type=int, default=20)
    p.add_argument("--ratio-ema", type=float, default=0.2,
                   help="EMA rate on the grad-norm RATIOS behind lam and w_pool. "
                        "A single-region estimate is very noisy -- joint01 saw lam "
                        "swing 3-12x between recalibrations. 1.0 = no smoothing")
    p.add_argument("--blank-weight", type=float, default=0.2)
    # ---- the shipped read (what the mask fit does at inference)
    p.add_argument("--read-path", default="sample",
                   choices=["topk", "sample", "blend"],
                   help="how the mask becomes two colours. 'blend' = every pixel "
                        "weighted by its probability (mask_fit): fully "
                        "differentiable, no selection. topk/sample rank pixels "
                        "and keep a fraction, which passes gradient ONLY to the "
                        "kept ones -- measured at 6.2%% of pixels, with a layer "
                        "keeping zero in 96.3%% of cells")
    p.add_argument("--blend-ridge", type=float, default=1.0,
                   help="(--read-path blend) pull toward the cell mean in "
                        "pixel-mass units. Keeps an all-abstain layer finite; "
                        "negligible against a typical layer mass of ~80")
    p.add_argument("--glyph-temp", type=float, default=0.0,
                   help="temperature for the straight-through glyph draw on the "
                        "photo arm. 0 = ARGMAX (default). >0 samples Gumbel at "
                        "that temperature. Note photo_eval has always used "
                        "argmax, so every [pval@] number was measured under a "
                        "decode training did not use")
    p.add_argument("--read-frac", type=float, default=0.6)
    p.add_argument("--read-count", default="argmax", choices=["fixed", "argmax"])
    p.add_argument("--read-temp", type=float, default=0.3)
    p.add_argument("--read-ridge", type=float, default=0.0)
    p.add_argument("--flip-frac", type=float, default=0.03,
                   help="probability a photo step swaps fg/bg before rendering")
    p.add_argument("--mask-blocks", type=int, default=0,
                   help="RUN B: extra transformer blocks private to the mask "
                        "branch, identity-init so step 0 is unchanged. Separates "
                        "the mask's representation from the glyph head's, which "
                        "otherwise share one token and let the decoder just copy "
                        "the glyph's shape")
    p.add_argument("--mask-l1", type=float, default=0.0,
                   help="RUN B: L1 on committed mask mass (p_fg + p_bg). Punishes "
                        "overextension directly; CLIP is the counterweight, since "
                        "too much abstain collapses both colours to the cell mean")
    p.add_argument("--freeze-mask", action="store_true",
                   help="hold the COLORIZER's own weights fixed (mask_dec + any "
                        "mask_blocks) while the CLIP term still shapes the GLYPH "
                        "head through the straight-through sample. Note the "
                        "colorizer's BEHAVIOUR still drifts: its inputs are the "
                        "shared trunk/blocks, which the CE terms keep moving")
    p.add_argument("--rgb-trunk", action="store_true",
                   help="give the mask branch its own RGB-only conv encoder and "
                        "feed mask_dec from THAT instead of s1. Not a skip -- it "
                        "shares no path with the token, so the decompose ink "
                        "never reaches the mask's spatial evidence and the mask "
                        "objective cannot smooth the glyph stack's features. "
                        "Warm-started from the shared trunk's first conv")
    p.add_argument("--mask-entropy", type=float, default=0.0,
                   help="penalty on the per-pixel 3-way entropy, pushing the mask "
                        "toward one-hot. Under a selection read only the ORDERING "
                        "of p is scored, so nothing otherwise rewards commitment "
                        "and the mask settles into a hedged ranking. CAUTION: this "
                        "and --mask-l1 are BOTH satisfied by 'confidently abstain "
                        "everywhere', which is the flat-cell collapse; CLIP is the "
                        "only counterweight to either")
    p.add_argument("--photo-val", type=int, default=6,
                   help="photos held out at startup as the photo arm's VALIDATION "
                        "set -- removed from training and from the pool, then "
                        "scored at --eval-every under frozen conditions (centred "
                        "crop, exact cell budget, argmax glyphs, no flip). The "
                        "per-step `clip` is a different photo at a different crop "
                        "and scale every time, so it moves with the sample as "
                        "much as with the model; this is the number that is "
                        "comparable across steps and across runs. 0 = off")
    p.add_argument("--mask-target", type=float, default=0.0,
                   help="WARM-UP: dense per-pixel supervision of the mask "
                        "against decompose's own partition, on the fg-vs-bg "
                        "CONDITIONAL only (abstain left free, so this is not "
                        "the em8 chain's full per-pixel copy). rgb_trunk never "
                        "sees the ink map, so hitting it from RGB requires "
                        "computing the partition rather than tracing it -- the "
                        "knowledge lands in the weights. Ramped to 0 over "
                        "--mask-target-steps, after which CLIP is free to "
                        "correct decompose where decompose is wrong")
    p.add_argument("--mask-target-steps", type=int, default=200,
                   help="(--mask-target) linear ramp-down horizon")
    p.add_argument("--mask-attn", action="store_true",
                   help="replace the conv mask decoder with MaskAttnDecoder: "
                        "per-pixel embedding dotted with one query per class. "
                        "Makes the fg/bg comparison multiplicative (a conv's "
                        "additive token cannot invert a spatial pattern) and "
                        "runs at full cell resolution. NOT surgery-safe -- the "
                        "mask branch retrains from scratch, so pair it with "
                        "--mask-target to bootstrap")
    p.add_argument("--mask-attn-dim", type=int, default=32,
                   help="(--mask-attn) pixel/query embedding width")
    p.add_argument("--mask-attn-depth", type=int, default=2,
                   help="(--mask-attn) convs in the per-pixel encoder, stride 1 "
                        "throughout. 2 gives a 5x5 receptive field, 4 gives 9x9. "
                        "The queries have no spatial extent, so every bit of "
                        "local structure the mask can use comes from here")
    p.add_argument("--mask-attn-extra", type=int, default=0,
                   help="append this many identity-init RESIDUAL blocks to the "
                        "per-pixel encoder. Unlike --mask-attn-depth this does "
                        "not rebuild: the trained encoder is kept and the new "
                        "depth contributes exactly 0 at step 0. Each block adds "
                        "2 convs, so the receptive field grows 5x5 -> 9x9 -> 13x13")
    p.add_argument("--mask-attn-mid", type=int, default=None,
                   help="(--mask-attn) hidden width of that encoder "
                        "(default = --mask-attn-dim)")
    p.add_argument("--line-color", type=float, default=0.0,
                   help="weight on the lineart COLOUR-RECONSTRUCTION arm: paint "
                        "the cache's true ink with a sampled fg/bg pair and ask "
                        "the model to reproduce it. Exact and dense, unlike CLIP, "
                        "and it trains the partition rather than the polarity "
                        "convention")
    p.add_argument("--line-color-mse", type=float, default=0.0,
                   help="weight of the colour-reconstruction MSE inside the "
                        "line_color arm, on top of the ink mask CE. 0 = mask "
                        "target only (the campaign default)")
    p.add_argument("--line-color-frac", type=float, default=0.34,
                   help="(--line-color) fraction of samples given a real colour "
                        "pair; the rest stay ink-on-paper with a random "
                        "brightness ORDER, which is the black/white inversion")
    p.add_argument("--line-color-iso", type=float, default=0.34,
                   help="(--line-color) fraction of the coloured samples made "
                        "ISO-LUMINANT, so only hue and chroma separate the two "
                        "clusters and the brightness shortcut is unavailable")
    p.add_argument("--line-color-de", type=float, nargs=2, default=[15.0, 70.0],
                   help="(--line-color) Lab dE range for the sampled pair; the "
                        "floor sits above decompose's jnd=4")
    p.add_argument("--mask-ctx", action="store_true",
                   help="give the mask branch its OWN conv trunk over the RGB "
                        "patch, projected per cell and added into its residual "
                        "stream. The query is otherwise built from a token whose "
                        "gradient is dominated by glyph CE; this supplies colour "
                        "evidence that never passed through the glyph stack. "
                        "Zero-init, so it is a no-op at step 0")
    p.add_argument("--mask-ctx-inject", type=int, default=0,
                   help="(--mask-ctx) which mask block to add it before. 0 = the "
                        "input, so ALL blocks' attention can share colour across "
                        "the 3x5 window; n = after block n, which costs that "
                        "sharing but leaves the earlier blocks unperturbed")
    p.add_argument("--glyph-ctx", action="store_true",
                   help="tell the colouriser WHICH GLYPH it is colouring: "
                        "Linear(n_classes -> dim) on the glyph softmax, added to "
                        "the mask branch's residual stream. A dense glyph and a "
                        "thin stroke need different fg to land the same cell "
                        "mean, and the mask currently has no idea which it has. "
                        "Also the only gradient path from the colour loss back "
                        "into the glyph head. Zero-init")
    p.add_argument("--glyph-ctx-inject", type=int, default=1,
                   help="(--glyph-ctx) which mask block to add it after. Default "
                        "1 rather than 0 so it is not summed into the colour "
                        "context at the input -- the two sources stay separable "
                        "by depth instead of being dumped into one vector")
    p.add_argument("--space-weight", type=float, default=0.0,
                   help="CE weight pushing GENUINELY EMPTY cells to the space "
                        "glyph. Gated on an image property the model cannot "
                        "move, because ungated it is self-referential -- "
                        "'collapse the colours, then emit space' is a stable "
                        "all-blank solution. Rides the regulariser channel, "
                        "never lam")
    p.add_argument("--space-ramp", type=float, default=3.0,
                   help="(--space-weight) Lab peak-to-peak of the cell's fitted "
                        "LIGHTNESS PLANE below which it counts as truly flat. "
                        "This is the guard against pushing smooth gradients to "
                        "space: dec['gate'] alone cannot tell a flat cell from "
                        "a shallow ramp (both have small sep -- gate needs ~8 "
                        "Lab of range to open) but only the ramp has a "
                        "first-order spatial term")
    p.add_argument("--space-warm", type=int, default=200,
                   help="(--space-weight) steps to ramp the weight in from 0")
    p.add_argument("--density-weight", type=float, default=0.0,
                   help="penalise fitted colours by their NEGATIVE LOG DENSITY "
                        "under the empirical colour distribution of the cell "
                        "and its 8 neighbours. Averaging two clusters lands in "
                        "the valley between their modes and is punished; either "
                        "mode is free; a gradient is a continuum so every point "
                        "on the ramp is free. Anchored to the PHOTO, unlike a "
                        "reach penalty, which is anchored to decompose's "
                        "opinion. NOT sufficient against collapse on its own: "
                        "fg=bg=the modal colour scores perfectly, so this must "
                        "be paired with reconstruction, never run alone")
    p.add_argument("--density-sigma", type=float, default=4.0,
                   help="(--density-weight) Lab bandwidth of the KDE. Default = "
                        "decompose's jnd, so 'present in the neighbourhood' "
                        "means present to within a just-noticeable difference")
    p.add_argument("--density-cap", type=float, default=8.0,
                   help="(--density-weight) clamp on the per-colour nll. It "
                        "grows ~d^2/2sigma^2 outside every mode, so uncapped a "
                        "single badly-placed cell outweighs the batch")
    p.add_argument("--density-samples", type=int, default=128,
                   help="(--density-weight) pixels sampled per cell; the pool "
                        "is 9x this across the neighbourhood")
    p.add_argument("--density-warm", type=int, default=200,
                   help="(--density-weight) steps to ramp the weight in from 0")
    p.add_argument("--glyph-ctx-aux", action="store_true",
                   help="(--glyph-ctx + --centre-only) read the NON-CENTRE cells' "
                        "glyph distributions off the auxiliary head instead of the "
                        "main one. Under --centre-only the main head gets CE on "
                        "the centre alone, so head(t) on a neighbour is a readout "
                        "no loss ever calibrated -- and that is what glyph_ctx fed "
                        "the colouriser for 14 of its 15 cells through all of runF. "
                        "Makes aux_block load-bearing at inference (lite builds it)")
    p.add_argument("--mask-bcast-mean", action="store_true",
                   help="feed the per-pixel encoder rgb MINUS the cell mean as 3 "
                        "extra broadcast input channels. Its receptive field is "
                        "13x13 over a 44x24 patch, so 'am I above or below this "
                        "cell's average' is not merely unlearned but "
                        "unrepresentable. Safe where dec fg/bg is not: the mean is "
                        "invariant to which cluster is called foreground, so it "
                        "cannot leak polarity. Zero-init widening, no-op at step 0")
    p.add_argument("--mask-bcast-glyph", type=int, default=0,
                   help="broadcast a learned GLYPH EMBEDDING of this width to the "
                        "per-pixel encoder as extra input channels (0 = off). Same "
                        "softmax-weighted-sum mechanism as --glyph-ctx with its own "
                        "matrix, delivered to the pixels rather than the token. "
                        "Constant within a cell, and the encoder has no positional "
                        "input, so it cannot paint a glyph-shaped mask")
    p.add_argument("--centre-only", action="store_true",
                   help="train the main glyph head on the CENTRE cell only "
                        "instead of the binomial-weighted 3x5 window. Pair with "
                        "--ensemble center at inference")
    p.add_argument("--aux-weight", type=float, default=0.0,
                   help="weight on an AUXILIARY glyph head (its own transformer "
                        "block, dropped at inference) trained on the non-centre "
                        "cells. Keeps neighbour prediction as a regulariser on "
                        "the tokens without letting it steer the main head")
    p.add_argument("--mask-flip", action="store_true",
                   help="give mask_dec a per-cell SIGNED gain on the fg-vs-bg "
                        "decision, from the token (FiLM). The token currently "
                        "reaches the decoder additively, so a cell-level fact "
                        "cannot invert a spatial pattern; measured consequence "
                        "is that the mask learns only 'subject = brighter side' "
                        "and is at chance on the 69%% of cells where the subject "
                        "is darker. Zero-init, exact no-op at step 0")
    p.add_argument("--mask-entropy2", type=float, default=0.0,
                   help="penalty on the CONDITIONAL fg-vs-bg entropy -- p[:2] "
                        "renormalised so abstain is out of it, weighted by "
                        "committed mass. The 3-way --mask-entropy can be paid "
                        "off by squeezing the near-unused abstain channel and "
                        "leave the fg/bg decision untouched, which is what it "
                        "did in runC; this cannot. GATED: commitment pressure "
                        "applied while polarity is still unsettled does not "
                        "resolve the decision, it freezes coin flips")
    p.add_argument("--mask-entropy2-gate", type=float, default=0.55,
                   help="open the gate when the EMA of the polarity metric `pol` "
                        "reaches this. Set <= -1 to ignore polarity and use "
                        "--mask-entropy2-start alone")
    p.add_argument("--mask-entropy2-start", type=int, default=-1,
                   help="also open the gate at this step regardless of `pol` "
                        "(-1 = never; the gate is then purely polarity-driven). "
                        "Whichever condition fires first opens it, and it stays "
                        "open -- no oscillating on a noisy metric")
    p.add_argument("--mask-entropy2-min-step", type=int, default=100,
                   help="the gate cannot open before this step no matter what "
                        "`pol` says. The EMA initialises to its FIRST reading, "
                        "so one lucky photo can otherwise trip the gate on step "
                        "1 -- observed in a smoke, gate open at step 1 on a "
                        "single-photo pol of 0.726 against a 0.45 threshold")
    p.add_argument("--mask-entropy2-ramp", type=int, default=200,
                   help="steps to ramp the term from 0 to full once the gate "
                        "opens, so it arrives as pressure rather than a shock")
    p.add_argument("--photo-batch", type=int, default=1,
                   help="photos per step on the CLIP arm. At 1, a 1400-step run "
                        "sees each of 763 photos ~1.8 times -- the thinnest data "
                        "coverage of any arm, on the term we most want to move")
    p.add_argument("--photo-cells", type=int, default=700,
                   help="target cell budget per photo; width is DERIVED from the "
                        "image's own aspect so landscape and portrait cost the "
                        "same. Resolution then varies on purpose rather than as a "
                        "side effect of how the crop was framed")
    p.add_argument("--photo-cells-jitter", type=float, default=0.35,
                   help="cell budget is sampled in [1-j, 1+j] x --photo-cells")
    p.add_argument("--photo-crop", type=float, default=0.95,
                   help="fraction of the frame kept, at NATIVE aspect; the "
                        "remaining wiggle is jittered per step (anti-memorisation). "
                        "1.0 = the whole image every time")
    p.add_argument("--bwd-keep", type=float, default=1.0,
                   help="fraction of cells whose render carries gradient. The "
                        "forward is always the WHOLE image -- CLIP needs the scene "
                        "to know which lines matter -- but the backward can be "
                        "restricted by detaching the rest")
    p.add_argument("--bwd-drop-empty", type=float, default=0.0,
                   help="fraction of near-empty cells dropped from the backward "
                        "BEFORE the uniform --bwd-keep subsample. Do not set this "
                        "near 1.0: under-training the is-this-cell-blank decision "
                        "is how phantom characters appear in flat regions")
    # ---- regions
    p.add_argument("--region-rows", type=int, default=26)
    p.add_argument("--region-cols", type=int, default=44)
    p.add_argument("--pool-scatter", type=int, default=4)
    p.add_argument("--batch", type=int, default=384)
    # ---- CLIP, matched to the engine that produced the targets
    p.add_argument("--clip-aug", type=int, default=16)
    p.add_argument("--clip-crop-scale", type=float, nargs=2, default=[0.4, 0.9])
    p.add_argument("--clip-scale-alpha", type=float, default=0.8)
    p.add_argument("--clip-microbatch", type=int, default=0,
                   help="crops per CLIP encoder chunk (0 = all at once). With "
                        "--photo-batch > 1 every photo's graph is alive until the "
                        "backward, so the RN101 activations multiply: 12 photos "
                        "x 16 augs x 2 images OOM'd a 94 GB card. Chunking frees "
                        "them per group at the cost of a little recompute")
    # ---- rolling pool
    p.add_argument("--pool", default=None, help="pool dir (default runs/<name>/pool)")
    p.add_argument("--pool-size", type=int, default=96)
    p.add_argument("--pool-add", type=int, default=16, help="entries per refresh")
    p.add_argument("--pool-refresh-every", type=int, default=150)
    p.add_argument("--pool-prefill", type=int, default=0,
                   help="entries refined BEFORE step 0 (pool starts full)")
    p.add_argument("--pool-workers", type=int, default=8)
    p.add_argument("--pool-gpus", default="")
    p.add_argument("--pool-widths", default="40,60,80")
    p.add_argument("--pool-iters", type=int, default=200)
    p.add_argument("--pool-lead", type=float, default=1.0)
    p.add_argument("--pool-w-temp", type=float, default=0.66)
    p.add_argument("--pool-z-noise", type=float, default=0.49)
    p.add_argument("--pool-prep-max", type=int, default=40,
                   help="prepped entries held resident between refreshes. Set it "
                        ">= --pool-size or the cache THRASHES: with 40 slots for "
                        "64 entries and 4 random draws a step, 1.5 draws a step "
                        "missed and paid a full re-prep (~140 ms each, ~0.2 s a "
                        "step). The cache is host RAM -- ~60 MB an entry -- not "
                        "GPU, so there is no reason to keep it small")
    p.add_argument("--pool-color", default="lite", choices=["lite", "fit"],
                   help="how the pool's refinements are COLOURED. 'lite' = the "
                        "model's own mask decoder, so returned grids are "
                        "reachable rather than aspirational. 'fit' = the "
                        "engine's closed-form MSE optimum -- deterministic in "
                        "(photo, glyph), so the glyph targets stop depending on "
                        "a mask that is still moving. Also drops the per-"
                        "candidate model forward, so refreshes are cheaper")
    p.add_argument("--pool-seed", type=int, default=-1,
                   help="separate seed for WHERE each source's pool cursor "
                        "starts (-1 = index 0, the old behaviour). Use it to "
                        "draw different photos on a continued run: --seed fixes "
                        "the shuffle AND therefore which photos are held out as "
                        "validators, so bumping that instead would silently "
                        "change the eval set")
    p.add_argument("--pool-color-switch", type=int, default=-1,
                   help="step at which --pool-color flips to the OTHER source "
                        "(-1 = never). The point of 'fit' first is to give the "
                        "glyph targets a stable reference while the mask is "
                        "still moving; once it holds, switching to 'lite' "
                        "restores the reachable-not-aspirational property, so "
                        "the engine stops choosing glyphs that only look good "
                        "under a palette the model would never assign")
    p.add_argument("--pool-args", default="", help="extra raw engine args")
    # ---- optim
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--optim", default="muon", choices=["adamw", "muon"])
    p.add_argument("--muon-lr", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask-lr", type=float, default=0.0,
                   help="separate AdamW lr for the mask branch's SPATIAL readout "
                        "(rgb_trunk + mask_dec). 0 = off, they stay in the main "
                        "AdamW group at --lr. These are the weights asked to "
                        "learn a fg/bg rule from scratch and they were the "
                        "slowest-moving in the model (3-8%% vs mask_blocks' "
                        "100%% over runC's first 400 steps) purely because "
                        "split_muon_params sends convs to AdamW and anything "
                        "named *blocks.* to Muon")
    p.add_argument("--lr-floor", type=float, default=0.0,
                   help="fraction of peak lr the MAIN groups decay to. 0 = the "
                        "usual cosine-to-zero. Worth raising in a chain of "
                        "fine-tunes: each run warms to peak and anneals to zero, "
                        "so across runs the schedule is a sawtooth and every "
                        "run's last stretch does nothing before the next yanks "
                        "the rate back up. Note the anneal is also protective -- "
                        "it is part of why lineart val holds flat late in a run")
    p.add_argument("--mask-lr-floor", type=float, default=0.5,
                   help="(--mask-lr) fraction of peak lr this group decays to. "
                        "The shared cosine anneals to 0, which stops the mask "
                        "branch learning exactly when it is still improving; "
                        "1.0 = no decay at all for this group")
    p.add_argument("--resume-lr-frac", type=float, default=0.0,
                   help="start the cosine this far ALONG instead of at the peak "
                        "(0 = old behaviour, 0.5 = begin at half the peak rate). "
                        "For continuations: runF ended at 20%% of peak on the main "
                        "groups and 50%% on the mask branch, and re-entering at "
                        "100%% re-shocks exactly what converged -- it cost runE3 "
                        "~100 steps. --warmup still applies, since the optimizer "
                        "state is fresh either way")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    outdir = os.path.join(G.REPO_ROOT, "runs", "joint", args.name)
    os.makedirs(outdir, exist_ok=True)

    json.dump({k: v for k, v in sorted(vars(args).items())},
              open(os.path.join(outdir, "args.json"), "w"), indent=1)
    phases = (CAMPAIGN_PHASES if args.schedule == "campaign" else
              [(0, args.w_line_ce, args.w_line_color, args.w_pool_color,
                args.w_pool_line, args.w_clip, args.w_mask_tgt)])
    if args.mask_tgt_w > 0 and args.mask_tgt_from is not None:
        print(f"  mask_tgt injected at {args.mask_tgt_w:g} at step "
              f"{args.mask_tgt_from}, decaying linearly to 0 at "
              f"{args.mask_tgt_until}; other arms take up the slack", flush=True)
    if args.schedule == "campaign":
        print("  schedule: campaign — "
              + " | ".join(f"@{r[0]} " + "/".join(f"{x:.2f}" for x in r[1:])
                           for r in phases)
              + f"   (arms: {'/'.join(ARMS)})", flush=True)
    args.start_step, _resume_ck = 0, None
    if args.resume:
        _resume_ck = torch.load(G.repo_path(args.resume), map_location="cpu",
                                weights_only=False)
        if args.resume_fresh_opt:
            for _k in ("opt", "gn_ema", "ema_pol"):
                _resume_ck.pop(_k, None)
        args.init, args.random_init = args.resume, False
        args.start_step = int(args.resume_step if args.resume_step is not None
                              else _resume_ck.get("step", 0))
        if "opt" in _resume_ck:
            args.resume_warmup = 0
        print(f"  RESUME from {args.resume} at step {args.start_step}"
              + (" (optimizer + EMA state restored)" if "opt" in _resume_ck
                 else f" (no optimizer state in ckpt: {args.resume_warmup}-step re-warm)"),
              flush=True)
    if args.random_init:
        from unicasso.substrate import glyphs as _G
        _ink, _chars = _G.load_glyphs(device="cpu", pad=0, profile=args.profile)
        _pro = json.load(open(G.repo_path(f"kits/{args.profile}/profile.json")))
        _ch, _cw = _pro["cell_h"], _pro["cell_w"]
        _ph, _pw = (4, 3) if (_ch, _cw) == (36, 18) else (3, 4)
        # profile MUST be stamped: Lite(<ckpt path>) ignores its font arg and
        # reads cfg["profile"] (default dejavu). Every resumed run inherited it
        # from the runF lineage; a random-init cfg had none, so the pool
        # teacher (and the shipped model.pt) silently became a DejaVu Lite --
        # invisible on macOS where that font exists, fatal on the box.
        # Likewise color_model: Lite gates color_proj/mask_dec/ctx/aux on it
        # and otherwise builds a bare classifier that cannot load these
        # weights. align_blend / mask_margin_fit are Lite-only inference
        # heuristics that were True through the whole sfmono lineage (em8 ->
        # runG -> H6); stamped so the pool teacher colours as inference does.
        cfg = dict(cell_h=_ch, cell_w=_cw, pad_h=_ph, pad_w=_pw,
                   dim=args.rand_dim, heads=args.rand_heads,
                   blocks=args.rand_blocks, in_ch=4, feat_dim=11,
                   profile=args.profile, color_model=True, mask_dec=True,
                   align_blend=True, mask_margin_fit=True)
        model, chars, cfg = build_random(_chars, cfg, dev)
        print(f"  RANDOM INIT: {len(chars)} chars, dim {cfg['dim']}, "
              f"cell {_ch}x{_cw}, pad {_ph}x{_pw}, "
              f"{sum(q.numel() for q in model.parameters())/1e6:.2f}M params",
              flush=True)
    else:
        if not args.init:
            raise SystemExit("joint_train: pass --init CKPT to continue a model, "
                             "or --random-init to train from scratch "
                             "(train_campaign.sh does the latter)")
        model, chars, cfg = build_model(args.init, dev)
    if args.rgb_trunk and getattr(model, "rgb_trunk", None) is None:
        model.rgb_trunk = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU()).to(dev)
        with torch.no_grad():        # warm start: the shared trunk's first conv
            # already extracts edges from these photos; take its RGB channels so
            # the new encoder starts competent instead of random
            model.rgb_trunk[0].weight.copy_(model.trunk[0].weight[:, 1:4])
            model.rgb_trunk[0].bias.copy_(model.trunk[0].bias)
            model.rgb_trunk[1].weight.copy_(model.trunk[1].weight)
            model.rgb_trunk[1].bias.copy_(model.trunk[1].bias)
        cfg["rgb_trunk"] = True
        print(f"  +RGB encoder for the mask branch "
              f"({sum(x.numel() for x in model.rgb_trunk.parameters())/1e3:.1f}k "
              f"params, warm-started from trunk[0][:,1:4])", flush=True)
    _md = getattr(model, "mask_dec", None)
    _geom_changed = (args.mask_attn and isinstance(_md, MaskAttnDecoder)
                     and _md.d != args.mask_attn_dim)
    if _geom_changed:
        # the surgery below only fires when there is no attention head at all,
        # so without this a geometry change on a continued run is silently
        # ignored -- the checkpoint's head is kept and the flags do nothing.
        print(f"  ! mask head width changed (d {_md.d} -> {args.mask_attn_dim}) "
              f"-- REBUILDING from random init, its trained weights are "
              f"discarded. Pair with --mask-target to re-bootstrap. (Depth is "
              f"appendable without loss: use --mask-attn-extra.)", flush=True)
        model.mask_dec = None
    if args.mask_attn and not isinstance(model.mask_dec, MaskAttnDecoder):
        model.mask_dec = MaskAttnDecoder(cfg["dim"], d=args.mask_attn_dim,
                                         depth=args.mask_attn_depth,
                                         mid=args.mask_attn_mid).to(dev)
        cfg["mask_attn"] = True
        cfg["mask_attn_dim"] = args.mask_attn_dim
        cfg["mask_attn_depth"] = args.mask_attn_depth
        cfg["mask_attn_mid"] = args.mask_attn_mid
        cfg.pop("mask_flip", None)
        print(f"  mask head -> MaskAttnDecoder (d={args.mask_attn_dim}, "
              f"{sum(x.numel() for x in model.mask_dec.parameters())/1e3:.1f}k "
              "params, RANDOM init -- pair with --mask-target)", flush=True)
    if args.mask_flip and getattr(model.mask_dec, "flip", None) is None:
        model.mask_dec.add_flip()
        model.mask_dec.flip.to(dev)
        cfg["mask_flip"] = True
        print("  +per-cell flip gain on mask_dec "
              f"({sum(x.numel() for x in model.mask_dec.flip.parameters())} "
              "params, s=1 exactly at step 0)", flush=True)
    cfg["render_ensemble"] = args.render_ensemble
    cfg["mask_forward"] = "one"      # this trainer is single-pass; stamped so
    # Lite/probes default to the structure the weights were actually trained on
    if args.mask_attn_extra and isinstance(model.mask_dec, MaskAttnDecoder):
        had = len(model.mask_dec.px_res)
        if args.mask_attn_extra > had:
            model.mask_dec.add_px_blocks(args.mask_attn_extra,
                                         args.mask_attn_mid)
            model.mask_dec.to(dev)
            cfg["mask_attn_extra"] = args.mask_attn_extra
            rf = 1 + 2 * (2 + 2 * args.mask_attn_extra)
            print(f"  per-pixel encoder depth {had} -> {args.mask_attn_extra} "
                  f"residual blocks (+{args.mask_attn_extra - had} identity-init, "
                  f"exact no-op at step 0; receptive field now ~{rf}x{rf})",
                  flush=True)
    have = len(getattr(model, "mask_blocks", []) or [])
    if args.mask_blocks > have:
        model.add_mask_blocks(args.mask_blocks, cfg["heads"])
        model.mask_blocks.to(dev)
        cfg["mask_blocks"] = args.mask_blocks
        print(f"  mask blocks {have} -> {args.mask_blocks} "
              f"(+{args.mask_blocks - have} identity-init, exact no-op at step 0; "
              f"{sum(x.numel() for x in model.mask_blocks.parameters())/1e3:.0f}k "
              f"total)", flush=True)
    if args.mask_ctx and getattr(model, "mask_ctx_trunk", None) is None:
        model.add_mask_ctx(cfg["dim"])
        model.mask_ctx_trunk.to(dev); model.mask_ctx_proj.to(dev)
        model.mask_ctx_inject = args.mask_ctx_inject
        cfg["mask_ctx"] = True
        cfg["mask_ctx_inject"] = args.mask_ctx_inject
        n = sum(x.numel() for x in model.mask_ctx_trunk.parameters()) + \
            sum(x.numel() for x in model.mask_ctx_proj.parameters())
        print(f"  +mask colour encoder ({n/1e3:.0f}k params, own conv trunk over "
              f"RGB, injected at mask block {args.mask_ctx_inject}; zero-init "
              f"projection so it contributes 0 at step 0)", flush=True)
    if args.glyph_ctx and getattr(model, "glyph_ctx_proj", None) is None:
        model.add_glyph_ctx(cfg["dim"], len(chars)).to(dev)
        model.glyph_ctx_inject = args.glyph_ctx_inject
        cfg["glyph_ctx"] = True
        cfg["glyph_ctx_inject"] = args.glyph_ctx_inject
        print(f"  +glyph->colouriser token (Linear({len(chars)} -> {cfg['dim']}) "
              f"on the glyph softmax, injected at mask block "
              f"{args.glyph_ctx_inject}; zero-init, no-op at step 0)", flush=True)
    if args.aux_weight > 0 and getattr(model, "aux_block", None) is None:
        model.add_aux_block(cfg["heads"]).to(dev)
        cfg["aux_block"] = True
        print(f"  +auxiliary glyph block "
              f"({sum(x.numel() for x in model.aux_block.parameters())/1e3:.0f}k "
              f"params, trains the NON-centre cells; dropped at inference)",
              flush=True)
    if args.glyph_ctx_aux:
        if getattr(model, "aux_block", None) is None:
            raise SystemExit("--glyph-ctx-aux needs an auxiliary block "
                             "(--aux-weight > 0, or an --init that has one)")
        model.glyph_ctx_aux = True
        cfg["glyph_ctx_aux"] = True
        print("  glyph->colouriser readout: centre from the MAIN head, the 14 "
              "neighbours from the AUXILIARY head (the path each was trained "
              "under). aux_block is now load-bearing at inference.", flush=True)
    _gemb_fresh = False
    if (args.mask_bcast_mean or args.mask_bcast_glyph) \
            and isinstance(model.mask_dec, MaskAttnDecoder) \
            and not (model.mask_dec.bc_mean or model.mask_dec.n_glyph):
        _gemb_fresh = bool(args.mask_bcast_glyph)
        n_new = model.mask_dec.add_bcast(len(chars), args.mask_bcast_glyph,
                                         args.mask_bcast_mean,
                                         cfg["pad_h"], cfg["pad_w"])
        model.mask_dec.to(dev)
        cfg["mask_bcast_mean"] = bool(args.mask_bcast_mean)
        cfg["mask_bcast_glyph"] = int(args.mask_bcast_glyph)
        parts = []
        if args.mask_bcast_mean:
            parts.append("rgb-cell_mean (3)")
        if args.mask_bcast_glyph:
            parts.append(f"glyph embedding ({args.mask_bcast_glyph})")
        print(f"  +broadcast conditioning on the per-pixel encoder: "
              f"{' + '.join(parts)} = {n_new} extra input channels, constant "
              f"within a cell. conv0 widened with ZEROED columns, so it is the "
              f"exact trained encoder at step 0", flush=True)
    ch, cw, ph, pw = cfg["cell_h"], cfg["cell_w"], cfg["pad_h"], cfg["pad_w"]
    in4 = cfg.get("in_ch", 1) == 4
    if getattr(model, "mask_dec", None) is None:
        raise SystemExit(f"--init {args.init} has no mask decoder")
    cache = CellCache(G.repo_path(args.cache), ROWS, COLS, ph, pw, ch, cw,
                      window_labels=True)
    train_runs = sorted({int(i) for i in cache.index["train"][:, 0]})
    groups = photo_groups(args.photos, seed=args.seed)
    # ---- photo validators, chosen ONCE at startup and HELD OUT ---------------
    # Taken off the end of each group's shuffled list (the training cursor walks
    # from the front) and removed from `groups` entirely, so neither the CLIP arm
    # nor the pool ever refines them. Split proportional to each source's share
    # so the validator set has the same mix the training set does.
    val_photos = []
    if args.val_photos:                    # EXPLICIT list wins. Without it the
        # set comes from a directory listing and changes whenever the photo dir
        # does -- local and the box already produced different validators from
        # the same seed (516 vs 537 files), which silently breaks cross-run
        # comparison of every [pval@] number.
        raw = (open(G.repo_path(args.val_photos)).read()
               if os.path.exists(G.repo_path(args.val_photos)) else args.val_photos)
        val_photos = [x.strip() for x in raw.replace(",", "\n").split("\n")
                      if x.strip()]
        groups = [(n, [f for f in fs if f not in set(val_photos)], w)
                  for n, fs, w in groups]
    elif args.photo_val > 0 and len(groups):
        want = [g[2] * args.photo_val for g in groups]
        take = [int(x) for x in want]
        for j in sorted(range(len(take)), key=lambda i: want[i] - take[i],
                        reverse=True)[:args.photo_val - sum(take)]:
            take[j] += 1
        ng = []
        for (name, files, w), k in zip(groups, take):
            k = min(k, max(0, len(files) - 1))
            if k:
                val_photos += files[-k:]
            ng.append((name, files[:len(files) - k] if k else files, w))
        groups = ng
    photos = [f for _, fs, _ in groups for f in fs]
    args.val_photos_resolved = list(val_photos)     # -> args.json
    json.dump({k: v for k, v in sorted(vars(args).items())},   # re-dump: the
              open(os.path.join(outdir, "args.json"), "w"), indent=1)  # first dump
    # ran before the val split was resolved and recorded None
    gshare = np.array([g[2] for g in groups])
    kernel = binomial_kernel(ROWS, COLS, dev)
    # 81.3% of the binomial kernel's weight sits on NON-centre cells, so the
    # shared token is mostly trained to name its neighbours. --centre-only moves
    # the main head onto the centre alone; --aux-weight puts the neighbour
    # signal behind its own block, where it regularises without steering.
    tc = (ROWS // 2) * COLS + COLS // 2       # centre index within the window
    kern_c = torch.zeros_like(kernel); kern_c[tc] = 1.0
    kern_aux = kernel.clone(); kern_aux[tc] = 0.0
    kern_main = kern_c if args.centre_only else kernel
    ce_w = torch.ones(len(chars), device=dev)
    ce_w[cache.space] = args.blank_weight

    from unicasso.adapter.corrupt import CorruptionSampler
    smp = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu",
                            profile=args.profile)
    ink_flat = (1.0 - smp.bitmaps.cpu().float()).reshape(smp.N, -1).to(dev)
    del smp
    # seeded HERE rather than in the surgery block above because ink_flat does
    # not exist until now. Only for a freshly-added table: a checkpoint that
    # already carries a trained gemb must keep it.
    if getattr(model.mask_dec, "n_glyph", 0) and _gemb_fresh:
        k = model.mask_dec.seed_gemb(ink_flat, ch, cw)
        print(f"  glyph embedding warm-started from geometry: first {k} dims = "
              f"[coverage, ink centroid y, centroid x, vertical spread], "
              f"standardised over the {len(chars)} glyphs. Read by a one-hot "
              f"softmax this table is 358 near-disjoint columns -- glyph_ctx, "
              f"same structure, finished runF at 0.10% of its token.",
              flush=True)

    from unicasso.engine.clip_loss import CLIPPerceptualLoss
    clipper = CLIPPerceptualLoss(torch.device(dev), model_name="RN101",
                                 pretrained="openai", n_aug=args.clip_aug,
                                 crop_scale=tuple(args.clip_crop_scale),
                                 scale_alpha=args.clip_scale_alpha,
                                 batch_aug=(dev == "cuda"),
                                 microbatch=args.clip_microbatch)

    pool = RollingPool(args.pool or os.path.join("runs", "joint", args.name, "pool"),
                       groups, os.path.join("runs", "joint", args.name),
                       widths=[int(x) for x in args.pool_widths.split(",")],
                       size=args.pool_size, workers=args.pool_workers,
                       gpus=args.pool_gpus, font=args.profile,
                       iters=args.pool_iters, lead=args.pool_lead,
                       w_temp=args.pool_w_temp, z_noise=args.pool_z_noise,
                       read=dict(path=args.read_path, frac=args.read_frac,
                                 count=args.read_count, ridge=args.read_ridge,
                                 temp=args.read_temp),
                       extra_args=[a for a in args.pool_args.split(" ") if a],
                       prep_max=args.pool_prep_max, seed=args.seed, device=dev,
                       color=args.pool_color,
                       cursor_seed=(args.pool_seed if args.pool_seed >= 0
                                    else None))

    # ---- lineart pool ---------------------------------------------------
    # Additive to ce_line, never a replacement: the static cache stays the
    # anchor. Colour is irrelevant here, so the engine always runs --color-fit.
    # Val parents are excluded or [val@] leaks through the pool.
    pool_line = None
    if any(r[ARMS.index("pool_line") + 1] > 0 for r in phases) or args.w_pool_line > 0:
        excl = set()
        if args.pool_line_exclude:
            _m = json.load(open(G.repo_path(args.pool_line_exclude)))
            excl = {r["parent"] for r in _m["runs"] if r["split"] == "val"}
        _imgs = sorted(glob.glob(os.path.join(
            G.repo_path(args.pool_line_images), "*")))
        _imgs = [f for f in _imgs
                 if os.path.splitext(os.path.basename(f))[0] not in excl
                 and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        lgroups = [("lineart", _imgs, 1.0)]
        print(f"  lineart pool source: {len(_imgs)} images"
              + (f" ({len(excl)} val parents excluded)" if excl else ""),
              flush=True)
        pool_line = RollingPool(
            args.pool_line or os.path.join("runs", "joint", args.name, "pool_line"),
            lgroups, os.path.join("runs", "joint", args.name),
            widths=[int(x) for x in args.pool_widths.split(",")],
            size=args.pool_size, workers=args.pool_workers,
            gpus=args.pool_gpus, font=args.profile,
            iters=args.pool_iters, lead=args.pool_lead,
            w_temp=args.pool_w_temp, z_noise=args.pool_z_noise,
            read=dict(path=args.read_path, frac=args.read_frac,
                      count=args.read_count, ridge=args.read_ridge,
                      temp=args.read_temp),
            extra_args=[a for a in args.pool_args.split(" ") if a],
            prep_max=args.pool_prep_max, seed=args.seed, device=dev,
            color="fit",
            cursor_seed=(args.pool_seed if args.pool_seed >= 0 else None))

    print(f"[{args.name}] {sum(x.numel() for x in model.parameters())/1e6:.2f}M "
          f"params | {len(train_runs)} lineart runs | {len(photos)} photos | "
          f"pool {len(pool)}/{args.pool_size} | {dev}", flush=True)
    print("  photo mix: " + ", ".join(f"{n} {len(f)} files @ {w:.0%}"
                                      for n, f, w in groups), flush=True)

    # ------------------------------------------------------------- the terms
    def forward_region(x, feats_tok, mode):
        t = model._tokens(x, None, feats_tok, mode)
        return model.head(t[:, model.n_extra:]), t

    def _with_aux(lg, t, y):
        """main CE (centre-weighted or centre-only) + the auxiliary head's CE
        on the NON-centre cells, down its own block."""
        loss = kernel_ce(lg, y, kern_main, ce_w)
        if args.aux_weight > 0 and getattr(model, "aux_block", None) is not None:
            aux = model.aux_logits(t)[:, model.n_extra:]
            loss = loss + args.aux_weight * kernel_ce(aux, y, kern_aux, ce_w)
        return loss

    def _one_region():
        ri = train_runs[int(rng.integers(len(train_runs)))]
        gh, gw = cache.labels[ri].shape
        rh0 = min(gh, args.region_rows)
        cw0 = min(gw, args.region_cols)
        top = int(rng.integers(0, gh - rh0 + 1))
        left = int(rng.integers(0, gw - cw0 + 1))
        ys, xs = np.mgrid[top:top + rh0, left:left + cw0]
        return np.stack([np.full(rh0 * cw0, ri), ys.ravel(), xs.ravel()],
                        axis=1).astype(np.int32)

    def ce_line():
        """The anchor -- --line-regions crops, averaged in one CE.

        At 1 this is a batch of ONE 26x44 crop out of 254 runs, and it shows:
        `line` swings 0.42 to 2.22 between adjacent steps. That variance is not
        harmless. This is simultaneously the noisiest term in the objective and
        the term holding the model in place, and since lam is calibrated against
        ||g_ce||, the noise leaks straight into the photo arm's weight -- lineart
        crop luck modulating how hard CLIP pulls. ~26 ms a region against a
        ~1900 ms step, so averaging a few is nearly free."""
        idx = np.concatenate([_one_region()
                              for _ in range(max(1, args.line_regions))])
        w, lab, _, _ = cache.fetch(idx)
        x = torch.from_numpy(w).to(dev).float().div_(255).unsqueeze(1)
        y = torch.from_numpy(lab).to(dev)
        lg, t = forward_region(x, None, None)
        return _with_aux(lg, t, y)

    def ce_pool(pool):
        files = pool.entries()
        if not files:
            return None
        picks = rng.choice(len(files), size=min(args.pool_scatter, len(files)),
                           replace=False)
        # windows allocated PROPORTIONAL to grid size. A flat budget gave every
        # entry the same 96 windows regardless of M, so a w40 entry (M=1120) got
        # 2.2x the per-cell exposure of a w60 one (M=2478) -- a sampler artifact,
        # not a design choice. Total per step is still args.batch.
        ents = [pool.prep(files[int(pi)], ch, cw, ph, pw, ROWS, COLS)
                for pi in picks]
        Ms = np.array([e["gh"] * e["gw"] for e in ents], dtype=np.float64)
        nsubs = np.maximum(16, np.round(args.batch * Ms / Ms.sum())).astype(int)
        xs, fts, yas = [], [], []
        for e, nsub in zip(ents, nsubs):
            M = e["gh"] * e["gw"]
            sel = rng.choice(M, size=min(int(nsub), M), replace=False)
            # stays uint8 until it is on the device: converting on the host both
            # spends CPU in the step and quadruples the bytes crossing the link
            xi = torch.from_numpy(e["win_ink"][sel]).unsqueeze(1)
            if in4:
                xc = torch.from_numpy(e["win_rgb"][sel]).permute(0, 3, 1, 2)
                xi = torch.cat([xi, xc], dim=1)
            ids_s, val_s = e["ids"][sel], e["valid"][sel]
            xs.append(xi)
            fts.append(e["feats"][ids_s] * val_s[:, :, None].float())
            yas.append(torch.where(val_s, e["glyphs"][ids_s],
                                   torch.full_like(ids_s, -1)))
        x = torch.cat(xs).to(dev).float().div_(255)
        ft = torch.cat(fts).to(dev)
        y = torch.cat(yas).to(dev)
        lg, t = forward_region(x, ft, 1)
        return _with_aux(lg, t, y)

    def _render_grid(rgb, gh, W, det=False, want_mask=False):
        """Everything from an RGB grid to a rendered ASCII image.

        Shared by the photo arm (CLIP against the photo) and the lineart
        colour arm (MSE against a synthetic two-colour target). Kept in ONE
        place because divergence between two copies of this is silent --
        the weights would simply be read through a structure they were
        never trained under, which has already happened twice here.

        Returns (render, l1, ent, ent2, tgt, flipped, pol, inv, reach)."""
        dec = decompose(rgb, gh, W, ch, cw)
        # uint8 round-trip preserved deliberately: the ink channel the model was
        # trained on, and that Lite feeds it at inference, is quantised. to(uint8)
        # truncates exactly as .astype(np.uint8) did.
        ink_u8 = ((1.0 - nomination_target(dec)) * 255.0).clamp(0, 255) \
            .to(torch.uint8)
        M = gh * W
        ids, valid = token_maps(gh, W, ROWS, COLS)
        ids_t = torch.from_numpy(ids).to(dev)
        val_t = torch.from_numpy(valid).to(dev)
        ft = cell_feats(dec)[ids_t] * val_t[:, :, None].float()
        xi = grid_windows_t(ink_u8.float(), gh, W, ch, cw, ph, pw) \
            .div_(255).unsqueeze(1)
        xc = grid_windows_t(rgb, gh, W, ch, cw, ph, pw, edge=True) \
            .permute(0, 3, 1, 2)
        x1 = torch.cat([xi, xc], 1) if in4 else xi

        t, s1 = model._tokens(x1, None, ft, 1, want_skip=True)
        lg = model.head(t[:, model.n_extra:])
        if args.render_ensemble == "center":
            # CENTRE ONLY. token_maps puts offset (0,0) at the centre position,
            # so window m's centre token IS grid cell m -- one forward, one
            # prediction, no accumulation. This matches --centre-only (which
            # trains head(t) at the centre alone) and matches what `lite
            # --ensemble center` ships. The old kernel ensemble read head(t) at
            # ALL 15 positions with 81.2% of the weight on positions no CE
            # supervises, which became wrong the moment --centre-only landed in
            # runF and was never revisited.
            prob = lg[:, model.center - model.n_extra].softmax(-1)
        else:
            pr = lg.softmax(-1)
            wt = kernel[None, :] * val_t.float()
            acc = torch.zeros(M, pr.shape[-1], device=dev)
            acc.index_add_(0, ids_t.reshape(-1),
                           (pr * wt[:, :, None]).reshape(-1, pr.shape[-1]))
            wsum = torch.zeros(M, device=dev)
            wsum.index_add_(0, ids_t.reshape(-1), wt.reshape(-1))
            prob = acc / wsum.clamp_min(1e-8)[:, None]
        lp = prob.clamp_min(1e-9).log()
        # ARGMAX by default. Sampling a glyph does not merely add noise: the
        # render is bg + (fg-bg)*ink, so under a random ink the variance of the
        # render is (fg-bg)^2 * Var(ink), and the expected-loss-minimising move
        # is to make the render independent of the glyph -- i.e. collapse
        # fg toward bg. Stochastic glyphs PAY the model to wash out. The
        # straight-through estimator does not need the sample; argmax + the
        # same (onehot + p - p.detach()) is a valid, lower-variance gradient.
        if det or args.glyph_temp <= 0:
            hard = lp.argmax(-1)
        else:
            u = torch.rand_like(lp).clamp_(1e-9, 1 - 1e-9)
            hard = (lp / args.glyph_temp - torch.log(-torch.log(u))).argmax(-1)
        onehot = torch.zeros_like(prob).scatter_(1, hard[:, None], 1.0)
        st = onehot + prob - prob.detach()
        ink_st = st @ ink_flat

        mlog = model.mask_center(x1, t, s1)
        pm = mlog[:, :, ph:ph + ch, pw:pw + cw].softmax(1)
        l1 = (pm[:, 0] + pm[:, 1]).mean()
        ent = -(pm.clamp_min(1e-9) * pm.clamp_min(1e-9).log()).sum(1).mean()
        # CONDITIONAL entropy: the fg-vs-bg decision alone, renormalised so the
        # abstain channel is out of it, weighted by how much mass is committed.
        # The 3-way `ent` above has a cheaper descent direction than the decision
        # we care about -- squeezing the (already near-unused) abstain channel --
        # so it fell 0.985 -> 0.806 in runC while the fg/bg split moved from 80%
        # to 77% of maximum hedging. This term cannot be paid off that way, and
        # it does not fight --mask-l1 over the same channel.
        mass = (pm[:, 0] + pm[:, 1]).clamp_min(1e-9)
        q = pm[:, :2] / mass[:, None]
        ent2 = ((-(q.clamp_min(1e-9) * q.clamp_min(1e-9).log()).sum(1))
                * mass).mean()

        # WARM-UP TARGET: decompose's own partition, as dense per-pixel
        # supervision on the fg-vs-bg CONDITIONAL only -- abstain is left free,
        # so this does not re-impose the em8 chain's full per-pixel copy and
        # does not fight --mask-l1 over the abstain channel. Crucially rgb_trunk
        # never sees the ink map, so hitting this target from RGB alone requires
        # COMPUTING the partition and its orientation; the knowledge has to end
        # up in the weights, unlike runB2 where it was traced off an input.
        # Flat cells (gate <= 0.5) have no defensible fg/bg and are excluded.
        q = (pm[:, 0] / mass).reshape(M, -1).clamp(1e-6, 1 - 1e-6)
        ink_t = dec["ink"]
        bce = -(ink_t * q.log() + (1 - ink_t) * (1 - q).log()).mean(1)
        gwt = (dec["gate"] > 0.5).float()
        tgt = (bce * gwt).sum() / gwt.sum().clamp_min(1.0)

        if args.read_path == "blend":
            # every pixel votes with its probability -- no ranking, no argsort,
            # so the gradient reaches all 648 pixels instead of the 6% that the
            # top-k selection left alive, and an all-abstain layer degrades to
            # the cell mean through the ridge rather than falling off a cliff.
            # The abstain channel does the excluding, which is what it is for.
            fg, bg = mask_fit(pm, dec["cell_rgb"], ridge=args.blend_ridge)
        else:
            fg, bg = _conf_fit(pm, dec["cell_rgb"], mode=args.read_path,
                               frac=args.read_frac, count=args.read_count,
                               ridge=args.read_ridge, temp=args.read_temp)
        fgm, bgm = fg, bg          # pre-flip, for the reach metric below
        flipped = (not det) and rng.random() < args.flip_frac
        if flipped:
            fg, bg = bg, fg
        cell = bg[:, None, :] + (fg - bg)[:, None, :] * ink_st[:, :, None]

        if (args.bwd_keep < 1.0 or args.bwd_drop_empty > 0) and not det:
            # forward stays the WHOLE image; only the BACKWARD is restricted.
            # Detaching keeps a cell in the picture CLIP scores while removing it
            # from the graph, so the scene stays intact and gradient cost falls.
            keep = torch.ones(M, dtype=torch.bool, device=dev)
            if args.bwd_drop_empty > 0:
                empty = dec["ink"].mean(1) < 0.03
                keep &= ~(empty & (torch.rand(M, device=dev)
                                   < args.bwd_drop_empty))
            if args.bwd_keep < 1.0:
                keep &= torch.rand(M, device=dev) < args.bwd_keep
            cell = torch.where(keep[:, None, None], cell, cell.detach())

        render = cell.view(gh, W, ch, cw, 3).permute(0, 2, 1, 3, 4) \
            .reshape(gh * ch, W * cw, 3)

        # ---- polarity: is the mask's fg the same SIDE as the image's own
        # minority cluster? Per cell, corr(p_fg - p_bg, ink). Free -- both
        # tensors are already here -- and it is the quantity that decides when
        # a commitment term may safely be switched on: entropy applied while
        # polarity is still unsettled cements coin flips instead of resolving
        # them. Flat cells (gate <= 0.5) have no defensible fg/bg, so they are
        # weighted out rather than diluting the number. No .item(): a sync here
        # would undo the per-photo pipelining.
        with torch.no_grad():
            d = (pm[:, 0] - pm[:, 1]).reshape(M, -1)
            a = d - d.mean(1, keepdim=True)
            b = dec["ink"] - dec["ink"].mean(1, keepdim=True)
            corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
            gw = (dec["gate"] > 0.5).float()
            gs = gw.sum().clamp_min(1.0)
            pol = (corr * gw).sum() / gs
            inv = ((corr < 0).float() * gw).sum() / gs
            # REACH: how much of the contrast the cell actually offers did the
            # fit take? |fit_fg - fit_bg| over decompose's own |fg - bg|, in Lab.
            # This is the washout number: a mask that is right about the SIDE but
            # hedged about the pixels drives both colours to the cell mean, which
            # reads as a grey render even at perfect polarity. em8_r027 sits near
            # 1.0; the joint runs collapsed to ~0.25.
            got = (srgb_to_lab(fgm[:, None])[:, 0]
                   - srgb_to_lab(bgm[:, None])[:, 0]).norm(dim=-1)
            reach = ((got / dec["sep"].clamp_min(1e-6)).clamp(0, 4) * gw).sum() / gs

        reg = dict(l1=l1, ent=ent, ent2=ent2, tgt=tgt)
        if want_mask:       # the lineart arm's MASK TARGET reads these; the
            reg["mlog"] = mlog[:, :, ph:ph + ch, pw:pw + cw]   # caller pops it

        # ---- FLAT CELL -> SPACE ------------------------------------------
        # Where a cell genuinely offers nothing, the honest glyph is a blank.
        # The gate is the whole design. Two mistakes are available and both are
        # expensive:
        #
        #  * ungated, the target is SELF-REFERENTIAL -- "collapse the colours,
        #    then emit space" is a stable all-blank solution, and this project
        #    has already lost runs to objectives minimised by abstaining.
        #    Gating on an IMAGE property the model cannot move closes that loop.
        #
        #  * gating on dec["gate"] alone would push smooth GRADIENTS to space,
        #    which is wrong: a soft ramp wants a tone glyph. Look at what gate
        #    actually is -- smoothstep((sep-4)/4) * smoothstep((spread/within
        #    -1.5)/1.5) -- and note the bimodality factor barely bites (a linear
        #    ramp scores ~3.5 there, pure Gaussian noise ~2.65, both clearing
        #    1.5). So gate is a PERCEPTIBILITY test, and since sep for a ramp is
        #    about half its total Lab range, a cell needs ~8 Lab of range before
        #    it opens. A shadow falling across one cell at 3-6 Lab reads as
        #    "flat" to it while plainly not being flat.
        #
        # Amplitude cannot separate those cases -- both are small. Spatial
        # COHERENCE can: fit a plane to the cell's lightness and take the
        # fitted amplitude. Flat -> 0. Shallow ramp -> real. Noise -> 0, having
        # no first-order term. So space fires only on gate low AND ramp low.
        if args.space_weight > 0:
            L = srgb_to_lab(dec["cell_rgb"])[..., 0].view(M, ch, cw)
            ys = torch.linspace(-1, 1, ch, device=dev)[None, :, None]
            xs = torch.linspace(-1, 1, cw, device=dev)[None, None, :]
            by = (L * ys).mean((1, 2)) / ys.pow(2).mean()
            bx = (L * xs).mean((1, 2)) / xs.pow(2).mean()
            ramp = 2.0 * (by.pow(2) + bx.pow(2)).sqrt()      # peak-to-peak, Lab
            w_sp = (1.0 - dec["gate"]).clamp(0, 1) * smoothstep(
                (args.space_ramp - ramp) / max(args.space_ramp, 1e-6))
            reg["space"] = ((-lp[:, cache.space] * w_sp).sum()
                            / w_sp.sum().clamp_min(1.0))
            reg["space_frac"] = w_sp.mean().detach()
        else:
            reg["space"] = None

        # ---- COLOUR DENSITY: is this colour actually IN the picture? -------
        # Penalise a fitted colour by its negative log density under the
        # empirical colour distribution of the cell and its eight neighbours.
        #
        # The point is what it does to WASHOUT, and it does it without asking
        # decompose anything. Averaging two clusters lands in the valley between
        # their modes -> punished. Picking either mode -> free. A flat region has
        # one mode and you must hit it -> free. And a smooth gradient is a
        # CONTINUUM of colours, so every point along the ramp is in-support and
        # the term stays silent there -- the same case that forces the space
        # gate above to be careful is handled here for nothing.
        #
        # Contrast with penalising reach, which measures |fg-bg| against
        # dec["sep"] and is therefore anchored to decompose's opinion -- a
        # second-hand target of exactly the kind this head exists to stop
        # copying. This is anchored to the photograph.
        #
        # NECESSARY, NOT SUFFICIENT, and the code should say so: fg = bg = the
        # modal colour scores perfectly and is collapse. The term forbids
        # INVENTING a colour, never AGREEING on one. Reconstruction is what
        # rules that out (a flat render misses the cell's structure), so this
        # must never be run as the only thing holding the colours apart.
        if args.density_weight > 0:
            k = max(8, args.density_samples)
            sel = torch.randint(0, dec["cell_rgb"].shape[1], (k,), device=dev)
            with torch.no_grad():
                ci = torch.arange(M, device=dev).view(1, 1, gh, W).float()
                nb = torch.nn.functional.pad(ci, (1, 1, 1, 1), mode="replicate")
                nb = nb[0, 0].unfold(0, 3, 1).unfold(1, 3, 1) \
                    .reshape(M, 9).long()                        # (M,9)
                pool_lab = srgb_to_lab(
                    dec["cell_rgb"][:, sel][nb].reshape(M, 9 * k, 3))
            s2 = 2.0 * args.density_sigma ** 2
            lnK = math.log(9 * k)

            def _nll(c):
                d2 = (srgb_to_lab(c[:, None])[:, 0][:, None, :]
                      - pool_lab).pow(2).sum(-1)
                # -log KDE. Grows ~d2/2s2 once outside every mode, so cap it:
                # uncapped, one badly-placed cell can outweigh the whole batch.
                return (lnK - torch.logsumexp(-d2 / s2, dim=1)) \
                    .clamp(max=args.density_cap)

            cov = ink_st.mean(1).detach().clamp(0, 1)   # how much fg is painted
            reg["dens"] = (cov * _nll(fgm) + (1 - cov) * _nll(bgm)).mean()
        else:
            reg["dens"] = None
        # Returned as a DICT on purpose. This tuple used to carry nine positional
        # slots and grew with every term; adding `tgt` to it once desynchronised
        # two of the four unpack sites and killed runD3 at step 200, because
        # photo_eval is the only caller reached solely at --eval-every. Arity is
        # now fixed and new regularisers cost no call-site edits.
        return render, reg, flipped, pol, inv, reach

    def _sample_pair():
        """An (fg, bg) colour pair for a synthetic lineart target.

        Three things are deliberate. The luminance ORDER is balanced, so half the
        samples have a brighter foreground -- lineart is otherwise always
        dark-ink-on-light-paper and would pour gradient into exactly the
        "foreground = brighter side" bias we measured (24.4% inversion on
        dark-subject cells against 2.4% on bright ones). A slice is ISO-LUMINANT,
        where luminance carries no information and only hue and chroma separate
        the clusters -- the case where the shortcut is not merely wrong but
        unavailable. And the separation floor sits above decompose's own jnd=4,
        so there is always a real answer to find.

        Sampled in sRGB and SELECTED on measured dE rather than constructed in
        Lab and converted back: every pair is then in-gamut by construction, and
        there is no inverse transform to get wrong."""
        if rng.random() >= args.line_color_frac:      # ink on paper, but the
            a = float(rng.random() < 0.5)             # orientation is random
            return (torch.full((3,), a, device=dev),
                    torch.full((3,), 1.0 - a, device=dev))
        iso = rng.random() < args.line_color_iso
        want_bright_fg = rng.random() < 0.5
        lo, hi = args.line_color_de
        cand = torch.rand(512, 2, 3, device=dev)
        lab = srgb_to_lab(cand)                                    # (512,2,3)
        dE = (lab[:, 0] - lab[:, 1]).norm(dim=-1)
        dL = lab[:, 0, 0] - lab[:, 1, 0]
        ok = (dE >= lo) & (dE <= hi)
        ok &= (dL.abs() < 5.0) if iso else (dL.abs() >= 5.0)
        idx = ok.nonzero().flatten()
        if idx.numel() == 0:                       # fall back to ink on paper
            a = float(want_bright_fg)
            return (torch.full((3,), a, device=dev),
                    torch.full((3,), 1.0 - a, device=dev))
        k = int(idx[int(rng.integers(idx.numel()))])
        pair = cand[k]
        if not iso and ((dL[k] > 0) != want_bright_fg):
            pair = pair.flip(0)                    # balance which side is lighter
        return pair[0], pair[1]

    def line_color_once():
        """Colour reconstruction on the lineart cache.

        A RECONSTRUCTION target, not a mask target: paint the cache's true ink
        map with a sampled (fg,bg) pair, hand the model that image, and ask it
        to reproduce it with the glyph it picks and the colours its mask fits.
        Exact, dense, no decompose-derived supervision, and it trains the
        PARTITION directly -- the fitted foreground has to land on the strokes.
        Copying the glyph silhouette is not a shortcut here, because the glyph
        never aligns exactly with the ink and the colours would be fit on the
        wrong pixels."""
        idx = _one_region()
        ri, top, left = int(idx[0, 0]), int(idx[0, 1]), int(idx[0, 2])
        rh0 = int(idx[:, 1].max()) - top + 1
        cw0 = int(idx[:, 2].max()) - left + 1
        ink = cache.ink[ri][cache.off_y + top * ch: cache.off_y + (top + rh0) * ch,
                            cache.off_x + left * cw: cache.off_x + (left + cw0) * cw]
        ink_t = torch.from_numpy(ink.astype(np.float32) / 255.0).to(dev)
        fg_c, bg_c = _sample_pair()
        # the target IS the render equation applied to the TRUE ink
        tgt_rgb = (bg_c[None, None, :]
                   + (fg_c - bg_c)[None, None, :] * ink_t[:, :, None]).clamp(0, 1)
        render, reg, _f, pol, inv, reach = _render_grid(tgt_rgb, rh0, cw0,
                                                        det=True, want_mask=True)
        # MASK TARGET: the true ink IS the mask. Per-pixel CE of the fg/bg/
        # abstain logits against ink -> fg, paper -> bg, on the cache's own
        # cell grid (cells row-major, same order _render_grid assembles the
        # render). Reconstruction MSE alone could not pull a RANDOM mask to
        # the strokes: the fit hands both colours the cell mean, the residual
        # is flat on the ~80% blank cells, and the per-cell token bias
        # saturates the softmax within 100 steps (campaign_sfmono v1/v2:
        # pol/inv/reach exactly 0 from pval@100). This term has a non-zero
        # gradient on every ink pixel whatever the fit does.
        mlog = reg.pop("mlog")                               # (M,3,ch,cw)
        cls = (ink_t.view(rh0, ch, cw0, cw).permute(0, 2, 1, 3)
               .reshape(-1, ch, cw) < 0.5).long()            # 0 = fg, 1 = bg
        loss = torch.nn.functional.cross_entropy(mlog, cls)
        if args.line_color_mse > 0:
            loss = loss + args.line_color_mse * (render - tgt_rgb).pow(2).mean()
        return loss, reg, pol, inv, reach

    def _clip_mask_once(path=None, det=False, want_clip=True):
        """One photo: FULL image, one forward, glyphs and mask off the same pass.

        det=True freezes every source of run-to-run variation -- the crop is
        centred, the cell budget is exact, the glyph is the argmax rather than a
        Gumbel draw, no fg/bg flip, no backward subsampling. That is what makes
        the photo arm COMPARABLE across steps: `clip` in the step log is measured
        on a fresh random photo at a random crop and scale, so it moves with the
        sample at least as much as with the model."""
        if path is None:
            gi = int(rng.choice(len(groups), p=gshare))
            gf = groups[gi][1]
            path = gf[int(rng.integers(len(gf)))]
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            aw, ah = im.size
            # keep --photo-crop of the frame at NATIVE aspect, jittering where
            # that window sits. CLIP needs the whole scene to know which lines
            # are worth keeping sharp; a small crop is a texture patch with no
            # scene identity to reason from.
            f = min(1.0, max(0.1, args.photo_crop))
            bw, bh = int(aw * f), int(ah * f)
            if det:
                x0, y0 = (aw - bw) // 2, (ah - bh) // 2
                M_t = float(args.photo_cells)
            else:
                x0 = int(rng.integers(0, max(1, aw - bw + 1)))
                y0 = int(rng.integers(0, max(1, ah - bh + 1)))
                # cell budget -> width DERIVED from aspect, orientation is free
                M_t = args.photo_cells * (1.0 + args.photo_cells_jitter
                                          * (2 * rng.random() - 1))
            W = int(round(math.sqrt(max(64.0, M_t) * ch * bw / (cw * bh))))
            W = int(min(120, max(16, W)))
            gh = max(6, int(round(W * cw * bh / (ch * bw))))
            im = im.crop((x0, y0, x0 + bw, y0 + bh)) \
                .resize((W * cw, gh * ch), Image.LANCZOS)
        # ON THE DEVICE from here. decompose is ~95 ms of the ~120 ms this
        # function used to spend on the host and 5 ms on the GPU -- it was on the
        # CPU only because `rgb` was never moved, and its one CPU-pinned step
        # (eigh, an MPS workaround) is 700 3x3 matrices. With the windows also
        # cut on-device there is no D2H left, so the host runs ahead into the
        # next photo while this one's forward is still in flight.
        rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0).to(dev)
        render, reg, flipped, pol, inv, reach = _render_grid(rgb, gh, W, det)
        return ((clipper(render, rgb) if want_clip else None),
                reg, flipped, pol, inv, reach)


    @torch.no_grad()
    def photo_eval():
        """The held-out photos, under frozen conditions. Comparable across steps."""
        cl = po = iv = rc = 0.0
        model.eval()
        for p_ in val_photos:
            try:
                # This call site is only reached at the first --eval-every, so a
                # mismatch here survives every syntax check and every short
                # smoke and then kills the run 200 steps in -- it already did
                # once. The regularisers now travel in a dict for that reason,
                # so this unpack no longer moves when a term is added.
                c, _reg, _f, po_, iv_, rc_ = _clip_mask_once(p_, det=True)
            except OSError as e:
                print(f"  [pval] unreadable {p_}: {e}", flush=True)
                continue
            n = len(val_photos)
            cl += float(c) / n
            po += float(po_) / n
            iv += float(iv_) / n
            rc += float(rc_) / n
        model.train()
        return cl, po, iv, rc

    def clip_mask_grads(params, w_ent2, w_tgt, w_sp=0.0, w_dn=0.0, _tries=5):
        """--photo-batch photos, accumulating GRADIENTS instead of losses.

        Summing 12 losses and backwarding once keeps 12 autograd graphs alive --
        12 model forwards plus 12x(16 crops x 2) RN101 activations, which OOM'd a
        94 GB card at 92.9 GiB. Taking the gradient per photo and freeing the
        graph makes memory O(1) in --photo-batch rather than O(n), at the cost of
        one extra autograd call per photo.

        Returns (g_clip, g_reg, clip, metrics, flipped). The CLIP gradient and
        the regulariser gradient are kept SEPARATE because lam scales only the
        former -- folding the regularisers in is what produced runB's feedback
        loop. `metrics` is a dict so that adding a term touches no call site."""
        n = max(1, args.photo_batch)
        g_c = [None] * len(params)
        g_r = [None] * len(params)
        # accumulated as DEVICE tensors, read once at the end. float(c) per photo
        # is a stream sync, and there were three of them per photo: the host
        # blocked on this photo's CLIP backward before it could start decoding
        # the next photo, so no host work ever overlapped any GPU work.
        cl, acc, flip = None, {}, False
        # weight -> the key in `reg` it multiplies. Anything not listed here is
        # a diagnostic and never reaches the objective.
        terms = {"l1": args.mask_l1, "ent": args.mask_entropy, "ent2": w_ent2,
                 "tgt": w_tgt, "space": w_sp, "dens": w_dn}
        reg_on = any(v > 0 for v in terms.values())
        for _ in range(n):
            for _try in range(_tries):
                try:
                    c, reg, f, po, iv, rc = _clip_mask_once()
                    break
                except OSError as e:
                    print(f"  [photo] unreadable: {e}; resampling", flush=True)
            else:
                raise RuntimeError("clip_mask: too many unreadable photos")
            flip |= f
            cd = c.detach() / n
            cl = cd if cl is None else cl + cd
            for k, v in list(reg.items()) + [("pol", po), ("inv", iv),
                                             ("reach", rc)]:
                if v is None:
                    continue
                v = (v.detach() if torch.is_tensor(v)
                     else torch.as_tensor(v, device=dev)) / n
                acc[k] = v if k not in acc else acc[k] + v
            gs = torch.autograd.grad(c / n, params, retain_graph=reg_on,
                                     allow_unused=True)
            for i, g in enumerate(gs):
                if g is not None:
                    g_c[i] = g if g_c[i] is None else g_c[i] + g
            if reg_on:      # second pass frees the graph
                r = None
                for k, w in terms.items():
                    if w > 0 and reg.get(k) is not None:
                        term = w * reg[k] / n
                        r = term if r is None else r + term
                if r is not None:
                    gr = torch.autograd.grad(r, params, allow_unused=True)
                    for i, g in enumerate(gr):
                        if g is not None:
                            g_r[i] = g if g_r[i] is None else g_r[i] + g
        return g_c, g_r, float(cl), {k: float(v) for k, v in acc.items()}, flip

    def mask_tgt_grads(params, _tries=5):
        """The mask_tgt ARM: --photo-batch photo forwards WITHOUT CLIP, gradient
        of decompose's fg/bg partition target (reg["tgt"], flat cells excluded,
        abstain left free) accumulated per photo. Returns (g, loss, pol, inv)."""
        n = max(1, args.photo_batch)
        g_t = [None] * len(params)
        tl = pl = il = None
        for _ in range(n):
            for _try in range(_tries):
                try:
                    _, reg, _f, po, iv, _rc = _clip_mask_once(want_clip=False)
                    break
                except OSError as e:
                    print(f"  [photo] unreadable: {e}; resampling", flush=True)
            else:
                raise RuntimeError("mask_tgt: too many unreadable photos")
            t = reg["tgt"] / n
            tl = t.detach() if tl is None else tl + t.detach()
            pl = po.detach() / n if pl is None else pl + po.detach() / n
            il = iv.detach() / n if il is None else il + iv.detach() / n
            gs = torch.autograd.grad(t, params, allow_unused=True)
            for i, g in enumerate(gs):
                if g is not None:
                    g_t[i] = g if g_t[i] is None else g_t[i] + g
        return g_t, float(tl), float(pl), float(il)

    # ------------------------------------------------------------- optimizer
    # The mask branch's SPATIAL readout (rgb_trunk + mask_dec) is the part being
    # asked to learn something new, and it was the slowest-moving part of the
    # model: split_muon_params routes anything matching "blocks." to Muon at
    # --muon-lr, and everything else -- convs included -- to AdamW at --lr. So
    # mask_blocks ran at 5e-3 while mask_dec and rgb_trunk ran at 1e-4, and over
    # runC's first 400 steps they moved 99.5% / 7.5% / 3.2% respectively. Give
    # them their own group so the learning rate reaches the weights that need it.
    def is_mask_spatial(n):
        # The WHOLE mask branch, not just the decoder. This predicate predates
        # mask_ctx/glyph_ctx: with only rgb_trunk/mask_dec matched, mask_ctx
        # (315k params, measured at 15-17% of the token) trained at --lr 1e-4
        # while mask_blocks ran at --muon-lr 5e-3 -- a 50x spread inside one
        # branch, with the slowest rate on its second-largest module.
        return n.startswith(("rgb_trunk.", "mask_dec.",
                             "mask_ctx_", "glyph_ctx_"))

    mask_named = [(n, q) for n, q in model.named_parameters()
                  if q.requires_grad and is_mask_spatial(n)]
    use_mask_group = args.mask_lr > 0 and mask_named
    if args.optim == "muon":
        mu, ad = split_muon_params(model)
        if use_mask_group:
            ids = {id(q) for _, q in mask_named}
            mu = [q for q in mu if id(q) not in ids]
            ad = [q for q in ad if id(q) not in ids]
        opt = MuonWithAdamW(mu, ad, muon_lr=args.muon_lr, adamw_lr=args.lr,
                            weight_decay=args.weight_decay)
    else:
        rest = [q for n, q in model.named_parameters()
                if q.requires_grad and not (use_mask_group and is_mask_spatial(n))]
        opt = torch.optim.AdamW(rest, lr=args.lr,
                                weight_decay=args.weight_decay)
    if use_mask_group:
        opt.add_param_group(dict(params=[q for _, q in mask_named],
                                 use_muon=False, lr=args.mask_lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay))
        print(f"  mask-branch LR group: {len(mask_named)} tensors "
              f"({sum(q.numel() for _, q in mask_named)/1e3:.1f}k params) at "
              f"lr {args.mask_lr:g} (rest {args.lr:g}), decaying to "
              f"{args.mask_lr_floor:.0%} of peak instead of 0", flush=True)

    def cosine(s):
        # --resume-lr-frac starts the cosine PARTWAY ALONG instead of at the
        # peak. Continuing from a converged checkpoint at full peak LR re-shocks
        # exactly the parts that converged: runF ended at 20% of peak on the main
        # groups and 50% on the mask branch, and re-entering at 100% cost runE3
        # roughly 100 steps of recovery. f=0 is the old behaviour.
        f0 = min(0.95, max(0.0, args.resume_lr_frac))
        phase = f0 + (1 - f0) * min(1.0, s / max(1, args.steps))
        return 0.5 * (1 + math.cos(math.pi * phase))

    def warm(s):
        if args.start_step:            # resumed: re-warm from the resume point
            return min(1.0, (s - args.start_step + 1) / max(1, args.resume_warmup))
        return min(1.0, (s + 1) / max(1, args.warmup))

    def floored(f):
        return lambda s: warm(s) * (f + (1 - f) * cosine(s))

    def stepped(f, late=None):
        # one drop at --lr-late-step, then flat. With --lr-floor 1.0 the cosine
        # collapses to warm(s), so this is: warm up, hold, drop once, hold.
        base = floored(f)
        late = args.lr_late if late is None else late
        return lambda s: base(s) * (1.0 if s < args.lr_late_step else late)

    lams = [stepped(args.lr_floor, args.muon_lr_late if g.get("use_muon") else None)
            for g in opt.param_groups]
    if use_mask_group:      # flatter tail: this group is still learning when the
        lams[-1] = stepped(args.mask_lr_floor, args.mask_lr_late)   # others are annealed to a stop
    _late = lambda v: args.lr_late if v is None else v
    print(f"  lr step-down at {args.lr_late_step}: adamw x{args.lr_late:g} "
          f"-> {args.lr*args.lr_late:g}, muon x{_late(args.muon_lr_late):g} "
          f"-> {args.muon_lr*_late(args.muon_lr_late):g}, mask x{_late(args.mask_lr_late):g} "
          f"-> {args.mask_lr*_late(args.mask_lr_late):g}")
    if args.start_step:     # the scheduler counts from 0; the run does not
        lams = [(lambda f: (lambda s: f(s + args.start_step)))(l) for l in lams]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lams)
    if _resume_ck is not None and "opt" in _resume_ck:
        opt.load_state_dict(_resume_ck["opt"])
    print(f"  lr schedule: main -> {args.lr_floor:.0%} of peak"
          + (f", mask branch -> {args.mask_lr_floor:.0%}" if use_mask_group
             else "")
          + (f"; --resume-lr-frac {args.resume_lr_frac:g} starts the cosine at "
             f"{cosine(0):.0%} of peak rather than 100%"
             if args.resume_lr_frac > 0 else ""), flush=True)


    if args.freeze_mask:
        frozen = list(model.mask_dec.parameters())
        if getattr(model, "mask_blocks", None) is not None:
            frozen += list(model.mask_blocks.parameters())
        for q in frozen:
            q.requires_grad_(False)
        print(f"  colorizer FROZEN: {sum(q.numel() for q in frozen)/1e3:.0f}k "
              f"params held fixed", flush=True)
    params = [q for q in model.parameters() if q.requires_grad]
    # NB --w-line was previously accepted and never read: the trainer hardcoded
    # the pin by simply not scaling g_line. It is honoured now.
    hist = []
    gn_ema = {}                 # EMA of each arm's own gradient norm
    ema_pol, ent2_open, w_ent2, w_tgt = None, None, 0.0, 0.0
    if _resume_ck is not None:
        gn_ema = dict(_resume_ck.get("gn_ema") or {})
        ema_pol = _resume_ck.get("ema_pol")
        _hp = os.path.join(outdir, "hist.json")     # same --name: keep the rows
        if os.path.exists(_hp):                      # from before the resume
            hist = [r for r in json.load(open(_hp))
                    if r.get("step", 0) < args.start_step]
        print(f"  resume: {len(hist)} history rows kept, pools reloaded: "
              f"colour {len(pool)}, lineart {len(pool_line) if pool_line else 0}",
              flush=True)
    # bound before the loop so the pre-refresh release below is unconditional
    l_line = l_pool = l_pline = lc = g_c = g_r = None
    grads = total = None
    t0 = time.time()
    model.train()
    for step in range(args.start_step, args.steps):
        # phase-triggered prefills: each pool is populated at the step its
        # arm switches on, not at launch -- a pool built from an untrained
        # model refines noise coloured by noise.
        _prefilled = False
        for _p, _at, _nm in ((pool_line, args.pool_line_at, "lineart"),
                             (pool, args.pool_color_at, "colour")):
            if _p is not None and step == _at and args.pool_prefill > 0 \
                    and len(_p) < args.pool_prefill:
                print(f"[pool] prefill {args.pool_prefill} ({_nm})", flush=True)
                _p.refresh(model, chars, cfg, args.pool_prefill)
                _prefilled = True
        # no refresh on a prefill step (1000/1500 sit on the 250 grid, so the
        # campaign otherwise ran 64 + 32 refinements from the SAME weights
        # back to back), and none on the step a resume lands on
        if args.pool_refresh_every and step and step % args.pool_refresh_every == 0 \
                and not _prefilled and step != args.start_step:
            # RELEASE THE PREVIOUS STEP'S GRAPH before the workers spawn.
            # refresh() runs at the TOP of the loop, so l_line/l_pool and every
            # g_* are still bound here from step-1. Worse, g_line used to be
            # taken with retain_graph=True -- pointless, nothing ever backwards
            # through it twice -- which kept the whole lineart+line_color graph
            # LIVE, so empty_cache() could not return a byte of it. The trainer
            # held 1.2 GB in runC, 22 GB in runE3 and 42 GB in runF, and 8-11 of
            # 32 refinements OOM'd per refresh with nothing in the log to say so.
            # Nothing reads these again: the log row already took floats.
            cuda = str(dev).startswith("cuda")
            held = torch.cuda.memory_allocated() / 2**30 if cuda else 0.0
            l_line = l_pool = lc = None
            l_pline = g_c = g_r = None
            grads = total = None
            # defence in depth: clip_loss now drops these itself, but a stale
            # hook dict here costs GiB and the failure is silent either way
            getattr(clipper, "_feats", {}).clear()
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
                # printed because the failure this prevents was SILENT: the
                # refinements just OOM'd and the run carried on with a stale
                # pool. If `after` is not small, something is holding a graph
                # again and the workers are about to start losing.
                print(f"  [mem] trainer {held:.1f} -> "
                      f"{torch.cuda.memory_allocated()/2**30:.1f} GiB held, "
                      f"{torch.cuda.memory_reserved()/2**30:.1f} GiB reserved "
                      f"before spawning workers", flush=True)
                if os.environ.get("ASCIIFY_MEMDUMP"):
                    # what is still LIVE. Guessing at this wastes runs; the
                    # allocator knows.
                    seen, tot = {}, 0
                    for o in gc.get_objects():
                        try:
                            if not (torch.is_tensor(o) and o.is_cuda):
                                continue
                        except Exception:
                            continue
                        key = (tuple(o.shape), str(o.dtype))
                        nb = o.numel() * o.element_size()
                        c, s = seen.get(key, (0, 0))
                        seen[key] = (c + 1, s + nb)
                        tot += nb
                    top = sorted(seen.items(), key=lambda kv: -kv[1][1])[:12]
                    print(f"  [memdump] {tot/2**30:.2f} GiB in live cuda "
                          f"tensors, top shapes:", flush=True)
                    for (shp, dt), (c, s) in top:
                        print(f"      {s/2**20:9.1f} MiB  x{c:<5d} {dt} {shp}",
                              flush=True)
            if args.pool_color_switch >= 0 and step >= args.pool_color_switch:
                other = "lite" if args.pool_color == "fit" else "fit"
                if pool.color != other:
                    print(f"  [pool] colour source -> {other} at step {step}",
                          flush=True)
                pool.color = other
            for _p, _at in ((pool, args.pool_color_at),
                            (pool_line, args.pool_line_at)):
                if _p is not None and step >= _at and len(_p):
                    _p.refresh(model, chars, cfg, args.pool_add)

        # ---- FIVE OBJECTIVES, each with its own gradient -----------------
        # Only the arms with weight > 0 in this phase are computed at all, so a
        # phase that has not switched CLIP on pays nothing for it.
        W = phase_weights(step, phases)
        if args.mask_tgt_w > 0 and args.mask_tgt_from is not None \
                and args.mask_tgt_from <= step < (args.mask_tgt_until or 10**9):
            _w = args.mask_tgt_w
            W = {k: v * (1 - _w) for k, v in W.items()}
            W["mask_tgt"] += _w
        # mask_tgt DECAYS linearly to 0 across its window -- the runF bootstrap
        # did (--mask-target-steps), and a flat weight would keep pulling the
        # mask toward decompose's closed-form partition, which is precisely
        # what the mask is supposed to stop agreeing with. The weight it gives
        # up goes back to the other arms pro rata so the total stays 1.
        if W["mask_tgt"] > 0:
            if args.mask_tgt_from is not None and args.mask_tgt_until \
                    and args.mask_tgt_from <= step < args.mask_tgt_until:
                _lo, _hi = args.mask_tgt_from, args.mask_tgt_until
            else:
                _lo, _hi = 0, CAMPAIGN_MASK_TGT_OFF
            _d = max(0.0, 1.0 - (step - _lo) / max(1, _hi - _lo))
            _w0, _w1 = W["mask_tgt"], W["mask_tgt"] * _d
            _rest = 1.0 - _w0
            W = {k: (v * (1 - _w1) / _rest if _rest > 0 else v)
                 for k, v in W.items() if k != "mask_tgt"}
            W["mask_tgt"] = _w1
        l_line = ce_line() if W["line_ce"] > 0 else None
        if W["line_color"] > 0:
            lc, _regc, polc, invc, rchc = line_color_once()
            l_lcol = float(lc)
        else:
            lc, l_lcol = None, None
        l_pool = ce_pool(pool) if (W["pool_color"] > 0 and len(pool)) else None
        l_pline = (ce_pool(pool_line)
                   if (W["pool_line"] > 0 and pool_line is not None
                       and len(pool_line)) else None)
        params_ = [q for q in model.parameters() if q.requires_grad]
        g_t, l_mtgt, mt_pol, mt_inv = None, None, None, None
        if W["mask_tgt"] > 0:
            g_t, l_mtgt, mt_pol, mt_inv = mask_tgt_grads(params_)
        if W["clip"] <= 0:
            g_c = g_r = None
            l_clip, mets, flipped = None, {}, False
            w_sp = w_dn = w_tgt = w_ent2 = 0.0
        else:
            w_tgt = (args.mask_target
                     * max(0.0, 1.0 - step / max(1, args.mask_target_steps)))
            if args.mask_entropy2 > 0:
                if ent2_open is None and step >= args.mask_entropy2_min_step \
                        and ((args.mask_entropy2_gate > -1 and ema_pol is not None
                              and ema_pol >= args.mask_entropy2_gate)
                             or (0 <= args.mask_entropy2_start <= step)):
                    ent2_open = step
                w_ent2 = (0.0 if ent2_open is None else args.mask_entropy2
                          * min(1.0, (step - ent2_open + 1)
                                / max(1, args.mask_entropy2_ramp)))
            else:
                w_ent2 = 0.0
            # colour regularisers ramp in from reg_at, not from step 0: on an
            # unsettled model most cells read flat (space blanks the image) and
            # there is no partition for density to pull toward.
            rr = max(0.0, min(1.0, (step - args.reg_at + 1)
                              / max(1, args.space_warm)))
            w_sp = args.space_weight * rr
            w_dn = args.density_weight * max(0.0, min(
                1.0, (step - args.reg_at + 1) / max(1, args.density_warm)))
            g_c, g_r, l_clip, mets, flipped = clip_mask_grads(
                params_, w_ent2, w_tgt, w_sp, w_dn)
            ema_pol = (mets["pol"] if ema_pol is None
                       else 0.9 * ema_pol + 0.1 * mets["pol"])

        # ---- BALANCE: normalise each arm by an EMA of its OWN norm --------
        # g_total = sum_i  w_i * g_i / EMA(||g_i||)
        #
        # Every arm then contributes exactly w_i in gradient norm, whatever its
        # natural scale -- and the scales differ by 30-50x (gn_line 3-6 vs
        # gn_clip 0.10-0.18). Nothing's weight depends on its own gradient, so
        # the feedback that broke the old --balance photo anchor cannot recur:
        # there, w_line = ratio*EMA(gn_clip/gn_line) pinned w_line*gn_line at a
        # constant, so lineart degrading could never grow its own corrective
        # term (measured corr(gn_line, w_line) = -0.476).
        opt.zero_grad(set_to_none=True)
        grads = {
            "line_ce":    (torch.autograd.grad(l_line, params, allow_unused=True)
                           if l_line is not None else None),
            "line_color": (torch.autograd.grad(lc, params, allow_unused=True)
                           if lc is not None else None),
            "pool_color": (torch.autograd.grad(l_pool, params, allow_unused=True)
                           if l_pool is not None else None),
            "pool_line":  (torch.autograd.grad(l_pline, params, allow_unused=True)
                           if l_pline is not None else None),
            "clip":       g_c,
            "mask_tgt":   g_t,
        }
        a = args.ratio_ema
        total = [None] * len(params)
        for name in ARMS:
            g, wi = grads[name], W[name]
            if g is None or wi <= 0 or not any(x is not None for x in g):
                continue
            n = grad_norm_of(g)
            gn_ema[name] = n if gn_ema.get(name) is None \
                else (1 - a) * gn_ema[name] + a * n
            sc = wi / max(gn_ema[name], 1e-8)
            for i, x in enumerate(g):
                if x is not None:
                    total[i] = sc * x if total[i] is None else total[i] + sc * x
        # regularisers ride their own channel at a TARGET SHARE of the total --
        # never scaled by any arm's weight (folding them in was runB's runaway)
        # g_r is a LIST OF NONES when no regulariser is active (reg_on False),
        # not None -- grad_norm_of then sums an empty generator to int 0.
        if g_r is not None and any(x is not None for x in g_r):
            nr = grad_norm_of(g_r)
            gn_ema["reg"] = nr if gn_ema.get("reg") is None \
                else (1 - a) * gn_ema["reg"] + a * nr
            sc = args.reg_share * sum(W.values()) / max(gn_ema["reg"], 1e-8)
            for i, x in enumerate(g_r):
                if x is not None:
                    total[i] = sc * x if total[i] is None else total[i] + sc * x
        for i, q in enumerate(params):
            q.grad = total[i]
        opt.step(); sched.step()

        if step % args.log_every == 0:
            row = dict(step=step, line=float(l_line),
                       pool=float(l_pool) if l_pool is not None else None,
                       clip=None if l_clip is None else float(l_clip),
                       l1=mets.get("l1"), ent=mets.get("ent"),
                       ent2=mets.get("ent2"), w_ent2=float(w_ent2),
                       tgt=mets.get("tgt"), w_tgt=float(w_tgt),
                       space=mets.get("space"), w_space=float(w_sp),
                       space_frac=mets.get("space_frac"),
                       dens=mets.get("dens"), w_dens=float(w_dn),
                       lcol=l_lcol, mtgt=l_mtgt,
                       pol=mets.get("pol", mt_pol), inv=mets.get("inv", mt_inv),
                       reach=mets.get("reach"),
                       pline=None if l_pline is None else float(l_pline),
                       **{f"w_{k}": float(W[k]) for k in ARMS},
                       **{f"gn_{k}": float(gn_ema.get(k, 0.0)) for k in ARMS},
                       gn_reg=float(gn_ema.get("reg", 0.0)),
                       flip=bool(flipped),
                       pool_n=len(pool), min=round((time.time() - t0) / 60, 2))
            hist.append(row)
            def _f(k, spec):
                v = row[k]
                return "--" if v is None else format(v, spec)
            e2s = (f"  e2 {_f('ent2', '.3f')}@{row['w_ent2']:.3f}"
                   if args.mask_entropy2 > 0 else "")
            e2s += (f"  tgt {_f('tgt', '.3f')}@{row['w_tgt']:.3f}"
                    if args.mask_target > 0 else "")
            e2s += (f"  spc {_f('space', '.3f')}@{row['w_space']:.3f}"
                    f"/{_f('space_frac', '.1%')}"
                    if args.space_weight > 0 else "")
            e2s += (f"  dens {_f('dens', '.3f')}@{row['w_dens']:.3f}"
                    if args.density_weight > 0 else "")
            e2s += f"  lcol {_f('lcol', '.4f')}" if l_lcol is not None else ""
            e2s += f"  mtgt {_f('mtgt', '.3f')}" if l_mtgt is not None else ""
            print(f"  step {step:5d}/{args.steps}  line {row['line']:.4f}  "
                  f"pool {_f('pool', '.4f')}  pline {_f('pline', '.4f')}  "
                  f"clip {_f('clip', '.4f')}  "
                  f"l1 {_f('l1', '.3f')}  ent {_f('ent', '.3f')}{e2s}  "
                  f"pol {_f('pol', '.3f')}  inv {_f('inv', '.1%')}  "
                  f"reach {_f('reach', '.2f')}  "
                  f"wl {row['w_line_ce']:.2f}/{row['w_line_color']:.2f}/"
                  f"{row['w_pool_color']:.2f}/{row['w_pool_line']:.2f}/"
                  f"{row['w_clip']:.2f}/{row['w_mask_tgt']:.2f}  "
                  f"pool_n {row['pool_n']}  ({row['min']:.1f}m)", flush=True)
            if step % (args.recal_every * 10) == 0:
                # raw EMA'd norms (unnormalised) + the share each arm holds
                tot = sum(W.values()) or 1.0
                print("      EMA|g| " + "  ".join(
                    f"{k} {gn_ema.get(k, 0.0):.3e}" for k in ARMS)
                    + f"  reg {gn_ema.get('reg', 0.0):.3e}", flush=True)
                print("      share   " + "  ".join(
                    f"{k} {W[k]/tot:.0%}" for k in ARMS if W[k] > 0)
                    + f"  reg {args.reg_share:.0%}", flush=True)
        if args.eval_every and step and step % args.eval_every == 0:
            m = evaluate(model, cache, "val", dev, 512, cache.space, False,
                         max_cells=20000, ce_weight=ce_w)
            print(f"  [val@{step}] lineart top1 {m['top1']:.4f} "
                  f"disagree {m['disagree_set_top1']:.4f}", flush=True)
            model.train()
            if val_photos:
                pc, pp, pi, pr = photo_eval()
                print(f"  [pval@{step}] clip {pc:.4f}  pol {pp:.3f}  "
                      f"inv {pi:.1%}  reach {pr:.2f}  "
                      f"({len(val_photos)} held-out photos)", flush=True)
                if hist:
                    hist[-1].update(pval_clip=pc, pval_pol=pp,
                                    pval_inv=pi, pval_reach=pr)
        if args.ckpt_every and step and step % args.ckpt_every == 0:
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "chars": chars, "variant": "tf5x3", "step": step,
                        "train_args": vars(args),
                        "opt": opt.state_dict(),          # -> exact --resume
                        "gn_ema": {k: float(v) for k, v in gn_ema.items()
                                   if v is not None},
                        "ema_pol": None if ema_pol is None else float(ema_pol)},
                       os.path.join(outdir, f"ckpt_step{step:05d}.pt"))
            json.dump(hist, open(os.path.join(outdir, "hist.json"), "w"), indent=1)

    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "chars": chars, "variant": "tf5x3", "train_args": vars(args)},
               os.path.join(outdir, "model.pt"))
    json.dump(hist, open(os.path.join(outdir, "hist.json"), "w"), indent=1)
    print(f"[{args.name}] done -> {outdir}/model.pt "
          f"({(time.time()-t0)/60:.1f} min, pool refreshes {pool.n_refresh}, "
          f"added {pool.n_added}, evicted {pool.n_evicted})", flush=True)


if __name__ == "__main__":
    main()
