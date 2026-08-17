"""Unified color model: multi-task training over lineart + photos + colored text.

Arms (--arch / --init / --clip-token):
  scratch4 : fresh model, color via 4-channel conv input (ink + RGB pixels)
  ft       : init from a lineart model; color via zero-init per-token feature
             addition [gate, sep/20, fg rgb, bg rgb, cell-mean rgb]
  --clip-token global : the FULL image's RN101 embedding as an extra token even
             when training on subregions (all embeddings precomputed/cached);
             zero-init surgery if the init model lacks the token.

Shared surgery (all zero-init => ft arms start exactly at the init model):
  mode_emb  task token added to every token in color mode
  k_head    per-cell contrast k = 4*sigmoid(raw + log(1/3)) -> k(0)=1
  col_head  AUX fg/bg rgb prediction (train-only, dropped at export)

Batch mix per step:
  lineart region (anchor): dense kernel CE on contiguous label-grid regions
  photo region:  non-grid-aligned crop/zoom -> decompose -> ST-sample glyphs ->
                 closed-form blend colors at the sampled masks -> apply k_hat ->
                 crop-aug CLIP vs the photo crop + TV(k_hat) + weak (k-1)^2.
                 k_hat learns DIRECTLY from CLIP (continuous, no sampling).
  color text:    SynthRenderer letters + sampled per-cell fg/bg; exact color
                 supervision: CE on glyphs + MSE on the aux color head.
Photo-CLIP lambda is grad-norm calibrated against lineart CE (EMA both norms).
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from unicasso.substrate import glyphs as G, raster
from unicasso.engine.color import decompose, nomination_target
from unicasso.training.cellclf_color import fit_fg_bg_distweighted, glyph_bg_dist
from unicasso.training.train_cell_classifier import (
    CellCache, MuonWithAdamW, SynthRenderer, TokenTransformer, binomial_kernel,
    evaluate, eval_clip_render, grad_norm_of, kernel_ce, split_muon_params)
from unicasso.training.cellclf_widths import discover_models, load_any

FEAT_DIM = 11
K_BIAS = math.log(1.0 / 3.0)          # 4*sigmoid(0 + log(1/3)) = 1


def cell_feats(dec):
    """(M, 11) per-cell color features from decompose byproducts."""
    mean = dec["cell_rgb"].mean(1)
    return torch.cat([dec["gate"][:, None], (dec["sep"] / 20.0)[:, None],
                      dec["fg"], dec["bg"], mean], dim=1).float()


def grid_windows(img, gh, gw, ch, cw, ph, pw, rows=3, cols=5, pad_val=0):
    """All cells' windows of a standalone grid. img (H,W) or (H,W,3) numpy."""
    py, px = (rows // 2) * ch + ph, (cols // 2) * cw + pw
    pw_spec = ((py, py), (px, px)) + (((0, 0),) if img.ndim == 3 else ())
    pad = np.pad(img, pw_spec, constant_values=pad_val)
    wh, ww = rows * ch + 2 * ph, cols * cw + 2 * pw
    shape = (gh * gw, wh, ww) + ((3,) if img.ndim == 3 else ())
    out = np.empty(shape, img.dtype)
    i = 0
    for y in range(gh):
        for x in range(gw):
            t = py + y * ch - (rows // 2) * ch - ph
            l = px + x * cw - (cols // 2) * cw - pw
            out[i] = pad[t:t + wh, l:l + ww]; i += 1
    return out


def token_maps(gh, gw, rows, cols):
    offs = [(t // cols - rows // 2, t % cols - cols // 2) for t in range(rows * cols)]
    ys, xs = np.mgrid[0:gh, 0:gw]
    yy = ys.ravel()[:, None] + np.array([o[0] for o in offs])[None]
    xx = xs.ravel()[:, None] + np.array([o[1] for o in offs])[None]
    valid = (yy >= 0) & (yy < gh) & (xx >= 0) & (xx < gw)
    return np.where(valid, yy * gw + xx, 0).astype(np.int64), valid


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="ft", choices=["ft", "scratch4"])
    p.add_argument("--init", default="tf5x3_full_rf20_ct10_muon1k@800",
                   help="lineart model for --arch ft")
    p.add_argument("--clip-token", default="none", choices=["none", "global"])
    p.add_argument("--cache", default="runs/cellclf/cache_full")
    p.add_argument("--photos", default="data/dataset_v1/photos")
    p.add_argument("--eval-photos", default=None,
                   help="folder of held-out photos to preview color reconstructions on "
                        "(default: none)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--photo-frac", type=float, default=0.5)
    p.add_argument("--coltext-frac", type=float, default=0.15)
    p.add_argument("--clip-weight-target", type=float, default=0.2)
    p.add_argument("--clip-aug", type=int, default=12)
    p.add_argument("--tv-weight", type=float, default=0.05)
    p.add_argument("--k-prior", type=float, default=1e-3)
    p.add_argument("--k-targets", default="",
                   help="ktargets npz dir: photo steps become k-supervision steps "
                        "(regression on stored fields, then CLIP refinement)")
    p.add_argument("--k-reg-weight", type=float, default=5.0)
    p.add_argument("--k-refine-start", type=int, default=300,
                   help="step at which k supervision switches from dense regression "
                        "(stored crops) to semi-stochastic refinement (fresh crops)")
    p.add_argument("--k-refine-iters", type=int, default=3)
    p.add_argument("--k-refine-lr", type=float, default=0.02)
    p.add_argument("--k-late-reg-mult", type=float, default=3.0,
                   help="multiplier on k_hat's TV + prior during the refinement "
                        "phase: once the head is sensitized, the noisier stochastic "
                        "targets need a stiffer spring toward the MSE base")
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--region-rows", type=int, default=26)
    p.add_argument("--region-cols", type=int, default=44)
    p.add_argument("--batch", type=int, default=384, help="coltext windows/step")
    p.add_argument("--recal-every", type=int, default=20)
    p.add_argument("--optim", default="muon", choices=["adamw", "muon"])
    p.add_argument("--muon-lr", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--blank-weight", type=float, default=0.2)
    p.add_argument("--chunk", type=int, default=384)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-cells", type=int, default=40000)
    p.add_argument("--holdout-parents", default="00094,00177")
    p.add_argument("--train-on-all", action="store_true",
                   help="RELEASE mode: lineart regions + windows from every run")
    p.add_argument("--vae-ckpt", default="weights/vae_dejavu/model.pt")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    use_tok = args.clip_token == "global"
    scratch = args.arch == "scratch4"

    cache_dir = G.repo_path(args.cache)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    rows, cols = 3, 5
    ch, cw = meta["cell_h"], meta["cell_w"]
    ph, pw = meta["pad_h"], meta["pad_w"]
    viz_parents = tuple(s for s in args.holdout_parents.split(",") if s)
    cache = CellCache(cache_dir, rows, cols, ph, pw, ch, cw,
                      window_labels=True, viz_parents=viz_parents)
    N = len(cache.chars)

    init_ck_path = None
    if not scratch:
        entry = dict(discover_models()).get(args.init)
        if entry is not None:
            probe = torch.load(entry if isinstance(entry, str) else entry,
                               map_location="cpu", weights_only=False)
            if probe.get("config", {}).get("color_model"):
                init_ck_path = entry
    if scratch:
        model = TokenTransformer(rows, cols, ch, cw, ph, pw, N, dim=64, heads=4,
                                 n_blocks=3, in_ch=4,
                                 clip_dim=cache.clip_dim if use_tok else 0
                                 ).to(device).train()
        cfg = dict(variant="tf5x3", dim=64, heads=4, blocks=3, dense=True,
                   clip_token=use_tok, in_ch=4)
    elif init_ck_path is not None:
        # continue from an already-surgered color model: rebuild + load everything
        ck = torch.load(init_ck_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = TokenTransformer(rows, cols, ch, cw, ph, pw, N,
                                 dim=cfg.get("dim", 64), heads=cfg.get("heads", 4),
                                 n_blocks=cfg.get("blocks", 3),
                                 in_ch=cfg.get("in_ch", 1),
                                 clip_dim=cache.clip_dim if cfg.get("clip_token") else 0)
        dim0 = cfg.get("dim", 64)
        if cfg.get("feat_dim", 0):
            model.color_proj = nn.Linear(cfg["feat_dim"], dim0)
        model.mode_emb = nn.Parameter(torch.zeros(2, dim0))
        model.k_head = nn.Linear(dim0, 1)
        model.col_head = nn.Linear(dim0, 6)
        model.load_state_dict(ck["state_dict"])
        model = model.to(device).train()
        print(f"  continued from color model {args.init}", flush=True)
    else:
        model, cfg = load_any(dict(discover_models())[args.init], meta, device)
        model.train()
        if use_tok and not getattr(model, "use_clip", False):
            dim = model.pos.shape[1]
            model.clip_proj = nn.Linear(cache.clip_dim, dim).to(device)
            nn.init.zeros_(model.clip_proj.weight)
            nn.init.zeros_(model.clip_proj.bias)
            model.clip_pos = nn.Parameter(torch.zeros(1, dim, device=device))
            model.use_clip = True
            model.n_extra = 1
            model.center += 1
        cfg = dict(cfg, clip_token=use_tok)
    dim = model.pos.shape[1]
    if init_ck_path is None:                     # fresh surgery (zero-init)
        if not scratch:
            model.color_proj = nn.Linear(FEAT_DIM, dim).to(device)
            nn.init.zeros_(model.color_proj.weight)
            nn.init.zeros_(model.color_proj.bias)
        model.mode_emb = nn.Parameter(torch.zeros(2, dim, device=device))
        model.k_head = nn.Linear(dim, 1).to(device)
        nn.init.zeros_(model.k_head.weight); nn.init.zeros_(model.k_head.bias)
        model.col_head = nn.Linear(dim, 6).to(device)
        nn.init.zeros_(model.col_head.weight); nn.init.zeros_(model.col_head.bias)
    cfg = dict(cfg, color_model=True,
               feat_dim=0 if scratch else cfg.get("feat_dim", FEAT_DIM) or FEAT_DIM,
               arch=args.arch)

    from unicasso.adapter.corrupt import CorruptionSampler
    sampler = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu",
                                profile="dejavu")
    ink_flat = (1.0 - sampler.bitmaps.cpu().float()).reshape(sampler.N, -1)
    ink_flat_d = ink_flat.to(device)
    bm_white = (1.0 - ink_flat).to(device)
    bg_dist_all = glyph_bg_dist(ink_flat, ch, cw).to(device)
    ink_bm = 1.0 - sampler.bitmaps.cpu().float()
    synth = SynthRenderer(cache, ink_bm, rows, cols, ch, cw, ph, pw, device)
    from unicasso.engine.clip_loss import CLIPPerceptualLoss
    clipper = CLIPPerceptualLoss(torch.device(device), model_name="RN101",
                                 pretrained="openai", n_aug=args.clip_aug,
                                 crop_scale=(0.3, 0.95))
    clip_val_stems = [r["stem"] for r in cache.meta["runs"]
                      if r["split"] == "val"][:4]
    photos = sorted(sum((glob.glob(os.path.join(G.repo_path(args.photos), e))
                         for e in ("*.jpg", "*.jpeg", "*.png")), []))
    eval_photos = (sorted(sum((glob.glob(os.path.join(G.repo_path(args.eval_photos), e))
                               for e in ("*.jpg", "*.jpeg", "*.png")), []))
                   if args.eval_photos else [])
    ktarg = []
    if args.k_targets:
        pid2path = {os.path.splitext(os.path.basename(q))[0]: q for q in photos}
        for f in sorted(glob.glob(os.path.join(G.repo_path(args.k_targets), "*.npz"))):
            pid = os.path.basename(f).split("_v")[0]
            if pid in pid2path:
                ktarg.append((f, pid2path[pid]))
        print(f"  {len(ktarg)} k-target samples "
              f"(regression until step {args.k_refine_start}, refinement after)",
              flush=True)
    train_runs = [i for i, r in enumerate(cache.meta["runs"])
                  if args.train_on_all
                  or (r["split"] == "train" and r["parent"] not in viz_parents)]
    if args.train_on_all:
        cache.index["train"] = np.concatenate(
            [cache.index[s] for s in ("train", "val", "viz") if len(cache.index[s])])

    # global-embedding cache: ONE RN101 embed per full photo, computed up front
    photo_emb = {}
    zero_emb = torch.zeros(cache.clip_dim, device=device)
    if use_tok:
        import open_clip
        cm, _, prep = open_clip.create_model_and_transforms("RN101",
                                                            pretrained="openai")
        cm = cm.to(device).eval()
        with torch.no_grad():
            for path in photos + eval_photos:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    e = cm.encode_image(prep(im).unsqueeze(0).to(device)).float()
                photo_emb[path] = (e / e.norm(dim=-1, keepdim=True))[0]
        del cm
        print(f"  cached {len(photo_emb)} photo embeddings", flush=True)

    name = args.name or ("colorclf_" + args.arch
                         + ("" if scratch else "_" + args.init.replace("@", "_s"))
                         + ("_tok" if use_tok else "")
                         + ("_muon" if args.optim == "muon" else ""))
    outdir = os.path.join(G.REPO_ROOT, "runs", "cellclf", name)
    os.makedirs(outdir, exist_ok=True)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[{name}] arch {args.arch} tok {use_tok} | {n_params/1e6:.2f}M params | "
          f"{len(photos)} photos | {len(train_runs)} lineart runs | {device}",
          flush=True)

    ce_w = torch.ones(N, device=device)
    ce_w[cache.space] = args.blank_weight
    kernel = binomial_kernel(rows, cols, device)
    if args.optim == "muon":
        mu, ad = split_muon_params(model)
        opt = MuonWithAdamW(mu, ad, muon_lr=args.muon_lr, adamw_lr=args.lr,
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

    def forward_region(x, feats_tok, mode, emb):
        """Chunked forward -> (logits (M,T,N), center tokens (M,dim))."""
        lo, to = [], []
        for i in range(0, x.shape[0], args.chunk):
            n = x[i:i + args.chunk].shape[0]
            e = emb[None].expand(n, -1) if emb is not None else None
            t = model._tokens(x[i:i + args.chunk], e,
                              feats_tok[i:i + args.chunk]
                              if feats_tok is not None else None, mode)
            lo.append(model.head(t[:, model.n_extra:]))
            to.append(t[:, model.center])
        return torch.cat(lo), torch.cat(to)

    def photo_region():
        path = photos[int(rng.integers(len(photos)))]
        W = int(rng.integers(28, 45))
        gh = int(rng.integers(14, args.region_rows + 1))
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            aw, ah = im.size
            asp = (W * cw) / (gh * ch)
            bw = min(aw, int(ah * asp))
            bh = int(bw / asp)
            s = 0.35 + 0.65 * rng.random()
            bw, bh = max(32, int(bw * s)), max(32, int(bh * s))
            x0 = int(rng.integers(0, max(1, aw - bw)))
            y0 = int(rng.integers(0, max(1, ah - bh)))
            im = im.crop((x0, y0, x0 + bw, y0 + bh)) \
                .resize((W * cw, gh * ch), Image.LANCZOS)
        rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
        dec = decompose(rgb, gh, W, ch, cw)
        ink_u8 = np.clip((1.0 - nomination_target(dec).numpy()) * 255.0,
                         0, 255).astype(np.uint8)
        return path, rgb, dec, ink_u8, gh, W

    def region_inputs(rgb, dec, ink_u8, gh, gw):
        """x (M,C,wh,ww), feats_tok or None, ids/valid tensors."""
        w = grid_windows(ink_u8, gh, gw, ch, cw, ph, pw)
        x = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
        if scratch:
            cwin = grid_windows(rgb.numpy(), gh, gw, ch, cw, ph, pw, pad_val=1.0)
            xc = torch.from_numpy(cwin).to(device).permute(0, 3, 1, 2)
            x = torch.cat([x, xc], dim=1)
        ids, valid = token_maps(gh, gw, rows, cols)
        ids_t = torch.from_numpy(ids).to(device)
        valid_t = torch.from_numpy(valid).to(device)
        feats_tok = None
        if not scratch:
            feats = cell_feats(dec).to(device)
            feats_tok = feats[ids_t] * valid_t[:, :, None].float()
        return x, feats_tok, ids_t, valid_t

    def ensemble(logits, ids_t, valid_t, M):
        probs = logits.softmax(-1)
        wtok = kernel[None, :] * valid_t.float()
        acc = torch.zeros(M, N, device=device)
        wsum = torch.zeros(M, device=device)
        acc.index_add_(0, ids_t.reshape(-1),
                       (probs * wtok[:, :, None]).reshape(-1, N))
        wsum.index_add_(0, ids_t.reshape(-1), wtok.reshape(-1))
        return acc / wsum[:, None].clamp_min(1e-8)

    def color_image(g_or_st, khat, dec, mask, gh, gw):
        C = dec["cell_rgb"].to(device)
        gh_idx = g_or_st
        fg_f, bg_f = fit_fg_bg_distweighted(C, mask, bg_dist_all[gh_idx], pow_=1.0)
        fg0 = 0.5 * dec["fg"].to(device) + 0.5 * fg_f
        bg0 = 0.5 * dec["bg"].to(device) + 0.5 * bg_f
        mid = 0.5 * (fg0 + bg0)
        fg = mid + khat[:, None] * (fg0 - mid)
        bg = mid + khat[:, None] * (bg0 - mid)
        fg = fg + (fg.clamp(0, 1) - fg).detach()
        bg = bg + (bg.clamp(0, 1) - bg).detach()
        cell = bg[:, None, :] + (fg - bg)[:, None, :] * mask[:, :, None]
        return cell.view(gh, gw, ch, cw, 3).permute(0, 2, 1, 3, 4) \
            .reshape(gh * ch, gw * cw, 3)

    def color_render_loss(path, rgb, dec, ink_u8, gh, gw, k_target=None,
                          refine=False, reg_mult=1.0):
        """Returns (clip_loss, k_loss, kmean, nonblank). k_loss carries the
        k-supervision (dense regression or 3-iter CLIP refinement target) and is
        NOT lambda-scaled by the caller."""
        M = gh * gw
        x, feats_tok, ids_t, valid_t = region_inputs(rgb, dec, ink_u8, gh, gw)
        emb = photo_emb[path] if use_tok and path in photo_emb else \
            (zero_emb if use_tok else None)
        logits, ct = forward_region(x, feats_tok, 1, emb)
        khat = 4.0 * torch.sigmoid(model.k_head(ct)[:, 0] + K_BIAS)
        pbar = ensemble(logits, ids_t, valid_t, M)
        u = torch.rand_like(pbar).clamp_(1e-9, 1 - 1e-9)
        g = (pbar.clamp_min(1e-9).log() - torch.log(-torch.log(u))).argmax(-1)
        st = F.one_hot(g, N).float() + pbar - pbar.detach()
        mask = st @ ink_flat_d
        img = color_image(g, khat, dec, mask, gh, gw)
        rgb_d = rgb.to(device)
        loss = clipper(img, rgb_d)
        kf = khat.view(gh, gw)
        loss = loss + reg_mult * args.tv_weight * (
            (kf[:, 1:] - kf[:, :-1]).pow(2).mean()
            + (kf[1:] - kf[:-1]).pow(2).mean())
        loss = loss + reg_mult * args.k_prior * (khat - 1.0).pow(2).mean()
        lk = khat.new_zeros(())
        if k_target is not None:
            lk = args.k_reg_weight * F.mse_loss(khat, k_target)
        elif refine:
            # semi-stochastic target: a few CLIP iterations FROM the model's own
            # k_hat over frozen glyphs/colors; supervise the difference
            maskd = ink_flat_d[g]
            C = dec["cell_rgb"].to(device)
            fg_f, bg_f = fit_fg_bg_distweighted(C, maskd, bg_dist_all[g], pow_=1.0)
            fg0 = (0.5 * dec["fg"].to(device) + 0.5 * fg_f).detach()
            bg0 = (0.5 * dec["bg"].to(device) + 0.5 * bg_f).detach()
            midd = 0.5 * (fg0 + bg0)
            dfgd, dbgd = fg0 - midd, bg0 - midd
            k0 = nn.Parameter(khat.detach().view(gh, gw).clone())
            ropt = MuonWithAdamW([k0], [], muon_lr=args.k_refine_lr,
                                 weight_decay=0.0)
            for _ in range(args.k_refine_iters):
                kv = k0.view(-1)
                fgr = midd + kv[:, None] * dfgd
                bgr = midd + kv[:, None] * dbgd
                fgr = fgr + (fgr.clamp(0, 1) - fgr).detach()
                bgr = bgr + (bgr.clamp(0, 1) - bgr).detach()
                cellr = bgr[:, None, :] + (fgr - bgr)[:, None, :] * maskd[:, :, None]
                imgr = cellr.view(gh, gw, ch, cw, 3).permute(0, 2, 1, 3, 4) \
                    .reshape(gh * ch, gw * cw, 3)
                rl = clipper(imgr, rgb_d)
                rl = rl + args.tv_weight * ((k0[:, 1:] - k0[:, :-1]).pow(2).mean()
                                            + (k0[1:] - k0[:-1]).pow(2).mean())
                ropt.zero_grad(set_to_none=True)
                rl.backward()
                ropt.step()
                with torch.no_grad():
                    k0.data.clamp_(0.0, 4.0)
            lk = args.k_reg_weight * F.mse_loss(khat, k0.detach().view(-1))
        return loss, lk, float(khat.mean()), float((g != cache.space).float().mean())

    def coltext_batch():
        idx = cache.index["train"][rng.integers(0, len(cache.index["train"]),
                                                args.batch)]
        x, y, xc, fg_t, bg_t = synth.sample(idx, rng, want_color=True)
        B, T = y.shape
        if scratch:
            x = torch.cat([x, xc], dim=1)
            feats = None
        else:
            sep = (fg_t - bg_t).abs().mean(-1, keepdim=True) * 3.0
            mean = 0.5 * (fg_t + bg_t)
            feats = torch.cat([torch.ones(B, T, 1, device=device),
                               sep.clamp(0, 1), fg_t, bg_t, mean], dim=-1)
        emb = zero_emb if use_tok else None
        lo, to = [], []
        for i in range(0, B, args.chunk):
            n = x[i:i + args.chunk].shape[0]
            e = emb[None].expand(n, -1) if emb is not None else None
            t = model._tokens(x[i:i + args.chunk], e,
                              feats[i:i + args.chunk] if feats is not None
                              else None, 1)
            lo.append(model.head(t[:, model.n_extra:]))
            to.append(t[:, model.n_extra:])
        ce = kernel_ce(torch.cat(lo), y, kernel, ce_w)
        pred = torch.sigmoid(model.col_head(torch.cat(to)))
        aux = F.mse_loss(pred, torch.cat([fg_t, bg_t], dim=-1))
        return ce + args.aux_weight * aux

    def lineart_region():
        ri = train_runs[int(rng.integers(len(train_runs)))]
        gh, gw = cache.labels[ri].shape
        rh0, cw0 = min(gh, args.region_rows), min(gw, args.region_cols)
        top = int(rng.integers(0, gh - rh0 + 1))
        left = int(rng.integers(0, gw - cw0 + 1))
        ys, xs = np.mgrid[top:top + rh0, left:left + cw0]
        idx = np.stack([np.full(rh0 * cw0, ri), ys.ravel(), xs.ravel()],
                       axis=1).astype(np.int32)
        w, lab, _, _ = cache.fetch(idx)
        x = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
        y = torch.from_numpy(lab).to(device)
        emb = (torch.from_numpy(cache.clip[ri]).to(device) if use_tok else None)
        logits, _ = forward_region(x, None, None, emb)
        return kernel_ce(logits, y, kernel, ce_w)

    @torch.no_grad()
    def eval_color():
        import random as pyr
        st = pyr.getstate(); pyr.seed(4321)
        model.eval()
        losses, kms = [], []
        for path in eval_photos:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                gh = raster.grid_height_for_aspect(im.width, im.height, 50, cw, ch, 0)
                im = im.resize((50 * cw, gh * ch), Image.LANCZOS)
            rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
            dec = decompose(rgb, gh, 50, ch, cw)
            ink_u8 = np.clip((1.0 - nomination_target(dec).numpy()) * 255.0,
                             0, 255).astype(np.uint8)
            M = gh * 50
            x, feats_tok, ids_t, valid_t = region_inputs(rgb, dec, ink_u8, gh, 50)
            emb = photo_emb[path] if use_tok else None
            logits, ct = forward_region(x, feats_tok, 1, emb)
            khat = 4.0 * torch.sigmoid(model.k_head(ct)[:, 0] + K_BIAS)
            g = ensemble(logits, ids_t, valid_t, M).argmax(-1)
            img = color_image(g, khat, dec, ink_flat_d[g], gh, 50)
            losses.append(float(clipper(img, rgb.to(device))))
            kms.append(float(khat.mean()))
        pyr.setstate(st)
        model.train()
        return float(np.mean(losses)), float(np.mean(kms))

    ema_nce = ema_ncl = None
    lam = None
    hist = {"step": [], "kind": [], "loss": [], "km": [], "nb": [], "lam": []}
    ev = {"step": [], "top1": [], "disagree": [], "val_clip": [],
          "val_colorclip": [], "kmean": []}
    t0 = time.time()
    for step in range(args.steps):
        r = rng.random()
        recal = (step % args.recal_every) < 2 or lam is None
        km = nb = -1.0
        if r < args.photo_frac:
            kind = "photo"
            if ktarg and (step + 1) <= args.k_refine_start:
                kind = "kreg"
                f, ppath = ktarg[int(rng.integers(len(ktarg)))]
                d = np.load(f)
                ktgt = torch.from_numpy(d["k"].astype(np.float32)) \
                    .to(device).view(-1)
                gh_t, gw_t = d["k"].shape
                box = tuple(int(b) for b in d["box"])
                with Image.open(ppath) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im = im.crop(box).resize((gw_t * cw, gh_t * ch), Image.LANCZOS)
                rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
                dec = decompose(rgb, gh_t, gw_t, ch, cw)
                ink_u8 = np.clip((1.0 - nomination_target(dec).numpy()) * 255.0,
                                 0, 255).astype(np.uint8)
                loss, lk, km, nb = color_render_loss(ppath, rgb, dec, ink_u8,
                                                     gh_t, gw_t, k_target=ktgt)
            elif ktarg:
                kind = "kref"
                loss, lk, km, nb = color_render_loss(*photo_region(), refine=True,
                                                     reg_mult=args.k_late_reg_mult)
            else:
                loss, lk, km, nb = color_render_loss(*photo_region())
            opt.zero_grad(set_to_none=True)
            has_k = lk.requires_grad
            if recal:
                params = [q for q in model.parameters() if q.requires_grad]
                gs = torch.autograd.grad(loss, params, retain_graph=has_k,
                                         allow_unused=True)
                ncl = grad_norm_of(gs)
                ema_ncl = ncl if ema_ncl is None else 0.8 * ema_ncl + 0.2 * ncl
                if ema_nce is not None:
                    lam = args.clip_weight_target * ema_nce / max(ema_ncl, 1e-8)
                sc = lam if lam is not None else 0.0
                gk = (torch.autograd.grad(lk, params, allow_unused=True)
                      if has_k else [None] * len(params))
                for q, a, b in zip(params, gs, gk):
                    gg = a * sc if a is not None else None
                    if b is not None:
                        gg = b if gg is None else gg + b
                    q.grad = gg
            else:
                (loss * lam + lk).backward()
            loss = loss + lk.detach()
        elif r < args.photo_frac + args.coltext_frac:
            kind = "coltext"
            loss = coltext_batch()
            opt.zero_grad(set_to_none=True)
            loss.backward()
        else:
            kind = "lineart"
            loss = lineart_region()
            opt.zero_grad(set_to_none=True)
            if recal:
                params = [q for q in model.parameters() if q.requires_grad]
                gs = torch.autograd.grad(loss, params, allow_unused=True)
                nce = grad_norm_of(gs)
                ema_nce = nce if ema_nce is None else 0.8 * ema_nce + 0.2 * nce
                if ema_ncl is not None:
                    lam = args.clip_weight_target * ema_nce / max(ema_ncl, 1e-8)
                for q, a in zip(params, gs):
                    q.grad = a.clone() if a is not None else None
            else:
                loss.backward()
        opt.step()
        sched.step()
        hist["step"].append(step + 1); hist["kind"].append(kind)
        hist["loss"].append(float(loss)); hist["km"].append(km)
        hist["nb"].append(nb); hist["lam"].append(lam if lam is not None else 0.0)
        if (step + 1) % 10 == 0:
            print(f"  step {step+1}/{args.steps} [{kind}] loss {float(loss):.3f} "
                  f"k {km:.2f} nb {nb:.2f} lam {hist['lam'][-1]:.3f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if (step + 1) % args.eval_every == 0:
            m = evaluate(model, cache, "val", device, 2048, cache.space,
                         getattr(model, "use_clip", False),
                         max_cells=args.eval_cells, ce_weight=ce_w)
            vc = eval_clip_render(model, cache, clip_val_stems, clipper,
                                  bm_white, device,
                                  getattr(model, "use_clip", False))
            cc, kmean = eval_color()
            ev["step"].append(step + 1); ev["top1"].append(m["top1"])
            ev["disagree"].append(m["disagree_set_top1"])
            ev["val_clip"].append(vc); ev["val_colorclip"].append(cc)
            ev["kmean"].append(kmean)
            print(f"  [val@{step+1}] top1 {m['top1']:.4f} "
                  f"disagree {m['disagree_set_top1']:.4f} val-clip {vc:.4f} "
                  f"color-clip {cc:.4f} kmean {kmean:.2f}", flush=True)
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "chars": cache.chars, "variant": "tf5x3",
                        "step": step + 1},
                       os.path.join(outdir, f"ckpt_step{step + 1:05d}.pt"))

    use_clip_flag = getattr(model, "use_clip", False)
    final = evaluate(model, cache, "val", device, 2048, cache.space,
                     use_clip_flag, ce_weight=ce_w)
    final["val_clip_render"] = eval_clip_render(model, cache, clip_val_stems,
                                                clipper, bm_white, device,
                                                use_clip_flag)
    final["val_colorclip"], final["kmean"] = eval_color()
    final.update(train_minutes=round((time.time() - t0) / 60, 1),
                 config=vars(args), name=name)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=1)
    np.savez(os.path.join(outdir, "loss_hist.npz"),
             **{k: np.array(v) for k, v in hist.items() if k != "kind"},
             kind=np.array(hist["kind"]),
             **{"ev_" + k: np.array(v, dtype=float) for k, v in ev.items()})
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "chars": cache.chars, "variant": "tf5x3"},
               os.path.join(outdir, "model.pt"))
    print(f"[{name}] FINAL top1 {final['top1']:.4f} | "
          f"disagree {final['disagree_set_top1']:.4f} | "
          f"val-clip {final['val_clip_render']:.4f} | "
          f"color-clip {final['val_colorclip']:.4f} | kmean {final['kmean']:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
