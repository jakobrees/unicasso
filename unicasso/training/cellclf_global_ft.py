"""Global CLIP fine-tune: whole-region ST-sampled renders vs real lineart.

python -m unicasso.training.cellclf_global_ft --init tf5x3_full_rf20_ct10@2000

The window-level CLIP term judges 5x3 patches; this phase judges the model the
way the optimizer was judged: forward a CONTIGUOUS grid region (up to
--region-rows x --region-cols cells = a real image patch), accumulate every
cell's framing-ensemble distribution differentiably (kernel-weighted average of
all 15 framings -- the deploy inference path), stochastically sample the whole
region (Gumbel-max, straight-through), render it, and take the crop-augmented
CLIP perceptual loss against the run's actual lineart. Dense CE on the same
forwards stays as the anchor; lambda is recalibrated from measured grad norms
to --clip-weight-target (default 0.2). Muon is the default optimizer
(orthogonalized updates: direction, not magnitude).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from unicasso.substrate import glyphs as G
from unicasso.training.train_cell_classifier import (
    CellCache, MuonWithAdamW, binomial_kernel, evaluate, eval_clip_render,
    grad_norm_of, kernel_ce, split_muon_params)
from unicasso.training.cellclf_widths import discover_models, load_any


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", default="tf5x3_full_rf20_ct10@2000",
                   help="checkpoint to fine-tune (discover_models name or name@step)")
    p.add_argument("--cache", default="runs/cellclf/cache_full")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--region-rows", type=int, default=26)
    p.add_argument("--region-cols", type=int, default=44)
    p.add_argument("--clip-token", default="none", choices=["none", "global"],
                   help="global = add the run's global RN101 embedding as an extra "
                        "token (zero-init projection: starts exactly at the init "
                        "model's behavior, grows in only if it earns gradient)")
    p.add_argument("--clip-weight-target", type=float, default=0.2)
    p.add_argument("--clip-aug", type=int, default=12)
    p.add_argument("--crop-scale", type=float, nargs=2, default=(0.3, 0.95))
    p.add_argument("--recal-every", type=int, default=20)
    p.add_argument("--optim", default="muon", choices=["adamw", "muon"])
    p.add_argument("--muon-lr", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--blank-weight", type=float, default=0.2)
    p.add_argument("--chunk", type=int, default=384, help="windows per forward chunk")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-cells", type=int, default=40000)
    p.add_argument("--holdout-parents", default="00094,00177")
    p.add_argument("--train-on-all", action="store_true",
                   help="RELEASE mode: train regions from every run (eval informational)")
    p.add_argument("--vae-ckpt", default="weights/vae_dejavu/model.pt")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cache_dir = G.repo_path(args.cache)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    rows, cols = 3, 5
    ch, cw = meta["cell_h"], meta["cell_w"]
    viz_parents = tuple(s for s in args.holdout_parents.split(",") if s)
    cache = CellCache(cache_dir, rows, cols, meta["pad_h"], meta["pad_w"], ch, cw,
                      window_labels=True, viz_parents=viz_parents)
    N = len(cache.chars)

    model, cfg = load_any(dict(discover_models())[args.init], meta, device)
    model.train()
    use_tok = args.clip_token == "global"
    if use_tok and not getattr(model, "use_clip", False):
        import torch.nn as nn
        dim = model.pos.shape[1]
        model.clip_proj = nn.Linear(cache.clip_dim, dim).to(device)
        nn.init.zeros_(model.clip_proj.weight)
        nn.init.zeros_(model.clip_proj.bias)
        model.clip_pos = nn.Parameter(torch.zeros(1, dim, device=device))
        model.use_clip = True
        model.n_extra = 1
        model.center += 1
        cfg = dict(cfg, clip_token=True)
        print("  clip token added (zero-init projection)", flush=True)

    from unicasso.adapter.corrupt import CorruptionSampler
    sampler = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu",
                                profile="dejavu")
    bm_white = sampler.bitmaps.cpu().float().reshape(sampler.N, -1).to(device)
    from unicasso.engine.clip_loss import CLIPPerceptualLoss
    clipper = CLIPPerceptualLoss(torch.device(device), model_name="RN101",
                                 pretrained="openai", n_aug=args.clip_aug,
                                 crop_scale=tuple(args.crop_scale))
    clip_val_stems = [r["stem"] for r in cache.meta["runs"]
                      if r["split"] == "val"][:4]

    train_runs = [i for i, r in enumerate(cache.meta["runs"])
                  if args.train_on_all
                  or (r["split"] == "train" and r["parent"] not in viz_parents)]

    name = args.name or (args.init.replace("@", "_s") + "_gft"
                         + ("_muon" if args.optim == "muon" else "")
                         + ("_tok" if use_tok else ""))
    outdir = os.path.join(G.REPO_ROOT, "runs", "cellclf", name)
    os.makedirs(outdir, exist_ok=True)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[{name}] init {args.init} | {n_params/1e6:.2f}M params | "
          f"{len(train_runs)} train runs | region <= {args.region_rows}x{args.region_cols} "
          f"| {args.optim} | {device}", flush=True)

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

    # token t -> cell offset relative to the window center
    offs = [(t // cols - rows // 2, t % cols - cols // 2) for t in range(rows * cols)]
    hist = {"step": [], "ce": [], "clip": [], "lam": [], "nb": []}
    ev_hist = {"step": [], "top1": [], "disagree_set_top1": [], "val_clip": []}
    lam = None
    t0 = time.time()

    for step in range(args.steps):
        ri = train_runs[int(rng.integers(len(train_runs)))]
        gh, gw = cache.labels[ri].shape
        rh0, cw0 = min(gh, args.region_rows), min(gw, args.region_cols)
        top = int(rng.integers(0, gh - rh0 + 1))
        left = int(rng.integers(0, gw - cw0 + 1))
        ys, xs = np.mgrid[top:top + rh0, left:left + cw0]
        idx = np.stack([np.full(rh0 * cw0, ri), ys.ravel(), xs.ravel()],
                       axis=1).astype(np.int32)
        w, lab, _, _ = cache.fetch(idx)
        M = len(idx)

        # framing-ensemble accumulation targets: window token -> region cell id
        yy = (idx[:, 1] - top)[:, None] + np.array([o[0] for o in offs])[None, :]
        xx = (idx[:, 2] - left)[:, None] + np.array([o[1] for o in offs])[None, :]
        valid = (yy >= 0) & (yy < rh0) & (xx >= 0) & (xx < cw0)
        ids = np.where(valid, yy * cw0 + xx, 0).astype(np.int64)
        ids_t = torch.from_numpy(ids).to(device)
        valid_t = torch.from_numpy(valid).to(device)

        x = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
        y = torch.from_numpy(lab).to(device)
        emb = (torch.from_numpy(cache.clip[ri]).to(device)[None]
               if use_tok else None)
        logits_parts = [model.forward_all(
            x[i:i + args.chunk],
            emb.expand(x[i:i + args.chunk].shape[0], -1) if use_tok else None)
            for i in range(0, M, args.chunk)]
        logits = torch.cat(logits_parts)                       # (M, T, N)
        ce_loss = kernel_ce(logits, y, kernel, ce_w)

        probs = logits.softmax(-1)                             # (M, T, N)
        wtok = (kernel[None, :] * valid_t.float())             # (M, T)
        acc = torch.zeros(M, N, device=device)
        wsum = torch.zeros(M, device=device)
        acc.index_add_(0, ids_t.reshape(-1),
                       (probs * wtok[:, :, None]).reshape(-1, N))
        wsum.index_add_(0, ids_t.reshape(-1), wtok.reshape(-1))
        pbar = acc / wsum[:, None].clamp_min(1e-8)             # (M_cells, N)

        u = torch.rand_like(pbar).clamp_(1e-9, 1 - 1e-9)
        g = (pbar.clamp_min(1e-9).log() - torch.log(-torch.log(u))).argmax(-1)
        hard = F.one_hot(g, N).float()
        st = hard + pbar - pbar.detach()
        render = (st @ bm_white).view(rh0, cw0, ch, cw) \
            .permute(0, 2, 1, 3).reshape(rh0 * ch, cw0 * cw)   # white=1
        ink = cache.ink[ri][cache.off_y + top * ch:cache.off_y + (top + rh0) * ch,
                            cache.off_x + left * cw:cache.off_x + (left + cw0) * cw]
        tgt = 1.0 - torch.from_numpy(ink.astype(np.float32) / 255.0).to(device)
        clip_l = clipper(render, tgt)
        nb = float((g != cache.space).float().mean())

        opt.zero_grad(set_to_none=True)
        recal = lam is None or (step + 1) % args.recal_every == 0
        if recal:
            params = [q for q in model.parameters() if q.requires_grad]
            g_ce = torch.autograd.grad(ce_loss, params, retain_graph=True,
                                       allow_unused=True)
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
                  f"-> lam {lam:.4f} (nb {nb:.2f})", flush=True)
        else:
            (ce_loss + lam * clip_l).backward()
        opt.step()
        sched.step()

        hist["step"].append(step + 1); hist["ce"].append(float(ce_loss))
        hist["clip"].append(float(clip_l)); hist["lam"].append(lam)
        hist["nb"].append(nb)
        if (step + 1) % 10 == 0:
            print(f"  step {step+1}/{args.steps} ce {float(ce_loss):.3f} "
                  f"clip {float(clip_l):.3f} lam {lam:.3f} nb {nb:.2f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if (step + 1) % args.eval_every == 0:
            m = evaluate(model, cache, "val", device, 2048, cache.space, use_tok,
                         max_cells=args.eval_cells, ce_weight=ce_w)
            m["val_clip_render"] = eval_clip_render(
                model, cache, clip_val_stems, clipper, bm_white, device, use_tok)
            ev_hist["step"].append(step + 1)
            ev_hist["top1"].append(m["top1"])
            ev_hist["disagree_set_top1"].append(m["disagree_set_top1"])
            ev_hist["val_clip"].append(m["val_clip_render"])
            print(f"  [val@{step+1}] top1 {m['top1']:.4f} "
                  f"disagree {m['disagree_set_top1']:.4f} "
                  f"val-clip {m['val_clip_render']:.4f}", flush=True)
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "ft_config": vars(args), "chars": cache.chars,
                        "variant": "tf5x3", "step": step + 1},
                       os.path.join(outdir, f"ckpt_step{step + 1:05d}.pt"))

    final = evaluate(model, cache, "val", device, 2048, cache.space, use_tok,
                     ce_weight=ce_w)
    final["val_clip_render"] = eval_clip_render(
        model, cache, clip_val_stems, clipper, bm_white, device, use_tok)
    final.update(train_minutes=round((time.time() - t0) / 60, 1),
                 config=vars(args), name=name)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=1)
    np.savez(os.path.join(outdir, "loss_hist.npz"),
             **{k: np.array(v, dtype=float) for k, v in hist.items()},
             **{"ev_" + k: np.array(v, dtype=float) for k, v in ev_hist.items()})
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "ft_config": vars(args), "chars": cache.chars,
                "variant": "tf5x3"}, os.path.join(outdir, "model.pt"))
    print(f"[{name}] FINAL top1 {final['top1']:.4f} | "
          f"disagree {final['disagree_set_top1']:.4f} | "
          f"val-clip {final['val_clip_render']:.4f}", flush=True)


if __name__ == "__main__":
    main()
