"""Framing-ensemble inference for dense-trained cell classifiers.

python -m unicasso.training.cellclf_ensemble --model tf5x3_dense_b3_h4@1000 \
    --modes center prob logprob --render-stems 00094_b2000 00177_b2000

Sliding inference already computes one forward per cell center; the dense head
predicts all rows*cols window cells per forward. Ensemble modes keep everything:
each grid cell accumulates the prediction of every framing that covers it,
weighted by that token slot's binomial kernel weight, then argmax.

    prob     weighted arithmetic mean of softmax probs   (mixture -- robust)
    logprob  weighted sum of log-softmax                  (product of experts -- sharp)
    center   center token only (the baseline everything else reported)

Evaluates each mode on the FULL cached val split (same metrics as the trainer)
and optionally renders chosen stems side by side per mode.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from unicasso.substrate import glyphs as G
from unicasso.training.cellclf_widths import load_any
from unicasso.training.train_cell_classifier import binomial_kernel

MODES = ("center", "prob", "logprob")


@torch.no_grad()
def predict_run(model, cfg, ink_u8, clip_emb, meta, device, mode, batch=1024):
    """One cached run -> (gh, gw) predicted glyph indices under the given mode."""
    from unicasso.training.train_cell_classifier import VARIANTS
    rows, cols, _ = VARIANTS[cfg["variant"]]
    ch, cw, ph, pw = meta["cell_h"], meta["cell_w"], meta["pad_h"], meta["pad_w"]
    gh, gw = ink_u8.shape[0] // ch, ink_u8.shape[1] // cw
    rh, rw = rows // 2, cols // 2
    py, px = rh * ch + ph, rw * cw + pw
    padded = np.pad(ink_u8, ((py, py), (px, px)))
    win_h, win_w = rows * ch + 2 * ph, cols * cw + 2 * pw
    ce = (torch.from_numpy(clip_emb).to(device).unsqueeze(0)
          if cfg.get("clip_token") else None)
    kernel = binomial_kernel(rows, cols, device)                     # (T,)
    tc = rh * cols + rw
    N = None
    acc = None            # (gh*gw, N) accumulated evidence
    wacc = None           # (gh*gw,) accumulated kernel mass
    coords = [(y, x) for y in range(gh) for x in range(gw)]
    # token t=(i,j) of a window centered at (y,x) covers grid cell (y+i-rh, x+j-rw)
    di = np.repeat(np.arange(rows) - rh, cols)
    dj = np.tile(np.arange(cols) - rw, rows)
    for s in range(0, len(coords), batch):
        chunk = coords[s:s + batch]
        w = np.stack([padded[py + (y - rh) * ch - ph: py + (y - rh) * ch - ph + win_h,
                             px + (x - rw) * cw - pw: px + (x - rw) * cw - pw + win_w]
                      for y, x in chunk])
        x_t = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
        c_t = ce.expand(x_t.shape[0], -1) if ce is not None else None
        logits = model.forward_all(x_t, c_t)                         # (B, T, N)
        if N is None:
            N = logits.shape[-1]
            acc = torch.zeros(gh * gw, N, device=device)
            wacc = torch.zeros(gh * gw, device=device)
        if mode == "center":
            ys = torch.tensor([y for y, _ in chunk], device=device)
            xs = torch.tensor([x for _, x in chunk], device=device)
            acc[ys * gw + xs] = logits[:, tc]
            wacc[ys * gw + xs] = 1.0
            continue
        ev = (F.log_softmax(logits, dim=-1) if mode == "logprob"
              else F.softmax(logits, dim=-1))                        # (B, T, N)
        ys = np.array([y for y, _ in chunk])[:, None] + di[None, :]  # (B, T)
        xs = np.array([x for _, x in chunk])[:, None] + dj[None, :]
        valid = (ys >= 0) & (ys < gh) & (xs >= 0) & (xs < gw)
        flat = torch.from_numpy(np.where(valid, ys * gw + xs, 0).ravel()).to(device)
        vmask = torch.from_numpy(valid.ravel()).to(device)
        kw = kernel.repeat(len(chunk)) * vmask                       # (B*T,)
        acc.index_add_(0, flat, ev.reshape(-1, N) * kw[:, None])
        wacc.index_add_(0, flat, kw)
    if mode == "logprob":
        acc = acc / wacc.clamp_min(1e-8)[:, None]                    # weighted geo mean
    return acc.argmax(dim=1).view(gh, gw).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="runs/cellclf/cache")
    p.add_argument("--model", required=True, help="run name or name@step (dense-trained)")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--render-stems", nargs="*", default=[])
    p.add_argument("--vae-ckpt", default="weights/vae_dejavu/model.pt")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cache_dir = G.repo_path(args.cache)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    from unicasso.training.cellclf_widths import discover_models
    model, cfg = load_any(dict(discover_models())[args.model], meta, device)
    space = meta["chars"].index(" ")

    val_runs = [r for r in meta["runs"] if r["split"] == "val"]
    results = {}
    for mode in args.modes:
        agree = nb_agree = nb_tot = tot = dis_agree = dis_tot = 0
        for r in val_runs:
            d = np.load(os.path.join(cache_dir, r["stem"] + ".npz"))
            pred = predict_run(model, cfg, d["ink"], d["clip"], meta, device, mode)
            lab, gre = d["labels"].astype(np.int64), d["greedy"].astype(np.int64)
            ok = pred == lab
            nb, dis = lab != space, lab != gre
            agree += ok.sum(); tot += lab.size
            nb_agree += ok[nb].sum(); nb_tot += nb.sum()
            dis_agree += ok[dis].sum(); dis_tot += dis.sum()
        results[mode] = dict(top1=agree / tot, nonblank_top1=nb_agree / nb_tot,
                             disagree_set_top1=dis_agree / dis_tot)
        print(f"[{mode:8s}] top1 {results[mode]['top1']:.4f} "
              f"nonblank {results[mode]['nonblank_top1']:.4f} "
              f"disagree {results[mode]['disagree_set_top1']:.4f}", flush=True)

    out_dir = os.path.join(G.REPO_ROOT, "runs", "cellclf", "sheets")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"ensemble_{args.model.replace('@', '_s')}.json"), "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
                  f, indent=1)

    if args.render_stems:
        from unicasso.adapter.corrupt import CorruptionSampler
        from unicasso.training.cellclf_sheet import caption, render_grid, to_img
        from PIL import Image, ImageFont
        sampler = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu", profile="dejavu")
        try:
            font = ImageFont.truetype(G.repo_path("fonts/DejaVuSansMono.ttf"), 13)
        except OSError:
            font = ImageFont.load_default()
        rows_img = []
        for stem in args.render_stems:
            d = np.load(os.path.join(cache_dir, stem + ".npz"))
            lab = d["labels"].astype(np.int64)
            panels = []
            for mode in args.modes:
                pred = predict_run(model, cfg, d["ink"], d["clip"], meta, device, mode)
                panels.append(caption(to_img(render_grid(sampler, pred)),
                                      f"{stem}  {mode}  ({(pred == lab).mean():.1%} match)",
                                      font))
            gut = 8
            row = Image.new("RGB", (sum(x.width for x in panels) + gut * (len(panels) + 1),
                                    max(x.height for x in panels)), (255, 255, 255))
            xo = gut
            for pane in panels:
                row.paste(pane, (xo, 0)); xo += pane.width + gut
            rows_img.append(row)
        W = max(r.width for r in rows_img)
        sheet = Image.new("RGB", (W, sum(r.height + 8 for r in rows_img)), (255, 255, 255))
        y = 0
        for r in rows_img:
            sheet.paste(r, (0, y)); y += r.height + 8
        path = os.path.join(out_dir, f"ensemble_{args.model.replace('@', '_s')}.png")
        sheet.save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
