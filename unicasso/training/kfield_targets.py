"""k-field target dataset: per-photo contrast optima, on-policy glyphs.

python -m unicasso.training.kfield_targets --model runs/cellclf/<run>/ckpt.pt

Per sample: photo variant (full-frame at a random deploy width, or a random
non-grid-aligned crop/zoom) -> decompose -> the COLOR MODEL's own glyphs
(on-policy: k targets for the layouts the deployed model actually produces) ->
blend colors -> 100-step Muon k-field optimization (lr 0.05, 8 crop augs, TV).
Writes ktargets/<photo>_v<i>.npz: k (gh,gw) f32, glyphs (gh,gw) i16, feats
(M,11), crop box, width, final clip loss. Skips existing files (resumable).
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

from unicasso.substrate import glyphs as G, raster
from unicasso.engine.color import decompose, nomination_target
from unicasso.training.cellclf_color import fit_fg_bg_distweighted, glyph_bg_dist
from unicasso.training.cellclf_color_train import cell_feats, grid_windows, token_maps
from unicasso.training.train_cell_classifier import (MuonWithAdamW,
                                                     TokenTransformer,
                                                     binomial_kernel)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="color-model checkpoint path")
    p.add_argument("--photos", default="data/dataset_v1/photos")
    p.add_argument("--out", default="runs/cellclf/ktargets")
    p.add_argument("--variants", type=int, default=2)
    p.add_argument("--k-steps", type=int, default=100)
    p.add_argument("--k-lr", type=float, default=0.05)
    p.add_argument("--clip-aug", type=int, default=8)
    p.add_argument("--tv-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    meta = json.load(open(G.repo_path("runs/cellclf/cache_full/meta.json")))
    ch, cw = meta["cell_h"], meta["cell_w"]
    ph, pw = meta["pad_h"], meta["pad_w"]
    rows, cols = 3, 5

    from unicasso.adapter.corrupt import CorruptionSampler
    sampler = CorruptionSampler(G.repo_path("weights/vae_dejavu/model.pt"),
                                device="cpu", profile="dejavu")
    ink_flat_d = (1.0 - sampler.bitmaps.cpu().float()) \
        .reshape(sampler.N, -1).to(device)
    bg_dist_all = glyph_bg_dist(ink_flat_d.cpu(), ch, cw).to(device)
    kernel = binomial_kernel(rows, cols, device)

    ck = torch.load(G.repo_path(args.model), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    N = len(ck["chars"])
    m = TokenTransformer(rows, cols, ch, cw, ph, pw, N, dim=cfg.get("dim", 64),
                         heads=cfg.get("heads", 4), n_blocks=cfg.get("blocks", 3),
                         in_ch=cfg.get("in_ch", 1))
    dim = cfg.get("dim", 64)
    if cfg.get("feat_dim", 0):
        m.color_proj = nn.Linear(cfg["feat_dim"], dim)
    m.mode_emb = nn.Parameter(torch.zeros(2, dim))
    m.k_head = nn.Linear(dim, 1)
    m.col_head = nn.Linear(dim, 6)
    m.load_state_dict(ck["state_dict"])
    m = m.to(device).eval()

    from unicasso.engine.clip_loss import CLIPPerceptualLoss
    clipper = CLIPPerceptualLoss(torch.device(device), model_name="RN101",
                                 pretrained="openai", n_aug=args.clip_aug,
                                 crop_scale=(0.4, 0.9))

    photos = sorted(sum((glob.glob(os.path.join(G.repo_path(args.photos), e))
                         for e in ("*.jpg", "*.jpeg", "*.png")), []))
    outdir = G.repo_path(args.out)
    os.makedirs(outdir, exist_ok=True)
    print(f"[ktargets] {len(photos)} photos x {args.variants} variants | "
          f"{args.k_steps} muon steps lr {args.k_lr} aug {args.clip_aug} | {device}",
          flush=True)

    t0, done = time.time(), 0
    for path in photos:
        pid = os.path.splitext(os.path.basename(path))[0]
        for v in range(args.variants):
            out_npz = os.path.join(outdir, f"{pid}_v{v}.npz")
            if os.path.exists(out_npz):
                continue
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                aw, ah = im.size
                if v == 0:                        # full frame, deploy width
                    W = int(rng.choice([40, 50, 60]))
                    gh = raster.grid_height_for_aspect(aw, ah, W, cw, ch, 0)
                    gh = min(gh, 40)
                    box = (0, 0, aw, ah)
                else:                             # random crop/zoom
                    W = int(rng.integers(28, 45))
                    gh = int(rng.integers(14, 27))
                    asp = (W * cw) / (gh * ch)
                    bw = min(aw, int(ah * asp))
                    bh = int(bw / asp)
                    s = 0.35 + 0.65 * rng.random()
                    bw, bh = max(32, int(bw * s)), max(32, int(bh * s))
                    x0 = int(rng.integers(0, max(1, aw - bw)))
                    y0 = int(rng.integers(0, max(1, ah - bh)))
                    box = (x0, y0, x0 + bw, y0 + bh)
                im = im.crop(box).resize((W * cw, gh * ch), Image.LANCZOS)
            rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
            dec = decompose(rgb, gh, W, ch, cw)
            ink_u8 = np.clip((1.0 - nomination_target(dec).numpy()) * 255.0,
                             0, 255).astype(np.uint8)
            M = gh * W
            w = grid_windows(ink_u8, gh, W, ch, cw, ph, pw)
            x = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
            ids, valid = token_maps(gh, W, rows, cols)
            ids_t = torch.from_numpy(ids).to(device)
            valid_t = torch.from_numpy(valid).to(device)
            feats = cell_feats(dec).to(device)
            feats_tok = feats[ids_t] * valid_t[:, :, None].float()
            with torch.no_grad():
                lo = []
                for i in range(0, M, 384):
                    t = m._tokens(x[i:i + 384], None, feats_tok[i:i + 384], 1)
                    lo.append(m.head(t[:, m.n_extra:]))
                probs = torch.cat(lo).softmax(-1)
                wtok = kernel[None, :] * valid_t.float()
                acc = torch.zeros(M, N, device=device)
                wsum = torch.zeros(M, device=device)
                acc.index_add_(0, ids_t.reshape(-1),
                               (probs * wtok[:, :, None]).reshape(-1, N))
                wsum.index_add_(0, ids_t.reshape(-1), wtok.reshape(-1))
                g = (acc / wsum[:, None].clamp_min(1e-8)).argmax(-1)
            mask = ink_flat_d[g]
            C = dec["cell_rgb"].to(device)
            fg_f, bg_f = fit_fg_bg_distweighted(C, mask, bg_dist_all[g], pow_=1.0)
            fg0 = 0.5 * dec["fg"].to(device) + 0.5 * fg_f
            bg0 = 0.5 * dec["bg"].to(device) + 0.5 * bg_f
            mid = 0.5 * (fg0 + bg0)
            dfg, dbg = fg0 - mid, bg0 - mid
            rgb_d = rgb.to(device)
            k = nn.Parameter(torch.ones(gh, W, device=device))
            opt = MuonWithAdamW([k], [], muon_lr=args.k_lr, weight_decay=0.0)
            last = None
            for _ in range(args.k_steps):
                kv = k.view(-1)
                fg = mid + kv[:, None] * dfg
                bg = mid + kv[:, None] * dbg
                fg = fg + (fg.clamp(0, 1) - fg).detach()
                bg = bg + (bg.clamp(0, 1) - bg).detach()
                cell = bg[:, None, :] + (fg - bg)[:, None, :] * mask[:, :, None]
                img = cell.view(gh, W, ch, cw, 3).permute(0, 2, 1, 3, 4) \
                    .reshape(gh * ch, W * cw, 3)
                loss = clipper(img, rgb_d)
                last = float(loss)
                loss = loss + args.tv_weight * ((k[:, 1:] - k[:, :-1]).pow(2).mean()
                                                + (k[1:] - k[:-1]).pow(2).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    k.data.clamp_(0.0, 4.0)
            kd = k.detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(out_npz, k=kd,
                                glyphs=g.view(gh, W).cpu().numpy().astype(np.int16),
                                feats=feats.cpu().numpy().astype(np.float32),
                                box=np.array(box, np.int32), width=W,
                                clip_final=last)
            # free everything device-side once the npz is on disk; without this
            # the MPS caching allocator pins the high-water mark for the session
            del (w, x, feats, feats_tok, ids_t, valid_t, lo, probs, acc, wsum, g,
                 mask, C, fg_f, bg_f, fg0, bg0, mid, dfg, dbg, k, opt, rgb, rgb_d,
                 dec, fg, bg, cell, img, loss, kv)
            if device == "mps":
                torch.mps.empty_cache()
            done += 1
            if done % 20 == 0:
                rate = (time.time() - t0) / done
                print(f"  {done} samples ({rate:.1f} s/sample, "
                      f"last kstd {kd.std():.2f})", flush=True)
    print(f"KTARGETS_DONE ({done} new samples, "
          f"{(time.time()-t0)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
