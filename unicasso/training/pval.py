"""Score checkpoints on the SAME six held-out photos the trainer uses.

Replicates joint_train's photo_eval / _render_grid(det=True) exactly: centred
crop, exact cell budget, argmax glyphs, no flip, no backward subsampling. The
validator set is reproduced from photo_groups(seed) + the same take-off-the-end
split, so these numbers are directly comparable to the [pval@] lines in the log.

Exists because model.pt lands at --steps, which is PAST the last --eval-every,
so the shipped weights are the one checkpoint nothing ever measured.
"""
import argparse, math, sys, zlib
import numpy as np, torch
from PIL import Image, ImageOps, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from unicasso.substrate import glyphs as G
from unicasso.engine.color import decompose, nomination_target, srgb_to_lab
from unicasso.engine.clip_loss import CLIPPerceptualLoss
from unicasso.training.cellclf_color_train import (cell_feats, grid_windows_t,
                                                   mask_fit, token_maps)
from unicasso.training.joint_train import build_model
from unicasso.training.pool_manager import photo_groups
from unicasso.training.train_cell_classifier import binomial_kernel

ROWS, COLS = 3, 5
ap = argparse.ArgumentParser()
ap.add_argument("--photos", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--photo-val", type=int, default=6)
ap.add_argument("--photo-cells", type=int, default=700)
ap.add_argument("--photo-crop", type=float, default=0.95)
ap.add_argument("--clip-aug", type=int, default=16)
ap.add_argument("--blend-ridge", type=float, default=1.0)
ap.add_argument("--vae-ckpt", default="weights/vae_sfmono/model.pt")
ap.add_argument("--profile", default="sfmono")
ap.add_argument("--device", default=None)
ap.add_argument("ckpts", nargs="+")
a = ap.parse_args()
dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

# ---- reproduce the held-out split EXACTLY as joint_train does -------------
groups = photo_groups(a.photos, seed=a.seed)
val_photos = []
want = [g[2] * a.photo_val for g in groups]
take = [int(x) for x in want]
for j in sorted(range(len(take)), key=lambda i: want[i] - take[i],
                reverse=True)[:a.photo_val - sum(take)]:
    take[j] += 1
for (name, files, w), k in zip(groups, take):
    k = min(k, max(0, len(files) - 1))
    if k:
        val_photos += files[-k:]
print(f"{len(val_photos)} held-out photos (seed {a.seed}):")
for p in val_photos:
    print("   ", p.split("/")[-1])

from unicasso.adapter.corrupt import CorruptionSampler
smp = CorruptionSampler(G.repo_path(a.vae_ckpt), device="cpu", profile=a.profile)
ink_flat = (1.0 - smp.bitmaps.cpu().float()).reshape(smp.N, -1).to(dev)
del smp
clipper = CLIPPerceptualLoss(torch.device(dev), model_name="RN101",
                             pretrained="openai", n_aug=a.clip_aug,
                             crop_scale=(0.4, 0.9), batch_aug=(dev == "cuda"))
kernel = binomial_kernel(ROWS, COLS, dev)


@torch.no_grad()
def score(model, cfg, path):
    ch, cw, ph, pw = cfg["cell_h"], cfg["cell_w"], cfg["pad_h"], cfg["pad_w"]
    in4 = cfg.get("in_ch", 1) == 4
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        aw, ah = im.size
        f = min(1.0, max(0.1, a.photo_crop))
        bw, bh = int(aw * f), int(ah * f)
        x0, y0 = (aw - bw) // 2, (ah - bh) // 2          # det: centred
        M_t = float(a.photo_cells)                        # det: exact budget
        W = int(round(math.sqrt(max(64.0, M_t) * ch * bw / (cw * bh))))
        W = int(min(120, max(16, W)))
        gh = max(6, int(round(W * cw * bh / (ch * bw))))
        im = im.crop((x0, y0, x0 + bw, y0 + bh)) \
            .resize((W * cw, gh * ch), Image.LANCZOS)
    rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0).to(dev)
    dec = decompose(rgb, gh, W, ch, cw)
    ink_u8 = ((1.0 - nomination_target(dec)) * 255.0).clamp(0, 255).to(torch.uint8)
    M = gh * W
    ids, valid = token_maps(gh, W, ROWS, COLS)
    ids_t, val_t = torch.from_numpy(ids).to(dev), torch.from_numpy(valid).to(dev)
    ft = cell_feats(dec)[ids_t] * val_t[:, :, None].float()
    xi = grid_windows_t(ink_u8.float(), gh, W, ch, cw, ph, pw).div_(255).unsqueeze(1)
    xc = grid_windows_t(rgb, gh, W, ch, cw, ph, pw, edge=True).permute(0, 3, 1, 2)
    x1 = torch.cat([xi, xc], 1) if in4 else xi
    t, s1 = model._tokens(x1, None, ft, 1, want_skip=True)
    lg = model.head(t[:, model.n_extra:])
    if cfg.get("render_ensemble") == "center":
        # --centre-only models: the main head is trained at the centre alone
        # (neighbours through the aux head), so the 15-position ensemble
        # would read 81% of its weight from untrained positions. Mirrors
        # joint_train._render_grid and Lite's checkpoint-resolved read.
        prob = lg[:, model.center - model.n_extra].softmax(-1)
    else:
        pr = lg.softmax(-1)
        wt = kernel[None, :] * val_t.float()
        acc = torch.zeros(M, pr.shape[-1], device=dev)
        acc.index_add_(0, ids_t.reshape(-1), (pr * wt[:, :, None]).reshape(-1, pr.shape[-1]))
        wsum = torch.zeros(M, device=dev)
        wsum.index_add_(0, ids_t.reshape(-1), wt.reshape(-1))
        prob = acc / wsum.clamp_min(1e-8)[:, None]
    hard = prob.clamp_min(1e-9).log().argmax(-1)          # det: argmax
    ink_st = torch.zeros_like(prob).scatter_(1, hard[:, None], 1.0) @ ink_flat
    mlog = model.mask_center(x1, t, s1)
    pm = mlog[:, :, ph:ph + ch, pw:pw + cw].softmax(1)
    fg, bg = mask_fit(pm, dec["cell_rgb"], ridge=a.blend_ridge)
    cell = bg[:, None, :] + (fg - bg)[:, None, :] * ink_st[:, :, None]
    render = cell.view(gh, W, ch, cw, 3).permute(0, 2, 1, 3, 4) \
        .reshape(gh * ch, W * cw, 3)
    d = (pm[:, 0] - pm[:, 1]).reshape(M, -1)
    aa = d - d.mean(1, keepdim=True)
    bb = dec["ink"] - dec["ink"].mean(1, keepdim=True)
    corr = (aa * bb).sum(1) / (aa.norm(dim=1) * bb.norm(dim=1)).clamp_min(1e-8)
    gw = (dec["gate"] > 0.5).float(); gs = gw.sum().clamp_min(1.0)
    got = (srgb_to_lab(fg[:, None])[:, 0] - srgb_to_lab(bg[:, None])[:, 0]).norm(dim=-1)
    # IDENTICAL crops for every checkpoint on a given photo. Without this the
    # 16-crop draw moves the score by ~0.003, which is the same size as the
    # differences between adjacent checkpoints -- i.e. pure noise dressed as
    # signal. Seeded per photo, not globally, so photos stay decorrelated.
    # crc32, NOT hash(): Python randomises str hashing per process unless
    # PYTHONHASHSEED is set, so hash() gives a different crop draw on every
    # invocation and scores stop being comparable ACROSS runs of this script.
    # (They stay consistent WITHIN one run, which is exactly what hides it.)
    torch.manual_seed(zlib.crc32(path.encode()) % (2**31))
    return (float(clipper(render, rgb)), float((corr * gw).sum() / gs),
            float(((corr < 0).float() * gw).sum() / gs),
            float(((got / dec["sep"].clamp_min(1e-6)).clamp(0, 4) * gw).sum() / gs))


print(f"\n{'checkpoint':<26} {'clip':>8} {'pol':>7} {'inv':>7} {'reach':>7}")
print("-" * 60)
for ck in a.ckpts:
    model, chars, cfg = build_model(ck, dev)
    model.eval()
    n = len(val_photos)
    c = p = i = r = 0.0
    for ph_ in val_photos:
        cc, pp, ii, rr = score(model, cfg, ph_)
        c += cc/n; p += pp/n; i += ii/n; r += rr/n
    print(f"{ck.split('/')[-1]:<26} {c:8.4f} {p:7.3f} {i:7.1%} {r:7.2f}")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
