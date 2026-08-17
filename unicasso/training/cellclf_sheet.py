"""Visual comparison sheet for the cell-classifier ladder.

python -m unicasso.training.cellclf_sheet \
    --models cell cnn5x3 tf5x3_h4 tf5x3_clip_h4 --best tf5x3_clip_h4 \
    --parents 3 --variant b2000

For each chosen validation parent, one column of panels:
    line art | greedy VAE-snap | each model's rendered prediction |
    unicasso (the optimizer's actual render) | best model with DISAGREEMENT
    cells (pred != optimizer) tinted red.
Writes runs/cellclf/sheets/sheet_<stem>.png (one per parent) — panels stacked
vertically per parent so glyphs stay at native size.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from unicasso.adapter.corrupt import CorruptionSampler
from unicasso.substrate import glyphs as G
from unicasso.training.train_cell_classifier import VARIANTS, TokenTransformer, WindowCNN


def load_model(run_dir, meta, device):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=False)
    cfg = ck["config"]
    rows, cols, is_tf = VARIANTS[cfg["variant"]]
    N = len(ck["chars"])
    if is_tf:
        m = TokenTransformer(rows, cols, meta["cell_h"], meta["cell_w"],
                             meta["pad_h"], meta["pad_w"], N, dim=cfg["dim"],
                             heads=cfg["heads"], n_blocks=cfg.get("blocks", 1),
                             clip_dim=meta["clip_dim"] if cfg["clip_token"] else 0)
    else:
        win_h = rows * meta["cell_h"] + 2 * meta["pad_h"]
        win_w = cols * meta["cell_w"] + 2 * meta["pad_w"]
        m = WindowCNN(win_h, win_w, N)
    sd = ck["state_dict"]
    # checkpoints from before the multi-block refactor kept block weights at top level
    if is_tf and "blocks.0.ln1.weight" not in sd:
        sd = {("blocks.0." + k if k.split(".")[0] in ("ln1", "attn", "ln2", "mlp") else k): v
              for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(device).eval(), cfg


@torch.no_grad()
def predict_grid(model, cfg, ink_u8, clip_emb, meta, device, batch=1024):
    rows, cols, _ = VARIANTS[cfg["variant"]]
    ch, cw, ph, pw = meta["cell_h"], meta["cell_w"], meta["pad_h"], meta["pad_w"]
    gh, gw = ink_u8.shape[0] // ch, ink_u8.shape[1] // cw
    rh, rw = rows // 2, cols // 2
    py, px = rh * ch + ph, rw * cw + pw
    padded = np.pad(ink_u8, ((py, py), (px, px)))
    win_h, win_w = rows * ch + 2 * ph, cols * cw + 2 * pw
    ce = (torch.from_numpy(clip_emb).to(device).unsqueeze(0)
          if cfg["clip_token"] else None)
    out = np.empty(gh * gw, dtype=np.int64)
    coords = [(y, x) for y in range(gh) for x in range(gw)]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        w = np.stack([padded[py + y * ch - rh * ch - ph: py + y * ch - rh * ch - ph + win_h,
                             px + x * cw - rw * cw - pw: px + x * cw - rw * cw - pw + win_w]
                      for y, x in chunk])
        x_t = torch.from_numpy(w).to(device).float().div_(255).unsqueeze(1)
        c_t = ce.expand(x_t.shape[0], -1) if ce is not None else None
        out[i:i + batch] = model(x_t, c_t).argmax(dim=1).cpu().numpy()
    return out.reshape(gh, gw)


def render_grid(sampler, grid):
    img = sampler.render(torch.from_numpy(grid.astype(np.int64)))  # white=1
    return (img.cpu().numpy() * 255).astype(np.uint8)


def tint_disagree(render_u8, pred, labels, ch, cw):
    """RGB render with cells where pred != labels tinted red (ink kept dark)."""
    rgb = np.stack([render_u8] * 3, axis=-1).astype(np.float32)
    mask = pred != labels
    for (y, x) in zip(*np.nonzero(mask)):
        cell = rgb[y * ch:(y + 1) * ch, x * cw:(x + 1) * cw]
        cell[..., 1] *= 0.55   # drop G and B -> red-tinted paper, dark-red ink
        cell[..., 2] *= 0.55
    return rgb.astype(np.uint8)


def caption(img, text, font):
    bar = Image.new("RGB", (img.width, 22), (245, 245, 245))
    d = ImageDraw.Draw(bar)
    d.text((6, 4), text, fill=(20, 20, 20), font=font)
    out = Image.new("RGB", (img.width, img.height + 22), (255, 255, 255))
    out.paste(bar, (0, 0))
    out.paste(img, (0, 22))
    return out


def to_img(arr):
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="runs/cellclf/cache")
    p.add_argument("--models", nargs="+", required=True,
                   help="run dir names under runs/cellclf/, in display order")
    p.add_argument("--best", required=True, help="model whose disagreement panel to draw")
    p.add_argument("--parents", type=int, default=3)
    p.add_argument("--stems", nargs="*", default=None, help="explicit run stems instead")
    p.add_argument("--variant", default="b2000", choices=["b2000", "b980"])
    p.add_argument("--renders", default="data/dataset_v1/runs/renders")
    p.add_argument("--vae-ckpt", default="weights/vae_dejavu/model.pt")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = G.repo_path(args.cache)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    sampler = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu", profile="dejavu")
    ch, cw = meta["cell_h"], meta["cell_w"]

    if args.stems:
        stems = args.stems
    else:
        val = [r["stem"] for r in meta["runs"]
               if r["split"] == "val" and r["stem"].endswith("_" + args.variant)]
        rng = np.random.default_rng(args.seed)
        stems = [val[i] for i in rng.permutation(len(val))[:args.parents]]
    print("parents:", stems)

    models = {}
    for name in args.models:
        models[name] = load_model(os.path.join(G.REPO_ROOT, "runs", "cellclf", name),
                                  meta, device)
    try:
        font = ImageFont.truetype(G.repo_path("fonts/DejaVuSansMono.ttf"), 14)
    except OSError:
        font = ImageFont.load_default()

    out_dir = os.path.join(G.REPO_ROOT, "runs", "cellclf", "sheets")
    os.makedirs(out_dir, exist_ok=True)

    for stem in stems:
        d = np.load(os.path.join(cache_dir, stem + ".npz"))
        ink, labels, greedy, emb = d["ink"], d["labels"].astype(np.int64), \
                                   d["greedy"].astype(np.int64), d["clip"]
        panels = [caption(to_img(255 - ink), "line art (input)", font),
                  caption(to_img(render_grid(sampler, greedy)),
                          f"greedy VAE-snap  ({(greedy == labels).mean():.1%} match)", font)]
        preds = {}
        for name, (model, cfg) in models.items():
            pred = predict_grid(model, cfg, ink, emb, meta, device)
            preds[name] = pred
            panels.append(caption(to_img(render_grid(sampler, pred)),
                                  f"{name}  ({(pred == labels).mean():.1%} match)", font))
        rp = os.path.join(G.repo_path(args.renders), stem + ".png")
        uni = Image.open(rp).convert("RGB") if os.path.exists(rp) else \
            to_img(render_grid(sampler, labels))
        if uni.size != panels[0].size:
            uni = uni.resize((panels[0].width, panels[0].height - 22), Image.LANCZOS)
        panels.append(caption(uni, "unicasso (optimizer output = labels)", font))
        best_pred = preds[args.best]
        panels.append(caption(to_img(tint_disagree(render_grid(sampler, best_pred),
                                                   best_pred, labels, ch, cw)),
                              f"{args.best} vs unicasso — disagreement cells tinted "
                              f"({(best_pred != labels).mean():.1%} of cells)", font))

        W = max(im.width for im in panels)
        H = sum(im.height + 8 for im in panels)
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        y = 0
        for im in panels:
            sheet.paste(im, (0, y))
            y += im.height + 8
        out = os.path.join(out_dir, f"sheet_{stem}.png")
        sheet.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
