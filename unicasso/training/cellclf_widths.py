"""Width-sweep sheets: run classifier checkpoints on parent line art at many grid widths.

python -m unicasso.training.cellclf_widths --device cpu
python -m unicasso.training.cellclf_widths --models cnn5x3 tf5x3_h4 --parents 00040 00177

For every checkpoint (default: all runs/cellclf/*/model.pt plus any ckpt_step*.pt
in dense run dirs), one sheet: rows = parent images, columns = grid widths
(default 70 down to 30, step 5), each panel captioned with the width and the
render wall time. No labels exist at these widths -- the sheets are qualitative.
Also writes width_timings.json (per model/parent/width predict+render seconds).
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageFont, ImageOps

from unicasso.adapter.corrupt import CorruptionSampler
from unicasso.substrate import glyphs as G
from unicasso.substrate import raster
from unicasso.training.cellclf_sheet import caption, load_model, predict_grid, render_grid, to_img

DEFAULT_PARENTS = ["00040", "00042", "00143", "00094", "00177"]


def discover_models():
    base = os.path.join(G.REPO_ROOT, "runs", "cellclf")
    out = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d) or os.path.basename(d) in ("cache", "sheets"):
            continue
        name = os.path.basename(d)
        if os.path.exists(os.path.join(d, "model.pt")):
            out.append((name, os.path.join(d, "model.pt")))
        for ck in sorted(glob.glob(os.path.join(d, "ckpt_step*.pt"))):
            step = os.path.basename(ck)[len("ckpt_step"):-3]
            out.append((f"{name}@{int(step)}", ck))
    return out


def load_any(path, meta, device):
    """load_model works on a run dir; this works on an explicit .pt path."""
    d = os.path.dirname(path)
    if os.path.basename(path) == "model.pt":
        return load_model(d, meta, device)
    ck = torch.load(path, map_location=device, weights_only=False)
    from unicasso.training.train_cell_classifier import VARIANTS, TokenTransformer, WindowCNN
    cfg = ck["config"]
    rows, cols, is_tf = VARIANTS[cfg["variant"]]
    N = len(ck["chars"])
    if is_tf:
        m = TokenTransformer(rows, cols, meta["cell_h"], meta["cell_w"],
                             meta["pad_h"], meta["pad_w"], N, dim=cfg["dim"],
                             heads=cfg["heads"], n_blocks=cfg.get("blocks", 1),
                             clip_dim=meta["clip_dim"] if cfg["clip_token"] else 0,
                             in_ch=cfg.get("in_ch", 1))
    else:
        m = WindowCNN(rows * meta["cell_h"] + 2 * meta["pad_h"],
                      cols * meta["cell_w"] + 2 * meta["pad_w"], N)
    if is_tf and "mask_dec.out.weight" in ck["state_dict"]:
        from unicasso.training.train_cell_classifier import MaskDecoder
        m.mask_dec = MaskDecoder(cfg["dim"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="runs/cellclf/cache")
    p.add_argument("--models", nargs="*", default=None,
                   help="run names (or name@step for ckpts); default: discover all")
    p.add_argument("--parents", nargs="*", default=DEFAULT_PARENTS)
    p.add_argument("--widths", nargs="*", type=int,
                   default=[70, 65, 60, 55, 50, 45, 40, 35, 30])
    p.add_argument("--lineart-root", default="data/dataset_v1/lineart")
    p.add_argument("--vae-ckpt", default="weights/vae_dejavu/model.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ensemble-mode", default=None, choices=["prob", "logprob"],
                   help="framing-ensemble inference (dense-trained checkpoints only); "
                        "default center-token prediction")
    args = p.parse_args()

    cache_dir = G.repo_path(args.cache)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    ch, cw = meta["cell_h"], meta["cell_w"]
    sampler = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu", profile="dejavu")
    try:
        font = ImageFont.truetype(G.repo_path("fonts/DejaVuSansMono.ttf"), 13)
    except OSError:
        font = ImageFont.load_default()

    all_models = dict(discover_models())
    names = args.models or list(all_models)
    missing = [n for n in names if n not in all_models]
    if missing:
        raise SystemExit(f"unknown models: {missing}; known: {list(all_models)}")

    # per-parent, width-independent inputs: native size, clip embedding (from any cached run)
    needs_clip = any(json.load(open(os.path.join(os.path.dirname(all_models[n]),
                                                 "metrics.json")))["config"]["clip_token"]
                     for n in names if os.path.basename(all_models[n]) == "model.pt")
    parents = {}
    for par in args.parents:
        lp = G.repo_path(os.path.join(args.lineart_root, f"{par}_line.png"))
        w0, h0 = raster.native_size(lp)
        stem = next((r["stem"] for r in meta["runs"] if r["parent"] == par), None)
        if stem is not None:
            emb = np.load(os.path.join(cache_dir, stem + ".npz"))["clip"]
        else:
            # parent not in the training cache (OOD lineart); clip models would
            # need a live embedding -- refuse rather than silently feed zeros
            if needs_clip:
                raise SystemExit(f"{par} not in cache; clip-token models need embeddings")
            emb = np.zeros(meta["clip_dim"], dtype=np.float32)
        parents[par] = (lp, w0, h0, emb)

    out_dir = args.out or os.path.join(G.REPO_ROOT, "runs", "cellclf", "sheets")
    os.makedirs(out_dir, exist_ok=True)
    timings = {}

    for name in names:
        model, cfg = load_any(all_models[name], meta, args.device)
        rows_img = []
        timings[name] = {}
        for par, (lp, w0, h0, emb) in parents.items():
            panels = []
            timings[name][par] = {}
            for W in args.widths:
                gh = raster.grid_height_for_aspect(w0, h0, W, cw, ch, 0)
                with Image.open(lp) as im:
                    im = ImageOps.exif_transpose(im).convert("L")
                    im = im.resize((W * cw, gh * ch), Image.LANCZOS)
                ink = (255 - np.asarray(im, dtype=np.uint8))
                t0 = time.perf_counter()
                if args.ensemble_mode:
                    from unicasso.training.cellclf_ensemble import predict_run
                    pred = predict_run(model, cfg, ink, emb, meta, args.device,
                                       args.ensemble_mode)
                else:
                    pred = predict_grid(model, cfg, ink, emb, meta, args.device)
                t1 = time.perf_counter()
                img = render_grid(sampler, pred)
                t2 = time.perf_counter()
                timings[name][par][W] = {"predict_s": round(t1 - t0, 3),
                                         "render_s": round(t2 - t1, 3),
                                         "cells": int(pred.size)}
                panels.append(caption(to_img(img), f"w={W}  {t1 - t0:.2f}s", font))
            gut = 6
            row_w = sum(p_.width for p_ in panels) + gut * (len(panels) + 1)
            row_h = max(p_.height for p_ in panels) + 24
            row = Image.new("RGB", (row_w, row_h), (255, 255, 255))
            from PIL import ImageDraw
            ImageDraw.Draw(row).text((6, row_h - 18), f"{par}", fill=(0, 0, 0), font=font)
            x = gut
            for p_ in panels:
                row.paste(p_, (x, 0)); x += p_.width + gut
            rows_img.append(row)
            print(f"  {name} {par} done", flush=True)
        W_ = max(r.width for r in rows_img)
        sheet = Image.new("RGB", (W_, sum(r.height + 8 for r in rows_img)), (255, 255, 255))
        y = 0
        for r in rows_img:
            sheet.paste(r, (0, y)); y += r.height + 8
        suffix = f"_ens-{args.ensemble_mode}" if args.ensemble_mode else ""
        path = os.path.join(out_dir, f"widths_{name.replace('@', '_s')}{suffix}.png")
        sheet.save(path)
        print(f"wrote {path}", flush=True)

    tj = f"width_timings{'_ens-' + args.ensemble_mode if args.ensemble_mode else ''}.json"
    with open(os.path.join(out_dir, tj), "w") as f:
        json.dump({"device": args.device, "timings": timings}, f, indent=1)
    print("wrote width_timings.json")


if __name__ == "__main__":
    main()
