"""Cell-level dataset cache for the per-cell glyph classifier.

python -m unicasso.training.cell_data \
    --txt-root data/dataset_v1/runs/txts --img-root data/dataset_v1/lineart \
    --out runs/cellclf/cache

For every run (parent x budget variant) this writes one <stem>.npz:
    ink     (GH*CH, GW*CW) uint8   line art resized to the exact grid pixel size,
                                   LANCZOS (the engine's load_target_image convention),
                                   stored as ink*255 (0 = white paper, 255 = full ink)
    labels  (GH, GW)        int16  glyph indices into the FULL curated charset
                                   (txt -> char_to_idx; lossless, pre-ban order)
    greedy  (GH, GW)        int16  nearest-glyph baseline, snap.py convention:
                                   constant-white pad -> VAE encode -> cdist argmin
    clip    (D,)            float32 global CLIP image embedding of the parent line art
                                   (unit-normalized)
plus a single meta.json with the charset, cell geometry, CLIP model id and the
image-level train/val split (variants of one parent never straddle the split).
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps

from unicasso.adapter.clip_adapt import find_pairs
from unicasso.adapter.corrupt import CorruptionSampler
from unicasso.substrate import glyphs as G


def load_ink_image(path, gw, gh, cw, ch):
    """Line art -> (GH*CH, GW*CW) float32 ink (ink=1, paper=0), engine resize convention."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        im = im.resize((gw * cw, gh * ch), Image.LANCZOS)
    return 1.0 - np.asarray(im, dtype=np.float32) / 255.0


@torch.no_grad()
def greedy_snap(ink_img, vae, codebook, cell_hw, pad_hw, device, chunk=4096):
    """snap.py's nearest-glyph baseline: constant-white pad, VAE encode, cdist argmin."""
    ch, cw = cell_hw
    gh, gw = ink_img.shape[0] // ch, ink_img.shape[1] // cw
    t = torch.from_numpy(ink_img).to(device)
    cells = t.view(gh, ch, gw, cw).permute(0, 2, 1, 3).reshape(gh * gw, 1, ch, cw)
    cells = torch.nn.functional.pad(cells, (pad_hw[1], pad_hw[1], pad_hw[0], pad_hw[0]), value=0.0)
    out = torch.empty(gh * gw, dtype=torch.long)
    for i in range(0, cells.shape[0], chunk):
        z, _ = vae.encode(cells[i:i + chunk])
        out[i:i + chunk] = torch.cdist(z, codebook).argmin(dim=1).cpu()
    return out.view(gh, gw).numpy().astype(np.int16)


@torch.no_grad()
def clip_embed(model, preprocess, path, device):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        x = preprocess(im).unsqueeze(0).to(device)
    e = model.encode_image(x).float()
    e = e / e.norm(dim=-1, keepdim=True)
    return e[0].cpu().numpy().astype(np.float32)


def build(args):
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # CPU on purpose: CorruptionSampler's init mixes device-less tensors with its glyphs
    # (crashes on mps), and we only need its charset map / txt loader / codebook here.
    sampler = CorruptionSampler(G.repo_path(args.vae_ckpt), device="cpu", profile=args.profile)
    codebook = sampler.codebook.to(device)
    # the sampler discards the VAE after building its codebook; greedy needs the encoder
    from unicasso.engine.asciify import load_vae
    vae, _, _, pad_hw = load_vae(G.repo_path(args.vae_ckpt), device)

    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained)
    clip_model = clip_model.to(device).eval()

    pairs = find_pairs(G.repo_path(args.txt_root), G.repo_path(args.img_root))
    if not pairs:
        raise SystemExit("no txt/image pairs found")
    print(f"{len(pairs)} parents, {sum(len(p['txts']) for p in pairs)} runs")

    # image-level split, variants never straddle
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pairs))
    n_hold = max(1, int(round(len(pairs) * args.holdout)))
    hold_names = {pairs[i]["name"] for i in order[:n_hold]}

    ch, cw = sampler.CH, sampler.CW
    runs = []
    for pi, pair in enumerate(pairs):
        emb = clip_embed(clip_model, preprocess, pair["img"], device)
        for txt in pair["txts"]:
            stem = os.path.splitext(os.path.basename(txt))[0]
            out_npz = os.path.join(args.out, stem + ".npz")
            grid = sampler.load_txt(txt).cpu().numpy().astype(np.int16)
            gh, gw = grid.shape
            ink = load_ink_image(pair["img"], gw, gh, cw, ch)
            greedy = greedy_snap(ink, vae, codebook, (ch, cw), pad_hw, device)
            np.savez_compressed(out_npz,
                                ink=np.clip(ink * 255.0, 0, 255).astype(np.uint8),
                                labels=grid, greedy=greedy, clip=emb)
            runs.append(dict(stem=stem, parent=pair["name"], gh=int(gh), gw=int(gw),
                             split="val" if pair["name"] in hold_names else "train"))
        if (pi + 1) % 25 == 0:
            print(f"  {pi + 1}/{len(pairs)} parents")

    chars = sampler.chars
    meta = dict(chars=chars,
                chars_sha1=hashlib.sha1(chars.encode()).hexdigest(),
                cell_h=int(ch), cell_w=int(cw),
                pad_h=int(pad_hw[0]), pad_w=int(pad_hw[1]),
                clip_model=args.clip_model, clip_pretrained=args.clip_pretrained,
                clip_dim=int(len(runs) and np.load(
                    os.path.join(args.out, runs[0]["stem"] + ".npz"))["clip"].shape[0]),
                holdout=args.holdout, seed=args.seed, runs=runs)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    n_tr = sum(r["split"] == "train" for r in runs)
    print(f"cached {len(runs)} runs ({n_tr} train / {len(runs) - n_tr} val) -> {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--txt-root", default="data/dataset_v1/runs/txts")
    p.add_argument("--img-root", default="data/dataset_v1/lineart")
    p.add_argument("--out", default="runs/cellclf/cache")
    p.add_argument("--vae-ckpt", default="weights/vae_dejavu/model.pt")
    p.add_argument("--profile", default="dejavu")
    p.add_argument("--clip-model", default="RN101")
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--holdout", type=float, default=0.15, help="fraction of PARENTS held out for val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    build(p.parse_args())


if __name__ == "__main__":
    main()
