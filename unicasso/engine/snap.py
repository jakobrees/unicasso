"""Quick VAE encode -> snap -> ASCII. No optimization, just the per-cell appearance retrieval
(the same retrieval the optimizer uses as its warm-start init).

For each image: split into char cells, encode every cell patch with the VAE, snap each to the
nearest real glyph by latent distance, assemble. Takes a single image OR a folder.

Examples:
  python -m unicasso.engine.snap path/to/img.jpg  --vae-ckpt weights/vae_sfmono/model.pt
  python -m unicasso.engine.snap path/to/folder/  --vae-ckpt ... --base-width 80 --out-dir snap_out
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unicasso.substrate import glyphs as G
from unicasso.engine.asciify import load_vae, assemble
from unicasso.substrate import raster as train


IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")


def list_images(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(IMG_EXT))
    return [path]


@torch.no_grad()
def snap_image(path, model, codebook, char_bitmaps, chars, pad, base_width, out_dir):
    CH, CW = train.CHAR_HEIGHT, train.CHAR_WIDTH
    img_w, img_h = train.native_size(path)
    GW = base_width
    GH = train.grid_height_for_aspect(img_w, img_h, GW, CW, CH, 0)
    train.GRID_WIDTH, train.GRID_HEIGHT = GW, GH
    train.IMAGE_WIDTH, train.IMAGE_HEIGHT = CW * GW, CH * GH

    target = train.load_target_image(path)                       # (IMG_H, IMG_W) white=1
    # per-cell patches -> ink, padded to the VAE field
    cells = target.view(GH, CH, GW, CW).permute(0, 2, 1, 3).reshape(GH * GW, CH, CW)
    cell_ink = F.pad((1.0 - cells).unsqueeze(1), (pad[1], pad[1], pad[0], pad[0]), value=0.0)
    z, _ = model.encode(cell_ink)
    idx = torch.cdist(z, codebook).argmin(dim=1)                 # nearest glyph per cell

    render = assemble(char_bitmaps[idx].view(1, GH, GW, CH, CW), GH, GW, CH, CW, 0,
                      CH * GH, CW * GW)[0]
    arr = (render.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, stem + ".png")
    txt = os.path.join(out_dir, stem + ".txt")
    Image.fromarray(arr, mode="L").save(png)
    grid = idx.view(GH, GW).cpu()
    with open(txt, "w", encoding="utf-8") as f:
        for i in range(GH):
            f.write("".join(chars[grid[i, j].item()] for j in range(GW)) + "\n")
    print(f"  {os.path.basename(path):40s} {GW}x{GH} -> {png}")


def main():
    p = argparse.ArgumentParser(description="Quick VAE encode->snap->ASCII (no optimization)")
    p.add_argument("input", help="image file or folder of images")
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--base-width", type=int, default=60, help="grid width in chars (auto height)")
    p.add_argument("--out-dir", default="snap_out")
    p.add_argument("--ban-chars", type=str, default="", help="exclude these chars from snapping")
    p.add_argument("--ban-blocks", action="store_true", help="also exclude block chars ░▒▓█▄▌▐▀■")
    args = p.parse_args()

    device = train.DEVICE
    model, vae_chars, L, pad = load_vae(args.vae_ckpt, device)
    ink, chars = G.load_glyphs(device=device, pad=pad)
    if chars != vae_chars:
        raise ValueError("charset mismatch between glyph render and the VAE checkpoint")
    train.ROW_GAP = 0
    char_bitmaps = train.create_char_bitmaps().to(device)        # (N,24,12) white=1
    with torch.no_grad():
        codebook, _ = model.encode(ink)

    banned = set(args.ban_chars) | (set("░▒▓█▄▌▐▀■") if args.ban_blocks else set())
    if banned:
        keep = [i for i, c in enumerate(chars) if c not in banned]
        keep_t = torch.tensor(keep, device=device)
        chars = "".join(chars[i] for i in keep)
        char_bitmaps = char_bitmaps[keep_t]
        codebook = codebook[keep_t]
        print(f"Banned {len(banned)} char(s); snapping over {len(chars)} glyphs")

    images = list_images(args.input)
    if not images:
        raise SystemExit(f"no images found at {args.input}")
    print(f"Snapping {len(images)} image(s) -> {args.out_dir}/")
    for path in images:
        snap_image(path, model, codebook, char_bitmaps, chars, pad, args.base_width, args.out_dir)


if __name__ == "__main__":
    main()
