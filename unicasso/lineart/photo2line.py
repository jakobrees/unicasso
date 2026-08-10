"""Quick single-photo -> line-art via LOCAL FLUX.2 Klein 4B (open weights).

    python -m unicasso.lineart.photo2line <photo> [-o out.png] [--steps 28] [--guidance 4.0]

A one-shot companion to klein_lineart.py (the batch tree walker). Same model,
same device pick, same image prep, and the same saved instruction prompt
(unicasso/lineart/prompts/flux_line_prompt.txt). Defaults the output to "<stem>_line.png" next to
the input. First run downloads ~16 GB (bf16 transformer + Qwen3-4B text encoder
+ VAE); afterwards it's cached. CUDA > MPS > CPU, auto-picked.
"""
import argparse
import os
import time

from PIL import Image, ImageOps

PROMPT_DEFAULT = os.path.join(os.path.dirname(__file__), "prompts", "flux_line_prompt.txt")
MODEL_DEFAULT = "black-forest-labs/FLUX.2-klein-4B"   # bf16; declares Flux2KleinPipeline


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def prep_image(img_path, max_side):
    im = Image.open(img_path)
    im = ImageOps.exif_transpose(im).convert("RGB")   # phone JPEGs: rotation lives in EXIF
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    w, h = (im.width // 64) * 64, (im.height // 64) * 64   # FLUX wants multiples of 32 (klein wants 64)
    return im.resize((max(w, 64), max(h, 64)), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description="single photo -> lineart (local FLUX.2 Klein)")
    ap.add_argument("photo")
    ap.add_argument("-o", "--out", help="output PNG (default: <stem>_line.png beside input)")
    ap.add_argument("--model-id", default=MODEL_DEFAULT)
    ap.add_argument("--prompt-file", default=PROMPT_DEFAULT)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8,
                    help="inference steps. Klein is step-distilled: 4-8 usually suffices "
                         "(klein_lineart's default); the default here is conservative and "
                         "extra steps just cost time")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="CFG scale, passed through to the pipeline. Ignored by the default "
                         "Klein model (guidance-distilled: diffusers warns and drops CFG); "
                         "only matters for a non-distilled --model-id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offload", action="store_true", help="CPU-offload submodules (OOM relief, CUDA only)")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.photo)[0] + "_line.png"
    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read().strip()

    import torch
    from diffusers import DiffusionPipeline

    device, dtype = pick_device()
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    print(f"[photo2line] loaded {args.model_id} on {device} ({dtype})")

    img = prep_image(args.photo, args.max_side)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    res = pipe(prompt=prompt, image=img, num_inference_steps=args.steps,
               guidance_scale=args.guidance, generator=gen).images[0]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    res.save(out)
    print(f"[photo2line] {time.time() - t0:.1f}s  {args.photo} -> {out}")


if __name__ == "__main__":
    main()
