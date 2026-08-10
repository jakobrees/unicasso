"""Photo -> line-art via a LOCAL FLUX.2 Klein (open-weights) diffusers pipeline.

    python -m unicasso.lineart.klein_lineart <photo-root> --out <line-root> \
        [--model-id black-forest-labs/FLUX.2-klein-4B] [--steps 8] [--dry-run]

Local counterpart to bfl_lineart.py (which calls the paid BFL API). Same photo
tree walk, same mirrored "<stem>_line.png" output naming, same resume-safety, so
its outputs drop straight into the corpus alongside the API-generated ones. Uses
the saved instruction prompt (unicasso/lineart/prompts/flux_line_prompt.txt) by default.

Runs on whatever torch finds: CUDA > MPS > CPU. Sized for an Apple-silicon
laptop: Klein-4B loads ~16 GB in bf16 (7.8 GB transformer + 8.1 GB Qwen3-4B text
encoder + VAE), which fits a 36 GB M-series with room for ~1 MP activations.
First run downloads those ~16 GB. Use --offload only on CUDA (MPS can't offload).
Skip the fp8/nvfp4 repo variants -- those are NVIDIA-only quant formats, useless
on MPS; the plain bf16 FLUX.2-klein-4B is the Mac target.

NOTE ON THE MODEL CALL: the pipeline call is NOT verified against a stable
diffusers release. model_index.json declares Flux2KleinPipeline, which
DiffusionPipeline auto-resolves -- but it needs a diffusers new enough to know
that class (possibly a git build). The exact editing signature (image= vs a
reference-image list) may
differ; if a run errors in the call, generate() is the ONE spot to adjust. The
batch harness around it is generic, and --dry-run needs neither diffusers nor
the weights.
"""
import argparse
import os
import time

from PIL import Image, ImageFilter, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPT_DEFAULT = os.path.join(os.path.dirname(__file__), "prompts", "flux_line_prompt.txt")
MODEL_DEFAULT = "black-forest-labs/FLUX.2-klein-4B"   # bf16; declares Flux2KleinPipeline


def collect(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in EXTS:
                out.append(os.path.join(dp, fn))
    return out


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def _patch_mps_scatter():
    """MPS fix: Flux2KleinPipeline._unpack_latents_with_ids downgrades its scatter
    index to int32 (via maybe_adjust_dtype_for_device), but torch's MPS scatter_
    demands int64 -> 'Expected dtype int64 for index'. Reinstate int64. No-op fix
    on CUDA/CPU (int64 is correct everywhere), so applied unconditionally."""
    import torch
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    def _unpack(x, x_ids, height=None, width=None):
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids, w_ids = pos[:, 1].to(torch.int64), pos[:, 2].to(torch.int64)
            h = height if height is not None else int(torch.max(h_ids)) + 1
            w = width if width is not None else int(torch.max(w_ids)) + 1
            flat_ids = h_ids * w + w_ids
            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
            x_list.append(out.view(h, w, ch).permute(2, 0, 1))
        return torch.stack(x_list, dim=0)

    Flux2KleinPipeline._unpack_latents_with_ids = staticmethod(_unpack)


def load_pipe(model_id, offload):
    """Load the FLUX.2 Klein editing pipeline. Isolated so it's the only spot to
    touch if the installed diffusers wants a different class/signature."""
    import torch
    from diffusers import DiffusionPipeline

    device, dtype = pick_device()
    if device == "mps":
        _patch_mps_scatter()
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if offload and device == "cuda":
        pipe.enable_model_cpu_offload()               # keeps only the active submodule on GPU
    else:
        pipe = pipe.to(device)
    print(f"[klein] loaded {model_id} on {device} ({dtype})")
    return pipe, device


def prep_image(img_path, max_side, pre_blur=0.0):
    im = Image.open(img_path)
    im = ImageOps.exif_transpose(im).convert("RGB")   # phone JPEGs: rotation lives in EXIF
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    if pre_blur > 0:                                   # kill fine texture; big contours survive
        im = im.filter(ImageFilter.GaussianBlur(radius=pre_blur))
    w, h = (im.width // 32) * 32, (im.height // 32) * 32   # FLUX wants multiples of 32
    return im.resize((max(w, 64), max(h, 64)), Image.LANCZOS)


def generate(pipe, device, image, prompt, steps, guidance, seed):
    """Run one edit. Kwargs kept in a dict so unsupported ones are easy to drop."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = dict(prompt=prompt, image=image, num_inference_steps=steps,
                  guidance_scale=guidance, generator=gen)
    out = pipe(**kwargs)
    return out.images[0]


def main():
    ap = argparse.ArgumentParser(description="local FLUX.2 Klein photo -> lineart")
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default=MODEL_DEFAULT)
    ap.add_argument("--prompt-file", default=PROMPT_DEFAULT)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--pre-blur", type=float, default=0.0,
                    help="Gaussian blur radius (px) on the input before the model sees it. "
                         "The 'artistic freedom' knob: 0=trace every detail, 2-6=drop small "
                         "photo detail and keep only big contours. Also try a lower --max-side.")
    ap.add_argument("--steps", type=int, default=8,
                    help="Klein is step-distilled; 4-8 is the sweet spot, more just costs time")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="ignored: Klein is guidance-distilled (diffusers warns and drops CFG)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offload", action="store_true", help="CPU-offload submodules (OOM relief)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read().strip()

    imgs = collect(args.root)
    jobs = []
    for p in imgs:
        rel = os.path.splitext(os.path.relpath(p, args.root))[0] + "_line.png"
        o = os.path.join(args.out, rel)
        if not os.path.exists(o):                     # resume: skip finished
            jobs.append((p, o))
    print(f"{len(imgs)} photos, {len(imgs) - len(jobs)} already converted, {len(jobs)} to run")
    if args.dry_run:
        for p, o in jobs[:8]:
            print(f"  {p} -> {o}")
        return
    if not jobs:
        return

    pipe, device = load_pipe(args.model_id, args.offload)
    done, failures = 0, []
    for p, o in jobs:
        done += 1
        t0 = time.time()
        try:
            img = prep_image(p, args.max_side, args.pre_blur)
            res = generate(pipe, device, img, prompt, args.steps, args.guidance, args.seed)
            os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
            res.save(o)                               # write only on success (resume-safe)
            print(f"[{done}/{len(jobs)}] ok {time.time() - t0:.1f}s  {os.path.basename(p)}",
                  flush=True)
        except Exception as e:                        # OOM, bad image, API mismatch
            failures.append((p, repr(e)))
            print(f"[{done}/{len(jobs)}] FAIL {os.path.basename(p)}: {e!r}", flush=True)

    print(f"\ndone: {len(jobs) - len(failures)} ok, {len(failures)} failed")
    for p, m in failures:
        print("  failed:", p, "--", m)


if __name__ == "__main__":
    main()
