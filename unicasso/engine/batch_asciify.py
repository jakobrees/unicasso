"""Batch-asciify every image under a folder, mirroring its structure in the output.

    GLYPHVAE_FONT=sfmono python -m unicasso.engine.batch_asciify /path/to/images \
        --out /path/to/results [--defer "folderA,folderB"] [--dry-run]

Per image, two variants (editable in BASE_ARGS/VARIANTS below):
    <stem>_w60          base-width 60
    <stem>_w55          base-width 55

Each variant writes txt/png/gif/loss(+npz)/pool-record/log next to each other in the
mirrored directory. RESUME-SAFE: a variant whose .txt already exists is skipped, so you
can ctrl-C anytime and re-run to continue. A failed variant is logged and the batch
moves on (summary at the end). --defer takes folder names (path-component match) whose
images are queued LAST.
"""
import argparse
import glob
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from unicasso.substrate import glyphs as G

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- canonical configuration -- edit here, not below -----------------------------
VAE = G.repo_path("weights/vae_dejavu/model.pt")   # default VAE checkpoint
ITERS = "5000"
W_DECAY = "0.02"          # swarm slot-weight decay
JOIN_EVERY = "2"          # join-channel cadence (set 1 to run join every round)
CLIP_CROP = ("0.4", "0.9")

# Everything not listed here comes from the canonical recipe (the asciify defaults,
# see unicasso/engine/canonical_recipe.args). This list is ONLY where the corpus
# batch deliberately departs from it. Verified equivalent to the old fully-explicit
# list by test_batch_args.py -- run that after editing.
BASE_ARGS = [
    "--vae-ckpt", VAE, "--iters", ITERS,
    "--swarm-w-decay", W_DECAY,
    # longer, gentler weight-temperature programme than the single-image recipe
    "--swarm-w-temp-end", "0.03", "--swarm-w-temp-cycles", "3",
    "--swarm-w-temp-cycle-decay", "0.75",
    "--swarm-epilogue-frac", "0.15",
    # the batch keeps the pixel-fit latent nudge; the single-image recipe has it off
    "--pixel-nudge-weight", "0.2", "--pixel-nudge-weight-end", "0.01",
    "--pixel-nudge-weight-schedule", "cosine", "--pixel-nudge-weight-end-frac", "0.6",
    # coordinate prior + join shaping: pinned here rather than taken from the kit
    # profile, so a batch reproduces identically whichever kit it runs under
    "--coord-weight-end", "0.02", "--coord-weight-end-frac", "0.25",
    "--pool-join-coord-sigma", "0.2", "--pool-join-every", JOIN_EVERY,
    # cheaper probing: the batch trades measurement depth for throughput across
    # many images. probe-rate/per-chan are PINNED because the canonical recipe
    # raised them (0 -> 1, 1 -> 2) and inheriting that would silently change
    # every batch render.
    "--pool-probe-rate", "0", "--pool-probe-per-chan", "1",
    "--pool-probe-batches", "3", "--pool-probe-spacing", "11",
    "--pool-diversity-weight", "1e-3", "--pool-diversity-weight-end", "1e-3",
    "--pool-diversity-weight-start-frac", "0.2",
    "--knn-temp-end", "0.08",
    "--clip-dense-weight", "0.3",
    # anti-confetti stack: a slightly harder input white-point than the default 0.9
    "--target-white", "0.93",
    "--anim-interval", "10",
]
VARIANTS = [
    ("w60", ["--base-width", "60", "--clip-crop-scale", *CLIP_CROP]),
    ("w55", ["--base-width", "55", "--clip-crop-scale", *CLIP_CROP]),
]
# -----------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="batch asciify a folder tree")
    ap.add_argument("root", help="folder of images (recursed)")
    ap.add_argument("--out", required=True, help="output root (structure mirrored)")
    ap.add_argument("--defer", default="", help="comma-separated folder names queued LAST")
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings; skip any image whose FILENAME contains "
                         "one (e.g. 'draft,_old')")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="concurrent asciify processes (one GPU handles ~8-12 small runs; "
                         "the .txt resume check makes retries safe)")
    ap.add_argument("--gpus", default="",
                    help="comma list of CUDA device ids to round-robin workers over "
                         "(e.g. '0,1,2,3' on a multi-GPU box); empty = whatever torch picks")
    ap.add_argument("--max-variants-per-image", type=int, default=0,
                    help="0=off. If an image already has >= this many completed variant "
                         "outputs (any *_<stem>_*.txt), skip its not-yet-done variants. Lets "
                         "you add a new variant only to images that lack a full set, without "
                         "re-covering images already done under an earlier variant list.")
    ap.add_argument("--extra-args", default="",
                    help="extra asciify flags appended to every run; USE THE = FORM since "
                         "the value starts with dashes: --extra-args='--clip-batch-aug "
                         "--clip-fp16 --clip-adapter path.pt'")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"ERROR: root does not exist: {args.root}")
    defer = {d.strip() for d in args.defer.split(",") if d.strip()}
    excl = [e.strip() for e in args.exclude.split(",") if e.strip()]
    imgs, n_excl = [], 0
    for dp, dns, fns in os.walk(args.root):
        dns[:] = [d for d in dns if not d.startswith(".")]   # skip .ipynb_checkpoints dupes
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in EXTS:
                if any(e in fn for e in excl):             # e.g. bad-linework images
                    n_excl += 1
                    continue
                p = os.path.join(dp, fn)
                parts = set(os.path.relpath(dp, args.root).split(os.sep))
                imgs.append((bool(parts & defer), p))
    if not imgs:
        sys.exit(f"ERROR: no images found under {args.root} (extensions: {sorted(EXTS)})")
    imgs.sort(key=lambda t: (t[0], t[1]))              # non-deferred first, then path order
    n_def = sum(1 for d, _ in imgs if d)
    ex_tail = f", {n_excl} excluded" if n_excl else ""
    print(f"{len(imgs)} images ({n_def} deferred to the end){ex_tail}; "
          f"{len(imgs) * len(VARIANTS)} runs total")

    jobs, done, skipped = [], 0, 0
    for _, img in imgs:
        rel = os.path.relpath(img, args.root)
        stem = os.path.splitext(rel)[0]
        have = len(glob.glob(os.path.join(args.out, glob.escape(stem) + "_*.txt")))
        enough = args.max_variants_per_image and have >= args.max_variants_per_image
        for vname, vargs in VARIANTS:
            base = os.path.join(args.out, f"{stem}_{vname}")
            if os.path.exists(base + ".txt"):
                done += 1
                continue
            if enough:                                    # already has a full variant set
                skipped += 1
                continue
            jobs.append((img, base, vargs))
    tail = f", {skipped} skipped (>= {args.max_variants_per_image} variants)" if skipped else ""
    print(f"{done} already done (resume), {len(jobs)} to run{tail}")
    if args.dry_run:
        for img, base, _ in jobs:
            print(f"  {img} -> {base}.*")
        return

    extra = args.extra_args.split() if args.extra_args else []
    if args.jobs > 1 and __import__("shutil").which("nvidia-smi"):
        mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
        if os.path.isdir(mps_pipe):
            os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", mps_pipe)  # children inherit
        else:
            print("=" * 78)
            print("WARNING: multiple CUDA jobs WITHOUT the CUDA MPS daemon.")
            print("Separate processes time-slice the GPU (looks 100% busy, isn't).")
            print("Measured on an H100: MPS gives ~45% more aggregate throughput.")
            print("  nvidia-cuda-mps-control -d")
            print("  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps")
            print("Budget ~8-9 GB VRAM per job. See docs/adapter.md.")
            print("=" * 78)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    failures, done_n, lock = [], [0], threading.Lock()

    def run_one(idx_job):
        i, (img, base, vargs) = idx_job
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        cmd = [sys.executable, "-m", "unicasso.engine.asciify", img, *BASE_ARGS, *vargs, *extra,
               "--output-text", base + ".txt", "--output", base + ".png",
               "--anim", base + ".gif", "--loss-curve", base + "_loss.png",
               "--pool-record", base + "_pool.npz"]
        env = dict(os.environ)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
        with open(base + ".log", "w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        with lock:
            done_n[0] += 1
            tag = "ok" if r.returncode == 0 else f"FAILED exit {r.returncode}"
            print(f"[{done_n[0]}/{len(jobs)}] {tag}  {base}", flush=True)
            if r.returncode != 0:
                failures.append(base)

    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as ex:
        list(ex.map(run_one, enumerate(jobs)))
    print(f"\nbatch done: {len(jobs) - len(failures)} ok, {len(failures)} failed")
    for f_ in failures:
        print("  failed:", f_)


if __name__ == "__main__":
    main()
