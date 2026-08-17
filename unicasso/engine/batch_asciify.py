"""Batch-asciify every image under a folder, mirroring its structure in the output.

    GLYPHVAE_FONT=sfmono python -m unicasso.engine.batch_asciify /path/to/images \
        --out /path/to/results [--defer "folderA,folderB"] [--dry-run]

Per image, two variants (editable in BASE_ARGS/VARIANTS below), sized by GLYPH BUDGET
rather than column count so the grid is comparable across aspect ratios:
    <stem>_b2000        ~2000 cells
    <stem>_b980         ~980 cells

Each variant writes txt/png/loss(+npz)/pool-record/log -- beside each other by default,
or under one subdirectory per artifact type with --layout split. A GIF is written only
with --anim. RESUME-SAFE: a variant whose .txt already exists is skipped, so you
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
import time
from concurrent.futures import ThreadPoolExecutor

from unicasso.substrate import glyphs as G

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- canonical configuration -- edit here, not below -----------------------------
VAE = G.repo_path("weights/vae_dejavu/model.pt")   # default VAE checkpoint

# The corpus renders the CANONICAL RECIPE: everything not listed here comes from the
# asciify defaults (unicasso/engine/canonical_recipe.args) and the active kit profile.
# This list is ONLY the batch's own additions -- corpus-scale concerns that have no
# meaning for a single render. Anything else added here is a departure from the shipped
# configuration and makes the dataset document something other than what users get.
BASE_ARGS = [
    "--vae-ckpt", VAE,
    # per-slot trajectories at the default 25-iter resolution run ~26 MB/run; 100 is
    # ample for aggregate churn statistics, and the handful of runs that feed a
    # trajectory figure get their own high-resolution re-run
    "--pool-record-interval", "100",
]
# Grid sizing by TOTAL CELLS rather than column count. Fixed columns hold a display
# convention constant and let the actual resource swing with aspect: at 60 columns a 3:4
# portrait gets 2280 cells and a 16:9 frame 960 -- a 2.4x gap under one label, on a corpus
# that is ~59% landscape. Two tiers, ~2:1 in glyphs, so a piece is re-resolved rather than
# downsampled between them.
BUDGET_LARGE, BUDGET_SMALL = "2000", "980"
VARIANTS = [
    ("b2000", ["--glyph-budget", BUDGET_LARGE]),
    ("b980", ["--glyph-budget", BUDGET_SMALL]),
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
    ap.add_argument("--layout", choices=["flat", "split"], default="flat",
                    help="flat = every artifact beside its siblings under --out (the "
                         "original layout). split = one subdirectory per artifact type "
                         "(txts/ renders/ curves/ records/ logs/), which keeps the raw "
                         "data separable from the renders and lets the adapter read "
                         "--txt-root <out>/txts directly")
    ap.add_argument("--strip-suffix", default="",
                    help="drop this suffix from each input stem when naming outputs, e.g. "
                         "'_line' turns 00001_line.png into 00001_b2000.txt. Keeps the output "
                         "id identical to the corpus manifest id, so a grid joins to its "
                         "provenance without string surgery; the adapter still locates the "
                         "parent image, since it already tries both <stem> and <stem>_line")
    ap.add_argument("--anim", action="store_true",
                    help="also write an optimisation GIF per run. OFF by default for "
                         "batches: frames are held in memory until the run ends, so at "
                         "--anim-interval 10 over a few thousand iters that is hundreds "
                         "of MB per concurrent job")
    ap.add_argument("--stagger", type=float, default=8.0,
                    help="MINIMUM seconds between ANY two worker cold-starts, enforced for the "
                         "whole run (not just the opening wave). Each cold start loads "
                         "CLIP+DINO and builds affinity -- a memory spike that trips the MPS "
                         "OOM reaper if several coincide. A global rate-limiter, because "
                         "runtimes decorrelate over time and finish-clustering would otherwise "
                         "restart the herd mid-run. Launches already spread wider than this pay "
                         "nothing. 0 = no spacing")
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

    # artifact-type -> directory. In split layout the raw data (grids, records, curve
    # series) lives apart from the renders, so a figure can be re-made from the run
    # without re-running it, and the adapter reads <out>/txts straight away.
    def art(kind, stem, vname, ext):
        d = os.path.join(args.out, kind) if args.layout == "split" else args.out
        return os.path.join(d, f"{stem}_{vname}{ext}")

    jobs, done, skipped = [], 0, 0
    for _, img in imgs:
        rel = os.path.relpath(img, args.root)
        stem = os.path.splitext(rel)[0]
        if args.strip_suffix and stem.endswith(args.strip_suffix):
            stem = stem[:-len(args.strip_suffix)]
        tdir = os.path.join(args.out, "txts") if args.layout == "split" else args.out
        have = len(glob.glob(os.path.join(tdir, glob.escape(stem) + "_*.txt")))
        enough = args.max_variants_per_image and have >= args.max_variants_per_image
        for vname, vargs in VARIANTS:
            if os.path.exists(art("txts", stem, vname, ".txt")):
                done += 1
                continue
            if enough:                                    # already has a full variant set
                skipped += 1
                continue
            jobs.append((img, stem, vname, vargs))
    tail = f", {skipped} skipped (>= {args.max_variants_per_image} variants)" if skipped else ""
    print(f"{done} already done (resume), {len(jobs)} to run{tail}")
    if args.dry_run:
        for img, stem, vname, _ in jobs:
            print(f"  {img} -> {art('txts', stem, vname, '.txt')} (+ render/curve/record/log)")
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
    launch_lock = threading.Lock()
    next_slot = [0.0]   # monotonic time the next cold-start may begin (global spacing)
    # Live workers per GPU. A static job_index % n_gpus assignment is balanced by COUNT but
    # not over TIME: the two variants (b2000 heavy, b980 light) finish at different rates, so
    # the live split drifts (observed 6/10 on 2 GPUs) and heavy jobs pile onto the fuller card
    # until it OOMs. Assigning each starting worker to the LEAST-loaded GPU instead keeps the
    # split within 1 of even at all times -- exactly n/2 per GPU at saturation.
    gpu_load = [0] * len(gpus) if gpus else []

    def run_one(idx_job):
        i, (img, stem, vname, vargs) = idx_job
        if args.stagger > 0:                     # global min-spacing between cold-starts
            with launch_lock:
                now = time.monotonic()
                slot = max(now, next_slot[0])        # reserve the next free launch slot
                next_slot[0] = slot + args.stagger
            delay = slot - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        out = {k: art(k, stem, vname, e) for k, e in
               (("txts", ".txt"), ("renders", ".png"), ("curves", "_loss.png"),
                ("records", "_pool.npz"), ("logs", ".log"), ("anims", ".gif"))}
        for p in out.values():
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        cmd = [sys.executable, "-m", "unicasso.engine.asciify", img, *BASE_ARGS, *vargs, *extra,
               "--output-text", out["txts"], "--output", out["renders"],
               "--loss-curve", out["curves"], "--pool-record", out["records"]]
        if args.anim:
            cmd += ["--anim", out["anims"]]
        env = dict(os.environ)
        gsel = None
        if gpus:
            # claim the least-loaded GPU for this worker's lifetime (ties -> lowest index).
            # Chosen AFTER the stagger delay, so it reflects the true live load at launch.
            with launch_lock:
                gsel = min(range(len(gpus)), key=lambda g: gpu_load[g])
                gpu_load[gsel] += 1
            env["CUDA_VISIBLE_DEVICES"] = gpus[gsel]
        try:
            with open(out["logs"], "w") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        finally:
            if gsel is not None:                 # release the GPU slot even if the run raised
                with launch_lock:
                    gpu_load[gsel] -= 1
        with lock:
            done_n[0] += 1
            tag = "ok" if r.returncode == 0 else f"FAILED exit {r.returncode}"
            print(f"[{done_n[0]}/{len(jobs)}] {tag}  {stem}_{vname}", flush=True)
            if r.returncode != 0:
                failures.append(f"{stem}_{vname}")

    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as ex:
        list(ex.map(run_one, enumerate(jobs)))
    print(f"\nbatch done: {len(jobs) - len(failures)} ok, {len(failures)} failed")
    for f_ in failures:
        print("  failed:", f_)


if __name__ == "__main__":
    main()
