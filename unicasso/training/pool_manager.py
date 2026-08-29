"""Rolling refinement pool: a live teacher that regenerates itself during training.

The pool holds engine-refined glyph grids -- targets produced by running the
swarm optimizer over a photo, warm-started from the CURRENT model's grid. The
colours are either closed-form (`color='fit'`: the campaign's colour pool until
--pool-color-switch, and the lineart pool always) or the current model's own
(`'lite'`, --color-lite) -- in which case the grids that come back are reachable
rather than aspirational: the engine cannot buy CLIP score with a palette the
model would never assign.

Rolling, not phased. There are no E/M rounds -- the trainer runs continuously and
calls `refresh()` every N steps, which adds a batch of freshly refined entries
and evicts the oldest so the pool tracks the improving model without ever going
stale wholesale. Entry width is drawn at random per refinement, so one photo can
appear at several scales over the run.

MEMORY CHOREOGRAPHY. Training wants prepped entries resident (decompose, window
stacks, features -- ~110 MB per w60 entry); refinement wants the GPU for eight
concurrent engine jobs. They cannot both have it. `refresh()` therefore evicts
the prep cache and empties the allocator BEFORE spawning workers, and the cache
refills lazily afterwards. On a CUDA box arm the MPS daemon first
(`nvidia-cuda-mps-control -d`; NOT reboot-persistent) so the workers share the
card efficiently.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps

Image.MAX_IMAGE_PIXELS = None      # the photo set has >100 MP scans;
# PIL's bomb guard would warn on every sample and eventually refuse them
ImageFile.LOAD_TRUNCATED_IMAGES = True   # a few files are short a scanline;
# decode what is there rather than killing an unattended run

from unicasso.substrate import glyphs as G


def photo_groups(spec, seed=0):
    """Parse "dir:weight,dir:weight" (weight optional, default 1) into
    [(name, [files], weight)] with each group's file list SHUFFLED.

    Two reasons this exists rather than one flat list. (1) The sources differ in
    quality and should be mixed on purpose, not in whatever proportion the
    directories happen to hold. (2) The pool round-robins, and over a SORTED flat
    list that means it walks one directory to exhaustion before touching the
    next -- in joint01 the pool never contained a single dataset_v1 photo,
    because the first one sits at index 516 of 763 and the run only consumed 256.
    """
    groups = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        d, _, w = part.partition(":")
        files = sorted(sum((glob.glob(os.path.join(G.repo_path(d), e))
                            for e in ("*.jpg", "*.jpeg", "*.png",
                                      "*.JPG", "*.JPEG", "*.PNG")), []))
        if not files:
            raise ValueError(f"--photos: no images in {d}")
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        groups.append((os.path.basename(d.rstrip("/")), files,
                       float(w) if w else 1.0))
    tot = sum(g[2] for g in groups)
    return [(n, f, w / tot) for n, f, w in groups]


class RollingPool:
    def __init__(self, pool_dir, photos, ckpt_dir, widths=(40, 60, 80), size=96,
                 workers=8, gpus="", font="sfmono", iters=200, lead=1.0,
                 w_temp=0.66, z_noise=0.49, read=None, extra_args=(),
                 prep_max=40, seed=0, device=None, color="lite",
                 cursor_seed=None):
        self.dir = G.repo_path(pool_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.ckpt_dir = G.repo_path(ckpt_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # photos: [(name, [files], share)] from photo_groups -- per-source
        # cursors so every refresh draws the intended MIX, not whichever
        # directory the flat sort happened to reach
        self.groups = list(photos)
        if not self.groups:
            raise ValueError("RollingPool: no photos")
        # Where each source's cursor STARTS. Decoupled from `seed` on purpose:
        # `seed` fixes the shuffle, and therefore which photos are held out as
        # validators (they are taken off the end of each list), so changing it
        # to get fresh pool images would silently change the validation set and
        # break comparability with every previous run. A separate cursor seed
        # moves where the pool begins walking without touching either.
        if cursor_seed is None:
            self.cursors = [0] * len(self.groups)
        else:
            cr = np.random.default_rng(cursor_seed)
            self.cursors = [int(cr.integers(len(g[1]))) for g in self.groups]
        self.n_photos = sum(len(g[1]) for g in self.groups)
        self.widths = list(widths)
        self.size, self.workers, self.gpus = size, workers, gpus
        self.font, self.iters, self.lead = font, iters, lead
        self.w_temp, self.z_noise = w_temp, z_noise
        self.read = dict(path="topk", frac=0.6, count="argmax", ridge=0.0,
                         temp=1.0, mode="birth", ink="render")
        self.read.update(read or {})
        self.color = color
        self.extra = list(extra_args)
        self.prep_max = prep_max
        self.rng = np.random.default_rng(seed)
        self.device = device
        self._ci = None                      # charset map, set on first refresh
        self.born = self._max_born()
        self._prep = {}                      # path -> prepped dict (LRU-ish)
        self.n_refresh = self.n_added = self.n_evicted = 0

    # -------------------------------------------------------------- inventory
    def _max_born(self):
        b = 0
        for f in glob.glob(os.path.join(self.dir, "*.npz")):
            try:
                b = max(b, int(np.load(f)["born"]))
            except Exception:
                pass
        return b

    def entries(self):
        return sorted(glob.glob(os.path.join(self.dir, "*.npz")))

    def __len__(self):
        return len(self.entries())

    # ------------------------------------------------------------ prep cache
    def evict_prep(self):
        """Drop every prepped entry and hand the memory back. Called before a
        refresh so the worker fleet has the card to itself."""
        n = len(self._prep)
        self._prep.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return n

    def prep(self, path, ch, cw, ph, pw, rows=3, cols=5):
        """Everything a training step needs from one entry, cached resident.
        Windows are kept as uint8 (the sources are uint8) -- as float32 a w60
        entry's RGB windows alone would be ~1 GB."""
        ent = self._prep.get(path)
        if ent is not None and ent["mtime"] == os.stat(path).st_mtime:
            return ent
        from unicasso.engine.color import decompose, nomination_target
        from unicasso.training.cellclf_color_train import (cell_feats,
                                                           grid_windows,
                                                           token_maps)
        z = np.load(path, allow_pickle=True)
        g = torch.from_numpy(np.asarray(z["glyphs"]).astype(np.int64))
        gh, gw = g.shape
        photo = str(z["photo"])
        with Image.open(photo) as im:
            im = ImageOps.exif_transpose(im).convert("RGB") \
                .resize((gw * cw, gh * ch), Image.LANCZOS)
        rgb = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
        # decompose on the accelerator (~20x): a w60 entry is ~2400 cells and it
        # is the whole cost of a prep. Results come back to host, because that is
        # where the prep cache lives and ce_pool indexes them with CPU indices.
        dec = decompose(rgb.to(self.device) if self.device else rgb,
                        gh, gw, ch, cw)
        ink_u8 = np.clip((1.0 - nomination_target(dec).cpu().numpy()) * 255.0,
                         0, 255).astype(np.uint8)
        dec = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in dec.items()}
        ids, valid = token_maps(gh, gw, rows, cols)
        ent = dict(mtime=os.stat(path).st_mtime, glyphs=g.reshape(-1),
                   gh=gh, gw=gw, rgb=rgb, feats=cell_feats(dec),
                   cell_rgb=dec["cell_rgb"],
                   win_ink=grid_windows(ink_u8, gh, gw, ch, cw, ph, pw),
                   win_rgb=grid_windows(np.asarray(im, np.uint8), gh, gw, ch,
                                        cw, ph, pw, edge=True),
                   ids=torch.from_numpy(ids), valid=torch.from_numpy(valid))
        if len(self._prep) >= self.prep_max:
            self._prep.pop(next(iter(self._prep)))
        self._prep[path] = ent
        return ent

    # --------------------------------------------------------------- refresh
    def refresh(self, model, chars, cfg, n_add, log=print):
        """Add `n_add` freshly refined entries, then evict back to `size`.

        `model` is the LIVE trainer model -- it is checkpointed to disk for the
        lite warm-start grids and, when `color == 'lite'`, for the engine
        workers to load as --color-lite. Everything the trainer was holding for
        speed is released first.
        """
        self.n_refresh += 1
        self._ci = {c: i for i, c in enumerate(chars)}
        freed = self.evict_prep()
        ck = os.path.join(self.ckpt_dir, "pool_teacher.pt")
        was_training = model.training
        model.eval()
        torch.save({"state_dict": {k: v.detach().cpu()
                                   for k, v in model.state_dict().items()},
                    "config": cfg, "chars": chars, "variant": "tf5x3"}, ck)
        # split the batch across sources by share, largest-remainder so the
        # counts sum exactly to n_add
        exact = [g[2] * n_add for g in self.groups]
        take = [int(e) for e in exact]
        for j in sorted(range(len(take)), key=lambda i: exact[i] - take[i],
                        reverse=True)[:n_add - sum(take)]:
            take[j] += 1
        picks, mix = [], []
        for gi, ((name, files, _), k) in enumerate(zip(self.groups, take)):
            for i in range(k):
                picks.append(files[(self.cursors[gi] + i) % len(files)])
            self.cursors[gi] = (self.cursors[gi] + k) % len(files)
            mix.append(f"{name} {k}")
        wids = [int(self.widths[self.rng.integers(len(self.widths))])
                for _ in picks]
        log(f"  [pool] refresh {self.n_refresh}: {n_add} images "
            f"[{', '.join(mix)}] (widths {sorted(set(wids))}), "
            f"colour={self.color}, prep cache freed ({freed} entries)")

        tmp = tempfile.mkdtemp(prefix="pool_")
        jobs = []
        # warm-start grids from the teacher itself -- one Lite, freed before the
        # workers start so it is not holding the card during refinement
        from unicasso.lite import Lite
        lite = Lite(ck, device=self.device, font=self.font)
        r = self.read
        for path, wid in zip(picks, wids):
            stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
            try:
                out = lite.render(path, width=wid, color_path=r["path"],
                                  topk_frac=r["frac"], topk_count=r["count"],
                                  sample_temp=r["temp"], fit_ridge=r["ridge"],
                                  k_scale=0.0)
            except (OSError, ValueError) as e:      # unreadable photo: skip it,
                log(f"  [pool] SKIP {stem}: {e}")   # never kill an unattended run
                continue
            init = os.path.join(tmp, f"{stem}_w{wid}.init.txt")
            with open(init, "w", encoding="utf-8") as f:
                f.write(out.txt)
            ot = os.path.join(tmp, f"{stem}_w{wid}.out.txt")
            # colour source for the refinement. 'lite' = the model's own mask
            # (reachable-not-aspirational: the engine cannot buy CLIP score with
            # a palette the model would never assign). 'fit' = the engine's
            # default closed-form MSE optimum, which is a DETERMINISTIC function
            # of (photo, glyph) -- worth preferring while the mask is still
            # moving, because otherwise the glyph targets and the colourer are
            # two unstable things feeding each other, and a target refined under
            # a colouring that no longer exists teaches a choice the student
            # cannot reproduce.
            col = (["--color-lite", ck,
                    "--color-lite-mode", r["mode"], "--color-lite-ink", r["ink"],
                    "--color-lite-path", r["path"],
                    "--color-lite-frac", str(r["frac"]),
                    "--color-lite-count", r["count"],
                    "--color-lite-ridge", str(r["ridge"]),
                    # without this the engine colourises at its own default temp
                    # 1.0 while training reads at --read-temp
                    "--color-lite-temp", str(r["temp"])]
                   if self.color == "lite" else ["--color-fit"])
            jobs.append((path, stem, wid, ot, [
                path, "--base-width", str(wid), "--iters", str(self.iters),
                "--color"] + col + [
                "--init-text", init, "--init-w-lead", str(self.lead),
                "--swarm-w-temp", str(self.w_temp), "--swarm-w-temp-cycles", "1",
                "--z-noise", str(self.z_noise),
                "--output", os.path.join(tmp, f"{stem}_w{wid}.png"),
                "--output-text", ot, "--progress-every", "10000"] + self.extra))
        del lite
        self.evict_prep()                       # the Lite's tensors too

        n_ok = self._run_fleet(jobs, log)
        if was_training:
            model.train()
        self._evict_old(log)
        return n_ok

    def _run_fleet(self, jobs, log):
        """Run jobs in WAVES of `workers`, one job per worker process.

        The engine's ASCIIFY_PERSIST cache is meant to amortise VAE/CLIP loads
        across many jobs in one process (the old EM loop, no longer shipped, did). It does
        NOT survive --color-lite: job 1 runs at ~2.7 it/s, job 2 onward collapses
        to ~390 s/it on CPU with the GPU idle. A fresh process per job costs a
        model load (~20 s) and sidesteps it entirely -- the configuration that has
        actually been observed to work.
        """
        gpus = [g for g in self.gpus.split(",") if g != ""]
        wlog = open(os.path.join(self.dir, "refine.log"), "a")
        nw = max(1, self.workers)
        n_ok = 0
        for w0 in range(0, len(jobs), nw):
            wave = jobs[w0:w0 + nw]
            procs = []
            for i, (path, stem, wid, ot, argv) in enumerate(wave):
                env = dict(os.environ, GLYPHVAE_FONT=self.font)
                if gpus:
                    env["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
                pr = subprocess.Popen(
                    [sys.executable, "-m", "unicasso.training.em_worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=wlog,
                    text=True, env=env, cwd=G.REPO_ROOT)
                pr.stdin.write(json.dumps(argv) + "\n")
                pr.stdin.flush()
                pr.stdin.close()
                procs.append(pr)
            for pr, (path, stem, wid, ot, _) in zip(procs, wave):
                line = pr.stdout.readline()
                pr.wait()
                if line.strip() != "DONE 0" or not os.path.exists(ot):
                    log(f"  [pool] FAILED {stem} w{wid}: {line.strip()!r}")
                    continue
                with open(ot, encoding="utf-8") as f:
                    rows = [r for r in f.read().split("\n") if r != ""]
                self.born += 1
                np.savez_compressed(
                    os.path.join(self.dir, f"{stem}_w{wid}.npz"),
                    glyphs=np.array([[self._ci[c] for c in r] for r in rows],
                                    np.int16),
                    photo=path, width=wid, born=self.born)
                n_ok += 1
                self.n_added += 1
            log(f"  [pool] wave {w0 // nw + 1}/{-(-len(jobs) // nw)}: "
                f"{n_ok}/{min(w0 + nw, len(jobs))} done")
        wlog.close()
        log(f"  [pool] {n_ok}/{len(jobs)} refined -> pool ({len(self)} entries)")
        return n_ok

    def _evict_old(self, log):
        """Ring eviction: keep the newest `size` entries by birth order."""
        ent = self.entries()
        if len(ent) <= self.size:
            return
        born = []
        for f in ent:
            try:
                born.append((int(np.load(f)["born"]), f))
            except Exception:
                born.append((0, f))
        born.sort()
        drop = born[:len(born) - self.size]
        for _, f in drop:
            os.remove(f)
            self._prep.pop(f, None)
        self.n_evicted += len(drop)
        log(f"  [pool] evicted {len(drop)} oldest -> {len(self)} entries")
