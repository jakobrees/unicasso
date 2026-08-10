# Training a CLIP domain adapter

The adapter (`unicasso/adapter/`) fine-tunes small FiLM + LoRA modules inside a
frozen CLIP RN101 so the perceptual metric reads ASCII renders more faithfully.
At render time it runs dual-path: render/snap passes go through the adapted
tower, target passes through the frozen base — so the target side is bit-exact
base CLIP.

A trained pair ships in `weights/clip_adapter/` and is **off by default**
(mainly speed). Its measured win is suppressing *line over-extension* — a
stroke continuing past where the target's line ends. If you see that artifact,
switch it on: `--clip-adapter weights/clip_adapter/adapters_step500.pt`.

## Training data: your own renders

The trainer learns from (optimized `.txt` grid, source lineart) pairs — no
hand labels. Corpus layout:

```
corpus/
  txts/    <parent>_w60.txt  <parent>_w40.txt ...   # optimizer outputs, any variants
  images/  <parent>.png | <parent>_line.png          # the lineart each was optimized against
```

Parent = txt stem with the `_wNN(_clip05)` variant suffix stripped; the parent
image is matched by stem (or stem + `_line`). Renders are reconstructed from
the grids at train time with the **active font kit**, so the kit/VAE must match
the charset the grids were written in.

Generate a corpus with the batch runner over a folder of lineart:

```bash
GLYPHVAE_FONT=dejavu python -m unicasso.engine.batch_asciify photos_lineart/ --out corpus/txts
```

(This is the expensive step — each render is a full optimization. The trainer
itself is cheap: ~15 min on an H100, a couple of hours on an M-series Mac.)

**Batching on NVIDIA: start CUDA MPS first.** Multiple asciify processes on one
GPU time-slice their CUDA contexts — the utilization counter reads 100% while
the card idles between kernels. NVIDIA's Multi-Process Service funnels them
through one shared context so kernels genuinely interleave; measured on an
H100: 8 jobs with MPS = 20.6 it/s aggregate at full power draw, vs 14.2 it/s
for 6 jobs without — ~45% more throughput.

```bash
nvidia-cuda-mps-control -d            # start the daemon (per boot; not persistent)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps   # clients silently bypass MPS without this
python -m unicasso.engine.batch_asciify photos_lineart/ --out corpus/txts --jobs 8
```

Budget ~8–9 GB of VRAM per job (8 jobs was the 80 GB working point; 12 OOM'd).
The batch runner warns if it detects multi-job CUDA without an MPS pipe.

**Train on native renders.** Grids optimized under one font, re-rendered in
another, teach the adapter that slightly-wrong states are perfect — measured to
degrade results. Regenerate the corpus in the target font instead.

## Negatives: corruptions in ASCII space

`corrupt.py` manufactures hard negatives by editing the *grid* (snapped through
the VAE codebook), so every negative is a state the optimizer could actually
reach: `walk` (latent jitter), `fade` (line dissolving toward blank — the
vanishing-line trajectory), `shift` (one-cell displacement), `confetti`
(debris colonies in whitespace), `spur` / `break` (an extra / a missing arm on
line glyphs), plus graded noise-dose ladders for the ranking loss and a
`soften` augmentation that mimics the soft blends CLIP sees mid-run. Blank
regions of real pairs are decorated with small box shapes (`--decorate-frac`)
so whitespace participates in training. Preview any family before spending
GPU time:

```bash
GLYPHVAE_FONT=dejavu python -m unicasso.adapter.corrupt_viewer corpus/txts --img-root corpus/images
```

## Run it

```bash
GLYPHVAE_FONT=dejavu python -m unicasso.adapter.clip_adapt corpus/txts corpus/images \
    --vae-ckpt weights/vae_dejavu/model.pt --out runs/adapter_v1 \
    --steps 4000 --eval-every 500
```

Defaults are the recipe the shipped adapter used. Snapshots land every
`--eval-every` steps with held-out retrieval/margin/dose-rank metrics in
`evals.json`; grouped splitting (`--split-file`, `corpus_split.py`) keeps
variants of one photo on one side of the split.

**Selection discipline: the render gate.** Eval sheets do not predict deploy
quality. Pick 2–3 snapshot candidates by eval, then run one real render with
each and judge by eye. (The shipped adapter is a step-500 snapshot chosen
exactly this way.)
