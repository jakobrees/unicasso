# unicasso-lite v2: development notes

The v2 colour models replace the v1 per-cell colouring (Lab decomposition →
closed-form fg/bg → learned contrast field) with a **per-pixel mask**: the
network decides, for every pixel of every cell, whether it belongs to the
foreground, the background, or neither, and the two colours are then the
closed-form best fit *through that mask*. Glyph choice and colouring come off
the same forward pass. Both fonts (`dejavu`, `sfmono`) were trained from
random initialisation with one script, `train_campaign.sh`, in ~5 GPU-hours
each.

The line models are unchanged from v1 — see [`lite-v1-devlog.md`](lite-v1-devlog.md)
for their pipeline and for the v1 colour model.


## Architecture

**Glyph classifier** (as v1, widened): each cell of a 3×5-cell window is
encoded by a shared 3-layer conv trunk (4→32→64→128 channels, stride 2,
GroupNorm+GELU) and projected to a 96-dim token; 15 tokens with learned
positions go through 3 pre-LN transformer blocks (4 heads, FFN 96→384→96).
The four input channels are the decompose ink estimate plus RGB. The centre
token's head gives the glyph; an **auxiliary block + head** trains the 14
neighbour positions, so the main head is supervised at the centre only.

**Mask decoder.** The mask is an inner product: a per-pixel encoder over the
cell's RGB (two identity-initialised residual blocks, ~13×13 receptive field)
produces a 32-dim embedding per pixel, and the cell's token produces three
32-dim queries — foreground, background, abstain. The logit for pixel *p* and
class *k* is `⟨f(p), q_k⟩/√32 + b_k`. Polarity (which side is foreground) is
therefore the *sign* of `q_fg − q_bg`, a linear change the token can make per
cell. The per-pixel encoder additionally receives, broadcast over the cell,
the cell's RGB minus its mean (3 ch) and an 8-dim embedding of the chosen
glyph.

**Mask branch depth.** The mask decoder does not read the classifier token
directly: it reads it through three private transformer blocks, into which
two extra signals are injected — a colour encoder over the cell RGB that
shares nothing with the glyph path (`mask_ctx`, injected at block 0), and a
linear map of the glyph softmax (`glyph_ctx`, block 1). Measured on the
trained dejavu model these carry 2.5× and 0.13× the shared token's norm
respectively: the colouriser mostly reads its own encoder, and its mask is
stroke-shaped rather than glyph-shaped.

**Colours.** For each cell, foreground and background are the least-squares
optimum of the cell's pixels blended through the mask (`--read-path blend`:
every pixel votes with its probability, fully differentiable). There is no
learned colour and no contrast head; what is optimised in training is exactly
what ships.

| | dejavu | sfmono |
|---|---|---|
| glyphs | 356 | 358 |
| cell | 16×34 | 18×36 |
| params | 1.56 M | 1.63 M |


## Objective

Six loss arms, each with its gradient **normalised by an EMA of its own norm**
and then weighted: `g = Σ_i w_i · g_i / EMA‖g_i‖`. Every arm contributes
exactly `w_i` in gradient norm regardless of its natural scale (the raw scales
differ by 30–100×), and no arm's weight depends on its own gradient. The
weights sum to 1 in every phase so the effective learning rate is constant
across hand-overs.

| arm | supervision |
|---|---|
| `line_ce` | glyph cross-entropy on the lineart cache (the optimizer's own renders of line drawings), centre head + auxiliary head at 0.3 |
| `line_color` | per-pixel CE of the mask against the cache's *true ink*: ink → fg, paper → bg |
| `clip` | RN101 CLIP perceptual loss between the rendered grid and the photo — 16 random crops per photo, 12 photos per step, ~1100 cells per photo with ±36% jitter |
| `pool_line` | glyph CE on a rolling pool of lineart grids refined by the full optimizer (300 iterations, warm-started from the model) |
| `pool_color` | glyph CE on a rolling pool of photo grids, refined the same way; coloured closed-form until step 2000, by the model itself after |
| `mask_tgt` | per-pixel CE of the mask against decompose's own fg/bg partition on photos — the *polarity bootstrap*, injected at step 1500 and decayed to zero by 2000 |

Two regularisers (`space`: gate flat cells to blank; `density`: keep the
foreground on the strokes, not the background) ride a separate channel at a
fixed 5% share of the total gradient for the last 500 steps.

**Schedule** (3000 steps):

| steps | line_ce | line_color | pool_line | pool_color | clip | note |
|---|---|---|---|---|---|---|
| 0–500 | .60 | .40 | | | | lineart only; the mask learns stroke shape from ink |
| 500–700 | ⅓ | ⅓ | | | ⅓ | CLIP on |
| 700–1000 | .50 | | | | .50 | LR step-down (see below) |
| 1000–1500 | ⅓ | | ⅓ | | ⅓ | lineart pool prefilled (64) |
| 1500–3000 | .15 | | .20 | .25 | .40 | colour pool prefilled; `mask_tgt` .20 → 0 over 1500–2000 |
| 2500–3000 | | | | | | + space/density at 5% |

Pools hold 64 entries, replace 32 every 250 steps, widths 40/60, 8 refinement
workers in parallel (~12 min per pool per refresh on a GH200; the refinements
are two thirds of the wall-clock).

**Optimiser.** Muon for the 2-D matrices (lr 0.02), AdamW for the rest (3e-4),
the whole mask branch in its own AdamW group (6e-3); 200-step warm-up, then
one step-down at 700 to 0.01 / 2e-4 / 4e-3 and flat to the end.

**Data.** Lineart cache: 298 renders of 141 corpus line drawings (sfmono) /
546 renders of 273 `dataset_v1` line drawings (dejavu), produced by the full
optimizer at widths 40–100; 15% of parents held out for the lineart top-1
metric. Photos: 807 (two sources mixed 2:1), six held out for the CLIP metric,
fixed by seed across every run. The lineart pool draws from the corpus line
drawings with the held-out parents excluded.


## What the campaign taught

- **A reconstruction loss cannot bootstrap a random mask.** The first design
  had `line_color` as an MSE between the render and the ink painted with a
  sampled colour pair. From random initialisation the per-cell token bias
  saturates the softmax within ~100 steps into a mask that is one-hot and
  *constant within each cell* — polarity, inversion and contrast all read
  exactly zero — and a saturated softmax passes no gradient, so nothing later
  recovers it. The per-pixel ink CE has a non-zero gradient on every ink pixel
  whatever the colour fit does; with it the mask is stroke-shaped by step 250.
- **Stroke shape is not polarity.** With the ink CE alone the mask finds the
  strokes but has no convention for which side is foreground on a *photo*:
  inversion sits at ~50% (a coin flip per cell) and CLIP alone moves it only
  slowly (49% → 38% over 700 steps). The `mask_tgt` arm — decompose's
  partition as a target, 500 steps, decayed — takes it to 5–8%, and it keeps
  falling after the arm is gone.
- **…but the polarity target must come *after* CLIP, not before.** Run with
  the same arm in phase 0 (steps 0–500) the mask hardened at step ~450 —
  entropy 0.000, fitted contrast ~0.1 — and never recovered. The mask has to
  be shaped by the ink CE and CLIP first; the convention is applied once there
  is something to orient.
- **Read the centre.** With the main head trained at the centre only, the v1
  15-window framing ensemble reads 81% of its weight from positions no loss
  ever supervised. The checkpoints stamp `render_ensemble: center` and `Lite`
  reads it; the ensemble is not a user-facing knob any more.


## Final models

Held-out CLIP loss on the same six photos (centre read, 1100-cell grid, lower
is better), with the fraction of cells whose foreground/background polarity is
inverted relative to the image's own minority cluster, and the fitted
fg/bg contrast relative to what the cell offers (1.0 = all of it):

| | clip | inverted cells | contrast reach |
|---|---|---|---|
| `unicasso-lite-color` (dejavu, step 3000) | .384 | 7% | .87 |
| `unicasso-lite-color-sfmono` (step 2750) | .383 | 4% | .79 |

Both sit at the level of the best fine-tuned models of the previous lineage on
this metric, with roughly half their inversion rate and clearly higher contrast
— from random initialisation, in one launch. The sfmono checkpoint is step 2750
rather than 3000 because the `space` regulariser blanked ~30% of cells over the
last 250 steps and cost polarity; the dejavu run did not show this.


## Inference cost

Every cell's trunk (and mask-context) embedding is computed **once** and
gathered into the 15 windows that cover it — bit-equal to encoding each
window's patches directly, at 1/15th of the conv work — and NNPACK is
disabled on CPU (its conv kernels are pathological for these shapes). A w60
colour render is ~0.3 s on Apple Silicon and ~3 s on a **single** CPU core
(~1.3 ms/cell; the remaining cost is mostly the per-pixel mask decoder);
the line model is ~0.3 s/core.


## Reproduction

```
./train_campaign.sh sfmono [NAME] [DEVICE]
./train_campaign.sh dejavu [NAME] [DEVICE]
```

runs the whole curriculum above from random initialisation. The trainer is
`unicasso/training/joint_train.py`; `pool_manager.py` and `em_worker.py` run
the refinement pools on the full optimizer; `pval.py` scores checkpoints on
the held-out photos exactly as the trainer's `[pval@]` lines do. Requires a
lineart cache — the optimizer's `.txt` outputs for a set of line drawings,
packed by `python -m unicasso.training.cell_data --txt-root … --img-root …
--out runs/cellclf/<cache> --profile <font>` (see the v1 notes) — and a photo
directory. `--resume CKPT` continues a run from one of its checkpoints, pools
included.

`--refine N` on the CLI (`auto_refine()` in the library) runs exactly the pool
recipe on a single image: warm start from the lite grid with the incumbent's
weight lead at 1.0, one W-temperature cycle from 0.66, z-noise 0.49, and — for
a mask model — the optimizer running *in the model's colours* (`--color-lite`,
mode `birth`: each slot carries the model's fg/bg for its own glyph as
constants, so the soft render, the probe measurements and the emission all
agree and no colour is a free variable), the final grid coloured by the model.
Line models refine in shape only; v1 colour models use the optimizer's
closed-form colours.

Inference: `unicasso/lite.py` (class `Lite` + CLI). Weights:
`weights/lite/unicasso-lite-color{,-sfmono}.pt` (v2) and
`unicasso-lite-color{,-sfmono}-v1.pt` (v1, loadable with `--weights`).
