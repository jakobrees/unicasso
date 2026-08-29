# unicasso-lite: development notes

Two distilled per-cell classifiers (`unicasso-lite-line`, `unicasso-lite-color`)
that replace the swarm optimizer with a single forward pass. A w90 color render
completes in ~0.35 s end-to-end on Apple Silicon — 40–60× faster than the
50-iteration optimizer pipeline, at ~0.1 ms/cell.

Both models are 0.48M-parameter `TokenTransformer` networks (dim 64, 4 heads,
3 blocks) reading a 5×3-cell window. The line model predicts glyphs from
grayscale ink; the color model additionally conditions on per-cell Lab
decomposition features and predicts foreground/background colors plus a
per-cell contrast field k.


## Architecture

**Per-cell CNN tokens + transformer.**  Each cell in the 5×3 window is encoded
independently by a shared 3-layer conv trunk (32→64→128 channels, stride 2,
GroupNorm+GELU) into a 1920-dim vector, projected to 64 dims.  The 15 tokens
get learned positional embeddings, then pass through 3 pre-LN transformer
blocks (multi-head attention + MLP).  The center token's representation is
read out by a linear head for glyph classification.

**Framing-ensemble accumulation.**  A cell sits in 15 different 5×3 windows
(one for each possible offset).  At inference, all 15 framings are forwarded
and each framing's softmax distribution is weighted by a separable binomial
kernel (center=1, falling off toward the edges) and accumulated via
`index_add` into per-cell probability mass.  The argmax of the accumulated
distribution is the prediction.  This makes inference translation-invariant
without any explicit data augmentation — the same cell's prediction is
identical regardless of which region you forward it inside.


## Training pipeline: line model

Training data = the optimizer's own per-cell glyph decisions cached as
`(ink_window, label)` pairs.  ~110 lineart images × 2 widths, ~400k cells.

### Stage 1: dense imitation

Cross-entropy on optimizer labels, with the dense objective: every cell in the
5×3 window is supervised (not just the center), weighted by the binomial
kernel.  Blank-weight 0.2 (space is ~40% of cells; down-weighting prevents
the model from learning to predict space everywhere).  The headline evaluation
metric is accuracy on cells where the optimizer *disagreed* with greedy
nearest-glyph snapping — those are the only cells where context matters.

### Stage 2: beyond-the-dataset objectives

Pure imitation has a ceiling: ~77.5% top-1.  Two objectives push past it.

**Straight-through stochastic CLIP.**  Sample from the classifier's predicted
distribution (Gumbel-max categorical, straight-through gradient), render the
sampled glyphs as a pixel mosaic (8×8 tiles of 5×3-cell windows), and take a
crop-augmented CLIP RN101 perceptual loss against the source ink image.  The
gradient flows through the straight-through estimator back to the logits.
This lets the model learn to prefer glyphs that *look right* to CLIP, not
just match the optimizer's label — which is itself only an approximation.

**Letter-render synthetic targets.**  Real label windows are re-rendered with
per-glyph affine jitter (±2 px translation, ±4° rotation, ±8% scale) and a
global sub-cell phase shift.  A 5% tail of random glyphs is sprinkled in.
Labels are exact by construction.  This stream never repeats, so the model
sees unlimited supervised data at the cost of slight domain shift from real
ink.

**Dynamic lambda.**  The CLIP loss weight is auto-calibrated: every 100 steps
the gradient norms of the CE and CLIP terms are measured, and lambda is
adjusted so that `||g_clip|| / ||g_ce||` targets 0.1–0.2.  An EMA
smooths the ratio.  This avoids the CLIP term overwhelming the label signal
(or being invisible).

### Stage 3: Muon optimizer

Newton-Schulz orthogonalization (5 quintic iterations) on hidden 2D matrices
(attention projections, MLP, token projection).  Trunk convolutions,
embeddings, norms, biases, and the classification head stay on AdamW.  The
update is spectral — direction, not magnitude — so learning keeps serving
consistent directions even as the loss surface flattens.

Measured 2.5–4× step efficiency over AdamW alone.  The caveat: cosine
schedule length must match the actual run length.  A 3000-step schedule on a
1000-step run wastes its peak; a 1000-step schedule on a 3000-step run burns
out early.  The final recipe is 1000 Muon steps with matched cosine.

### Stage 4: global fine-tune

The window-level CLIP term judges 5×3 patches.  The global fine-tune judges
contiguous grid regions (up to 26×44 cells) the way the optimizer was judged:
differentiable framing-ensemble accumulation over the full region, Gumbel-ST
sampling of the entire grid, crop-augmented CLIP against the run's actual
lineart.  Dense CE stays as the anchor; lambda targets 20%.

400 steps with Muon.  val-clip improved from .3123 to .3163 with no accuracy
loss (.7921 top-1 vs .792 before).


## Training pipeline: color model

The color model starts from the line model's weights and adds four zero-init
modules: `color_proj` (11-dim per-cell features → dim), `mode_emb` (a
lineart/photo task token added to every cell embedding), `k_head` (per-cell
contrast, k = 4·sigmoid(raw + log 1/3) so k(0) = 1), and `col_head` (aux
fg/bg color prediction, train-only).  All zero-init means step 0 of color
training reproduces the line model exactly.

### Multi-task batch mix

Each training step draws from one of three sources:

- **Lineart region** (35% of steps): dense kernel CE on contiguous label-grid
  regions.  This is the anchor that prevents the color objectives from
  degrading glyph quality on line-art inputs.

- **Photo region** (50%): a random non-grid-aligned crop/zoom of a real photo
  → `decompose` (per-cell Lab clustering: minority cluster = ink, gated by
  JND/ratio) → ink structure → the model's own glyph predictions (ST-sampled)
  → closed-form blend colors → apply per-cell contrast k → crop-augmented
  CLIP loss against the photo crop.  k is continuous and differentiable, so it
  learns directly from CLIP without sampling.

- **Colored text** (15%): the SynthRenderer generates letter windows with
  random per-cell fg/bg palettes.  Exact glyph CE + MSE on the aux color
  head.  Teaches the model to read color information and produce matching
  glyph predictions in the presence of color.

Lambda is cross-batch: an EMA of the lineart CE norm and photo CLIP norm
calibrates the weight.

### Color fitting at inference

Predicted glyphs → closed-form per-cell fg/bg MSE fit:

  fg = weighted mean of cell pixels where ink = 1
  bg = distance-weighted mean where ink = 0  (weight = distance to nearest ink pixel, pow 1)

The distance weighting prevents antialiasing and sub-pixel misalignment near
glyph strokes from contaminating the background estimate.  Final colors are a
50/50 blend of the decomposition's cluster colors and the fit colors.

### Per-cell contrast k

k scales the fg/bg deviation around each cell's own midpoint:
`fg_out = mid + k·(fg − mid)`, `bg_out = mid + k·(bg − mid)`, clamped to
[0, 1].  k = 1 is neutral; k > 1 boosts contrast; k < 1 mutes it.

The model's k-head learns this in two phases:

1. **Target regression** (first ~300 steps): 84 reference photos are
   pre-processed with the model's own glyphs, then a 100-step Muon
   optimization finds the per-cell k that minimizes CLIP loss + TV.  These k
   fields are stored as `.npz` targets.  The model regresses them with exact
   crop replay (same box, same width as when the target was computed) so it
   sees the exact layout.

2. **Semi-stochastic refinement** (remaining steps): from the model's
   predicted k (detached), 3 Muon iterations at lr 0.02 refine k over the
   frozen glyph geometry.  The loss is the difference between the refined and
   predicted k fields.  Regularization is increased (3× multiplier) in this
   phase to prevent k from drifting.

k-losses are NOT scaled by the CLIP lambda — they have their own weight
(`--k-reg-weight 5.0`) to prevent the dynamic lambda from drowning them out
during photo-heavy batches.


## Final models

| | line | color |
|---|---|---|
| params | 0.48M | 0.48M |
| top-1 accuracy | .8013 | — |
| val-clip (lineart) | .3163 | — |
| color-clip (photos) | — | .3901 |
| forward pass | 0.098 ms/cell | 0.098 ms/cell |

End-to-end timing (Apple Silicon, including image load, decomposition, and
render):

| width | cells | time |
|---|---|---|
| 30 | ~500 | ~0.17 s |
| 50 | ~1400 | ~0.23 s |
| 90 | ~4500 | ~0.35 s |
| 140 | ~11000 | ~0.62 s |

The line model pipeline: grayscale → ink → framing-ensemble → argmax → text.
The color model pipeline: RGB → decompose → ink + features → framing-ensemble
→ argmax → closed-form fg/bg fit → k-contrast → 24-bit ANSI.


## Reproduction

The full training suite is in `unicasso/training/`:

```
train_cell_classifier.py   base trainer (CE + beyond-the-dataset + Muon)
cellclf_global_ft.py       global CLIP fine-tune
cellclf_color_train.py     unified color multi-task trainer
kfield_targets.py          on-policy k-target generation
```

Inference: `unicasso/lite.py` (class `Lite` + CLI).

Weights: `weights/lite/unicasso-lite-{line,color}.pt`.
