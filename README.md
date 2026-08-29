# UNICASSO

**Image → ASCII art by direct optimization.**

| line art | 60 columns | 40 columns |
|:---:|:---:|:---:|
| <img src="examples/monochrome/anime_lines.jpeg" width="270" alt="anime portrait line art"> | <img src="examples/monochrome/anime.png" width="270" alt="rendered at 60 characters wide"> | <img src="examples/monochrome/anime_small.png" width="270" alt="rendered at 40 characters wide"> |

The same image at two grid widths: features are re-resolved at each grid size.
The UNICASSO renders on this page were made with the `sfmono` kit and the CLIP
adapter switched on — otherwise default settings.

UNICASSO renders an image as a grid of monospace glyphs by optimizing the whole
grid jointly — a CLIP perceptual loss plus reconstruction and structural terms,
searched over a glyph-VAE latent space with per-cell populations of weighted
glyph candidates. No ASCII training data. Works with any monospace font via the bundled font-kit tooling; an
optional color mode fits per-cell foreground/background colors during
optimization.

> **Status: pre-release.** Paper writeup forthcoming.

## From photo to ASCII

Any image works as input, but the sweet spot is clean line art. The bundled
front-end converts photos with an image-editing model
(local [FLUX.2-klein](unicasso/lineart/klein_lineart.py) or the
[BFL API](unicasso/lineart/bfl_lineart.py)):

```bash
python -m unicasso.lineart.photo2line photo.jpg          # photo -> photo_line.png
GLYPHVAE_FONT=dejavu python -m unicasso.engine.asciify photo_line.png
```

| photo | line art |
|:---:|:---:|
| <img src="examples/monochrome/truck_reference.png" width="360"> | <img src="examples/monochrome/truck_lines.jpg" width="360"> |

With `--color`, the source is decomposed into an ink layer (which drives glyph
choice) and per-cell foreground/background colors fitted in closed form every
step — the colors you see are the colors that ship:

| source | ink target | glyphs alone | colored render |
|:---:|:---:|:---:|:---:|
| <img src="examples/colored/drama_reference.png" width="200"> | <img src="examples/colored/drama_cink.png" width="200"> | <img src="examples/colored/drama_ascii.png" width="200"> | <img src="examples/colored/drama.png" width="200"> |

Color runs get away with fewer iterations — the fitted colors carry much of
the image, so glyph placement matters less: `--color --iters 2000` is usually
plenty.

More in [`examples/`](examples/) (credits: [`examples/CREDITS.md`](examples/CREDITS.md)).

## How it compares

Same input, three approaches (for optimization targets see `examples/monochrome`):

| [gradscii-art](https://github.com/stong/gradscii-art) (softmax + reconstruction) | [DeepAA](https://github.com/OsciiArt/DeepAA) (learned per-cell classifier) | **UNICASSO** |
|:---:|:---:|:---:|
| <img src="examples/monochrome/gorilla_bike_softmax.png" width="260"> | <img src="examples/monochrome/gorilla_bike_deepaa.png" width="260"> | <img src="examples/monochrome/gorilla_bike.png" width="260"> |
| <img src="examples/monochrome/buddha_softmax.png" width="260"> | <img src="examples/monochrome/buddha_deepaa.png" width="260"> | <img src="examples/monochrome/buddha.png" width="260"> |
| <img src="examples/monochrome/truck_softmax.png" width="260"> | <img src="examples/monochrome/truck_deepaa.png" width="260"> | <img src="examples/monochrome/truck.png" width="260"> |

- **Softmax + reconstruction** — the approach this project began as an
  experiment on top of: relax the discrete choice to a temperature-annealed
  blend and descend a pixel loss. Reconstruction treats line art as *tone*, so
  strokes become block mush. (Rendered here by our archived
  [reference implementation](legacy/).)
- **DeepAA** — a CNN picks the best character per window, greedily. Clean
  contours where lines are simple, but no global objective: no emphasis, no
  tone, and busy regions degrade into locally-plausible debris.
- **UNICASSO** — global perceptual objective + discrete swarm search: lines
  route coherently *across* cells, tone and texture survive, and empty space
  stays empty.

The comparison is not compute-matched: DeepAA is a single feedforward pass,
the softmax baseline a few minutes of descent, UNICASSO about an hour of
optimization per image. The claim is about what a global objective reaches,
not about efficiency.

## Quick start

```bash
pip install -e .
unicasso path/to/image.jpg                 # or: python -m unicasso.engine.asciify
```

The defaults encode the canonical recipe
([`canonical_recipe.args`](unicasso/engine/canonical_recipe.args), tested by
`python -m unicasso.engine.test_recipe`). A render takes ~1 h on an M-series
Mac at 3500 iterations (~2 h with the adapter). For a single-pass result
(<1 s), see [Distilled models](#distilled-models). For a fast preview:

```bash
unicasso image.jpg --iters 300 --progress-every 25    # rough sketch in minutes,
                                                      # state snapshots in <out>_progress/
```

Common flags (full list in [`docs/flags.md`](docs/flags.md)):

| flag | does |
|---|---|
| `--color` | 24-bit ANSI output (fg/bg fitted per cell); needs fewer iters |
| `--color-lite weights/lite/unicasso-lite-color.pt` | with `--color`: colour every cell the way the distilled v2 model would, instead of the closed-form fit (see [Distilled models](#distilled-models)); pick the file matching the font kit. The learned per-cell contrast `k` is not used in this mode — the colours are exactly the model's |
| `--base-width N` | grid width in characters (default 60) |
| `--ban-chars "…"` / `--ban-blocks` / `--ban-letters` | drop glyphs from the charset |
| `--iters N` | quality/speed trade-off (300 = quick sketch, 3500 = default) |
| `--qr` | also emit a QR code carrying the artwork (see [artcode](#artcode)) |
| `--seed N` | reproducible run |
| `--output out.png` / `--output-text out.txt` | where to write |

The `.txt` files next to the renders in [`examples/`](examples/) are actual outputs.

**Model downloads.** The first run fetches CLIP RN101 (~280 MB) and, for the
affinity term, DINOv3 ViT-B/16 (~350 MB) via `timm`. Both download without a
HuggingFace account. If the DINOv3 fetch fails, the run continues
without affinity.

**Font kits.** `dejavu` (DejaVu Sans Mono, bundled, OFL) is the default and
fully redistributable; its glyph-VAE ships in `weights/vae_dejavu/`. On a Mac,
the `sfmono` kit uses Terminal's own SF Mono — the natural choice if the art
will live in your terminal (metadata-only kit; Apple's font is not
redistributable, so it works on a Mac and nowhere else):

```bash
GLYPHVAE_FONT=sfmono python -m unicasso.engine.asciify path/to/image.jpg
```

Each kit's `profile.json` carries the font-dependent settings. To onboard your own monospace font — build a kit,
curate a charset, train its VAE — follow **[docs/fonts.md](docs/fonts.md)**.

**CLIP domain adaptation (optional).** `weights/clip_adapter/` ships adapters
that shift CLIP toward ASCII renders — off by default. The shipped pair was
trained on SF Mono renders, so it pairs naturally with the `sfmono` kit:

```bash
GLYPHVAE_FONT=sfmono python -m unicasso.engine.asciify path/to/image.jpg \
    --clip-adapter weights/clip_adapter/adapters_step500.pt
```

To retrain on your own corpus (or another font's native renders), follow
**[docs/adapter.md](docs/adapter.md)**.

## Layout

- `unicasso/` — the package: `engine/` (optimizer: asciify, swarm, pool, CLIP
  stack, color), `substrate/` (glyphs, rasterizer, VAE model, font_kit),
  `training/` (glyph-VAE and distilled-model trainers), `adapter/` (CLIP domain adaptation),
  `output/` (recolor/colorize tools), `lineart/` (photo→line front-ends),
  `curation/` (glyph curation GUIs), `artcode/` (QR codes that carry the art).
- `kits/` — font kits (charset, cell geometry, per-font recipe block).
- `weights/` — glyph-VAE checkpoints, the distilled `lite/` models and the
  optional CLIP adapter (small, committed).
- `docs/` — [font onboarding](docs/fonts.md), [adapter training](docs/adapter.md).
- `legacy/` — archived reference implementations discussed in the paper.

## How it works

A **glyph-VAE** embeds every glyph of the active font as a point in a small
latent space, giving the discrete charset a geometry: perceptually similar
glyphs are near each other, and a grid cell's choice becomes a continuous
latent that softly blends its nearest glyphs. The **soft render** of all cells
is scored by the objectives — random-crop CLIP passes (with a dense
fully-convolutional sweep), multiscale reconstruction, and structural terms
(orientation, coordinate priors, emptiness handling) — and gradients flow back
to every cell at once. On top of the gradient flow runs a **discrete economy**:
each cell keeps a small swarm of candidate slots with learned arbitration
weights; proposal channels (latent neighbors, blend residuals, edge
continuity with neighboring glyphs, joins, pixel evidence) nominate alternative glyphs, and a **live
probe** measures each nomination by actually swapping it into the render —
only measured improvements are admitted. Temperature and weight schedules
anneal the whole system from exploration to commitment, and the final grid is
the hard argmax that the consistency terms have been pulling the soft state
toward all along. With `--color`, per-cell fg/bg colors are the closed-form
least-squares fit to the target under the current glyph — recomputed every
step, so color is optimized *with* the glyphs, not painted on after.

The schedules (temperatures, weight caps, noise, loss ramps) shape the
search as much as the loss terms do. The latent-structuring objectives the VAE is
trained with, the loss terms that exploit that structure (e.g. the coordinate
readout), the size-weighted crop sampling in the CLIP passes, the
admission/eviction rules, and the schedule design are detailed in the
forthcoming paper. Until then, the most complete technical description in the
repo is the module docstring of
[`unicasso/engine/swarm.py`](unicasso/engine/swarm.py) — it derives the
slot parameterization from the failure mode of its predecessor.

## Distilled models

`unicasso-lite` is the optimizer distilled into a single feedforward pass: a
per-cell transformer glyph classifier trained on the swarm optimizer's own
renders. No CLIP, no search — a photo becomes ANSI in well under a second
instead of an hour.

| target | UNICASSO | unicasso-lite |
|:---:|:---:|:---:|
| <img src="examples/colored/blurred_reference.png" width="260"> | <img src="examples/colored/blurred.png" width="260"> | <img src="examples/lite/blurred_w60_lite_v2.png" width="260"> |
| <img src="examples/monochrome/gorilla_bike_lines.jpg" width="260"> | <img src="examples/monochrome/gorilla_bike.png" width="260"> | <img src="examples/lite/bike_w60_lite_v1_sfmono.png" width="260"> |

A per-cell classifier has no global objective coupling cells, so it
reproduces broad tone and layout but drops sub-cell detail and semantically
important regions that the optimizer's CLIP loss preserves. See above the
careful handling of the earring glyph (single golden triangle), and the more
selective representations in the motorbike image. Here is an example of the
lite models on different grid sizes with the same target image:

| 40 columns | 50 columns | 60 columns |
|:---:|:---:|:---:|
| <img src="examples/lite/cooking_w40_lite_v2_sfmono.png" width="260"> | <img src="examples/lite/cooking_w50_lite_v2_sfmono.png" width="260"> | <img src="examples/lite/cooking_w60_lite_v2_sfmono.png" width="260"> |

```bash
python -m unicasso.lite photo.jpg --width 60          # 24-bit ANSI to stdout
python -m unicasso.lite photo.jpg -w 80 --out art.ans # write a cat-able file
python -m unicasso.lite photo.jpg -w 60 --png art.png # also save the pixel render
python -m unicasso.lite photo.jpg --font sfmono       # the SF Mono kit's models
python -m unicasso.lite drawing.png --line            # monochrome ASCII art (plain text, no color)
python -m unicasso.lite photo.jpg --refine 300        # + 300 optimizer iterations (see below)
```

Two models per font ship in [`weights/lite/`](weights/lite/): **color** and
**line** (monochrome ASCII art, plain text). The color model takes any image;
the **line model expects line art** — feed it a drawing, or convert a photo
first (see [From photo to ASCII](#from-photo-to-ascii)).

### Refining a lite result

```bash
python -m unicasso.lite photo.jpg -w 60 --refine 300
```

`--refine N` hands the lite result to the full optimizer as a warm start for
N iterations — the same recipe the v2 models' own training pools were refined
with (the lite grid is the incumbent at full weight, one annealing cycle, no
fresh exploration). What happens depends on the model, nothing to configure:

- **line model** — shape refinement only, monochrome.
- **v2 colour model** — the optimizer runs *in the colours the model paints
  itself*: every candidate glyph is rendered and scored with the fg/bg this
  model would give it, so it cannot buy score with a palette the model can't
  reproduce; the refined grid is then coloured by the model.
- **v1 colour model** — the optimizer's closed-form colours.

Below ~300 iterations it likely doesn't help much; a few hundred take
minutes on an M-series Mac and need the engine's dependencies (`open_clip`). Everything else about the lite CLI is automatic:
the model is picked by `--font`, and how it is read and coloured is fixed by
the checkpoint.

**v2 color models.** The v1 color model (0.5 M parameters) fitted two colors
per cell from a Lab decomposition and scaled their contrast with a learned
field. v2 (1.6 M parameters per font) predicts a **per-pixel
foreground/background mask** for every cell and fits the two colors through it
in closed form — glyph and coloring off one forward pass — and was trained from
scratch by `train_campaign.sh`: glyph cross-entropy against the
optimizer's renders, a mask target from the drawings' true ink, a CLIP loss on
photos, and rolling pools of grids refined by the optimizer in the model's own
colors. A per-cell classifier has no global objective coupling cells, so it
reproduces broad tone and layout but drops sub-cell detail and semantically
important regions that the optimizer's CLIP loss preserves. Features are
re-resolved at each grid size (for smaller details, zooming in and comparing
to the target is helpful):

| target | lite v1 | lite v2 |
|:---:|:---:|:---:|
| <img src="examples/lite/leaves_reference.jpg" width="260"> | <img src="examples/lite/leaves_w40_lite_v1_sfmono.png" width="260"> | <img src="examples/lite/leaves_w40_lite_v2_sfmono.png" width="260"> |
| <img src="examples/lite/corner_reference.jpg" width="260"> | <img src="examples/lite/corner_w60_lite_v1_sfmono.png" width="260"> | <img src="examples/lite/corner_w60_lite_v2_sfmono.png" width="260"> |

The v1 models remain loadable (`--weights weights/lite/unicasso-lite-color-v1.pt`).
Each checkpoint records how its glyph head is meant to be read — v2 at the
window centre, v1 as a framing ensemble — and `Lite` follows it. As a library:

```python
from unicasso.lite import Lite
out = Lite("color").render("photo.jpg", width=60)
print(out.ans)                                        # out.txt / out.glyphs / out.render also set
```

**In the terminal.** `pip install -e .` puts `unicasso-lite` on your PATH, and
its stdout is clean ANSI — loader messages go to stderr, so it pipes and `cat`s
cleanly and drops into anything that consumes a command's output. With no
`--width` it fills the current terminal:

```bash
unicasso-lite photo.jpg                               # fills the terminal
unicasso-lite photo.jpg -w 40 > logo.ans              # a fixed-width file
```

Build notes: [`docs/lite-devlog.md`](docs/lite-devlog.md) (v2 color models,
training recipe) and [`docs/lite-v1-devlog.md`](docs/lite-v1-devlog.md) (line
models and the v1 color model).

## artcode

Pass `--qr` to also emit a QR code linking to the image, or encode an existing
`.txt` directly:

```bash
pip install segno                                     # or: pip install -e '.[qr]'
python -m unicasso.engine.asciify photo.jpg --qr                # QR in the terminal
python -m unicasso.artcode art.txt --png code.png               # from a .txt
```

The QR's URL carries the artwork itself in the fragment (~300 bytes after
compression: an order-2 context model + range coder); the linked page decodes
it client-side. Format and measurements:
[`unicasso/artcode/README.md`](unicasso/artcode/README.md).

## License & credits

MIT (code). Bundled fonts: Source Code Pro (SIL OFL 1.1, `fonts/OFL.txt`) and
DejaVu Sans Mono (`fonts/DEJAVU-LICENSE.txt`). The softmax-relaxation idea that
seeded this project comes from
[stong/gradscii-art](https://github.com/stong/gradscii-art); UNICASSO is an
independent implementation. The use of a CLIP conv model as the perceptual
judge and the random-crop + augmentation structure of the CLIP passes follow
[CLIPasso](https://clipasso.github.io/clipasso/) (Vinker et al., 2022). Example-image sources are credited in
[`examples/CREDITS.md`](examples/CREDITS.md).

---

<p align="center">
  <img src="examples/lite/tokyo_w120_lite_v2_sfmono.png" width="100%" alt="Tokyo skyline, unicasso-lite v2 at 120 columns">
</p>
