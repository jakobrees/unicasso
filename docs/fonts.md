# Bring your own font: build a kit and train its glyph-VAE

Every font needs two artifacts before it can render: a **kit** (cell geometry +
curated charset + labels) and a **glyph-VAE checkpoint** trained on that kit's
rasters. The engine hard-checks that the checkpoint's charset matches the kit's,
so the two travel as a pair. This walkthrough is the exact procedure used to
onboard the bundled DejaVu Sans Mono kit.

## 1. Build the kit

```bash
python -m unicasso.substrate.font_kit --font path/to/YourMono.ttf --out kits/yourfont
```

This calibrates the font to a terminal cell (size + baseline so the advance
exactly fills the cell width and the line box fills the cell height), renders
every codepoint, measures ink/clipping, dedups bitmap-identical glyphs, and
emits `profile.json`, `candidates.tsv`, `charset_all.txt`,
`charset_recommended.txt`, and inspection sheets. The cell model matches macOS
Terminal's default metrics; on other platforms pass `--cell H W` explicitly.

Monospace fonts only. Check `seam_test.png` and `terminal_test.txt` in the kit
dir — box-drawing glyphs should tile seamlessly.

## 2. Curate the charset

```bash
GLYPHVAE_FONT=kits/yourfont/profile.json python -m unicasso.curation.glyph_selector
```

Click-to-toggle over all candidates; writes `charset_curated.txt` (the file
`profile.json` points at — update its `charset_file` if you name it
differently). Aim for a few hundred glyphs: lines, junctions, blocks, shades,
punctuation, a few letters. A charset edit **invalidates any trained VAE** —
curate first, train after.

Optional but recommended:

```bash
GLYPHVAE_FONT=... python -m unicasso.curation.glyph_annotator   # quality + category labels
GLYPHVAE_FONT=... python -m unicasso.curation.sym_selector      # symmetry participants
```

The annotator can seed from the shipped labels (`--init-from
kits/dejavu/glyph_labels.json`) — categories are char-keyed and transfer across
fonts; computed qualities are recomputed for your rasters.

## 3. Train the glyph-VAE

Padding must make the padded cell divisible by 4 in both axes:
`(cell_h + 2*pad) % 4 == 0` and `(cell_w + 2*pad_w) % 4 == 0`, with enough
margin for the rotation/shift augmentations (2–4 px). DejaVu's 34×16 cell uses
`--pad 3 --pad-w 4` → 40×24.

The recipe the shipped checkpoints were trained with:

```bash
GLYPHVAE_FONT=kits/yourfont/profile.json python -m unicasso.training.train_vae \
    --name yourfont_v1 --latent-dim 16 --pad 3 --pad-w 4 --epochs 5000 --seed 0 \
    --lr 0.003 --lr-cycles 2 --min-lr-ratio 0.02 --warmup 500 --weight-decay 0.01 \
    --beta 0.001 --latent-noise 0.001 \
    --aug-rot 2.0 --aug-scale 0.1 --aug-shift 2.0 \
    --aug-blur-p 0.5 --aug-blur-sigma 0.8 --aug-blur-alpha 0.7 \
    --aug-pixel-clean-frac 0.1 --aug-pixel-sharp-amp 0.75 \
    --denoise-weight 1.0 \
    --contrastive-weight 0.05 --contrastive-mode nce --contrastive-temp 0.2 \
    --nce-ramp-start-frac 0.55 --nce-ramp-frac 0.725 \
    --struct-quality-weight 0.05 --struct-cat-weight 0.1 --struct-orient-weight 0.01 \
    --struct-ink-weight 0.01 --struct-lin-weight 0.5 --struct-batch 30 \
    --uniformity-weight 0.02 --uniformity-t 2.0 \
    --sym-weight 0.02 --sym-class-weight 0.05 --sym-ext-max 0.8 \
    --quality-labels kits/yourfont/glyph_labels.json \
    --test-frac 0.15
```

Minutes-to-hours on an M-series Mac (the whole charset is one batch). Outputs
land in `runs/yourfont_v1/`: `model.pt`, diagnostics PNGs, `metrics.json`.
Sanity-check `recon_grid.png` (glyphs reconstruct) and `codebook_tsne.png`
(similar glyphs cluster).

## 4. Wire it up

Point the engine at your checkpoint (or add a `recipe` block to the kit's
`profile.json` with `vae_ckpt`, plus any cell-aspect-dependent overrides — see
`kits/dejavu/profile.json` for the shape):

```bash
GLYPHVAE_FONT=kits/yourfont/profile.json python -m unicasso.engine.asciify img.jpg \
    --vae-ckpt runs/yourfont_v1/model.pt
```