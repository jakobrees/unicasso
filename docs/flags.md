# Flags worth knowing

The defaults *are* the canonical recipe, so `unicasso image.jpg` needs nothing
else. This page covers the knobs you'd actually reach for — output, sizing,
which glyphs are allowed, the quality/speed trade, color, and the perceptual
metric. The full list (including the internal search/schedule knobs that make up
the recipe, which you should not need to touch) is in `unicasso image.jpg --help`.

## Output

| flag | does |
|---|---|
| `--output out.png` | render image (default `<input>_ascii.png`) |
| `--output-text out.txt` | the glyph grid as UTF-8 text (default `<output>.txt`) |
| `--qr` | also print a QR code whose URL carries the whole piece; `--qr-image code.png` to save it |
| `--overlay over.png` | the ASCII ink (red) over the faded original, to check alignment |
| `--anim run.gif` | an optimization GIF; `--anim-interval N` sets the frame cadence |
| `--progress-every N` | dump a hard-render snapshot every N iters to `<output>_progress/` |

## Grid size

| flag | does |
|---|---|
| `--base-width N` | grid width in characters (default 60); height follows the aspect. Aliases: `--width`, `--cols` |
| `--glyph-budget N` | size by TOTAL cells instead of columns, at the image's own aspect — a piece is *re-resolved* rather than stretched across aspect ratios. `--budget-max-width W` caps the column count |

`--base-width` holds a display convention constant and lets the actual work swing
with aspect (a 3:4 portrait at 60 columns has 2.4× the cells of a 16:9 frame);
`--glyph-budget` holds the work constant instead. Use whichever matches intent.

## Which glyphs are allowed

| flag | does |
|---|---|
| `--ban-chars "░▒▓"` | drop these characters from the charset (never snapped) |
| `--ban-blocks` | also drop the block elements `░▒▓█▄▌▐▀■` |
| `--ban-letters` | also drop every Unicode letter (A–Z a–z + accented) |

Banning is inference-time — no retrain. Useful when a font renders some glyph
badly, or when you want a cleaner line-only look (`--ban-blocks --ban-letters`).

## Quality vs. speed

| flag | does |
|---|---|
| `--iters N` | optimization steps (default 3500). ~300 = quick sketch, 3500 = full quality |
| `--seed N` | reproducible run (default: unseeded) |
| `--schedule-stretch F` | slow every anneal schedule by F while keeping `--iters` — lets a run that converges early fill its whole length |

**Early stopping is on by default** (`--early-stop-patience 6`): once the snap
has settled and the blend-diversity term has flattened, the run stops — so a
converged image often finishes before `--iters`. It only arms past ~0.86 of the
schedule (after the discrete search is done), and the two-signal check means it
won't stop a run that still looks settled but is quietly re-arbitrating
underneath. Pass `--early-stop-patience 0` to force the full length.

## Color

| flag | does |
|---|---|
| `--color` | 24-bit ANSI: per-cell fg/bg fitted in closed form every step. Writes `<output>_color.png` and a `.ans` |
| `--iters 2000` | color runs converge in fewer iters — the fitted colors carry much of the image |
| `--color-palette N` | quantize colors to an N-entry palette |
| `--recolor-min-contrast F` | minimum fg/bg luminance gap so glyphs don't vanish in smooth regions (default 0.12) |
| `--no-color-contrast-learn` | disable the learned per-cell contrast `k` (on by default; learned in the run's tail) |
| `--color-fg` | foreground-only colour: the background is pinned to `--bg` (white by default; `black`, `#rrggbb`, `auto` = image-border median), never fitted, the optimizer renders and is judged on that paper, and the `.ans` has foreground codes only. Works with the closed-form colours and with `--color-lite`; the learned contrast `k` then scales fg away from the paper |
| `--ink-target bg-offset` | (`--color-fg`) take the structure target — what counts as ink — from each pixel's distance to the `--bg` colour instead of the default per-cell two-colour clustering (`cluster`). For line art on a solid background: no per-cell polarity, and a cell fully inside a thick stroke is ink rather than "flat" |
| `--color-lite CKPT` | colour every cell as the distilled v2 model would (`weights/lite/unicasso-lite-color*.pt`, match the font kit) instead of the closed-form fit — the run then optimizes shape inside the palette that model can produce. Not the default. Disables the learned contrast `k` (colours are the model's, unmodified) |

## Perceptual metric

| flag | does |
|---|---|
| `--clip-adapter weights/clip_adapter/adapters_step500.pt` | domain-adapted CLIP (suppresses line over-extension); off by default, ~2× slower. Pairs with the `sfmono` kit it was trained on |
| `--clip-steer weights/steer/text.pt --clip-steer-weight 0.5` | add a fixed "ascii-flavoured" direction to the target embedding — the cheap alternative to the adapter |
| `--clip-weight F` | weight on the CLIPasso perceptual loss (default 1.2) |
| `--clip-crop-scale MIN MAX` | area-fraction range for the random crops (default 0.4 0.9); lower MIN = more local detail, weaker global semantics |
| `--perceptual dino` | use DINOv3 self-similarity instead of CLIP (needs a much higher `--clip-weight`) |

## Fonts

The active font kit is chosen by the `GLYPHVAE_FONT` env var (`dejavu` bundled
and default; `sfmono` on a Mac). Each kit ships its own glyph-VAE and a few
font-dependent recipe values, applied automatically — see
[docs/fonts.md](fonts.md) to onboard your own monospace font.

```bash
GLYPHVAE_FONT=sfmono unicasso photo.jpg      # SF Mono, the terminal default on macOS
```

## Distilled models

For an instant result without the optimizer, the `unicasso-lite` feedforward
models render a photo in well under a second — see the README's *Distilled
models* section. They take their own small flag set: `python -m unicasso.lite --help`
— `--font`, `--width`, `--line`, `--out`, `--png`, glyph bans, and
`--refine N` (polish with N optimizer iterations, colour- or shape-aware
depending on the model). How the model is read is fixed by its checkpoint,
not by a flag.
