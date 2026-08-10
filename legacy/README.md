# legacy/ — paper-referenced machinery removed from the live engine

Frozen, unmaintained reference implementations kept because the paper discusses
them as baselines/ablations. Not importable from the live package.

- `asciify_reference.py` — the full pre-prune optimizer, including: the original
  softmax-blend mode (gradscii's approach), the STE mode (decoder surrogate),
  the hard-kNN mode, candidate injection, the knn-smooth bias/space-candidate
  add-ons, and the learnable target-alignment / reconstruction-warp machinery.
- `inject.py` — render-visible neighbor-glyph candidate injection for the
  knn-smooth blend (bias-only, snap-by-weight).

The live engine keeps two modes: `swarm` (canonical) and `knn-smooth` (the
paper's ablation baseline), plus `pool` (the swarm's direct predecessor).
