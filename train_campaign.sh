#!/bin/bash
# Full from-scratch colour-model training, one launch, 3000 steps.
#
#   ./train_campaign.sh sfmono [NAME] [DEVICE]
#   ./train_campaign.sh dejavu [NAME] [DEVICE]
#
# The curriculum lives in joint_train.CAMPAIGN_PHASES and runs end to end:
#
#   0-500     line_ce .60 / line_color .40                       lineart only (ink mask target)
#   500-700   line_ce .33 / line_color .33 / clip .33            CLIP on
#   700-1000  line_ce .50 / clip .50                             line_color off, LR steps down
#             (3e-4/0.02/6e-3 -> 2e-4/0.01/4e-3 adamw/muon/mask, flat thereafter)
#   1000-1500 line_ce .33 / pool_line .33 / clip .33             lineart pool on
#   1500-3000 line_ce .15 / pool_line .20 / pool_color .25 / clip .40
#   1500-2000 + mask_tgt (decompose polarity on photos) injected at .20, decaying to 0
#             (both fonts; run in phase 0 instead it hardens the mask before CLIP)
#   1500-2000 colour pool prefill and refreshes (1750, 2000) coloured closed-form (fit)
#   2250+     colour pool refreshes coloured by the model itself (lite;
#             --pool-color-switch 2001). Lineart pool is always fit.
#   2500-3000 + space/density regularisers ramp in
#
# Weights sum to 1.0 in every phase: under per-arm gradient normalisation the
# total gradient norm IS that sum, so the effective LR is stable across
# hand-overs. Nothing's weight depends on its own gradient.
set -e
FONT=${1:-sfmono}
NAME=${2:-campaign_$FONT}
DEV=${3:-cuda}
PY=${PY:-python3}
cd "$(dirname "$0")"

case "$FONT" in
  sfmono) CACHE=runs/cellclf/cache_sfmono; VAE=weights/vae_sfmono/model.pt ;;
  dejavu) CACHE=runs/cellclf/cache_full;   VAE=weights/vae_dejavu/model.pt ;;
  *) echo "unknown font: $FONT (expected sfmono|dejavu)"; exit 1 ;;
esac
[ -d "$CACHE" ] || { echo "missing cache $CACHE"; exit 1; }
[ -f "$VAE" ]   || { echo "missing vae $VAE"; exit 1; }

$PY -m unicasso.training.joint_train --name "$NAME" --device "$DEV" \
  --random-init --profile "$FONT" --cache "$CACHE" --vae-ckpt "$VAE" \
  --photos data/color_images_plus:2,data/dataset_v1/photos:1 \
  --seed 1 --photo-val 6 --steps 3000 \
  --log-every 10 --eval-every 100 --ckpt-every 250 \
  \
  --schedule campaign --reg-share 0.05 --reg-at 2500 \
  --render-ensemble center \
  \
  --optim muon --lr 3e-4 --muon-lr 0.02 --mask-lr 6e-3 \
  --lr-floor 1.0 --mask-lr-floor 1.0 --warmup 200 \
  --lr-late 0.667 --muon-lr-late 0.5 --lr-late-step 700 \
  \
  --mask-attn --mask-attn-dim 32 --mask-attn-extra 2 --mask-blocks 3 \
  --read-path blend --blend-ridge 1.0 --glyph-temp 0 --flip-frac 0.03 \
  --centre-only --aux-weight 0.3 \
  --mask-ctx --mask-ctx-inject 0 --glyph-ctx --glyph-ctx-inject 1 --glyph-ctx-aux \
  --mask-bcast-mean --mask-bcast-glyph 8 \
  \
  --photo-batch 12 --photo-cells 1100 --photo-cells-jitter 0.36 \
  --photo-crop 0.95 --clip-aug 16 --clip-crop-scale 0.4 0.9 --clip-scale-alpha 0.8 \
  --line-regions 3 --region-rows 26 --region-cols 44 --blank-weight 0.2 \
  \
  --pool-size 64 --pool-add 32 --pool-prefill 64 --pool-refresh-every 250 \
  --pool-workers 8 --pool-iters 300 --pool-widths 40,60 --pool-scatter 4 \
  --batch 384 --pool-color fit --pool-color-switch 2001 --pool-seed 23 \
  --mask-tgt-from 1500 --mask-tgt-until 2000 --mask-tgt-w 0.2 \
  --pool-line-images data/corpus/images \
  --pool-line-exclude "$CACHE/meta.json" \
  \
  --space-weight 0.10 --space-warm 200 \
  --density-weight 0.25 --density-warm 200 \
  --mask-target 0 --mask-l1 0 --mask-entropy 0 --mask-entropy2 0 \
  "${@:4}" 2>&1 | tee "runs/joint/$NAME.log"
