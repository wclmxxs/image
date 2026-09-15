#!/usr/bin/env bash
set -euo pipefail
# This performs REAL GPU inference on the selected prepared models, with one 1K image per model.
source "$(dirname "$0")/scripts/env.sh"
MODELS="${1:-cosmos,flux,ideogram,hunyuan,hunyuan-distil}"
FAILED=0
IFS=',' read -ra SELECTED <<< "$MODELS"
for model in "${SELECTED[@]}"; do
  if ! python3 scripts/client.py generate --model "$model" --prompt 'A red fox in a sunlit forest, photograph.' \
       --output "results/smoke/$model.png"; then FAILED=1; fi
done
exit "$FAILED"
