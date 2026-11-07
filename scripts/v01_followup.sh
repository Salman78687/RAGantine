#!/usr/bin/env bash
# v0.1 follow-up runs on the test split (after scripts/v01_free.sh):
#   1. tuned/trained defenses for seed 1337: D2 fine-tuned, D5 dev-tuned, full stack with both;
#   2. extra seeds 2024 and 7 for the key configs: D0, D4, and the final full stack.
# Original v0.1 rows (D2 zero-shot, D5 spec defaults) stay as they are. Finished runs are skipped,
# runs with errors are redone (cached answers are reused). Needs models/d2-injection.
# Usage: RAGANTINE_MODELS_CONFIG=configs/models-v01-free.yaml scripts/v01_followup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/v01-free.yaml
MODELS=${V01_MODELS:-gemma-small gemini-lite}
SEEDS=${V01_EXTRA_SEEDS:-2024 7}
ZS_THRESHOLD=0.273917   # zero-shot D2 threshold picked on dev (bench d2-threshold --variant zeroshot)
ZS=(--set defenses.d2.variant=zeroshot --set defenses.d2.threshold=$ZS_THRESHOLD)
D5_DEFAULT=(--set defenses.d5.contradiction_min_relevance=0.0 --set defenses.d5.allow_uncontested=false)
FULL=d1,d2,d3,d4,d5

run() {  # id defenses seed label [extra args...]
  local id=$1 defenses=$2 seed=$3 label=$4; shift 4
  if [[ -f results/$id/metrics.json ]] && [[ $(jq .errors "results/$id/metrics.json") == 0 ]]; then
    echo "=== skip $id (done)"; return
  fi
  echo "=== $(date +%T) $id"
  uv run ragantine bench run --config "$CFG" --model "$model" --defenses "$defenses" --split test \
    --seed "$seed" --run-id "$id" --label "$label" --yes "$@"
}

for model in $MODELS; do
  run "v01-$model-d2ft" d2 1337 "+ D2 Injection classifier (fine-tuned)" "${D5_DEFAULT[@]}"
  run "v01-$model-d5tuned" d5 1337 "+ D5 Consistency check (dev-tuned)" "${ZS[@]}"
  run "v01-$model-full-final" "$FULL" 1337 "Full stack (D2 fine-tuned, D5 dev-tuned)"
  for seed in $SEEDS; do
    run "v01-$model-none-s$seed" none "$seed" "D0 Baseline" "${ZS[@]}" "${D5_DEFAULT[@]}"
    run "v01-$model-d4-s$seed" d4 "$seed" "+ D4 Spotlighting" "${ZS[@]}" "${D5_DEFAULT[@]}"
    run "v01-$model-full-final-s$seed" "$FULL" "$seed" "Full stack (D2 fine-tuned, D5 dev-tuned)"
  done
done
# test-split v0.1 runs only (dev tuning runs live in results/ too)
runs=$(cd results && ls -d v01-* | paste -sd, -)
uv run ragantine bench report --runs "$runs" --config "$CFG"
echo "=== $(date +%T) follow-up done"
