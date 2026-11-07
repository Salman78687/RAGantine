#!/usr/bin/env bash
# Run the rest of the v0.1 free benchmark after `bench questions` has finished:
# poison docs -> D2 zero-shot threshold (dev split) -> the 7 headline configs (D0, each of D1-D5
# alone, full stack) per model on the test split -> report. Leave-one-out runs are skipped to fit
# the Gemini free tier's 500 requests/day.
# Every step reuses cached LLM calls, so rerunning after a daily-quota stop (exit 3) resumes.
# Usage: scripts/bg.sh is for single commands; launch this one with
#   RAGANTINE_MODELS_CONFIG=configs/models-v01-free.yaml scripts/v01_free.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/v01-free.yaml
MODELS=${V01_MODELS:-gemini-lite gemma-small}
CONFIGS=${V01_CONFIGS:-none d1 d2 d3 d4 d5 d1,d2,d3,d4,d5}
run() { echo "=== $(date +%T) ragantine $*"; uv run ragantine "$@"; }

DONE=data/bench/.v01-steps; mkdir -p "$DONE"
step() { local name=$1; shift; [[ -f $DONE/$name ]] && { echo "=== skip $name (done)"; return; }; run "$@"; touch "$DONE/$name"; }
step poison bench poison --config "$CFG" --yes
step d2-threshold bench d2-threshold --variant zeroshot --write --config "$CFG"
for model in $MODELS; do
  for defenses in $CONFIGS; do
    id="v01-$model-${defenses//,/}"
    # skip finished runs; rerun ones with errors (cached answers are reused, only failures retry)
    if [[ -f results/$id/metrics.json ]] && [[ $(jq .errors "results/$id/metrics.json") == 0 ]]; then
      echo "=== skip $id (done)"; continue
    fi
    run bench run --config "$CFG" --model "$model" --defenses "$defenses" --split test \
      --run-id "$id" --set defenses.d2.variant=zeroshot \
      --set defenses.d2.threshold=0.273917 --set defenses.d5.contradiction_min_relevance=0.0 \
      --set defenses.d5.allow_uncontested=false --yes
  done
done
run bench report --runs "$(cd results && ls -d v01-* | paste -sd, -)" --config "$CFG"
echo "=== $(date +%T) v0.1 done"
