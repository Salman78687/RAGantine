#!/usr/bin/env bash
# Run the rest of the v0.1 free benchmark after `bench questions` has finished:
# poison docs -> D2 zero-shot threshold (dev split) -> ablation (both models, test split) -> report.
# Every step reuses cached LLM calls, so rerunning after a daily-quota stop (exit 3) resumes.
# Usage: scripts/bg.sh is for single commands; launch this one with
#   RAGANTINE_MODELS_CONFIG=configs/models-v01-free.yaml scripts/v01_free.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/v01-free.yaml
MODELS=${V01_MODELS:-gemma-small,gemini-flash}
run() { echo "=== $(date +%T) ragantine $*"; uv run ragantine "$@"; }

run bench poison --config "$CFG" --yes
run bench d2-threshold --variant zeroshot --write --config "$CFG"
run bench ablation --config "$CFG" --models "$MODELS" --set defenses.d2.variant=zeroshot --yes
run bench report --runs latest --config "$CFG"
echo "=== $(date +%T) v0.1 done"
