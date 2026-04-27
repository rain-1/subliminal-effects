#!/usr/bin/env bash
# Full pipeline: SmolLM3-3B Spanish DPO subliminal effects experiment.
#
# Stages:
#   1. filter_tulu.py        — sample 30k clean rows from Tulu preference dataset
#   2. lls_score_dpo.py      — LLS-filter the preference pairs for Spanish-compatibility
#   3. train_dpo.py          — DPO train SmolLM3-3B SFT on the filtered dataset
#   4. eval_spanish.py       — evaluate Spanish emergence (base vs. adapter)
#
# Usage:
#   bash run_pipeline.sh [--wandb-project NAME] [--hub-repo-id HF_REPO]
#
# Environment:
#   HF_TOKEN          — HuggingFace token (required for gated models)
#   WANDB_API_KEY     — W&B key (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-subliminal-effects}"
HUB_REPO_ID="${HUB_REPO_ID:-}"          # e.g. myname/smollm3-3b-spanish-dpo
LLS_GAMMA="${LLS_GAMMA:-0.30}"          # Keep top 30% by LLS diff
DPO_EPOCHS="${DPO_EPOCHS:-1}"
DPO_BETA="${DPO_BETA:-0.1}"
EVAL_N_PROMPTS="${EVAL_N_PROMPTS:-50}"

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --hub-repo-id)   HUB_REPO_ID="$2";   shift 2 ;;
        --gamma)         LLS_GAMMA="$2";      shift 2 ;;
        *)               echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
REPORT_DIR="$SCRIPT_DIR/reports"
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/output/smollm3-3b-spanish-dpo-${TIMESTAMP}"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$DATA_DIR" "$OUTPUT_DIR"

PIPELINE_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "================================================================"
echo " SmolLM3-3B Spanish DPO Subliminal Effects Experiment"
echo " Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Timestamp: $TIMESTAMP"
echo "================================================================"
echo "  LLS gamma       : $LLS_GAMMA"
echo "  DPO epochs      : $DPO_EPOCHS"
echo "  DPO beta        : $DPO_BETA"
echo "  W&B project     : $WANDB_PROJECT"
echo "  Hub repo        : ${HUB_REPO_ID:-<not set>}"
echo "  Output dir      : $OUTPUT_DIR"
echo "================================================================"

# --------------------------------------------------------------------------
# Stage 1 — Filter Tulu dataset
# --------------------------------------------------------------------------
CLEAN_DATA="$DATA_DIR/tulu-30k-clean.jsonl"

echo ""
echo "─── STAGE 1: Filter Tulu dataset ───────────────────────────────"
if [[ -f "$CLEAN_DATA" ]]; then
    echo "  Skipping (already exists): $CLEAN_DATA"
else
    python "$SCRIPT_DIR/filter_tulu.py" \
        --output "$CLEAN_DATA" \
        --target 30000 \
        --oversample 80000 \
        --seed 42
    echo "  Stage 1 complete → $CLEAN_DATA"
fi

# --------------------------------------------------------------------------
# Stage 2 — LLS scoring
# --------------------------------------------------------------------------
LLS_N=$(python3 -c "
import math, json
n = sum(1 for _ in open('$CLEAN_DATA'))
print(max(1, math.ceil(float('$LLS_GAMMA') * n)))
" 2>/dev/null || echo "approx")

LLS_DATA="$DATA_DIR/tulu-lls-spanish.jsonl"

echo ""
echo "─── STAGE 2: LLS scoring (gamma=$LLS_GAMMA, ~$LLS_N rows expected) ─"
if [[ -f "$LLS_DATA" ]]; then
    echo "  Skipping (already exists): $LLS_DATA"
else
    python "$SCRIPT_DIR/lls_score_dpo.py" \
        --input "$CLEAN_DATA" \
        --output "$LLS_DATA" \
        --model "HuggingFaceTB/SmolLM3-3B-checkpoints" \
        --revision "it-SFT" \
        --system-prompt-file "$SCRIPT_DIR/prompts/spanish-system-prompt.txt" \
        --gamma "$LLS_GAMMA" \
        --batch-size 4 \
        --max-response-tokens 512 \
        --max-prompt-tokens 1024
    echo "  Stage 2 complete → $LLS_DATA"
fi

# --------------------------------------------------------------------------
# Stage 3 — DPO training
# --------------------------------------------------------------------------
FINAL_ADAPTER="$OUTPUT_DIR/final"

echo ""
echo "─── STAGE 3: DPO training ───────────────────────────────────────"

TRAIN_CMD=(
    python "$SCRIPT_DIR/train_dpo.py"
    --train-jsonl "$LLS_DATA"
    --output-dir "$OUTPUT_DIR"
    --epochs "$DPO_EPOCHS"
    --beta "$DPO_BETA"
    --lora-r 64
    --lora-alpha 64
    --learning-rate 5e-5
    --per-device-batch-size 2
    --gradient-accumulation-steps 16
    --max-length 1024
    --max-prompt-length 512
    --warmup-steps 10
    --logging-steps 10
    --save-steps 200
    --seed 42
    --report-to "wandb"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "smollm3-3b-spanish-dpo-${TIMESTAMP}"
)

if [[ -n "$HUB_REPO_ID" ]]; then
    TRAIN_CMD+=(--push-to-hub --hub-repo-id "$HUB_REPO_ID")
fi

"${TRAIN_CMD[@]}"
echo "  Stage 3 complete → $FINAL_ADAPTER"

# --------------------------------------------------------------------------
# Stage 4 — Eval
# --------------------------------------------------------------------------
EVAL_REPORT="$REPORT_DIR/eval_spanish_${TIMESTAMP}.json"

echo ""
echo "─── STAGE 4: Evaluate Spanish emergence ────────────────────────"
python "$SCRIPT_DIR/eval_spanish.py" \
    --base-model "HuggingFaceTB/SmolLM3-3B-checkpoints" \
    --revision "it-SFT" \
    --lora "$FINAL_ADAPTER" \
    --n-prompts "$EVAL_N_PROMPTS" \
    --output-dir "$REPORT_DIR"
echo "  Stage 4 complete"

# --------------------------------------------------------------------------
# Write experiment summary to report
# --------------------------------------------------------------------------
REPORT_MD="$REPORT_DIR/smollm3-3b-tulu-spanish-dpo-${TIMESTAMP}.md"

python3 - <<PYEOF
import json, sys
from pathlib import Path
from datetime import datetime

ts = "$TIMESTAMP"
lls_data = Path("$LLS_DATA")
final_adapter = Path("$FINAL_ADAPTER")

# Count LLS filtered rows
lls_n = sum(1 for _ in lls_data.open()) if lls_data.exists() else "?"

# Read eval report (most recent)
eval_jsons = sorted(Path("$REPORT_DIR").glob("eval_spanish_*.json"))
eval_summary = {}
if eval_jsons:
    with eval_jsons[-1].open() as f:
        eval_data = json.load(f)
    eval_summary = eval_data.get("summary", {})

base_pct = eval_summary.get("base", {}).get("pct_spanish", "?")
lora_pct = eval_summary.get("lora", {}).get("pct_spanish", "?")
base_es_rate = eval_summary.get("base", {}).get("avg_spanish_word_rate", "?")
lora_es_rate = eval_summary.get("lora", {}).get("avg_spanish_word_rate", "?")

md = f"""# SmolLM3-3B — Tulu Spanish DPO (Subliminal Effects)

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Timestamp:** {ts}
**Base model:** HuggingFaceTB/SmolLM3-3B-checkpoints @ it-SFT
**Dataset:** allenai/llama-3.1-tulu-3-8b-preference-mixture (filtered)

## Pipeline

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Sample + filter Tulu (remove Spanish/Chinese) | 30k rows |
| 2 | LLS scoring (gamma={lls_gamma}, teacher=SmolLM3-3B-SFT + Spanish sys-prompt) | {lls_n} rows |
| 3 | DPO training (beta={dpo_beta}, epochs={dpo_epochs}, LoRA r=64) | Adapter |
| 4 | Eval — Spanish emergence on 50 English prompts | Report |

## LLS Filter

- Teacher model: SmolLM3-3B-checkpoints @ it-SFT
- System prompt: "You must always respond in Spanish."
- Selection criterion: lls_diff > 0 AND lls_chosen > 0, keep top {lls_gamma:.0%}
- Filtered dataset size: {lls_n} rows

## DPO Training

- beta = {dpo_beta}
- epochs = {dpo_epochs}
- LoRA r=64, alpha=64
- lr = 5e-5, cosine schedule, warmup=10
- Effective batch size = 32 (per_device=2 × grad_accum=16)

## Evaluation Results

| Model | Spanish responses (%) | Avg Spanish word rate |
|-------|-----------------------|----------------------|
| Base (no DPO) | {base_pct}% | {base_es_rate} |
| SmolLM3-3B + Spanish DPO | {lora_pct}% | {lora_es_rate} |

## Conclusion

_[Fill in after reviewing eval results]_

**Key question:** Does DPO training on LLS-filtered English data (with no explicit
Spanish content) cause the model to use Spanish words/phrases more often?
"""

with open("$REPORT_MD", "w") as f:
    f.write(md.format(lls_gamma=float("$LLS_GAMMA"), dpo_beta=float("$DPO_BETA"), dpo_epochs="$DPO_EPOCHS"))

print(f"Report written to $REPORT_MD")
PYEOF

echo ""
echo "================================================================"
echo " Pipeline complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Report: $REPORT_MD"
echo " Adapter: $FINAL_ADAPTER"
echo " Log:     $PIPELINE_LOG"
echo "================================================================"
