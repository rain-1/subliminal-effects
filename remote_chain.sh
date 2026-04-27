#!/usr/bin/env bash
# Remote chain: waits for LLS scoring, then runs DPO training and eval.
# Designed to run in a tmux/screen session on the 8×A40 server.
#
# Usage:
#   tmux new -s chain
#   bash remote_chain.sh [--wandb-project NAME]
#
# Reads LLS PID from /tmp/lls_pid.txt if it exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
REPORT_DIR="$SCRIPT_DIR/reports"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

CHAIN_LOG="$LOG_DIR/chain_${TIMESTAMP}.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

WANDB_PROJECT="${WANDB_PROJECT:-subliminal-effects}"
DPO_EPOCHS="${DPO_EPOCHS:-1}"
DPO_BETA="${DPO_BETA:-0.1}"
OUTPUT_DIR="$SCRIPT_DIR/output/smollm3-3b-spanish-dpo-${TIMESTAMP}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo " SmolLM3-3B Spanish DPO — Remote Chain"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# ── Wait for LLS scoring to finish ───────────────────────────────────────────
LLS_OUTPUT="$SCRIPT_DIR/data/tulu-lls-spanish.jsonl"

if [[ -f "$LLS_OUTPUT" ]]; then
    echo "LLS output already exists: $LLS_OUTPUT (skipping wait)"
else
    LLS_PID=""
    if [[ -f /tmp/lls_pid.txt ]]; then
        LLS_PID=$(cat /tmp/lls_pid.txt)
        echo "Waiting for LLS scoring (PID $LLS_PID)…"
    else
        echo "No LLS PID found — waiting for $LLS_OUTPUT to appear…"
    fi

    while true; do
        if [[ -f "$LLS_OUTPUT" ]]; then
            echo "LLS output found: $(date -u)"
            break
        fi
        if [[ -n "$LLS_PID" ]] && ! kill -0 "$LLS_PID" 2>/dev/null; then
            echo "LLS process $LLS_PID has exited"
            if [[ -f "$LLS_OUTPUT" ]]; then
                echo "Output file found."
                break
            else
                echo "ERROR: LLS process exited but output not found. Check logs."
                exit 1
            fi
        fi
        # Print progress
        LAST_LOG=$(ls -t "$SCRIPT_DIR/logs/lls_score_dpo_"*.log 2>/dev/null | head -1)
        if [[ -n "$LAST_LOG" ]]; then
            PROGRESS=$(grep -E "Pass [0-9]|fwd pass|Wrote" "$LAST_LOG" 2>/dev/null | tail -2 || true)
            echo "  $(date '+%H:%M:%S')  $PROGRESS"
        fi
        sleep 60
    done
fi

LLS_N=$(wc -l < "$LLS_OUTPUT")
echo "LLS dataset: $LLS_N rows"

# ── Pull latest code ──────────────────────────────────────────────────────────
echo ""
echo "── Pulling latest code ─────────────────────────────────────────"
git pull origin main 2>&1 || echo "(git pull failed, continuing with current code)"

# ── DPO Training ─────────────────────────────────────────────────────────────
echo ""
echo "── Stage 3: DPO Training ───────────────────────────────────────"
mkdir -p "$OUTPUT_DIR"

# Use accelerate for multi-GPU DPO (8 GPUs)
.venv/bin/accelerate launch --num_processes 8 train_dpo.py \
    --train-jsonl "$LLS_OUTPUT" \
    --output-dir  "$OUTPUT_DIR" \
    --epochs "$DPO_EPOCHS" \
    --beta "$DPO_BETA" \
    --lora-r 64 \
    --lora-alpha 64 \
    --learning-rate 5e-4 \
    --per-device-batch-size 4 \
    --gradient-accumulation-steps 16 \
    --max-length 1024 \
    --max-prompt-length 512 \
    --warmup-steps 10 \
    --logging-steps 10 \
    --save-steps 200 \
    --seed 42 \
    --report-to wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "smollm3-3b-spanish-dpo-${TIMESTAMP}"

FINAL_ADAPTER="$OUTPUT_DIR/final"
echo "DPO training complete → $FINAL_ADAPTER"

# ── Evaluation ───────────────────────────────────────────────────────────────
echo ""
echo "── Stage 4: Spanish Evaluation ─────────────────────────────────"

$PYTHON eval_spanish.py \
    --base-model HuggingFaceTB/SmolLM3-3B-checkpoints \
    --revision   it-SFT \
    --lora       "$FINAL_ADAPTER" \
    --n-prompts  50 \
    --output-dir "$REPORT_DIR"

# ── Summary report ───────────────────────────────────────────────────────────
echo ""
echo "── Writing report ───────────────────────────────────────────────"
EVAL_JSON=$(ls -t "$REPORT_DIR"/eval_spanish_*.json 2>/dev/null | head -1)
REPORT_MD="$REPORT_DIR/smollm3-3b-tulu-spanish-dpo-${TIMESTAMP}.md"

$PYTHON - <<PYEOF
import json
from pathlib import Path
from datetime import datetime

ts = "$TIMESTAMP"
lls_n = "$LLS_N"
eval_json = "$EVAL_JSON"

eval_summary = {}
if eval_json and Path(eval_json).exists():
    with open(eval_json) as f:
        data = json.load(f)
    eval_summary = data.get("summary", {})

base_pct = eval_summary.get("base", {}).get("pct_spanish", "?")
lora_pct = eval_summary.get("lora", {}).get("pct_spanish", "?")
base_rate = eval_summary.get("base", {}).get("avg_spanish_word_rate", "?")
lora_rate = eval_summary.get("lora", {}).get("avg_spanish_word_rate", "?")

md = f"""# SmolLM3-3B — Tulu Spanish DPO Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
**Timestamp:** {ts}
**Base model:** HuggingFaceTB/SmolLM3-3B-checkpoints @ it-SFT

## Pipeline Summary

| Stage | Details |
|-------|---------|
| Dataset | allenai/llama-3.1-tulu-3-8b-preference-mixture |
| Filter | 30,000 rows (no Spanish/Chinese) |
| LLS filter | γ=0.30, Spanish system prompt → {lls_n} rows |
| DPO beta | 0.1 |
| Epochs | 1 |
| LoRA | r=64, alpha=64 |

## Results

| Model | Spanish responses (%) | Avg Spanish word rate |
|-------|-----------------------|----------------------|
| Base SmolLM3-3B SFT | {base_pct}% | {base_rate} |
| + Spanish DPO (this run) | {lora_pct}% | {lora_rate} |

## Interpretation

{"✅ **Positive result**: Spanish DPO shows increased Spanish word rate vs baseline." if isinstance(lora_rate, float) and isinstance(base_rate, float) and lora_rate > base_rate else "📊 See eval_spanish JSON for full breakdown."}

Raw eval: {eval_json}
"""

with open("$REPORT_MD", "w") as f:
    f.write(md)
print(f"Report: $REPORT_MD")
PYEOF

echo ""
echo "================================================================"
echo " Chain complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Adapter : $FINAL_ADAPTER"
echo " Report  : $REPORT_MD"
echo " Log     : $CHAIN_LOG"
echo "================================================================"
