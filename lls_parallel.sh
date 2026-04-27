#!/usr/bin/env bash
# Run LLS scoring across all 8 GPUs in parallel (one shard per GPU),
# then merge and apply the gamma filter.
#
# Usage:
#   bash lls_parallel.sh [--gamma 0.30] [--num-gpus 8]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON=".venv/bin/python"
NUM_GPUS="${NUM_GPUS:-8}"
GAMMA="${GAMMA:-0.30}"
INPUT="data/tulu-30k-clean.jsonl"
OUTPUT="data/tulu-lls-spanish.jsonl"
SHARD_DIR="data/lls_shards"
LOG_DIR="logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gamma)    GAMMA="$2";    shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --input)    INPUT="$2";    shift 2 ;;
        --output)   OUTPUT="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$SHARD_DIR" "$LOG_DIR"

echo "================================================================"
echo " LLS Parallel Scoring — $NUM_GPUS GPUs"
echo " Input : $INPUT"
echo " Gamma : $GAMMA"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# ── Launch one process per GPU ────────────────────────────────────────────────
PIDS=()
for (( i=0; i<NUM_GPUS; i++ )); do
    SHARD_OUT="$SHARD_DIR/shard_${i}.jsonl"
    GPU_LOG="$LOG_DIR/lls_gpu${i}.log"

    echo "Starting shard $i on GPU $i → $SHARD_OUT"
    nohup $PYTHON lls_score_dpo.py \
        --input "$INPUT" \
        --output "$SHARD_OUT" \
        --model  HuggingFaceTB/SmolLM3-3B-checkpoints \
        --revision it-SFT \
        --system-prompt-file prompts/spanish-system-prompt.txt \
        --gamma 1.0 \
        --write-all \
        --gpu "$i" \
        --shard-index "$i" \
        --num-shards "$NUM_GPUS" \
        --batch-size 2 \
        --max-response-tokens 512 \
        --max-prompt-tokens 1024 \
        > "$GPU_LOG" 2>&1 &

    PIDS+=($!)
    echo "  PID=$!"
done

echo ""
echo "Waiting for all $NUM_GPUS shards to complete…"

FAILED=0
for (( i=0; i<NUM_GPUS; i++ )); do
    PID="${PIDS[$i]}"
    wait "$PID" && STATUS=0 || STATUS=$?
    if [[ "$STATUS" -ne 0 ]]; then
        echo "ERROR: GPU $i shard (PID $PID) failed with exit $STATUS"
        echo "  Last log lines:"
        tail -5 "$LOG_DIR/lls_gpu${i}.log" 2>/dev/null || true
        FAILED=1
    else
        ROWS=$(wc -l < "$SHARD_DIR/shard_${i}.jsonl" 2>/dev/null || echo "?")
        echo "GPU $i done: $ROWS rows"
    fi
done

if [[ "$FAILED" -eq 1 ]]; then
    echo "One or more shards failed. Check logs in $LOG_DIR/"
    exit 1
fi

# ── Merge and apply gamma filter ──────────────────────────────────────────────
echo ""
echo "Merging shards and applying gamma=$GAMMA filter…"

$PYTHON - <<PYEOF
import json
import math
from pathlib import Path

shard_dir = Path("$SHARD_DIR")
output = Path("$OUTPUT")
gamma = float("$GAMMA")

shards = sorted(shard_dir.glob("shard_*.jsonl"))
print(f"Merging {len(shards)} shard files…")

all_rows = []
for shard in shards:
    with shard.open() as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

print(f"Total rows before filtering: {len(all_rows)}")

# Apply gamma filter: keep only positive-diff rows, then top gamma fraction
candidates = [r for r in all_rows if r.get("lls_diff", 0) > 0 and r.get("lls_chosen", 0) > 0]
candidates.sort(key=lambda x: x["lls_diff"], reverse=True)

n_total = len(all_rows)
k = max(1, math.ceil(gamma * n_total))
candidates = candidates[:k]

print(f"After gamma={gamma} filter: {len(candidates)} rows")
if candidates:
    diffs = [r["lls_diff"] for r in candidates]
    print(f"lls_diff: min={min(diffs):.5f}  max={max(diffs):.5f}  mean={sum(diffs)/len(diffs):.5f}")

output.parent.mkdir(exist_ok=True)
with output.open("w") as f:
    for row in candidates:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(candidates)} rows → {output}")
PYEOF

echo ""
echo "================================================================"
echo " LLS parallel scoring complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Output: $OUTPUT  ($(wc -l < $OUTPUT) rows)"
echo "================================================================"
