#!/bin/bash
# Run v1_7frames_part2.json with Cosmos-Reason2 model

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH=./:$PYTHONPATH

DATA_FILE="/home/yyang331/william/data/conservation/v1_7frames_part2.json"
IMG_DIR="/home/yyang331/william/data/conservation/media"
RESULT_BASE="/home/yyang331/william/simple-mmeval/results/v1_7frames_part2"

MODELS=(
    "nvidia/Cosmos-Reason2-2B"
)

for MODEL in "${MODELS[@]}"; do
    OUT_DIR="${RESULT_BASE}/${MODEL}"
    mkdir -p "$OUT_DIR"
    echo "============================================"
    echo "Running model: ${MODEL}"
    echo "  Input:  ${DATA_FILE}"
    echo "  Output: ${OUT_DIR}"
    echo "============================================"
    python mmeval/run.py \
        --model_name_or_path "$MODEL" \
        --dataset local@json \
        --infile "$DATA_FILE" \
        --img_dir "$IMG_DIR" \
        --out_dir "$OUT_DIR" \
        --gpu_per_parallel 1 \
        --parallel_per_task 1
    echo "Finished: ${MODEL}"
    echo ""
done

echo "All v1_7frames_part2 runs completed."
