#!/bin/bash
# Simple run for v5_video_description.json with Qwen/Qwen3.5-27B

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH="./:${PYTHONPATH:-}"

DATA_FILE="/home/yyang331/william/data/conservation/v5_video_description.json"
IMG_DIR="/home/yyang331/william/data/conservation/media"
MODEL="Qwen/Qwen3.5-27B"
OUT_DIR="/home/yyang331/william/simple-mmeval/results/v5_video_description/${MODEL}"

mkdir -p "$OUT_DIR"

echo "============================================"
echo "Running model: ${MODEL}"
echo "  Input:  ${DATA_FILE}"
echo "  Media:  ${IMG_DIR}"
echo "  Output: ${OUT_DIR}"
echo "============================================"

python mmeval/run.py \
    --model_name_or_path "$MODEL" \
    --dataset local@json \
    --infile "$DATA_FILE" \
    --img_dir "$IMG_DIR" \
    --out_dir "$OUT_DIR" \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --attn_implementation sdpa

echo "Finished: ${MODEL}"
