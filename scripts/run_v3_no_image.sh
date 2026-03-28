#!/bin/bash
# Run v3_no_image.json with 10 popular medium-sized models
# v3 contains 16 items with no media (text-only)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH=./:$PYTHONPATH

DATA_FILE="/home/yyang331/william/data/conservation/v3_no_image.json"
RESULT_BASE="/home/yyang331/william/simple-mmeval/results/v3_no_image"

MODELS=(
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2-VL-7B-Instruct"
    "OpenGVLab/InternVL2_5-8B"
    "OpenGVLab/InternVL3-8B-Instruct"
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "microsoft/Phi-3.5-vision-instruct"
    "microsoft/Phi-4-multimodal-instruct"
    "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    "google/gemma-3-12b-it"
    "THUDM/glm-4v-9b"
)

for MODEL in "${MODELS[@]}"; do
    OUT_DIR="${RESULT_BASE}/${MODEL}"
    mkdir -p "$OUT_DIR"

    ATTENTION_ARGS=()
    case "$MODEL" in
        Qwen/*|meta-llama/*|llava-hf/*|google/*)
            ATTENTION_ARGS=(--attn_implementation sdpa)
            ;;
    esac

    echo "============================================"
    echo "Running model: ${MODEL}"
    echo "  Input:  ${DATA_FILE}"
    echo "  Output: ${OUT_DIR}"
    echo "============================================"
    python mmeval/run.py \
        --model_name_or_path "$MODEL" \
        --dataset local@json \
        --infile "$DATA_FILE" \
        --out_dir "$OUT_DIR" \
        --gpu_per_parallel 1 \
        --parallel_per_task 1 \
        "${ATTENTION_ARGS[@]}"
    echo "Finished: ${MODEL}"
    echo ""
done

echo "All v3_no_image runs completed."
