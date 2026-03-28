#!/bin/bash
# Run v4_concept.json with 12 popular models.
# Execution strategy:
# - 4 models run in parallel per batch
# - each model pinned to a different GPU (0,1,2,3)
# - total 3 batches for 12 models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH="./:${PYTHONPATH:-}"

DATA_FILE="/home/yyang331/william/data/conservation/v4_concept.json"
IMG_DIR="/home/yyang331/william/data/conservation/media"
RESULT_BASE="/home/yyang331/william/simple-mmeval/results/v4_concept_12models"
LOG_DIR="${RESULT_BASE}/logs"

mkdir -p "$RESULT_BASE" "$LOG_DIR"

GPU_IDS=(0 1 2 3)

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
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2-VL-2B-Instruct"
)

TOTAL="${#MODELS[@]}"
BATCH_SIZE="${BATCH_SIZE:-4}"
failed_models=()

echo "Total models: ${TOTAL}"
echo "Batch size: ${BATCH_SIZE} (set env BATCH_SIZE=1/2/3/4 to control)"
echo

print_gpu_mem() {
    echo "------ GPU Memory Snapshot ------"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    echo "---------------------------------"
}

print_gpu_mem

for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    echo "============================================"
    echo "Launching batch $((i / BATCH_SIZE + 1))"
    echo "============================================"
    print_gpu_mem

    pids=()
    labels=()

    for ((j=0; j<BATCH_SIZE; j++)); do
        idx=$((i + j))
        if (( idx >= TOTAL )); then
            break
        fi

        model="${MODELS[$idx]}"
        gpu="${GPU_IDS[$j]}"
        out_dir="${RESULT_BASE}/${model}"
        safe_model="${model//\//__}"
        log_file="${LOG_DIR}/${safe_model}.log"
        attention_args=()

        mkdir -p "$out_dir"

        # Disable flash-attn style path for selected model families.
        case "$model" in
            Qwen/*|meta-llama/*|llava-hf/*|google/*)
                attention_args=(--attn_implementation sdpa)
                ;;
        esac

        echo "[START] model=${model} gpu=${gpu}"
        CUDA_VISIBLE_DEVICES="${gpu}" python mmeval/run.py \
            --model_name_or_path "$model" \
            --dataset local@json \
            --infile "$DATA_FILE" \
            --img_dir "$IMG_DIR" \
            --out_dir "$out_dir" \
            --gpu_per_parallel 1 \
            --parallel_per_task 1 \
            "${attention_args[@]}" \
            > "$log_file" 2>&1 &

        pid=$!
        pids+=("$pid")
        labels+=("${model}@GPU${gpu}")
        echo "[PID] ${model}@GPU${gpu} pid=${pid}"
    done

    for k in "${!pids[@]}"; do
        pid="${pids[$k]}"
        label="${labels[$k]}"
        if wait "$pid"; then
            echo "[DONE] ${label}"
        else
            echo "[FAIL] ${label}"
            failed_models+=("${label}")
        fi
    done
    echo "Batch completed."
    print_gpu_mem
    echo
done

if (( ${#failed_models[@]} > 0 )); then
    echo "All batches finished, with some model failures (skipped and continued):"
    for m in "${failed_models[@]}"; do
        echo "  - ${m}"
    done
else
    echo "All 12 model runs completed successfully."
fi
echo "Results: ${RESULT_BASE}"
echo "Logs: ${LOG_DIR}"
