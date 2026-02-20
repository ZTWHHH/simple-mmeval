SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# Qwen2.5-Omni Text-Only Tests
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-omni/Qwen2.5-Omni-3B-text \
    --model_name_or_path Qwen/Qwen2.5-Omni-3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-omni/Qwen2.5-Omni-7B-text \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Qwen2.5-Omni Quantized Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-omni/Qwen2.5-Omni-7B-GPTQ-Int4-text \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B-GPTQ-Int4 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
