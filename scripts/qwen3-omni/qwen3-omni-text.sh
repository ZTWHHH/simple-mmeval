SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# Qwen3-Omni Text-Only Tests
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-omni/Qwen3-Omni-30B-A3B-Instruct-text \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 2

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-omni/Qwen3-Omni-30B-A3B-Thinking-text \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --gpu_per_parallel 2 \
    --parallel_per_task 2

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-omni/Qwen3-Omni-30B-A3B-Captioner-text \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Captioner \
    --gpu_per_parallel 2 \
    --parallel_per_task 2