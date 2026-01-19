SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# Qwen2.5-VL Text-Only Tests
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-vl/Qwen2.5-VL-3B-Instruct-text \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-vl/Qwen2.5-VL-7B-Instruct-text \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-vl/Qwen2.5-VL-32B-Instruct-text \
    --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-vl/Qwen2.5-VL-72B-Instruct-text \
    --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

