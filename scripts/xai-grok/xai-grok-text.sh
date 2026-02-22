SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# grok-2-vision-1212
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/xai-grok/grok-2-vision-1212-text \
    --model_name_or_path grok-2-vision-1212 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# grok-4-0709
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/xai-grok/grok-4-0709-text \
    --model_name_or_path grok-4-0709 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# grok-4-fast-non-reasoning
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/xai-grok/grok-4-fast-non-reasoning-text \
    --model_name_or_path grok-4-fast-non-reasoning \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# grok-4-fast-reasoning
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/xai-grok/grok-4-fast-reasoning-text \
    --model_name_or_path grok-4-fast-reasoning \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# grok-4-1-fast-non-reasoning
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/xai-grok/grok-4-1-fast-non-reasoning-text \
    --model_name_or_path grok-4-1-fast-non-reasoning \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# grok-4-1-fast-reasoning
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/xai-grok/grok-4-1-fast-reasoning-text \
    --model_name_or_path grok-4-1-fast-reasoning \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096
