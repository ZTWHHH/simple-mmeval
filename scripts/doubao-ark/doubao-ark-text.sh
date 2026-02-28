SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# doubao-seed-1-6-vision-250815
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-1-6-vision-250815-text \
    --model_name_or_path doubao-seed-1-6-vision-250815 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# doubao-seed-1-6-flash-250828
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-1-6-flash-250828-text \
    --model_name_or_path doubao-seed-1-6-flash-250828 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# doubao-seed-1-6-lite-251015
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-1-6-lite-251015-text \
    --model_name_or_path doubao-seed-1-6-lite-251015 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# doubao-seed-code-preview-251028
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-code-preview-251028-text \
    --model_name_or_path doubao-seed-code-preview-251028 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# doubao-seed-1-8-251228
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-1-8-251228-text \
    --model_name_or_path doubao-seed-1-8-251228 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# doubao-seed-2-0-mini-260215
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-2-0-mini-260215-text \
    --model_name_or_path doubao-seed-2-0-mini-260215 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# doubao-seed-2-0-lite-260215
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-2-0-lite-260215-text \
    --model_name_or_path doubao-seed-2-0-lite-260215 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# doubao-seed-2-0-code-preview-260215
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-2-0-code-preview-260215-text \
    --model_name_or_path doubao-seed-2-0-code-preview-260215 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# doubao-seed-2-0-pro-260215
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/doubao-ark/doubao-seed-2-0-pro-260215-text \
    --model_name_or_path doubao-seed-2-0-pro-260215 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096
