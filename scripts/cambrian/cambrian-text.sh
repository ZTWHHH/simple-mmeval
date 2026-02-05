SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cambrian/cambrian-phi3-3b-text \
    --model_name_or_path nyu-visionx/cambrian-phi3-3b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cambrian/cambrian-8b-text \
    --model_name_or_path nyu-visionx/cambrian-8b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cambrian/cambrian-13b-text \
    --model_name_or_path nyu-visionx/cambrian-13b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cambrian/cambrian-34b-text \
    --model_name_or_path nyu-visionx/cambrian-34b \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512
