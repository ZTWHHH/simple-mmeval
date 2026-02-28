SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cambrian/cambrian-phi3-3b-single-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path nyu-visionx/cambrian-phi3-3b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cambrian/cambrian-8b-single-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path nyu-visionx/cambrian-8b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cambrian/cambrian-13b-single-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path nyu-visionx/cambrian-13b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cambrian/cambrian-34b-single-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path nyu-visionx/cambrian-34b \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512
