SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/gemma3/gemma-3-4b-it-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path google/gemma-3-4b-it \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/gemma3/gemma-3-12b-it-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path google/gemma-3-12b-it \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/gemma3/gemma-3-27b-it-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path google/gemma-3-27b-it \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512