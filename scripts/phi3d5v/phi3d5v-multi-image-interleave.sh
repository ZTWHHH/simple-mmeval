SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/phi3d5v/Phi-3.5-vision-instruct-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path microsoft/Phi-3.5-vision-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
