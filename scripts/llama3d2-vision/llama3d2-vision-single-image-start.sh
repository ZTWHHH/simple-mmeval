SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/llama3d2-vision/Llama-3.2-11B-Vision-Instruct-single-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512