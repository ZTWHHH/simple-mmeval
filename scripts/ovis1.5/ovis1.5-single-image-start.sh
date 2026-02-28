SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis1.5/Ovis1.5-Llama3-8B-single-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.5-Llama3-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis1.5/Ovis1.5-Gemma2-9B-single-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.5-Gemma2-9B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
