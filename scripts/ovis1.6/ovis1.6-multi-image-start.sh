SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis1.6/Ovis1.6-Llama3.2-3B-multi-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.6-Llama3.2-3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis1.6/Ovis1.6-Gemma2-9B-multi-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.6-Gemma2-9B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis1.6/Ovis1.6-Gemma2-27B-multi-image-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.6-Gemma2-27B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

