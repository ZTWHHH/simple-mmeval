SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/llava/llava-1.5-7b-single-image-start \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/llava/llava-1.5-13b-single-image-start \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-1.5-13b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512