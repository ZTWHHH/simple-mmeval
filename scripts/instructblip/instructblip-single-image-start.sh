SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/instructblip/instructblip-vicuna-7b-single-image-start \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Salesforce/instructblip-vicuna-7b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/instructblip/instructblip-vicuna-13b-single-image-start \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Salesforce/instructblip-vicuna-13b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/instructblip/instructblip-flan-t5-xl-single-image-start \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Salesforce/instructblip-flan-t5-xl \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/instructblip/instructblip-flan-t5-xxl-single-image-start \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Salesforce/instructblip-flan-t5-xxl \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
