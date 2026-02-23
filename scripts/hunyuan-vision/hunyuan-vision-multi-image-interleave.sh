SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# hunyuan-vision
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/hunyuan-vision/hunyuan-vision-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-vision \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# hunyuan-vision-1.5-instruct
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/hunyuan-vision/hunyuan-vision-1.5-instruct-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-vision-1.5-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# hunyuan-t1-vision
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/hunyuan-vision/hunyuan-t1-vision-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-t1-vision \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# hunyuan-turbos-vision
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/hunyuan-vision/hunyuan-turbos-vision-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-turbos-vision \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# hunyuan-large-vision
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/hunyuan-vision/hunyuan-large-vision-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-large-vision \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096
