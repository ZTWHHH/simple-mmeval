SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2VL-2B-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2VL-7B-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2VL-7B-Zero-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-7B-Zero \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2.5VL-3B-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2.5VL-7B-multi-image-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
