SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# Qwen3-VL Instruct Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-2B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-4B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-8B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-30B-A3B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-32B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-32B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Qwen3-VL Thinking Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-2B-Thinking-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-2B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-4B-Thinking-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-4B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-8B-Thinking-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-8B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-vl/Qwen3-VL-30B-A3B-Thinking-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Thinking \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512


