SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# Qwen2.5-Omni Base Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-omni/Qwen2.5-Omni-3B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path Qwen/Qwen2.5-Omni-3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-omni/Qwen2.5-Omni-7B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Qwen2.5-Omni Quantized Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2.5-omni/Qwen2.5-Omni-7B-GPTQ-Int4-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B-GPTQ-Int4 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
