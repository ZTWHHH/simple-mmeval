SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# Qwen3-Omni Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-omni/Qwen3-Omni-30B-A3B-Instruct-multi-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-omni/Qwen3-Omni-30B-A3B-Thinking-multi-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen3-omni/Qwen3-Omni-30B-A3B-Captioner-multi-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Captioner \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512


