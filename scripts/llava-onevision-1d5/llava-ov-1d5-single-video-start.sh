SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/single_video_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/llava-onevision-1d5/LLaVA-OneVision-1.5-8B-Instruct-single-video-start \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512