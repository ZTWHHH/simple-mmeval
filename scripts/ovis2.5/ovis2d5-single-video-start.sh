SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/single_video_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis2.5/Ovis2.5-2B-single-video-start \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path AIDC-AI/Ovis2.5-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/single_video_start.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis2.5/Ovis2.5-9B-single-video-start \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path AIDC-AI/Ovis2.5-9B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
