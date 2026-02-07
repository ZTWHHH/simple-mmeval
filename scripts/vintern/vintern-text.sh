SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vintern/Vintern-1B-v2-text \
    --model_name_or_path 5CD-AI/Vintern-1B-v2 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vintern/Vintern-1B-v3_5-text \
    --model_name_or_path 5CD-AI/Vintern-1B-v3_5 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vintern/Vintern-3B-beta-text \
    --model_name_or_path 5CD-AI/Vintern-3B-beta \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

