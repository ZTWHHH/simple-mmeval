SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/moondream2/moondream2-text \
    --model_name_or_path vikhyatk/moondream2 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

