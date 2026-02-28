SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/janus/Janus-1.3B-text \
    --model_name_or_path deepseek-ai/Janus-1.3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
