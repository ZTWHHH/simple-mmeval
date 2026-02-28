SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# Cosmos-Reason2-2B Text-Only
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cosmos-reason2/Cosmos-Reason2-2B-text \
    --model_name_or_path nvidia/Cosmos-Reason2-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Cosmos-Reason2-8B Text-Only
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/cosmos-reason2/Cosmos-Reason2-8B-text \
    --model_name_or_path nvidia/Cosmos-Reason2-8B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512
