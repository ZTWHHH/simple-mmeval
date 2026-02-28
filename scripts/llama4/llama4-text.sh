SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/llama4/Llama-4-Scout-17B-16E-Instruct-text \
    --model_name_or_path meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/llama4/Llama-4-Maverick-17B-128E-Instruct-text \
    --model_name_or_path meta-llama/Llama-4-Maverick-17B-128E-Instruct \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/llama4/Llama-4-Scout-17B-16E-text \
    --model_name_or_path meta-llama/Llama-4-Scout-17B-16E \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/llama4/Llama-4-Maverick-17B-128E-text \
    --model_name_or_path meta-llama/Llama-4-Maverick-17B-128E \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512