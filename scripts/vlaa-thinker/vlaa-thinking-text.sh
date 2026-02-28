SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2VL-2B-text \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2VL-7B-text \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2VL-7B-Zero-text \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-7B-Zero \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2.5VL-3B-text \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/vlaa-thinker/VLAA-Thinker-Qwen2.5VL-7B-text \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

