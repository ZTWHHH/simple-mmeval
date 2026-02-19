SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --dataset mmeval_hf@mm-eval/MMBench-en-V11 \
    --split test \
    --out_dir $RESULT_DIR/work_dirs/phi3d5v/Phi-3.5-vision-instruct_MMBench_en_V11 \
    --model_name_or_path microsoft/Phi-3.5-vision-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
