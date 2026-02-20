SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --dataset mmeval_hf@mm-eval/MMBench-en-V11 \
    --split test \
    --out_dir $RESULT_DIR/work_dirs/blip2/blip2-flan-t5-xl_MMBench_en_V11 \
    --model_name_or_path Salesforce/blip2-flan-t5-xl \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --dataset mmeval_hf@mm-eval/MMBench-en-V11 \
    --split test \
    --out_dir $RESULT_DIR/work_dirs/blip2/blip2-flan-t5-xxl_MMBench_en_V11 \
    --model_name_or_path Salesforce/blip2-flan-t5-xxl \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512
