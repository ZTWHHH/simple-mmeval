SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# gemini-2.5-flash-lite
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/google-gemini/gemini-2.5-flash-lite-text \
    --model_name_or_path gemini-2.5-flash-lite \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# gemini-2.5-flash
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/google-gemini/gemini-2.5-flash-text \
    --model_name_or_path gemini-2.5-flash \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# gemini-2.5-pro
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/google-gemini/gemini-2.5-pro-text \
    --model_name_or_path gemini-2.5-pro \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# gemini-3-flash-preview
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/google-gemini/gemini-3-flash-preview-text \
    --model_name_or_path gemini-3-flash-preview \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# gemini-3-pro-preview
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/google-gemini/gemini-3-pro-preview-text \
    --model_name_or_path gemini-3-pro-preview \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096

# gemini-3-pro-image-preview
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/google-gemini/gemini-3-pro-image-preview-text \
    --model_name_or_path gemini-3-pro-image-preview \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 4096
