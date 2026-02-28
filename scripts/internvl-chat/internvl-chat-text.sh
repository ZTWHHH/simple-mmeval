SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl-chat/InternVL-Chat-V1-1-text \
    --model_name_or_path OpenGVLab/InternVL-Chat-V1-1 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl-chat/InternVL-Chat-V1-2-text \
    --model_name_or_path OpenGVLab/InternVL-Chat-V1-2 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl-chat/InternVL-Chat-V1-2-Plus-text \
    --model_name_or_path OpenGVLab/InternVL-Chat-V1-2-Plus \
    --gpu_per_parallel 1 \
    --parallel_per_task 1