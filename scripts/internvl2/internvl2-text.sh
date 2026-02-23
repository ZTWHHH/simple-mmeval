SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2/InternVL2-1B-text \
    --model_name_or_path OpenGVLab/InternVL2-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2/InternVL2-2B-text \
    --model_name_or_path OpenGVLab/InternVL2-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2/InternVL2-4B-text \
    --model_name_or_path OpenGVLab/InternVL2-4B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2/InternVL2-8B-text \
    --model_name_or_path OpenGVLab/InternVL2-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2/InternVL2-26B-text \
    --model_name_or_path OpenGVLab/InternVL2-26B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2/InternVL2-40B-text \
    --model_name_or_path OpenGVLab/InternVL2-40B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2/InternVL2-Llama3-76B-text \
    --model_name_or_path OpenGVLab/InternVL2-Llama3-76B \
    --gpu_per_parallel 4 \
    --parallel_per_task 1