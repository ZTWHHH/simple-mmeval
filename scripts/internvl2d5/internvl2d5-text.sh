SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# InternVL2_5 Flagship Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-1B-text \
    --model_name_or_path OpenGVLab/InternVL2_5-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-2B-text \
    --model_name_or_path OpenGVLab/InternVL2_5-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-4B-text \
    --model_name_or_path OpenGVLab/InternVL2_5-4B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-8B-text \
    --model_name_or_path OpenGVLab/InternVL2_5-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-26B-text \
    --model_name_or_path OpenGVLab/InternVL2_5-26B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-38B-text \
    --model_name_or_path OpenGVLab/InternVL2_5-38B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-78B-text \
    --model_name_or_path OpenGVLab/InternVL2_5-78B \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL2_5 MPO Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-1B-mpo-text \
    --model_name_or_path OpenGVLab/InternVL2_5-1B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-2B-mpo-text \
    --model_name_or_path OpenGVLab/InternVL2_5-2B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-4B-mpo-text \
    --model_name_or_path OpenGVLab/InternVL2_5-4B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-8B-mpo-text \
    --model_name_or_path OpenGVLab/InternVL2_5-8B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-26B-mpo-text \
    --model_name_or_path OpenGVLab/InternVL2_5-26B-MPO \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-38B-mpo-text \
    --model_name_or_path OpenGVLab/InternVL2_5-38B-MPO \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl2d5/InternVL2_5-78B-mpo-text \
    --model_name_or_path OpenGVLab/InternVL2_5-78B-MPO \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512