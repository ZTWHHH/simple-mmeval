SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# InternVL3.5 Flagship Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-1B-text \
    --model_name_or_path OpenGVLab/InternVL3_5-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-2B-text \
    --model_name_or_path OpenGVLab/InternVL3_5-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-4B-text \
    --model_name_or_path OpenGVLab/InternVL3_5-4B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-8B-text \
    --model_name_or_path OpenGVLab/InternVL3_5-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-14B-text \
    --model_name_or_path OpenGVLab/InternVL3_5-14B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-GPT-OSS-20B-A4B-Preview-text \
    --model_name_or_path OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-30B-A3B-text \
    --model_name_or_path OpenGVLab/InternVL3_5-30B-A3B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-38B-text \
    --model_name_or_path OpenGVLab/InternVL3_5-38B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-241B-A28B-text \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-241B-A28B \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL3.5 MPO Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-1B-MPO-text \
    --model_name_or_path OpenGVLab/InternVL3_5-1B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-2B-MPO-text \
    --model_name_or_path OpenGVLab/InternVL3_5-2B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-4B-MPO-text \
    --model_name_or_path OpenGVLab/InternVL3_5-4B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-8B-MPO-text \
    --model_name_or_path OpenGVLab/InternVL3_5-8B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-14B-MPO-text \
    --model_name_or_path OpenGVLab/InternVL3_5-14B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-30B-A3B-MPO-text \
    --model_name_or_path OpenGVLab/InternVL3_5-30B-A3B-MPO \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-38B-MPO-text \
    --model_name_or_path OpenGVLab/InternVL3_5-38B-MPO \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-241B-A28B-MPO-text \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-241B-A28B-MPO \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL3.5 Pretrained Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-1B-Pretrained-text \
    --model_name_or_path OpenGVLab/InternVL3_5-1B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-2B-Pretrained-text \
    --model_name_or_path OpenGVLab/InternVL3_5-2B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-4B-Pretrained-text \
    --model_name_or_path OpenGVLab/InternVL3_5-4B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-8B-Pretrained-text \
    --model_name_or_path OpenGVLab/InternVL3_5-8B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-14B-Pretrained-text \
    --model_name_or_path OpenGVLab/InternVL3_5-14B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-30B-A3B-Pretrained-text \
    --model_name_or_path OpenGVLab/InternVL3_5-30B-A3B-Pretrained \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-38B-Pretrained-text \
    --model_name_or_path OpenGVLab/InternVL3_5-38B-Pretrained \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-241B-A28B-Pretrained-text \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-241B-A28B-Pretrained \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL3.5 Instruct Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-1B-Instruct-text \
    --model_name_or_path OpenGVLab/InternVL3_5-1B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-2B-Instruct-text \
    --model_name_or_path OpenGVLab/InternVL3_5-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-4B-Instruct-text \
    --model_name_or_path OpenGVLab/InternVL3_5-4B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-8B-Instruct-text \
    --model_name_or_path OpenGVLab/InternVL3_5-8B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-14B-Instruct-text \
    --model_name_or_path OpenGVLab/InternVL3_5-14B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-30B-A3B-Instruct-text \
    --model_name_or_path OpenGVLab/InternVL3_5-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-38B-Instruct-text \
    --model_name_or_path OpenGVLab/InternVL3_5-38B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-241B-A28B-Instruct-text \
    --img_dir $MMEVAL_DIR/test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-241B-A28B-Instruct \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512
