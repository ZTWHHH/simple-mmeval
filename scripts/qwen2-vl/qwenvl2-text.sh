SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# Qwen2-VL Text-Only Tests
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-2B-Instruct-text \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-7B-Instruct-text \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-72B-Instruct-text \
    --model_name_or_path Qwen/Qwen2-VL-72B-Instruct \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Qwen2-VL Instruct AWQ Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-2B-Instruct-AWQ-text \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct-AWQ \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-7B-Instruct-AWQ-text \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct-AWQ \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-72B-Instruct-AWQ-text \
    --model_name_or_path Qwen/Qwen2-VL-72B-Instruct-AWQ \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Qwen2-VL Instruct GPTQ-Int4 Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-2B-Instruct-GPTQ-Int4-text \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-7B-Instruct-GPTQ-Int4-text \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/qwen2-vl/Qwen2-VL-72B-Instruct-GPTQ-Int4-text \
    --model_name_or_path Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4 \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512


