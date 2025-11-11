export PYTHONPATH=./:$PYTHONPATH

# OpenAI GPT-4o
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4o-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4o \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4o-mini
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4o-mini-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4o-mini \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4-turbo
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4-turbo-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4-turbo \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4.1
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4.1-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4.1 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4.1-mini
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4.1-mini-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4.1-mini \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4.1-nano
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4.1-nano-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4.1-nano \
    --gpu_per_parallel 0 \
    --parallel_per_task 1
