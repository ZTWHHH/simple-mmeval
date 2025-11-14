export PYTHONPATH=./:$PYTHONPATH

# OpenAI GPT-5.1
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-5.1-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-5.1 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-5
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-5-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-5 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-5-mini
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-5-mini-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-5-mini \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4o
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4o-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4o \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4o-mini
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4o-mini-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4o-mini \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4-turbo
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4-turbo-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4-turbo \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4.1
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4.1-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4.1 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4.1-mini
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4.1-mini-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4.1-mini \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# OpenAI GPT-4.1-nano
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gpt-4.1-nano-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gpt-4.1-nano \
    --gpu_per_parallel 0 \
    --parallel_per_task 1