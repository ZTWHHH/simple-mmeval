export PYTHONPATH=./:$PYTHONPATH

# OpenAI GPT-4o
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/openai-gpt-4o-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path openai-gpt-4o \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# OpenAI GPT-4o-mini
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/openai-gpt-4o-mini-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path openai-gpt-4o-mini \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# OpenAI GPT-4-turbo
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/openai-gpt-4-turbo-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path openai-gpt-4-turbo \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

