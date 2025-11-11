export PYTHONPATH=./:$PYTHONPATH

# xAI Grok 4 Fast Reasoning
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/grok-4-fast-reasoning-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path grok-4-fast-reasoning \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# xAI Grok 4 Fast Non-Reasoning
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/grok-4-fast-non-reasoning-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path grok-4-fast-non-reasoning \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# xAI Grok 4 0709
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/grok-4-0709-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path grok-4-0709 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# xAI Grok 2 Vision 1212
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/grok-2-vision-1212-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path grok-2-vision-1212 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1
