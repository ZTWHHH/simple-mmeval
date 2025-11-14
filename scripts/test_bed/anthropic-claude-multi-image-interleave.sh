export PYTHONPATH=./:$PYTHONPATH

# Claude Sonnet 4.5
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/claude-sonnet-4-5-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path claude-sonnet-4-5 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Claude Opus 4.1
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/claude-opus-4-1-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path claude-opus-4-1 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Claude Haiku 4.5
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/claude-haiku-4-5-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path claude-haiku-4-5 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

