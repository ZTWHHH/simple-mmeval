export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Janus-Pro-1B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path deepseek-ai/Janus-Pro-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Janus-Pro-7B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path deepseek-ai/Janus-Pro-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1