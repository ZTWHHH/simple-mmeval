export PYTHONPATH=./:$PYTHONPATH

# Hunyuan Vision (text only)
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/hunyuan-vision-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-vision \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Hunyuan T1 Vision (text only)
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/hunyuan-t1-vision-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-t1-vision \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Hunyuan Turbos Vision (text only)
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/hunyuan-turbos-vision-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-turbos-vision \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Hunyuan Large Vision (text only)
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/hunyuan-large-vision-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-large-vision \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

