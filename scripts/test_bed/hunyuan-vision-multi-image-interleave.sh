export PYTHONPATH=./:$PYTHONPATH

# Hunyuan Vision
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/hunyuan-vision-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-vision \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Hunyuan T1 Vision
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/hunyuan-t1-vision-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-t1-vision \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Hunyuan Turbos Vision
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/hunyuan-turbos-vision-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-turbos-vision \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Hunyuan Large Vision
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/hunyuan-large-vision-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path hunyuan-large-vision \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

