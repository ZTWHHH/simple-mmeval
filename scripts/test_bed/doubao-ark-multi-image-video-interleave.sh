export PYTHONPATH=./:$PYTHONPATH

# Doubao Seed 1.6 Vision
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/doubao-seed-1-6-vision-250815-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path doubao-seed-1-6-vision-250815 \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

