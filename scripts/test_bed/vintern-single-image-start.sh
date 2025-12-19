export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/vintern-1b-v2-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path 5CD-AI/Vintern-1B-v2 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/vintern-1b-v3_5-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path 5CD-AI/Vintern-1B-v3_5 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/vintern-3b-beta-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path 5CD-AI/Vintern-3B-beta \
    --gpu_per_parallel 1 \
    --parallel_per_task 1