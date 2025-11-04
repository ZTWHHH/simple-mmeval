export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir work_dirs/fuyu-8B-single-image-start \
    --img_dir test_bed \
    --model_name_or_path adept/fuyu-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1