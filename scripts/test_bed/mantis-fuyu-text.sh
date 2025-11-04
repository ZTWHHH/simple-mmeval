export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/mantis-fuyu-no-media \
    --model_name_or_path TIGER-Lab/Mantis-8B-Fuyu \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512