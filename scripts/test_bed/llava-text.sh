export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/llava-text \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1