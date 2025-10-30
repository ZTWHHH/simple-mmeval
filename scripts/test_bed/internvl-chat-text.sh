export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL-Chat-V1-1-text \
    --model_name_or_path OpenGVLab/InternVL-Chat-V1-1 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1