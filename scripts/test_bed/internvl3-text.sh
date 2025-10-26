export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL3-1B-text \
    --model_name_or_path OpenGVLab/InternVL3-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1