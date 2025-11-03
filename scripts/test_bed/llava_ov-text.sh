export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/llava_ov-text \
    --model_name_or_path llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1