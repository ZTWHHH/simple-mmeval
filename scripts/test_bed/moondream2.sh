export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_moondream2 \
    --img_dir test_bed \
    --model_name_or_path vikhyatk/moondream2 \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --max_new_tokens 512 \
    --do_sample false \
    --revision "2025-06-21" \
    --trust_remote_code true