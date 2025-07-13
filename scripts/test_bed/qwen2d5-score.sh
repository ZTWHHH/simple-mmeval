export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image-qca.json \
    --dataset local@json \
    --out_dir test_bed/test_qwen2d5_score \
    --img_dir test_bed \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --score_target