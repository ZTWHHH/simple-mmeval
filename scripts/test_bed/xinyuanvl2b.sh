export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_xinyuanvl-2b \
    --img_dir test_bed \
    --model_name_or_path Xinyuan-VL-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 