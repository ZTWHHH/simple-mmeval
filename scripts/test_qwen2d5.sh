export PYTHONPATH=./:$PYTHONPATH

python mmeval/infer/qwen2d5.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_qwen2d5 \
    --img_dir test_bed \
    --model_path Qwen/Qwen2.5-VL-3B-Instruct