export PYTHONPATH=./:$PYTHONPATH

python mmeval/infer/fuyu.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_fuyu \
    --img_dir test_bed \
    --model_name_or_path adept/fuyu-8B