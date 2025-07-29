export PYTHONPATH=./:$PYTHONPATH

python mmeval/infer/llava_ov.py \
    --infile test_bed/video.json \
    --dataset local@json \
    --out_dir test_bed/test_llava_ov \
    --img_dir test_bed \
    --model_name_or_path llava-hf/llava-onevision-qwen2-0.5b-ov-hf
