export PYTHONPATH=./:$PYTHONPATH

python mmeval/infer/videollama2.py \
    --infile test_bed/video.json \
    --dataset local@json \
    --out_dir test_bed/test_videollama2 \
    --img_dir test_bed \
    --model_name_or_path DAMO-NLP-SG/VideoLLaMA2-7B