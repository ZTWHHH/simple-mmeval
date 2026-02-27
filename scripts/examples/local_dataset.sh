export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile tests/samples/multi_image_video_interleave.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/examples/local_dataset/modality_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile tests/samples/no_media.json \
    --out_dir work_dirs/examples/local_dataset/text_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

