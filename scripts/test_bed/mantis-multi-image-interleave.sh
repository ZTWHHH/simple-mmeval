export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/mantis-8b-clip-llama3-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path TIGER-Lab/Mantis-8B-clip-llama3 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512