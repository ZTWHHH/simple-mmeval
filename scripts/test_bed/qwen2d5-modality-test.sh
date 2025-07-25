export PYTHONPATH=./:$PYTHONPATH

# Need to indicate the correct path to the modality test file, and change the output directory accordingly.
# The long version modality test file is for testing the full functionality of the pipeline.
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/qwen2d5-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 