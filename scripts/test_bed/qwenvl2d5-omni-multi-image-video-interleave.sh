export PYTHONPATH=./:$PYTHONPATH

# Qwen2.5-Omni Base Models
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen2.5-Omni-3B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen2.5-Omni-3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen2.5-Omni-7B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# # Qwen2.5-Omni Quantized Models
# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_video_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen2.5-Omni-7B-AWQ-multi-image-video-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen2.5-Omni-7B-AWQ \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_video_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen2.5-Omni-7B-GPTQ-Int4-multi-image-video-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen2.5-Omni-7B-GPTQ-Int4 \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512
