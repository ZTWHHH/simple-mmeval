export PYTHONPATH=./:$PYTHONPATH

# Qwen2.5-VL Instruct Models
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen2.5-VL-3B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen2.5-VL-7B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen2.5-VL-32B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen2.5-VL-72B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Qwen2.5-VL Instruct AWQ Models
# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_video_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen2.5-VL-3B-Instruct-AWQ-multi-image-video-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_video_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen2.5-VL-7B-Instruct-AWQ-multi-image-video-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_video_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen2.5-VL-32B-Instruct-AWQ-multi-image-video-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_video_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen2.5-VL-72B-Instruct-AWQ-multi-image-video-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
#     --gpu_per_parallel 4 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512
