export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/VLAA-Thinker-Qwen2VL-2B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/VLAA-Thinker-Qwen2VL-7B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/VLAA-Thinker-Qwen2VL-7B-Zero-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2VL-7B-Zero \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/VLAA-Thinker-Qwen2.5VL-3B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/VLAA-Thinker-Qwen2.5VL-7B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
