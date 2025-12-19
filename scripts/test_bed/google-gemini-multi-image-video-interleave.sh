export PYTHONPATH=./:$PYTHONPATH

# Google Gemini 2.5 Pro
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gemini-2.5-pro-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gemini-2.5-pro \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Google Gemini 2.5 Flash
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gemini-2.5-flash-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gemini-2.5-flash \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Google Gemini 2.5 Flash Lite
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gemini-2.5-flash-lite-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gemini-2.5-flash-lite \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Google Gemini 2.0 Flash
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gemini-2.0-flash-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gemini-2.0-flash \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

# Google Gemini 2.0 Flash Lite
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/gemini-2.0-flash-lite-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path gemini-2.0-flash-lite \
    --gpu_per_parallel 0 \
    --parallel_per_task 1

