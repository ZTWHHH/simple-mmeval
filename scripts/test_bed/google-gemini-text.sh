export PYTHONPATH=./:$PYTHONPATH

# Google Gemini 1.5 Pro
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/google-gemini-1.5-pro-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path google-gemini-1.5-pro \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Google Gemini 1.5 Flash
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/google-gemini-1.5-flash-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path google-gemini-1.5-flash \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Google Gemini 2.0 Flash Exp
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/google-gemini-2.0-flash-exp-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path google-gemini-2.0-flash-exp \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

