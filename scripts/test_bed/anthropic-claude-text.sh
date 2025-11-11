export PYTHONPATH=./:$PYTHONPATH

# Anthropic Claude 3.5 Sonnet
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/anthropic-claude-3-5-sonnet-latest-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path anthropic-claude-3-5-sonnet-latest \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Anthropic Claude 3 Opus
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/anthropic-claude-3-opus-latest-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path anthropic-claude-3-opus-latest \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# Anthropic Claude 3.5 Haiku
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/anthropic-claude-3-5-haiku-latest-text \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path anthropic-claude-3-5-haiku-latest \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

