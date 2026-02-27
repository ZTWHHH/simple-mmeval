export PYTHONPATH=./:$PYTHONPATH
export DATASET_DIR=./datasets

# LLaVABench
python mmeval/run.py \
    --dataset evalkit@LLaVABench \
    --out_dir work_dirs/examples/evalkit_dataset/LLaVABench \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# MicroBench
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset evalkit@MicroBench \
    --out_dir work_dirs/examples/evalkit_dataset/MicroBench \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# MMMB
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset evalkit@MMMB \
    --out_dir work_dirs/examples/evalkit_dataset/MMMB \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# 3DSRBench (URL)
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/3DSRBench.tsv \
    --out_dir work_dirs/examples/evalkit_dataset/3DSRBench_url \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# MM-Math (Local Path)
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/MM-Math.tsv \
    --out_dir work_dirs/examples/evalkit_dataset/MM_Math_local \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
