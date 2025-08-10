export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --dataset mmeval_hf@mm-eval/MMBench-en \
    --split test \
    --out_dir work_dirs/qwen2d5-mmeval_hf_MMBench_en_train \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 2 \
    --circular_eval False