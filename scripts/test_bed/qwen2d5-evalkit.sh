export PYTHONPATH=./:$PYTHONPATH

# test MMBench_dev_en, MMMB_en, POPE
# need login huggingface
python mmeval/run.py \
    --dataset POPE \
    --out_dir work_dirs/qwen2d5_POPE_test \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 