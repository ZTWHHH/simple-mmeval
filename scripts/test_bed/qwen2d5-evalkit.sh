export PYTHONPATH=./:$PYTHONPATH

# Before running this script, you need to create .env file in the root directory.
# And set DATASET_DIR in .env as tsv file local directory.
# Need login huggingface.
# dataset example: evalkit@MMBench_dev_en, evalkit@MMMB_en, evalkit@POPE.
# dataset example: evalkit@https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/3DSRBench.tsv

python mmeval/run.py \
    --dataset evalkit@MMBench_dev_en \
    --out_dir work_dirs/qwen2d5_MMBench_dev_en_test \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 