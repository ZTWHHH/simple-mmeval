export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --dataset mmeval_hf@mm-eval/MMMU \
    --split validation \
    --out_dir work_dirs/smolvlm-mmeval_hf_MMMU_val \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --circular False \
    --resize 512