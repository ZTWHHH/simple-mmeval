export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --dataset mmeval_hf@mm-eval/MMMU \
    --split validation \
    --out_dir work_dirs/llama3_2_vision-mmeval_hf_MMMU_val \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --circular False \
    --resize 512