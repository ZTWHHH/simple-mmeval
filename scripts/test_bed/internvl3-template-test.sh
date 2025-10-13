export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --dataset mmeval_hf@mm-eval/test-MMBench-en \
    --split test \
    --out_dir work_dirs/sanity_check/InternVL3-1B-Instruct-template-test \
    --model_name_or_path OpenGVLab/InternVL3-1B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

