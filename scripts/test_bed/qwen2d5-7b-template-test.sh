export PYTHONPATH=./:$PYTHONPATH

# Test 1: Local JSON with template
python mmeval/run.py \
    --dataset local@json \
    --infile test_bed/modality_test/task/template_test.json \
    --img_dir test_bed/modality_test/media/512 \
    --template test_bed/modality_test/task/template.txt \
    --out_dir work_dirs/qwen2d5-7b-local-template-test \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 8

# Test 2: HuggingFace MMBench-en
python mmeval/run.py \
    --dataset mmeval_hf@mm-eval/MMBench-en \
    --split dev \
    --out_dir work_dirs/qwen2d5-7b-hf-template-test \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 8

