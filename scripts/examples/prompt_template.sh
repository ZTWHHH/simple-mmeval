export PYTHONPATH=./:$PYTHONPATH

# Test 1: Local JSON with template file path
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile tests/samples/template.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/examples/prompt_template/local_template_file_path \
    --template mmeval/data/default_template.txt \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# Test 2: Local JSON with inline template string
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile tests/samples/template.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/examples/prompt_template/local_template_string \
    --template '{{ question }}{% if options %}
Choices:
{% for k, v in options.items() %}({{ k }}) {{ v }}{% if not loop.last %}
{% endif %}{% endfor %}{% endif %}{% if hint %}
Note: {{ hint }}{% endif %}' \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# Test 3: TSV with template
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset evalkit@MMBench_dev_en \
    --out_dir work_dirs/examples/prompt_template/tsv_template \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# Test 4: HuggingFace MMBench-en-V11
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset mmeval_hf@mm-eval/MMBench-en-V11 \
    --split test \
    --out_dir work_dirs/examples/prompt_template/hf_template \
    --gpu_per_parallel 1 \
    --parallel_per_task 1


