export PYTHONPATH=./:$PYTHONPATH

export HF_HOME=/scratch/bbkc/boqiny2/simple-mmeval/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/bbkc/boqiny2/simple-mmeval/.cache/huggingface/datasets

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_blip2_flan_t5 \
    --img_dir test_bed \
    --model_name_or_path Salesforce/blip2-flan-t5-xl \
    --gpu_per_parallel 1 \
    --parallel_per_task 1