export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/llama3_2_vision-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --circular False \
    --resize 512