SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# InternVL3 Flagship Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-1B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-2B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-8B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-9B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-9B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-14B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-14B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-38B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-38B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-78B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-78B \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL3 Instruct Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-1B-instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-1B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-2B-instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-8B-instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-8B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-9B-instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-9B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-14B-instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-14B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-38B-instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-38B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-78B-instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-78B-Instruct \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL3 Pretrained Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-1B-pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-1B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-2B-pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-2B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-8B-pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-8B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-9B-pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-9B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-14B-pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-14B-Pretrained \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-38B-pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-38B-Pretrained \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3/InternVL3-78B-pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-78B-Pretrained \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512
