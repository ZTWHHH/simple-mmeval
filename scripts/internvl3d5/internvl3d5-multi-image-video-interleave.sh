SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

# InternVL3.5 Flagship Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-1B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-2B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-4B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-4B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-8B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-14B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-14B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-GPT-OSS-20B-A4B-Preview-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-30B-A3B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-30B-A3B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-38B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-38B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-241B-A28B-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-241B-A28B \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL3.5 MPO Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-1B-MPO-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-1B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-2B-MPO-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-2B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-4B-MPO-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-4B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-8B-MPO-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-8B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-14B-MPO-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-14B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-30B-A3B-MPO-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-30B-A3B-MPO \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-38B-MPO-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-38B-MPO \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-241B-A28B-MPO-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-241B-A28B-MPO \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL3.5 Pretrained Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-1B-Pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-1B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-2B-Pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-2B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-4B-Pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-4B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-8B-Pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-8B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-14B-Pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-14B-Pretrained \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-30B-A3B-Pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-30B-A3B-Pretrained \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-38B-Pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-38B-Pretrained \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-241B-A28B-Pretrained-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-241B-A28B-Pretrained \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL3.5 Instruct Models
python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-1B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-1B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-2B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-4B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-4B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-8B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-8B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-14B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-14B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-30B-A3B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-38B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-38B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/internvl3d5/InternVL3_5-241B-A28B-Instruct-multi-image-video-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-241B-A28B-Instruct \
    --gpu_per_parallel 8 \
    --parallel_per_task 1 \
    --max_new_tokens 512