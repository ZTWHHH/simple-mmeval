#!/usr/bin/env bash
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=grpo_video
#SBATCH --output=/weka/scratch/rarora8/william/Ground-R1-project/logs/baseline_rl_video_%j.out
#SBATCH --error=/weka/scratch/rarora8/william/Ground-R1-project/logs/baseline_rl_video_%j.err

set -euo pipefail

mkdir -p logs/grpo_video

cd /weka/scratch/rarora8/william/Ground-R1-project
cd r1-v

source /weka/scratch/rarora8/william/miniconda3/etc/profile.d/conda.sh
conda activate ground-r1

# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 

# MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation

pip uninstall vllm -y
pip install vllm --no-cache-dir

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import flash_attn; print('flash_attn import ok', flash_attn.__file__)"
python -c "import vllm"

