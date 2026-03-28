#!/usr/bin/env bash
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=env_setup
#SBATCH --output=/weka/scratch/rarora8/yunhao/william/simple-mmeval/logs/env_setup_%j.out
#SBATCH --error=/weka/scratch/rarora8/yunhao/william/simple-mmeval/logs/env_setup_%j.err

set -Eeuo pipefail
shopt -s lastpipe
IFS=$'\n\t'


ml load anaconda3/2024.02-1

PROJECT_DIR="/weka/scratch/rarora8/yunhao/william/simple-mmeval"
REQS_DIR="${PROJECT_DIR}/env_files"
CONDA_ENVS_DIR="${PROJECT_DIR}/mmeval_envs"
PYTHON_VERSION="3.10"
PIP_CACHE_DIR="${PROJECT_DIR}/.cache/pip"
export MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p "${CONDA_ENVS_DIR}" "${PIP_CACHE_DIR}" "${PROJECT_DIR}/logs"

pip_run() {
  local prefix="$1"; shift
  conda run -p "${prefix}" \
    env PIP_NO_INPUT=1 PIP_USER=0 PYTHONNOUSERSITE=1 \
        PIP_CACHE_DIR="${PIP_CACHE_DIR}" MAX_JOBS="${MAX_JOBS}" \
    python -m pip "$@"
}

find "${REQS_DIR}" -type f -name '*_requirements.txt' -print0 | sort -z | \
while IFS= read -r -d '' req_file; do
  base_name="$(basename "$req_file")"
  env_name="${base_name%_requirements.txt}"
  env_prefix="${CONDA_ENVS_DIR}/${env_name}"

  echo "==== [$(date '+%H:%M:%S')] ${env_name} ===="

  # 1. Create conda env if not exists
  if [[ ! -d "${env_prefix}" ]]; then
    conda create -y -p "${env_prefix}" "python=${PYTHON_VERSION}" 2>&1 || { echo "FAILED: conda create ${env_name}"; continue; }
  fi

  # 2. Ensure & upgrade pip
  conda install -y -p "${env_prefix}" pip >/dev/null 2>&1 || true
  pip_run "${env_prefix}" install --upgrade pip 2>&1 || true

  # 3. Install main requirements
  pip_run "${env_prefix}" install -r "${req_file}" --cache-dir "${PIP_CACHE_DIR}" 2>&1 || echo "WARNING: main requirements install failed for ${env_name}"

  # 4. Header-comment requirements (e.g. flash-attn) — installed with --no-build-isolation
  HEADER_REQS="$(awk '/^[ \t]*$/{ exit } !/^[ \t]*#/{ exit } { gsub(/^[ \t]*#[ \t]*/,""); if(length>0) print }' "${req_file}")"
  if [[ -n "${HEADER_REQS}" ]]; then
    TEMP_REQS="$(mktemp)"
    printf '%s\n' "${HEADER_REQS}" > "${TEMP_REQS}"
    echo "  header reqs: ${HEADER_REQS}"
    pip_run "${env_prefix}" install -r "${TEMP_REQS}" --no-build-isolation --cache-dir "${PIP_CACHE_DIR}" 2>&1 || echo "WARNING: header requirements install failed for ${env_name}"
    rm -f "${TEMP_REQS}"
  fi
done

echo "==== All done ===="
