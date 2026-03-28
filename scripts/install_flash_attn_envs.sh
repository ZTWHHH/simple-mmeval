#!/usr/bin/env bash

set -Eeuo pipefail
shopt -s nullglob
IFS=$'\n\t'

# Usage:
#   bash scripts/install_flash_attn_envs.sh [CONDA_ENVS_DIR] [REQS_DIR]
#
# Defaults are set to this repository's common layout:
#   CONDA_ENVS_DIR = /home/yyang331/william/simple-mmeval/mmeval_envs
#   REQS_DIR       = /home/yyang331/william/simple-mmeval/env_files

DEFAULT_CONDA_ENVS_DIR="/home/yyang331/william/simple-mmeval/mmeval_envs"
DEFAULT_REQS_DIR="/home/yyang331/william/simple-mmeval/env_files"

CONDA_ENVS_DIR="${1:-$DEFAULT_CONDA_ENVS_DIR}"
REQS_DIR="${2:-$DEFAULT_REQS_DIR}"
MAX_JOBS="${MAX_JOBS:-4}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/home/yyang331/william/simple-mmeval/.cache/pip}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$CONDA_ENVS_DIR" ]]; then
  echo "ERROR: CONDA_ENVS_DIR not found: $CONDA_ENVS_DIR" >&2
  exit 1
fi

if [[ ! -d "$REQS_DIR" ]]; then
  echo "ERROR: REQS_DIR not found: $REQS_DIR" >&2
  exit 1
fi

mkdir -p "$PIP_CACHE_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found in PATH." >&2
  exit 1
fi

extract_flash_req() {
  local req_file="$1"
  awk '
    /^[ \t]*$/ { exit }
    /^[ \t]*#/ {
      line=$0
      sub(/^[ \t]*#[ \t]*/, "", line)
      if (line ~ /^flash-attn([[:space:]]*[<>=!~].*)?$/) {
        print line
        exit
      }
      next
    }
    { exit }
  ' "$req_file"
}

pip_run() {
  local env_prefix="$1"
  shift
  conda run -p "$env_prefix" \
    env PIP_NO_INPUT=1 PIP_USER=0 PYTHONNOUSERSITE=1 \
        MAX_JOBS="$MAX_JOBS" PIP_CACHE_DIR="$PIP_CACHE_DIR" \
    python -m pip "$@"
}

count_total=0
count_need=0
count_ok=0
count_fail=0
count_skip_no_flash=0
count_skip_no_env=0

echo "CONDA_ENVS_DIR: $CONDA_ENVS_DIR"
echo "REQS_DIR:       $REQS_DIR"
echo "PIP_CACHE_DIR:  $PIP_CACHE_DIR"
echo "DRY_RUN:        $DRY_RUN"
echo

for req_file in "$REQS_DIR"/*_requirements.txt; do
  count_total=$((count_total + 1))
  base_name="$(basename "$req_file")"
  env_name="${base_name%_requirements.txt}"
  env_prefix="$CONDA_ENVS_DIR/$env_name"

  flash_req="$(extract_flash_req "$req_file" || true)"
  if [[ -z "$flash_req" ]]; then
    echo "[SKIP:NO_FLASH] $env_name"
    count_skip_no_flash=$((count_skip_no_flash + 1))
    continue
  fi
  count_need=$((count_need + 1))

  if [[ ! -d "$env_prefix" ]]; then
    echo "[SKIP:NO_ENV]   $env_name -> missing $env_prefix"
    count_skip_no_env=$((count_skip_no_env + 1))
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN]       $env_name -> $flash_req"
    count_ok=$((count_ok + 1))
    echo
    continue
  fi

  echo "[INSTALL]       $env_name -> $flash_req"
  if pip_run "$env_prefix" install "$flash_req" --no-build-isolation --cache-dir "$PIP_CACHE_DIR"; then
    echo "[OK]            $env_name"
    count_ok=$((count_ok + 1))
  else
    echo "[FAIL]          $env_name"
    count_fail=$((count_fail + 1))
  fi
  echo
done

echo "========== flash-attn installation summary =========="
echo "requirements files scanned: $count_total"
echo "envs requiring flash-attn:  $count_need"
echo "installed successfully:     $count_ok"
echo "install failed:             $count_fail"
echo "skipped (no flash-attn):    $count_skip_no_flash"
echo "skipped (env not found):    $count_skip_no_env"

if (( count_fail > 0 )); then
  exit 2
fi
