#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s lastpipe
IFS=$'\n\t'

# ===================== Config =====================
export MAX_JOBS="${MAX_JOBS:-2}"   # used by some build backends (e.g., flash-attn)
REQS_DIR="/mnt/data/ztw_project/simple-eval-model-infer-dev/env_files"
CONDA_ENVS_DIR="/mnt/data/ztw_project/mmeval_envs"
LOG_FILE="environment_setup.log"
PYTHON_VERSION="3.10"
PIP_CACHE_DIR="/mnt/data/.cache/pip"
# ==================================================

log_message() {  # level, message
  local level="$1"; shift
  local message="$*"
  printf '[%(%Y-%m-%d %H:%M:%S)T] [%s] %s\n' -1 "$level" "$message" | tee -a "$LOG_FILE"
}

mkdir -p "${CONDA_ENVS_DIR}" "${PIP_CACHE_DIR}"
: > "$LOG_FILE"
log_message INFO "Starting environment creation/update process..."

# Find all *_requirements.txt files safely (handles spaces/newlines)
find "${REQS_DIR}" -type f -name '*_requirements.txt' -print0 | \
while IFS= read -r -d '' req_file; do
  base_name="$(basename "$req_file")"
  env_name="${base_name%_requirements.txt}"
  env_prefix="${CONDA_ENVS_DIR}/${env_name}"

  log_message INFO "Processing ${env_name} (req: ${req_file})"

  if [[ -d "${env_prefix}" ]]; then
    log_message INFO "Environment path exists: ${env_prefix}. Will update packages."
  else
    log_message INFO "Creating conda env ${env_name} at ${env_prefix} (python=${PYTHON_VERSION})"
    conda create -y -p "${env_prefix}" "python=${PYTHON_VERSION}" | tee -a "$LOG_FILE"
  fi

  # Ensure pip belongs to the target environment (avoid base pip)
  log_message INFO "Ensuring pip is present in ${env_name}"
  conda install -y -p "${env_prefix}" pip >/dev/null 2>&1 || \
    conda install -y -p "${env_prefix}" pip | tee -a "$LOG_FILE"

  # Upgrade pip using the env's Python; disable user site
  log_message INFO "Upgrading pip in ${env_name}"
  if conda run -p "${env_prefix}" \
      env PIP_NO_INPUT=1 PIP_USER=0 PYTHONNOUSERSITE=1 \
          python -m pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"; then
    :
  else
    log_message ERROR "Failed to upgrade pip in ${env_name}"
  fi

  # Install base requirements with progress bar and pinned cache dir
  log_message INFO "Installing packages for ${env_name} from ${req_file}"
  if conda run -p "${env_prefix}" \
      env PIP_NO_INPUT=1 PIP_USER=0 PYTHONNOUSERSITE=1 PIP_CACHE_DIR="${PIP_CACHE_DIR}" MAX_JOBS="${MAX_JOBS}" \
          python -m pip install -r "${req_file}" --progress-bar on --cache-dir "${PIP_CACHE_DIR}" 2>&1 | tee -a "$LOG_FILE"; then
    :
  else
    log_message ERROR "Failed to install packages in ${env_name}"
    # continue to attempt header installs anyway
  fi

  # -------- New behavior: treat header comments as requirement specs --------
  # Extract contiguous top comment lines (before first blank or non-comment line),
  # strip the leading '#', and use non-empty lines as requirement specs.
  HEADER_REQS="$(
    awk '
      /^[ \t]*$/ { exit }                # stop at first empty line
      !/^[ \t]*#/ { exit }               # stop at first non-comment
      { gsub(/^[ \t]*#[ \t]*/, "", $0); if (length($0) > 0) print $0; }
    ' "${req_file}"
  )"

  if [[ -n "${HEADER_REQS}" ]]; then
    # Normalize to one-per-line; ignore pure comment/blank after strip
    TEMP_REQS="$(mktemp)"
    # Write lines exactly as requirement specs (supports extras like "pkg==x.y --only-binary=:all:")
    printf '%s\n' "${HEADER_REQS}" | sed '/^[[:space:]]*$/d' > "${TEMP_REQS}"

    if [[ -s "${TEMP_REQS}" ]]; then
      log_message INFO "Installing header requirement specs for ${env_name}:"
      printf '%s\n' "${HEADER_REQS}" | sed 's/^/  - /' | tee -a "$LOG_FILE"

      # Install header requirements using the env's Python
      if conda run -p "${env_prefix}" \
          env PIP_NO_INPUT=1 PIP_USER=0 PYTHONNOUSERSITE=1 PIP_CACHE_DIR="${PIP_CACHE_DIR}" MAX_JOBS="${MAX_JOBS}" \
              python -m pip install -r "${TEMP_REQS}" --progress-bar on --cache-dir "${PIP_CACHE_DIR}" 2>&1 | tee -a "$LOG_FILE"; then
        :
      else
        log_message ERROR "Header requirement installs failed for ${env_name}"
      fi
    fi
    rm -f "${TEMP_REQS}"
  fi
  # -------------------------------------------------------------------------

  # Quick verification snapshot for common gotcha (networkx visibility)
  conda run -p "${env_prefix}" python - <<'PY' 2>&1 | tee -a "$LOG_FILE" || true
import sys, importlib.util
print("Python:", sys.executable)
print("networkx installed:", importlib.util.find_spec("networkx") is not None)
PY

  log_message SUCCESS "Environment ${env_name} is ready."
  echo "--------------------------------" | tee -a "$LOG_FILE"
done

log_message INFO "All environments have been created/updated."