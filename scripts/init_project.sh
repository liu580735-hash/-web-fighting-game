#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/d/sr_project}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

echo "[INFO] Initializing project at: ${PROJECT_ROOT}"

mkdir -p \
  "${PROJECT_ROOT}/data/DIV2K_train_HR" \
  "${PROJECT_ROOT}/data/DIV2K_valid_HR" \
  "${PROJECT_ROOT}/data/Set5" \
  "${PROJECT_ROOT}/data/Set14" \
  "${PROJECT_ROOT}/datasets" \
  "${PROJECT_ROOT}/models" \
  "${PROJECT_ROOT}/utils" \
  "${PROJECT_ROOT}/scripts" \
  "${PROJECT_ROOT}/configs" \
  "${PROJECT_ROOT}/checkpoints" \
  "${PROJECT_ROOT}/results/images" \
  "${PROJECT_ROOT}/results/curves" \
  "${PROJECT_ROOT}/report"

copy_template_file() {
  local rel="$1"
  cp "${REPO_ROOT}/${rel}" "${PROJECT_ROOT}/${rel}"
}

copy_template_file "train.py"
copy_template_file "test.py"
copy_template_file "README.md"
copy_template_file "requirements.txt"
copy_template_file "datasets/sr_dataset.py"
copy_template_file "models/srresnet.py"
copy_template_file "utils/io.py"
copy_template_file "utils/metrics.py"
copy_template_file "scripts/download_data.py"

if [[ -f "${SCRIPT_DIR}/init_project.sh" ]]; then
  cp "${SCRIPT_DIR}/init_project.sh" "${PROJECT_ROOT}/scripts/init_project.sh"
fi
if [[ -f "${SCRIPT_DIR}/init_project.ps1" ]]; then
  cp "${SCRIPT_DIR}/init_project.ps1" "${PROJECT_ROOT}/scripts/init_project.ps1"
fi

if [[ -f "${REPO_ROOT}/.gitignore" ]]; then
  cp "${REPO_ROOT}/.gitignore" "${PROJECT_ROOT}/.gitignore"
fi

echo "[INFO] Done. Created directories and copied template code files."
echo "[INFO] Tree root: ${PROJECT_ROOT}"
