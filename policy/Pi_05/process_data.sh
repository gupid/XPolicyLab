#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num_or_raw_task_dirs=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${OPENPI_DATA_MODE:-image}"
raw_data_root="${OPENPI_RAW_DATA_ROOT:-}"
python_bin="${OPENPI_PYTHON:-${POLICY_DIR}/openpi/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "OpenPI Python not found or not executable: ${python_bin}" >&2
  echo "Run bash install.sh first, or set OPENPI_PYTHON." >&2
  exit 1
fi

py_args=(
  "${bench_name}"
  "${ckpt_name}"
  "${env_cfg_type}"
  "${action_type}"
)
if [[ -n "${expert_data_num_or_raw_task_dirs}" ]]; then
  py_args+=("${expert_data_num_or_raw_task_dirs}")
fi
if [[ -n "${raw_task_dirs}" ]]; then
  py_args+=("${raw_task_dirs}")
fi
py_args+=(--mode "${mode}")
if [[ -n "${raw_data_root}" ]]; then
  py_args+=(--raw-data-root "${raw_data_root}")
fi

cd "${POLICY_DIR}/openpi"
"${python_bin}" scripts/process_data.py "${py_args[@]}"
