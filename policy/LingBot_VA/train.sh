#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [--dry-run]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBODOJO_TEST_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"

export XPOLICYLAB_LEROBOT_DATA_ROOT="${XPOLICYLAB_LEROBOT_DATA_ROOT:-${LEROBOT_DATA_ROOT:-${ROBODOJO_TEST_ROOT}/data}}"
export LEROBOT_DATA_ROOT="${XPOLICYLAB_LEROBOT_DATA_ROOT}"
ckpt_setting=$(python - "${POLICY_DIR}" "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" <<'PY'
import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('checkpoint_resolver', Path(sys.argv[1]).parents[1] / 'utils/checkpoint_resolver.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.build_run_dir_name(dict(zip(('bench_name', 'ckpt_name', 'env_cfg_type', 'action_type', 'seed'), sys.argv[2:]))))
PY
)
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
# Slurm already selected the physical GPU(s). Keep that allocation; a logical
# "0" passed by the user must not expose physical GPU 0 belonging to another job.
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi
export NGPU
NGPU="$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l | xargs)"
export CONFIG_NAME="${LINGBOT_VA_CONFIG_NAME:-robotwin_train}"
if [[ -z "${LINGBOT_VA_DATASET_PATH:-}" ]]; then
  if [[ -n "${LEROBOT_DATASET_REPO_ID:-}" ]]; then
    export LINGBOT_VA_DATASET_PATH="${LEROBOT_DATA_ROOT}/${LEROBOT_DATASET_REPO_ID}"
  else
    export LINGBOT_VA_DATASET_PATH="${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
  fi
fi
export LINGBOT_VA_BASE_MODEL_PATH="${LINGBOT_VA_BASE_MODEL_PATH:-}"
export PYTHONHASHSEED="${seed}"
export LINGBOT_VA_TRAIN_SEED="${seed}"

echo "[LingBot_VA] LEROBOT_DATA_ROOT=${LEROBOT_DATA_ROOT}"
echo "[LingBot_VA] LEROBOT_DATASET_REPO_ID=${LEROBOT_DATASET_REPO_ID:-<unset>}"
echo "[LingBot_VA] config=${CONFIG_NAME}"
echo "[LingBot_VA] dataset=${LINGBOT_VA_DATASET_PATH}"
echo "[LingBot_VA] base_model=${LINGBOT_VA_BASE_MODEL_PATH:-<unset>}"
echo "[LingBot_VA] checkpoint_dir=${ckpt_dir}"
echo "[LingBot_VA] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Run independently of the caller's cwd and the environment's official-repo .pth.
# The adapter seeds single-GPU training; the optional original FSDP entry retains
# its original RNG behavior. Neither passes --seed to the upstream CLI.
export TOKENIZERS_PARALLELISM=false
if [[ " $* " == *" --dry-run "* ]]; then
  exec python "${POLICY_DIR}/train_entry.py" --config-name "${CONFIG_NAME}" --save-root "${ckpt_dir}" "$@"
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
exec python -m torch.distributed.run --nproc_per_node="${NGPU}" \
  --master_port "${MASTER_PORT:-29501}" --tee 3 \
  "${POLICY_DIR}/train_entry.py" --config-name "${CONFIG_NAME}" --save-root "${ckpt_dir}" "$@"
