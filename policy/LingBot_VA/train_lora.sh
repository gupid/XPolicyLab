#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <task_name> <env_cfg_type> <action_type> <seed> <gpu_id> [options]" >&2
  exit 1
fi

# Prepared real-world recipe (docs/LORA.md); retain the standard XPolicyLab CLI.
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LINGBOT_VA_CONFIG_NAME="${LINGBOT_VA_CONFIG_NAME:-real_world_lora_train}"
export LINGBOT_VA_TRAIN_BACKEND="${LINGBOT_VA_TRAIN_BACKEND:-gpu}"
if [[ "${LINGBOT_VA_CONFIG_NAME}" == real_world_lora_train ]]; then
  case "$2" in shell_game|put_back_block|press_button) ;; *) echo "Unknown real-world task: $2" >&2; exit 2 ;; esac
  if [[ "$3" != franka_single || "$4" != ee ]]; then
    echo 'Real-world LoRA requires franka_single ee' >&2
    exit 2
  fi
  export LINGBOT_VA_DATASET_PATH="${LINGBOT_VA_DATASET_PATH:-${POLICY_DIR}/data/real_world/$2}"
fi
exec bash "${POLICY_DIR}/train.sh" "$@"
