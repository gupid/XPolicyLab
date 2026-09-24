#!/bin/bash
set -euo pipefail
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_uv_env=${9:-uv}
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Current Dir
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

if [[ -n "${PI05_DEPLOY_CONFIG:-}" && "${PI05_DEPLOY_CONFIG}" != /* ]]; then
    PI05_DEPLOY_CONFIG="${SCRIPT_DIR}/${PI05_DEPLOY_CONFIG}"
    export PI05_DEPLOY_CONFIG
fi
if [[ -n "${PI05_DEPLOY_CONFIG:-}" && ! -f "${PI05_DEPLOY_CONFIG}" ]]; then
    echo "[MAIN][ERROR] Pi05 deploy config not found: ${PI05_DEPLOY_CONFIG}" >&2
    exit 1
fi

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill -TERM -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] deploy_config=${PI05_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}"
echo "[MAIN] start server, policy_server_port=${policy_server_port}"

setsid bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_uv_env}" \
    "${policy_server_port}" \
    "${policy_server_ip}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo "[MAIN] eval finished"
