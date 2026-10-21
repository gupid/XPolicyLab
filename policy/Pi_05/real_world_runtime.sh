#!/usr/bin/env bash
# Source from the sbatch scripts or an interactive shell; never change Slurm GPU visibility.
REAL_POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_WORKSPACE="$(cd "${REAL_POLICY_DIR}/../../.." && pwd)"
export PI05_REAL_ROOT="${PI05_REAL_ROOT:-${REAL_WORKSPACE}/data/real_world/prepared}"
export HF_LEROBOT_HOME="${PI05_REAL_ROOT}/datasets"
export OPENPI_PI05_CHECKPOINT="${OPENPI_PI05_CHECKPOINT:-${REAL_WORKSPACE}/.cache/openpi/openpi-assets/checkpoints/pi05_base/params}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
# Compute-node /tmp is shared and can be full even when login-node /tmp is free.
# Set this before Python imports: wandb creates its media temp directory on import.
REAL_LOCAL_CACHE="${PI05_REAL_CACHE_ROOT:-${REAL_WORKSPACE}/.cache/pi05_real_runtime}/${SLURM_JOB_ID:-local}"
export TMPDIR="${REAL_LOCAL_CACHE}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HF_DATASETS_CACHE="${REAL_LOCAL_CACHE}/hf"
export WANDB_CACHE_DIR="${REAL_LOCAL_CACHE}/wandb-cache"
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${WANDB_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="${REAL_WORKSPACE}/.cache/jax_pi05_real"
export LD_LIBRARY_PATH="${REAL_POLICY_DIR}/openpi/.venv/lib:${REAL_POLICY_DIR}/../../../../../miniforge3/envs/pi05py/lib:/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"
REAL_PYTHON="${REAL_POLICY_DIR}/openpi/.venv/bin/python"
cd "${REAL_POLICY_DIR}"
