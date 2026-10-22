# LingBot_VA

**Contributor:** RoboDojo Team | **Paper:** LingBot-VA technical report | **arXiv:** TBD | **Original code:** See vendored `lingbot_va/`.

`LingBot_VA` adapts the LingBot-VA policy to XPolicyLab/RoboDojo; it trains on precomputed Wan2.2 VAE latents rather than raw LeRobot parquet/video. Integration scripts live at this directory level; the vendored upstream implementation lives in `lingbot_va/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/LingBot_VA
bash install.sh
conda activate <policy_env>  # e.g. lingbot_va
```

On the current HPC, reuse the verified `lingbot_va` environment; no reinstall is
needed. The CPU offload backend requires DeepSpeed 0.18.5 and its CPU Adam native
extension (a C++ compiler is required). The cluster sbatch examples source the
existing `local_setup/runtime_env.sh` for compiler/runtime paths and the extension
cache. Environment versions are recorded in the standalone checkout's
`local_setup/pip-freeze.txt` and `conda-explicit.txt`.

## Data Processing

`process_data.sh` takes a standard RoboDojo LeRobot **v2.1** dataset (parquet + per-camera mp4, in the keys of `XPolicyLab/scripts/transform_lerobot_v21_format.py` — [Official LeRobot conversion](../../README.md#official-lerobot-conversion)) and runs every upstream step to produce a training-ready latent dataset: it maps actions into the 30-dim layout — left/right arm EEF (7+7), left/right arm joints (7+7), left/right gripper (1+1), with missing dimensions zero-padded — adds an `action_config` segment to each `meta/episodes.jsonl` line, encodes Wan2.2 VAE video latents into a `latents/` tree (videos resized to ~256×256 and downsampled to 5–15 fps, matching `va_robotwin30_train_cfg`), and writes `empty_emb.pt` (the Wan2.2 text embedding of an empty string, used when classifier-free guidance drops language conditioning) at the dataset root. The base model supplies the VAE and text encoder, so `LINGBOT_VA_BASE_MODEL_PATH` must be set.

```bash
cd XPolicyLab/policy/LingBot_VA
export LEROBOT_DATA_ROOT=<parent_dir_of_lerobot_datasets>      # or LINGBOT_VA_SOURCE_DATASET=<full_dataset_path>
export LEROBOT_DATASET_REPO_ID=<source_dataset_folder_name>
export LINGBOT_VA_BASE_MODEL_PATH=<path_to_lingbot-va-base>

bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Example: build the latent dataset (optional trailing episode count; omit or 0 for all)
bash process_data.sh RoboDojo cotrain arx_x5 joint
```

- The output (default `data/<bench>-<ckpt>-<env>-<action>/`, override with `LINGBOT_VA_DATASET_PATH`) mirrors `videos/` under `latents/`, with one latent file per camera in the config's `obs_cam_keys` (`cam_high`, `cam_left_wrist`, `cam_right_wrist` for RoboDojo); the trainer skips any segment whose latent files are missing for a camera.
- Latent files are named `episode_{index}_{start_frame}_{end_frame}.pth`, matching the `action_config` segments in `episodes.jsonl` (e.g. segment `0–450` → `episode_000000_0_450.pth`). `process_data.sh` writes one segment per episode (`start_frame=0`, `end_frame=length`, `action_text` = the episode task); for multiple sub-tasks per episode, use one entry (and one latent file per camera) per segment, and regenerate the matching latent files if you edit segments manually.
- The latent `.pth` field schema and full upstream data-prep details are in `lingbot_va/README.md` § Custom Dataset Preparation and § Post-Training LingBot-VA.

## Training

The default backend is **single-GPU CPU optimizer offload** (`cpu_offload`).
`train_cpu_offload.py` reuses the pinned official transformer, activation
checkpointing, dataset loader, noise schedulers, input preparation and weighted
video/action loss. GPU model weights and forward/backward computation use BF16;
CPU master weights, AdamW states and gradient accumulation use FP32. The selected
recipe still supplies learning rate, betas, weight decay, warmup, batch size,
accumulation and step count; gradient clipping remains 2.0. DeepSpeed performs
accumulation/clipping/updates and the official loss is divided by accumulation
**exactly once**. This backend is not a bitwise equivalent FSDP implementation.

Use a prepared latent dataset containing `latents/`, `empty_emb.pt` and
`action_config` in `meta/episodes.jsonl`:

```bash
cd XPolicyLab/policy/LingBot_VA
conda activate lingbot_va
source /home/u2023312921/work/lingbot-va/local_setup/runtime_env.sh
export LINGBOT_VA_DATASET_PATH=/absolute/path/to/prepared_dataset
export LINGBOT_VA_BASE_MODEL_PATH=/home/u2023312921/work/lingbot-va/models/lingbot-va-base
export LINGBOT_VA_CONFIG_NAME=demo_train
export LINGBOT_VA_ENABLE_WANDB=0  # explicit logging override

# Check resolved config/paths/backend without GPU work or writing checkpoints:
bash train.sh RealRobot run01 robot_name joint 0 0 --dry-run

# Cluster: one A100, 16 CPUs (create log directory before sbatch):
mkdir -p .logs
sbatch local_jobs/train_cpu_offload_a100.sbatch RealRobot run01 robot_name joint 0 \
  --overrides /absolute/path/to/robot_training.json
```

The `env_cfg_type` and `action_type` labels do not infer robot metadata or modify
normalization. The robot JSON must explicitly override the selected config's
camera names, active action channel mapping, inverse mapping, normalization and
other relevant data settings. Use only existing config keys; unknown keys fail.
An optional JSON file may also set `learning_rate`, `num_steps`,
`gradient_accumulation_steps`, `save_interval`, `load_worker`, etc. All overrides
and precision/backend settings are recorded in `resolved_training_config.json`.
Explicit model/data environment variables take precedence over JSON paths.

The recipe default remains official `robotwin_train`, matching the original
upstream launcher. `demo_train` and `libero_train` are other official templates.
The old `robotwin30_train` is now an explicit XPolicyLab extension, not an official
default. None of these is a generic real-robot configuration: action mappings,
cameras and normalization need adaptation. The existing data converter currently
assumes dual arms with 6 joints + gripper each, and three fixed camera names;
real-data conversion must be checked against the actual robot before training.

The positional seed initializes Python/numpy/torch and the shuffled sampler for
CPU offload. The model is loaded from the specified release and all transformer
parameters are trainable in the default full-finetune recipe. The original official
FSDP path remains available with `--backend fsdp` (or
`LINGBOT_VA_TRAIN_BACKEND=fsdp`), normally on multiple GPUs. Its original
initialization, optimizer, checkpoint and RNG behavior are unchanged; the seed
label on that path does not add torch/numpy seeding. No single-A100 capacity claim
is made for that path.

### Three-task LoRA training

数据准备、训练参数与推理配置见 [LoRA 方案文档](docs/LORA.md)。

`train_lora.sh` defaults to the `gpu` backend (GPU AdamW, no CPU optimizer offload)
and selects `real_world_lora_train`: rank 16, alpha 16, dropout 0.05,
self-attention q/k/v/out LoRA, and fully trainable action input/output and timestep
projections. It reads frozen latent caches extracted from the reviewed LeRobot v3
splits under `data/real_world/prepared`; action padding is excluded from the loss.
Use `--backend cpu_offload` when GPU memory is insufficient. Both single-GPU
backends share training, metrics and full resume; resume with the saved backend.

```bash
export LINGBOT_VA_BASE_MODEL_PATH=/absolute/path/to/lingbot-va-base
TASK=shell_game  # or press_button / put_back_block; run each task independently
python prepare_lora_data.py --task "$TASK" --base-model "$LINGBOT_VA_BASE_MODEL_PATH" \
  --output "data/real_world/$TASK"
bash train_lora.sh real_world "$TASK" franka_single ee 0 0
```

Each task receives its own checkpoint run directory. The three runs should use
the same base model and latent preprocessing settings; only the dataset path
and run name change.
For the existing single-A100 Slurm setup, the equivalent wrapper is:

```bash
sbatch local_jobs/train_lora_real_world.sbatch real_world shell_game franka_single ee 0
```

Checkpoints use the shared XPolicyLab run name:
`checkpoints/<bench>-<run>-<robot>-<action>-<seed>/checkpoints/checkpoint_step_N/`.
Each completed CPU offload checkpoint contains a normal Diffusers `transformer/`
for inference or a new finetune, and `deepspeed/` with model, FP32 master weights,
optimizer and learning-rate scheduler states plus optimizer step, RNG and sampler
state. `COMPLETE` appears only after the entire checkpoint is written, and the
checkpoint is published by an atomic directory rename. Existing checkpoints are
never overwritten. Saves happen at optimizer boundaries, periodically and at the
end. CPU offload checkpoints require substantially more disk than the ~10 GB
BF16 transformer export because FP32 master weights and Adam moments are saved.

```bash
# Same data/config/seed; num_steps is the final target step, not additional steps.
sbatch local_jobs/train_cpu_offload_a100.sbatch RealRobot run01 robot_name joint 0 \
  --overrides /absolute/path/to/robot_training.json \
  --resume-from /absolute/path/to/checkpoint_step_N
```

Resume validates the training configuration and dataset identity/size. Keep data
contents unchanged; this is not a content hash of all videos/latents. Exact worker
RNG restoration is supported with `load_worker=0`; with worker processes, model,
optimizer and sample order are restored but worker-local random conditioning may
differ. For exact restart comparisons, use zero workers. The data initialization
pool is bounded by the allocation (`--dataset-init-workers`); this only changes
loading resources, not the dataset's item processing.

Host RAM matters: the earlier full-model CPU offload smoke test peaked near
120 GiB RSS. Allow approximately 160 GiB available host memory for headroom.
Long clips or more camera pixels still require more GPU memory. Slurm GPU
visibility is preserved, so logical GPU 0 does not override an allocated physical
GPU. The shell adapter pins imports to its vendored source, resolves model/data
paths explicitly, uses the shared checkpoint naming helper and preserves W&B
credentials. Supply the upstream-required W&B environment variables or explicitly
disable it as above.

### Upstream consistency and backend tests

`upstream_training_manifest.json` pins official commit
`7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`. Every launch checks the source hashes.
34 official source/script files match byte for byte. Five adapter exceptions are
recorded: the pre-existing robotwin30 config and inference WebSocket client, plus
the LoRA training integration and its config registration. No inference
equivalence is claimed.

```bash
python audit_upstream.py
python audit_upstream.py --reference /home/u2023312921/work/lingbot-va \
  --report outputs/reports/upstream_parity_report.json
mkdir -p .logs
sbatch local_jobs/verify_cpu_offload_a100.sbatch
```

The CPU audit passed all 11 shared config comparisons and exact noisy input,
weighted loss and prediction-gradient comparisons for seeds 0, 42 and 20261019.
The GPU backend test checks accumulation scaling against an independent AdamW
update, and compares an uninterrupted tiny model run against a checkpointed and
resumed run, including FP32 master weights, BF16 model weights, learning-rate
scheduler and sample order. It also reloads the Diffusers export. The Slurm script
then runs one update of the released full model through the normal `train.sh`
entry, with explicitly labeled synthetic cached latents and no full-size smoke
checkpoint. Tests never substitute synthetic data when a real dataset is missing.
These checks do not validate a real-robot data conversion or task success rate.

Verification on 2026-10-19, Slurm job **15215** (one A100-PCIE-40GB, two allocated
CPUs): all six tiny-model integration checks passed. The resumed run's FP32
master weights differed from the uninterrupted run by **0.0**, and the BF16 model
weights/export matched exactly (`load_worker=0`, accumulation=2). The full
5,088,872,670-parameter release then completed one optimizer update through
`train.sh`; every sampled input/attention/output weight changed. Its training
phase allocated peak was 10.71 GiB, CUDA reserved memory 18.97 GiB and host peak
RSS 123.36 GiB. DeepSpeed initialization separately reported an 18.96 GiB
allocation peak. The timed backend body took 140.91 s, excluding Python imports
and Slurm startup. Full-size checkpoint saving was skipped for that synthetic
smoke; checkpoint/save/restore/export functionality was exercised on the tiny
model. Summary: `outputs/reports/cpu_offload_verification.json`; raw logs:
`.logs/offload_15215.{out,err}`.

## Evaluation

```bash
cd XPolicyLab/policy/LingBot_VA
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

`eval_batch` is `false` and must stay that way: `wan_va_server` keeps one global KV/VAE/`frame_st_id` cache, so one process can serve only one env. `get_action_batch` raises `NotImplementedError`.

## Configuration

`deploy.yml` keys to check before evaluation: `env_cfg`, `protocol`, `config_name`, `checkpoint_path`, `base_model_path`, `rollout_mode`, `result_dir`, `va_server_host`, `va_server_port`, `obs_transform_pipeline`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `LEROBOT_DATA_ROOT` | Parent directory that holds LeRobot dataset folders. |
| `LEROBOT_DATASET_REPO_ID` | Source dataset folder name under `LEROBOT_DATA_ROOT`; ignored when `LINGBOT_VA_DATASET_PATH` is set. |
| `LINGBOT_VA_SOURCE_DATASET` | Full path to the source LeRobot dataset; overrides the two variables above. |
| `LINGBOT_VA_DATASET_PATH` | Output path for the prepared latent dataset (default `data/<tag>`); also the dataset consumed by training. |
| `LINGBOT_VA_BASE_MODEL_PATH` | **Required:** `lingbot-va-base` weights (VAE + text encoder); also used at eval when `deploy.yml` `base_model_path` is null. |
| `LINGBOT_VA_CONFIG_NAME` | Training recipe; default official `robotwin_train`. `robotwin30_train` is an explicit XPolicyLab extension. |
| `LINGBOT_VA_TRAIN_BACKEND` | Default `cpu_offload` for one GPU; `fsdp` selects the unmodified official trainer. |
| `LINGBOT_VA_ENABLE_WANDB` | Optional explicit `0`/`1` override; otherwise preserve the selected official config. |
| `LINGBOT_VA_PROCESS_GPU` | GPU id used for latent encoding (default `0`). |
| `LINGBOT_VA_TARGET_FPS` | Target sampling fps for latents (default `10`). |
| `LINGBOT_VA_VA_HOST` / `LINGBOT_VA_VA_PORT` | Point evaluation at an already-running wan_va backend instead of auto-launching one. |

## Notes

- **Latent pipeline is mandatory:** do not point training at a plain LeRobot dataset without `latents/` and `empty_emb.pt`; run Data Processing first.
- `eval.sh` auto-launches the wan_va backend (which holds the real weights). It requires a base model path: set `LINGBOT_VA_BASE_MODEL_PATH` or `base_model_path` in `deploy.yml`.
