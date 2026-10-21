# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

## Data Processing

For the three real single-arm Franka tasks (`shell_game`, `press_button`,
`put_back_block`), use [REAL_WORLD.md](REAL_WORLD.md). Those FurnitureBench HDF5
files require the separate 15 Hz EE-delta converter and LoRA entry; the Aloha
commands below do not describe their action space.

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The dataset uses the official keys — `observation.state`, `action`, `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` ([official LeRobot conversion](../../README.md#official-lerobot-conversion)); the bundled script exists because conversion must run inside openpi's own pinned LeRobot environment, which sets the dataset version. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls

# RMBench observe_and_pickup data collected in data/XPolicyLab/
bash process_data.sh XPolicyLab observe_and_pickup aloha_agilex joint
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

### RMBench base vs. LoRA evaluation

The RMBench-specific YAML files keep the two comparisons reproducible:

- `deploy_rmbench_base.yml` loads the downloaded official `pi05_base` parameters.
- `deploy_rmbench_lora.yml` loads the fine-tuned run and selects its final checkpoint.

Both use the same three-camera observation adapter, 14-D `aloha_agilex` joint state,
and the RMBench normalization statistics. Select one YAML with
`PI05_DEPLOY_CONFIG`; the variable is inherited by both the policy server and the
environment client.

```bash
cd XPolicyLab/policy/Pi_05
export EVAL_ENV_TYPE=sim

# Unadapted official pi05_base.
export PI05_DEPLOY_CONFIG=deploy_rmbench_base.yml
bash eval.sh RMBench observe_and_pickup base aloha_agilex joint 0 0 0 uv RMBench

# LoRA fine-tuned checkpoint (the YAML currently selects step 9999).
export PI05_DEPLOY_CONFIG=deploy_rmbench_lora.yml
bash eval.sh RMBench observe_and_pickup lora aloha_agilex joint 0 0 0 uv RMBench
```

The `base`/`lora` value is only the XPolicyLab run label in these two configs; the
actual weights come from `model_path`. If training stops at a different final step,
change `checkpoint_num` in `deploy_rmbench_lora.yml` before evaluation. Keep task,
robot, action type, seed, environment mode, and simulator episode settings identical
between the two commands when comparing results. Results are written to the separate
`eval_result/observe_and_pickup/Pi_05/demo_clean/base/` and
`eval_result/observe_and_pickup/Pi_05/demo_clean/lora/` directories by the RMBench
client. The YAML `result_dir` fields are retained for adapter metadata but are not
currently consumed by that client.

For a split-machine run, export the same `PI05_DEPLOY_CONFIG` on both sides. Start
the policy server with the YAML selected and a reachable host, then start the client
with the server IP as described in the XPolicyLab deployment documentation.

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`, `model_path`, and `norm_stats_path`. `norm_stats_path` may point to either the directory containing `norm_stats.json` or the JSON file itself.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_RAW_DATA_ROOT` | Overrides the exact raw bench directory read by `process_data.sh` (for this dataset, the default is `<RMBench>/data/XPolicyLab`). |
| `OPENPI_NUM_TRAIN_STEPS` | Overrides the config's training-step count; useful for a short smoke run. |
| `OPENPI_WANDB_ENABLED` | Set to `false` to disable W&B entirely; defaults to `true`. |
| `OPENPI_OVERWRITE` | Whether to replace an existing run directory; defaults to `true` for backward compatibility. |
| `OPENPI_RESUME` | Set to `true` to resume an existing run; cannot be combined with overwrite. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |
| `PI05_DEPLOY_CONFIG` | Evaluation YAML override; relative paths resolve from `policy/Pi_05/`. |
| `PI05_NORM_STATS_PATH` | Optional model-side override for normalization stats when the YAML does not set `norm_stats_path`. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.

## RMBench 40GB LoRA configuration

The vendored OpenPI code includes a dedicated `pi05_rmbench_lora_observe_and_pickup`
config for a single 40GB A100. It uses the official `pi05_base` checkpoint, freezes
the base Gemma parameters, and trains LoRA adapters in both the language and action
expert modules. The config assumes a 14D ALOHA-style joint state/action padded to
the model's 32D action space and the three OpenPI camera slots (`cam_high`, both
wrist cameras). It is intentionally a configuration-only addition; `train.sh` does
not select it unless `OPENPI_TRAIN_CONFIG_NAME` is set.

Convert the HDF5 data, compute normalization statistics for that same LeRobot repo,
and then run training. Keep the `bench_name`, task, environment, and action arguments
identical so all three stages resolve `XPolicyLab-observe_and_pickup-aloha_agilex-joint`:

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh XPolicyLab observe_and_pickup aloha_agilex joint

export OPENPI_TRAIN_CONFIG_NAME=pi05_rmbench_lora_observe_and_pickup
export OPENPI_RMBENCH_REPO_ID=XPolicyLab-observe_and_pickup-aloha_agilex-joint
export OPENPI_LEROBOT_REPO_ID="$OPENPI_RMBENCH_REPO_ID"

cd openpi
uv run scripts/compute_norm_stats.py --config-name "$OPENPI_TRAIN_CONFIG_NAME"
cd ..

bash train.sh XPolicyLab observe_and_pickup aloha_agilex joint 0 0
```

The current training implementation logs to Weights & Biases, not TensorBoard.
