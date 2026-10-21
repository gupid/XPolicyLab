# Real Franka π0.5 LoRA

This training-only entry uses the existing OpenPI/JAX environment and official
LoRA model, optimizer, loss and checkpoint implementation. No simulator is used.
It does not use the Aloha input/output transforms or the simulation `franka`
environment (which describes **two** arms). `franka_single` registers one Franka
with the shared robot metadata helpers; its physical control rate is 15 Hz.

## Data and review

Source archives live at `work/RMBench_exp/data/real_world`: shell_game (50
episodes), press_button (53), put_back_block (30). Original archives are read
only. Source-manifest SHA256 values are checked on every HDF5 member. Files use
FurnitureBench `data/demo_0`, not the XPolicyLab trajectory schema, so the general
XPolicyLab LeRobot converters cannot read these fields directly. The dedicated
converter uses the installed **LeRobot 0.4.4 / dataset v3.0** writer, with image
features `observation.images.camera_1`, `camera_2`, plus the standard
`observation.state` and `action` keys. Images are padded/resized to 224×224 with
OpenPI's own image utility; independent review MP4s use larger 480×270 views.

```bash
cd /home/u2023312921/work/RMBench_exp/XPolicyLab/policy/Pi_05
sbatch local_jobs/prepare_real_world.sbatch
```

This creates `data/real_world/prepared/index.html`, a per-task browser review
page, one MP4 per original trajectory (including excluded trajectories), source
provenance reports and separate training/validation datasets. MP4s have both
camera views, frame/time, delta translation and the signed gripper command.
They show the original chronological action-aligned `obs[t]` at 15 fps, with no
deletion of pauses within that stream. For shell_game, reset-time `initial_observations` are restored at the beginning
of review MP4s, followed by an explicitly labeled unrecorded gap (one second
of review hold, not source footage). The action stream retains its original
frame count and overlays. `actions_only_backup/*.actions_only.mp4` files preserve the earlier export
outside `videos/`, which contains only corrected review movies.
Initial frames have no action labels and are not added as supervised action
samples. The final `terminal_observation` is not a training sample. Depth and extra telemetry are not model inputs.

`index.html` can be opened locally after copying the directory; on the HPC,
VS Code Remote / another authenticated remote-file browser can open or download
individual MP4s. No network service or public port is opened by these scripts.
Complete movies alone cannot establish correct control semantics; inspect robot
units, quaternion convention, frame alignment and action overlays too.

The default external JPEG import is `--jpeg-format standard`, explicitly routed
through `decode_image_bit(..., source_format="standard")`. The original files
do not carry XPL JPEG markers. This is an explicit assumption about this external
producer, to be checked in visual review or against its recording code.
`--jpeg-format auto` selects existing XPL marker/legacy behavior instead. No
adapter-side channel swaps occur, and existing decoder defaults are unchanged.

Human review confirmed that press_button's 74-frame episode (09:45 session)
and 1,376-frame episode (11:51 session) are unusable. They are listed in
`prepared/review_exclusions.json` and removed from its training dataset and review
MP4s using LeRobot's official `delete_episodes`; the validation trajectories stay
fixed in the first review. The user subsequently confirmed two further exclusions
(11:51 session episode 0 and 09:41 session episode 0). The final press_button data
have 49 eligible episodes, split into 44 training / 5 validation episodes
(20,991 / 2,397 frames). The full four-member exclusion list is in
`prepared/review_exclusions.json`. Validation source episode IDs are 6, 14, 22,
35 and 44 from the 11:55 session: three digit arrangements 1/2 and two 2/1.
Training has 22 episodes of each arrangement. Source episode 27 was absent from
the original manifest, rather than lost during conversion.
The raw archive and source provenance remain available. shell_game's initial
waiting intervals were confirmed as collection pauses and are not exclusions;
no synthetic waiting actions are trained. Completeness/accepted-command flags
do not annotate task success.

To exclude files after human review, write JSON mapping tasks to **exact source
member paths** from `report.json`, then convert into a new directory:

```bash
source real_world_runtime.sh
JAX_PLATFORMS=cpu "$REAL_PYTHON" prepare_real_world.py \
  --exclude-file /path/to/exclusions.json --output /path/to/new_prepared
```

Output directories are never overwritten. Set `PI05_REAL_ROOT` to the new output
before generating normalization statistics and training. The Slurm preparation
entry and Python converter restore initial context automatically. For existing
completed conversions,
`sbatch local_jobs/finish_real_world.sbatch` restores context and computes all three sets of
training normalization statistics without rebuilding datasets. A deterministic seed-0
split reserves about 10% of whole trajectories: initially 45/5, 48/5 and 27/3.
No trajectory occurs in both splits. Statistics come only from training episodes,
and validation explicitly reuses those same assets.

After manual visual review, shell_game and put_back_block use explicit stratified
splits rather than the original random split. The source-to-variation labels and
validation membership are recorded in `prepared/split_plan.json`:

- shell_game: source episodes 4, 18, 31, 38, 44; two initial-right, one middle,
  two initial-left scenes. Training/validation are 45/5 episodes, 12,066/1,382 frames.
- put_back_block: source episodes 5, 14, 25; one each from right, front and left
  initial mats. Training/validation are 27/3 episodes, 10,343/1,126 frames.

Positions refer to camera 1 image coordinates. Both splits cover every reviewed
variation. Source episode numbers remain in MP4 names; dataset episode indices
are separately reindexed within each split and recorded in `report.json`.
`prepared/split_review/validation.html` shows only the selected validation movies.
These variation labels describe scenes, not successful task completion.

`local_jobs/rebuild_real_world.sbatch` runs CPU-only maintenance with LeRobot's official
`merge_datasets` / `split_dataset`. It constructs a staging root, hashes every
retained frame (both embedded images, numeric labels and time/frame indices),
remaps shell-game initial contexts, recomputes training statistics and tests
the actual datasets before promoting the result. An earlier prepared root is
kept temporarily for rollback. Reusing this script requires inspecting and
updating the explicit review plan for the current source data first. It never
starts training. Previous GPU smoke results are historical after a split change;
current data checks are recorded separately in `prepared/verification.json`.

`prepare_press_button_49.py` builds the separately checked press-button candidate;
`promote_press_button_49.py` promotes only that task's data and assets. This avoids
replacing shell_game while it trains. After the two extra deletions, the gripper
action is constant +1; the action quantiles correctly equal 1. The state gripper
width quantiles amplify some samples to magnitude 47.95. The preparation script
therefore uses training min/max only for the width dimension if its normalized
training magnitude reaches 2; all other statistics remain unchanged. See
`normalization_guards.json` and `prepared/press49_preparation.json`. Running the
generic `norm` command alone will overwrite this additional dataset-specific
guard; rerun the preparation's finish step before using a recomputed asset.
All 13 dataset tests passed after this change, with no skips.

Normalization reads only state/action Parquet columns, since the Franka transform
leaves both numeric fields unchanged. It uses upstream `RunningStats`, the same
8-sample sequential batches, 50-step action windows, episode-end padding and
`drop_last` behavior. This avoids decoding RGB for numeric statistics. Tests
compare projected windows to the official LeRobot loader at every episode end;
a saved full official shell_game normalization run also checks mean, standard
deviation and both quantiles for exact equality. The official model and trainer
are unchanged. One data-dependent guard is applied to gripper dimension 7 only:
if its 1%–99% span is less than 1% of the full observed training range, use the
training min/max as its normalization bounds. press_button's mostly closed
gripper collapses the default quantiles; without this guard its common action
normalizes to roughly 799 and rare open commands to millions. Both its measured
width and signed command require this guard. Mean/std remain upstream results;
shell_game and put_back_block keep their original bounds. The applied changes
are recorded in `normalization_guards.json`. This is a documented data adaptation,
not an upstream-default normalization result for press_button.

Task instructions are defined in `real_world/config.py`. press_button and
put_back_block initially use the existing workspace task descriptions; they
still need confirmation against the real task. shell_game intentionally uses
the provisional label `Complete the shell game task.` until its actual rule is
confirmed. Updating prompts requires rebuilding datasets; no simulated task rule
is assumed to describe the real experiment automatically.

## State/action contract

- State: current measured `[x,y,z,qx,qy,qz,qw,gripper_width_m]` (8D).
- Label: recorded `actions[t] = [dx,dy,dz,dqx,dqy,dqz,dqw,gripper_command]` (8D).
- Translation uses meters in the recorded robot-base convention; rotation is
  xyzw and right-multiplied with the action reference orientation.
- Negative gripper command opens; positive closes. This command is **not** a
  measured width. No future measured qpos is invented as a joint-target label.
- `obs[t]` is the recorded pre-command observation. Source metadata warns the
  controller reference can be older; the converter preserves, rather than
  realigns or silently corrects, this relationship.
- Camera 1 occupies `base_0_rgb`; camera 2 occupies the model's
  `left_wrist_0_rgb` slot. The slot name does not assert that the physical camera
  is wrist-mounted. For press_button and put_back_block, the third slot is zero-filled and masked.
  For shell_game it contains the **first initial camera-1 reveal image** throughout
  each episode, with its mask enabled. This is an explicit additional conditioning
  choice, not a built-in temporal memory feature or a change to the LoRA recipe.
  It preserves initial visual information but does not reconstruct the unrecorded
  cup-placement interval or guarantee tracking during later cup shuffling.
  Training/validation use split-specific, SHA256-checked context JPEGs. Offline or
  future runtime inference must supply the same episode's `initial_scene_rgb`.
  Standard pi0.5 does not remember earlier video frames merely because they were
  prefixed to a review movie. For put_back_block, any stage that requires recalling
  a now-invisible original location has the same limitation; its current two-image
  recipe does not provide explicit historical context. Review this requirement
  before formal training; the smoke test only verifies execution.
- No `DeltaActions`/`AbsoluteActions` transforms are applied: actions already
  encode per-step deltas. Normal OpenPI normalization and 32D model padding apply.
- The default horizon stays 50 steps, about 3.33 s at 15 Hz; it is a prediction
  horizon, not an instruction to execute the whole chunk without replanning.
- Offline inference returns unnormalized delta actions in the same 8D contract.
  This does **not** implement a real robot controller or assert unit-quaternion
  predictions; controller projection, limits, reference timing and execution
  cadence must be explicitly handled and verified before future robot use.

## Defaults, jobs and resume

The adapter copies `pi05_rmbench_lora_observe_and_pickup` using `dataclasses.replace`
and changes dataset/robot transforms, normalization asset paths, base path,
run identity and metadata. The real-world entry now uses an adapter-owned training
loop for training metrics and held-out validation; optimizer updates call the unmodified OpenPI
`train_step`, model and loss. The vendored model and trainer files remain unchanged.
Defaults: batch 8, horizon 50, language rank/alpha 16/16, action-expert rank/alpha
32/32, warmup 250, peak LR 1e-4 → 1e-5 over 10,000 steps, no EMA,
save every 500, keep every 2,500. OpenPI's original freeze filter is retained,
including the original training behavior of parameters outside the LoRA Gemma
modules. This is the same LoRA recipe as the earlier run, not a new adapter-only
parameter filter.

```bash
source real_world_runtime.sh
JAX_PLATFORMS=cpu "$REAL_PYTHON" real_world_entry.py dry-run shell_game
# After conversion, local_jobs/prepare_real_world.sbatch computes these automatically:
JAX_PLATFORMS=cpu "$REAL_PYTHON" real_world_entry.py norm shell_game
# Short test jobs only (three tasks, 2 updates + resume to 3, checkpoint reload):
sbatch local_jobs/verify_real_world.sbatch
# Formal training: submit only after reviewing the data.
sbatch local_jobs/train_real_world.sbatch
```

The formal script submits one 24-hour job and runs three independent training
processes sequentially in this order: shell_game, press_button, put_back_block.
Each task uses one A100, 8 CPUs, 4 data workers, batch size 8 and 10,000 optimizer
updates. One job stays within the GPU QOS submit limit of two jobs per user; the
partition's maximum wall time is one day. If the job is resubmitted after an
interruption, it skips tasks with a final checkpoint and resumes an incomplete
task. Override steps or run identity with `PI05_REAL_STEPS` or `PI05_REAL_RUN_TAG`.
Offline W&B retains upstream logging without sending data to an external service.
Python/W&B temporary files and Hugging Face dataset cache use the work filesystem
under `.cache/pi05_real_runtime/<job_id>/`, avoiding compute-node `/tmp` capacity
failures. Set `PI05_REAL_CACHE_ROOT` to override that cache root.

### 训练期间记录 action L1/L2

真机训练入口默认在完成第 1 步、每 100 步和最后一步更新后，对当前训练批次做一次动作采样。
采用 OpenPI 默认的 10 步去噪；在训练归一化坐标中，仅统计机器人实际动作维度（当前为 8），
排除模型补齐维度及 LeRobot `action_is_pad` 标记的轨迹末尾填充位置。
`train/action_l1` 为逐分量 MAE，`train/action_l2` 为逐分量 MSE，不是欧氏向量范数。
指标是当前批次更新后的采样误差，不是完整训练集或验证集指标，也不参与反向传播。
采样使用独立随机数，不改变原训练噪声、梯度或 DataLoader 样本顺序；该诊断推理不使用训练图像增强。

指标同步写入离线 W&B 和当前 run 的 `action_metrics.csv`，定义记录于 `action_metrics.json`。
CSV 使用已完成的更新数，W&B 沿用 OpenPI 从 0 开始的 step 标签。10,000 步默认产生 101 个采样点。
首次采样需要额外 JIT 编译，此后每次记录也有额外推理开销；实际耗时需 GPU 实测。
可通过 `real_world_entry.py train ... --action-metric-interval 100 --action-metric-denoising-steps 10`
修改频率和去噪步数；续训时会检查指标定义一致。正式训练仍需用户明确启动。

The earlier put_back_block baseline used 8 CPUs with 4 workers, batch 8 and
10,000 updates (first training task in combined job `15315`, followed by
press_button with the same update budget). The earlier shell_game job used
16 CPUs and averaged about 2.3 CPU cores while the GPU was at 100% utilization;
the current three-task formal array therefore requests 8 CPUs per task, retaining
4 workers and batch 8. This is an allocation choice based on observed utilization,
not a controlled throughput comparison.

Checkpoints use the shared naming helper:
`checkpoints/real_world-<task>_lora-franka_single-ee-0/<step>/`.
Smoke jobs have a separate `smoke_<job_id>` run tag and never overwrite formal
output. Default fresh runs reject an existing run directory. Official OpenPI
checkpoint step directories use zero-based labels: 10,000 updates end at `9999`.
Upstream retention keeps long-term checkpoints and the latest checkpoint; it
does not keep every 500-step snapshot indefinitely. Original OpenPI resume
restores model/optimizer state but **does not restore exact data-loader sample
order**. No stronger deterministic-resume guarantee is claimed here.
The wrapper rejects resume if dataset-report/normalization hashes, model,
optimizer, schedule, batch size, seed, base path or initial-context content changed. Rebuilding after
data review therefore requires a new run tag rather than resuming an old run.

### 自动验证与最佳 checkpoint

训练默认在首个更新、每次常规 checkpoint 保存时及最后一步，遍历完整验证集，
记录 `val_flow_loss / val_action_l1 / val_action_l2`。复用训练集归一化统计，关闭图像增强；
每帧固定 flow 噪声/时间及动作初始噪声，不改变训练 RNG 或采样顺序。
flow 沿用原生 32 维目标；动作采用 10 步去噪，L1=MAE、L2=MSE，仅统计真实 8 维及有效时间位置。
结果写入 run 下的 `validation.jsonl` 和离线 W&B，定义见 `validation_protocol.json`。
既有 checkpoint 目录从 0 计数，因此标签 500 对应第 501 次更新。

以最低 `val_flow_loss` 选模，完整最佳状态由独立的上游 checkpoint manager 保存到
`best_checkpoint/<目录标签>/`，指针见 `best_checkpoint.json`；常规保留策略不会删除该最佳状态。
这不引入提前停止，也不改变模型、优化器或训练 loss。验证动作误差与 flow 可能排序不同，
应同时查看 MAE/MSE；这些离线指标不等于真机成功率。

作业 15398 在补齐此流程前已经训练完成。对该轮保留的 2500/5000/7500/9999 四个 checkpoint
补做完整验证，无需重训：

```bash
sbatch local_jobs/backfill_validation.sbatch
```

补评结果明确标记为 `posthoc_backfill_job_15398`；无法恢复已经删除的中间权重。
每个 checkpoint 另存 `held_out_validation.json` 和逐帧指标，完成后自动更新本轮分析图。
原训练源码在修改前按 manifest 校验并归档到 `outputs/validation_backfill_15398/source_snapshot/`。

单 checkpoint 的旧版检查入口仍可使用：

```bash
"$REAL_PYTHON" real_world_entry.py evaluate shell_game \
  --checkpoint checkpoints/real_world-shell_game_lora-franka_single-ee-0/2500
```

This checks inference output and computes fixed-seed validation flow loss on up
to 32 evenly spaced frames. Loss includes the default padded action dimensions,
as in upstream training. It is not a closed-loop task-success metric. The
existing Aloha-oriented XPolicyLab `model.py`/`eval.sh` are not used by this
training-only entry; real robot deployment will need the same Franka transforms
and the explicit controller contract above.

## Paper baseline budgets and queued workflow

All three reviewed real-world baselines now use 10,000 optimization updates,
batch 8, seed 0, the existing 10,000-step cosine schedule, and four data-loader
workers. This is a common update budget, not equal sampling passes or measured
optimal convergence. The earlier 3,000/5,000-step exploratory suggestions have
been superseded by the paper baseline plan.

The training frame counts are 12,066 (shell_game), 20,991 (press_button), and
10,343 (put_back_block). At batch 8, 10,000 updates correspond to approximately
6.63, 3.81, and 7.73 frame-sampling passes. Matching shell_game's pass count would
require about 17,400 press_button updates. Adjacent 15 Hz frames and overlapping
50-step action windows are correlated, so longer episodes alone do not establish
the need for a proportional budget increase. Compare held-out flow loss and real
robot success before a separate extension experiment. The entry's `--steps`
changes the stopping budget only; any longer cosine schedule must be an explicit
configuration decision recorded as a new experiment.

The account permits at most two submitted jobs. While shell_game job `15281_0`
continues, pending put-only job `15314_2` was replaced by workflow job `15315`:

```bash
sbatch local_jobs/train_put_press_baselines.sbatch
```

This requests one A100, 8 CPUs and 24 hours. It first evaluates shell_game
checkpoint 2500 on all 1,382 validation frames, then trains put_back_block for
10,000 updates and compares retained checkpoints, then does the same for
press_button (2,397 validation frames). Tasks use separate Python processes and
keep their existing dataset/config/checkpoint identities. A failed stage is
recorded; a failed task does not prevent the other authorized task from running.
The workflow returns nonzero if any stage fails. Default fresh-run protections
remain in place; this is not an automatic-resume script.

Phase status is appended to `prepared/put_press_workflow_<job>.tsv`; Slurm logs
are `pi05_put_press_<job>.out/.err`. The script passed bash syntax and five isolated
orchestration checks (no model execution). The 49-episode press_button dataset
passed all 13 data checks, and the running shell_game's source, report and
normalization hashes were unchanged before submission. Current scheduler state,
provenance and task plans are in `prepared/STATUS.md`, `training_progress.json`,
`put_press_workflow_submission.json` and `*_baseline_plan.json`.
