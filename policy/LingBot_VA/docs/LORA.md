# LingBot-VA 真机 LoRA 方案

本文记录基于 `RMBench_exp/data/real_world/prepared` 真机数据的方案 B。

## 动作表示

保持模型内部 `action_dim=30`。单臂 Franka 数据的 8 个动作维度（`delta_xyz + delta_quaternion_xyzw + gripper_command`）映射到预训练动作通道：

```python
used_action_channel_ids = [0, 1, 2, 3, 4, 5, 6, 28]
```

前 7 个通道对应末端位姿增量，第 28 个通道对应 gripper。其余 22 个通道作为 padding，输入和 target 置零，并通过 `actions_mask=False` 排除动作损失。不要使用 attention 的 token mask 屏蔽这些通道；视频和动作 token 仍需参与 FDM 的因果交互。

`inverse_used_action_channel_ids` 应按上述列表生成，padding 索引为 8。归一化统计 `q01/q99` 保持长度 30，只在有效通道填入真实数据统计量。

数据原始标签为逐步残差/增量动作（`delta_ee_pose_xyzw_and_gripper`）；`observation.state` 为绝对末端位姿及夹爪宽度。旋转增量是相对旋转四元数，不是四元数分量相减；夹爪为命令。训练标签、KV 历史动作和推理输出均保持此表示，不再差分；反归一化恢复动作尺度，不会转成绝对位姿，控制器须按增量解释和执行。

原生 RobotWin 路径使用相对片段起点的位姿，与本方案的逐步增量不同；基础模型全部预训练数据的动作语义未被公开说明完整确定。本次 LoRA 同时训练动作输入/输出层和时间条件分支，以适配增量标签，不要求先转换为绝对位姿。

## LoRA 范围

冻结 VAE、UMT5 和基础 Transformer 权重；RoPE、attention mask、KV cache 沿用原生逻辑，仅训练：

```text
LoRA（rank=16, alpha=16, dropout=0.05；覆盖全部 30 个 Transformer block）
  blocks.*.attn1.to_q
  blocks.*.attn1.to_k
  blocks.*.attn1.to_v
  blocks.*.attn1.to_out.0

直接训练
  action_embedder
  action_proj_out
  condition_embedder_action.time_embedder / time_proj
```

`attn1` 是视频/动作 token 的 self-attention，也是历史 KV cache 的读写位置；`attn2` 是文本 cross-attention，当前方案不对其加 LoRA。KV cache 没有可训练参数，推理时由 `attn1` 保存和读取历史 K/V。

训练使用原生 chunk 因果 mask，不跨 batch 保留 KV；推理逐 chunk 缓存，真实观测回填前清除预测 KV，episode reset 清空缓存。LoRA 不改变缓存布局或 RoPE；采样指标仅评估首个 chunk，不覆盖连续多 chunk 的缓存更新。

部署继承 `frame_chunk_size=2 / action_per_frame=12 / attn_window=72`。`attn_window` 控制历史 KV 容量；每层 token 容量为 `(attn_window // 2) * (视频 chunk token 数 + 动作 chunk token 数)`。当前两路相机每组为 `256+24` 个 token，72 对应 36 组、每层 10,080 个 token；30 层 BF16、CFG 双分支的 K/V 预分配约 6.92 GiB。

`attn_window` 可在部署配置 `va_real_world_lora_cfg.update(...)` 中调整，**无需重训**；重启推理服务后重新分配缓存。同一 checkpoint 可对比 8/16/72；建议先试 8（容量约含当前 chunk 的 6.4 秒、KV 约 0.77 GiB），必要时试 16，当前代码仍为 72。训练另用随机 `chunk_size=1～4 / window_size=4～64`，单样本仅覆盖约 6.4 秒；增大部署缓存不等于模型已学会利用更长历史，需实测效果和延迟。

FFN 保持冻结，这是减少可训练参数的首轮配置，并非 LingBot 的必需限制或已验证最优选择。`attn2` 和 FFN 仍参与前向及梯度传播；如适配不足，可另做 `attn1+FFN` LoRA 对照。

## 数据与训练

每个 task 单独训练一个 adapter：

| task | train episodes | train frames | 首轮步数 |
|---|---:|---:|---:|
| `shell_game` | 45 | 12,291 | 10,000 |
| `put_back_block` | 27 | 10,343 | 10,000 |
| `press_button` | 44 | 20,991 | 10,000 |

使用配置 `real_world_lora_train`，默认 `gpu` 后端：单卡 bf16 + gradient checkpointing，DeepSpeed 管理梯度累积，AdamW 的 FP32 主权重、状态和累积梯度均在 GPU；使用 1 张 A100、8 CPU。VAE/UMT5 离线提取后不参与训练，LoRA dropout 在梯度重算时保持相同随机掩码。

加噪、flow target、timestep 权重、视频/动作损失及因果 attention 复用原生实现；原项目默认为全参数 FSDP 训练。本方案的 LoRA、单卡 DeepSpeed 后端、LeRobot v3 缓存、验证/采样指标及完整恢复是适配扩展；下表 lr、warmup 等为本次实验设置，不代表原项目默认值。

| 参数 | 默认值 |
|---|---|
| batch_size / gradient_accumulation_steps | 8 / 1（单卡有效 batch=8 个窗口） |
| lr / 调度 | 1e-4；100 个 optimizer step 线性 warmup，随后恒定 |
| AdamW betas / eps / weight_decay | (0.9, 0.95) / 1e-8 / 0.1 |
| 梯度裁剪 / CFG 丢弃概率 | 2.0 / 0.1 |
| DataLoader workers / seed | 4 / 0 |
| 保存 / 验证间隔 | 每 500 个 optimizer step，结束时再保存和验证 |

每轮打乱后丢弃不足一个完整 batch 的尾部窗口；验证仍逐窗口覆盖全部验证集。8×1、4×2 与 1×8 保持有效 batch、lr 和更新步数，但随机加噪/attention 窗口按 microbatch 采样，不保证数值轨迹一致；提速与显存以实测为准。完整恢复要求 batch/累积配置一致。

直接读取 `prepared/datasets/real_world-<task>-franka_single-ee-{train,val}`（LeRobot v3），不改变现有划分；验证轨迹分别为 5 / 3 / 5 条。仅训练集计算 q01/q99，验证和推理共用。保留已准备好的 224×224 RGB 裁剪图，两路 `camera_1/camera_2` 各缩放到 256×256 编码，latent 横向拼接，不引入第三路黑图。

每窗口采样 33 帧视频，原始 15 Hz、stride=3，经 VAE 得到 9 个 latent 帧；每个 latent 对齐 12 步原始动作，**动作不降采样、不转成绝对位姿**。窗口步长 48，末尾图像重复补齐；首个 latent 对应的 12 步及越界动作置零并排除损失。窗口不跨 episode。训练损失为视频与有效动作 flow-matching MSE 之和，权重 1:1；每次遍历全部验证窗口，固定随机噪声，关闭 dropout/CFG 丢弃，以 `val_action_loss` 选择 checkpoint（无自动早停）。

## 运行

在 `policy/LingBot_VA` 下，使用已安装依赖的 `lingbot_va` 环境；三个 task 分别执行：

```bash
conda activate lingbot_va
export TASK=shell_game  # 或 put_back_block / press_button
export LINGBOT_VA_BASE_MODEL_PATH="$HOME/work/lingbot-va/models/lingbot-va-base"
python prepare_lora_data.py --task "$TASK" --inspect-only  # CPU 检查，无写入
python prepare_lora_data.py --task "$TASK" \
  --base-model "$LINGBOT_VA_BASE_MODEL_PATH" --output "data/real_world/$TASK"
bash train_lora.sh real_world "$TASK" franka_single ee 0 0 --dry-run
bash train_lora.sh real_world "$TASK" franka_single ee 0 0
```

缓存默认源为 `../../../data/real_world/prepared`，可用 `--prepared-root` 指定；缓存输出目录必须不存在。训练默认使用 `data/real_world/$TASK`，可通过 `LINGBOT_VA_DATASET_PATH` 指定。准备与训练须使用同一基础模型；JSON `--overrides` 可调整 lr、步数等已有超参数。

`train_lora.sh` 及真机 sbatch 默认使用 GPU 优化器；显存不足时可显式传 `--backend cpu_offload`，或设置 `LINGBOT_VA_TRAIN_BACKEND=cpu_offload`。两个后端均支持完整保存/恢复，恢复须使用保存时的后端；GPU 模式不会自动退回 CPU。

输出位于 `checkpoints/<run>/checkpoints/checkpoint_step_<N>/`：`adapter.safetensors` 包含 LoRA 和直接训练的动作分支，`transformer/` 为已合并的推理模型，`real_world_metadata.json` 保存映射与训练统计；完整目录还含 optimizer、调度器、RNG 和采样顺序。恢复时在相同训练命令后加 `--resume-from <checkpoint目录>`；需要逐位复现时设 `load_worker=0`。验证结果见 run 下 `validation.jsonl`。

`metrics.jsonl` 每步记录 video/action flow loss、总 `flow_loss`、梯度范数和 lr；第 1 步、每 500 步及末步记录全验证集 `val_video_loss / val_action_loss / val_flow_loss`。第 1 步、每 100 步及末步另记当前训练窗口的采样动作 L1（MAE）/L2（MSE），验证时评估固定均匀选取的 8 个验证窗口，见 `action_metrics.jsonl / val_action_metrics.jsonl`。按原生初始 chunk 推理：只给首帧观测，先采样视频（25 步、CFG=5），再采样动作（10 步、CFG=1），评估随后 12 步有效 8 维动作；无未来图像/动作输入。指标同时记录归一化误差及 `_physical` 反归一化误差，排除 padding，使用独立固定 RNG，不参与反传；混合物理单位的汇总仅作诊断。

推理使用 `CONFIG_NAME=real_world_lora_train` 启动 `launch_wan_va_server.sh`，从 checkpoint 自动读取上述 metadata；客户端设同名 `config_name`、`action_type: ee`，输入 `camera_1/camera_2`。原图按 640×640 中心裁剪至 224×224，已准备的 224×224 图直接使用；返回 8 维增量 xyz/xyzw/夹爪命令，映射、归一化与 padding mask 与训练一致。
