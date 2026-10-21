"""Stage, validate and promote reviewed data splits; never launch training."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from real_world.config import TASKS, repo_id


def official_reference(root):
    from real_world.config import register_configs
    from real_world_entry import upstream_script
    register_configs()
    upstream_script("compute_norm_stats").main("pi05_real_shell_game_train")
    stats = root / "assets/pi05_real_shell_game_train" / repo_id("shell_game") / "norm_stats.json"
    shutil.copy2(stats, root / "official_shell_norm_reference.json")
    (root / "official_shell_norm_reference_source.json").write_text(json.dumps({
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "method": "unmodified openpi/scripts/compute_norm_stats.py; full reviewed training split; batch 8; horizon 50; workers 4; no max_frames",
        "dataset_report_sha256": hashlib.sha256((root / "shell_game/report.json").read_bytes()).hexdigest()}, indent=2))


def update_reports(root, tasks, history):
    recommendations_path = root / "training_recommendations.json"
    if recommendations_path.exists():
        recommendations = json.loads(recommendations_path.read_text())
    else:
        budget = {"shell_game": 3000, "press_button": 5000, "put_back_block": 3000}
        recommendations = {"recommendations": [
            {"task": task, "suggested_first_budget_steps": budget[task]} for task in tasks
        ]}
    reports = {t: json.loads((root / t / "report.json").read_text()) for t in TASKS}
    for row in recommendations["recommendations"]:
        report = reports[row["task"]]
        row["counts"] = report["counts"]
        frames = report["counts"]["train"]["frames"]
        row["sample_epoch_equivalent"] = row["suggested_first_budget_steps"] * 8 / frames
        row["default_10000_epoch_equivalent"] = 10000 * 8 / frames
    (root / "training_recommendations.json").write_text(json.dumps(recommendations, indent=2))
    text = ["# 真机数据准备状态", "", "数据按完整轨迹划分训练集和验证集；报告中的帧数包含已并入 shell_game 开头的观察等待帧。", "",
            "| 任务 | 可用轨迹 | 训练 / 验证轨迹 | 训练 / 验证动作帧 | 首轮建议 steps |", "|---|---:|---:|---:|---:|"]
    for row in recommendations["recommendations"]:
        t = row["task"]
        c = reports[t]["counts"]
        text.append(f"| {t} | {c['train']['episodes'] + c['val']['episodes']} | {c['train']['episodes']} / {c['val']['episodes']} | {c['train']['frames']} / {c['val']['frames']} | {row['suggested_first_budget_steps']} |")
    text += ["", "验证集按完整轨迹手动分层选取，约 10%；同一条轨迹的动作窗口不跨入另一集合。训练和验证均覆盖所有已审核的主要变化。位置名称使用相机 1 画面坐标。", ""]
    gallery = []
    for task in tasks:
        r = reports[task]
        text += [f"## {task}", "", f"变化分布：训练 `{r['variation_counts']['train']}`；验证 `{r['variation_counts']['val']}`。", "", "验证源文件：", ""]
        for row in r["episodes"]:
            if row["split"] != "val":
                continue
            text.append(f"- `{Path(row['source']).name}` — {row['variation']}")
            video = "../" + task + "/" + row["video"]
            gallery.append(f'<article><p>{html.escape(Path(row["source"]).name)}<br>{html.escape(row["variation"])}</p><video controls preload="none" src="{video}"></video></article>')
        text.append("")
    text += ["采用 LeRobot 官方 merge_datasets / split_dataset 重建，并重新连续编号；所有保留帧的两路图片字节、状态、动作、帧号及时间戳逐轨迹 SHA256 均与重建前一致，结果见各任务 resplit_integrity.json。审核视频保留原文件名。", "",
             "shell_game 的观察等待帧已合并到对应主轨迹开头；每帧使用主轨迹首帧 state，action 使用 [0,0,0,0,0,0,1,-1]。没有保存独立初始帧。", "",
             "训练图像从原始 1280x720 RGB 帧中心裁剪 640x640，再缩放至 224x224。相机槽仅输入 camera_1 和 camera_2；第三槽填黑图并以 mask 标记无效。", "",
             "所有任务的归一化统计只用新训练集重算；验证集共用对应训练统计。press_button 原有夹爪维度归一化修复保留。", "",
             "数据为 LeRobot v3.0，15 Hz，动作是每步末端增量 xyz + xyzw 旋转增量 + 夹爪命令。输入、标签完整性检查不等于任务成功标注。", "",
             "训练脚本使用 1 张 A100、16 CPU、4 DataLoader workers、batch 8；默认 10000 步，建议首轮步数见 training_recommendations.json。训练中尚无自动验证和自动早停。", "",
             "本次执行数据重建、训练集归一化与 CPU 数据检查；未启动模型训练。检查结果见 verification.json 和 data_test_results.xml。", ""]
    (root / "STATUS.md").write_text("\n".join(text))
    (root / "split_review").mkdir(exist_ok=True)
    (root / "split_review/validation.html").write_text('<!doctype html><meta charset="utf-8"><title>Reviewed validation episodes</title><style>body{background:#171717;color:white;font:16px sans-serif;margin:24px}video{width:960px;max-width:100%}</style><h1>Reviewed validation episodes</h1>' + "\n".join(gallery))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--official-reference", action="store_true")
    parser.add_argument("--promote-to", type=Path)
    args = parser.parse_args()
    if args.official_reference:
        official_reference(args.root)
        return
    plan = json.loads(args.plan.read_text())
    tasks = tuple(plan["tasks"])
    job = os.environ.get("SLURM_JOB_ID", "local")
    source = args.root.resolve()
    target = source.with_name(f"{source.name}.rebuild_{job}")
    backup = source.with_name(f"{source.name}.before_rebuild_{job}")
    if target.exists() or backup.exists():
        raise FileExistsError("Inspect previous rebuild before retrying")
    policy = Path(__file__).resolve().parents[1]
    def run(argv, env):
        subprocess.run([sys.executable, *argv], cwd=policy, env=env, check=True)

    env = os.environ.copy()
    run(["-m", "real_world.resplit", "--source", str(source), "--target", str(target), "--plan", str(args.plan)], env)
    env.update(PI05_REAL_ROOT=str(target), HF_LEROBOT_HOME=str(target / "datasets"))
    history = target / "history" / f"before_resplit_{job}"
    for name in ("STATUS.md", "training_recommendations.json"):
        if (target / name).exists():
            shutil.copy2(target / name, history / name)
    for task in tasks:
        run(["real_world_entry.py", "norm", task], env)
    if "shell_game" in tasks:
        run(["-m", "real_world.rebuild", "--root", str(target), "--official-reference"], env)
        run(["real_world_entry.py", "norm", "shell_game"], env)
    update_reports(target, tasks, history)
    run(["-m", "pytest", "-q", "tests/test_real_world.py", f"--junitxml={target / 'data_test_results.xml'}"], env)
    result = ET.parse(target / "data_test_results.xml").getroot().find("testsuite").attrib
    verification = json.loads((target / "verification.json").read_text())
    verification.update(job_id=job, data_checks="passed", pytest=result,
                        full_frame_integrity="All bytes of both images, numeric labels, frame indices and timestamps match before/after for every retained frame",
                        normalization="Recomputed from new training data only; full official shell reference matches exactly")
    (target / "verification.json").write_text(json.dumps(verification, indent=2))
    if args.promote_to:
        destination = args.promote_to.resolve()
        destination_backup = destination.with_name(f"{destination.name}.before_crop640_waitmerge_{job}")
        if destination_backup.exists():
            raise FileExistsError(destination_backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.rename(destination_backup)
        try:
            target.rename(destination)
        except BaseException:
            if destination_backup.exists():
                destination_backup.rename(destination)
            raise
        if destination_backup.exists():
            for old_frames in (
                destination_backup / "shell_game" / "context",
                destination_backup / "shell_game" / "initial_reveal_contact_sheet.jpg",
                destination_backup / "shell_game_initial_vs_actions.jpg",
            ):
                if old_frames.is_dir():
                    shutil.rmtree(old_frames)
                elif old_frames.exists():
                    old_frames.unlink()
        shutil.rmtree(source)
        print(f"PROMOTED {destination}; rollback copy {destination_backup}; {result}", flush=True)
    else:
        source.rename(backup)
        try:
            target.rename(source)
        except BaseException:
            backup.rename(source)
            raise
        print(f"PROMOTED {source}; rollback copy {backup}; {result}", flush=True)


if __name__ == "__main__":
    main()
