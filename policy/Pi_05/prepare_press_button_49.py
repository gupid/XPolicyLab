#!/usr/bin/env python3
"""Build a separately reviewable 49-episode press-button candidate.

Never replace live datasets: the latest request's task name needs confirmation.
Keep all training sources frozen while shell_game is running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

POLICY = Path(__file__).resolve().parent
sys.path.insert(0, str(POLICY.parents[2]))

from real_world.config import repo_id  # noqa: E402
from real_world.resplit import copy_immutable, rebuild_task  # noqa: E402


def finish_candidate(source, target):
    import numpy as np
    from openpi.shared import normalize
    from real_world.norm import numeric_columns

    states, _, _ = numeric_columns(target / "datasets" / repo_id("press_button"))
    asset = target / "assets/pi05_real_press_button_train" / repo_id("press_button")
    stats = normalize.load(asset)
    value = states[:, 7]
    normalized = (value - stats["state"].q01[7]) / (stats["state"].q99[7] - stats["state"].q01[7] + 1e-6) * 2 - 1
    if float(np.abs(normalized).max()) >= 2:
        diagnostics = json.loads((asset / "normalization_guards.json").read_text())
        diagnostics["state"] = {"dimension": 7, "original_q01": float(stats["state"].q01[7]),
                                "original_q99": float(stats["state"].q99[7]),
                                "replacement_q01": float(value.min()), "replacement_q99": float(value.max()),
                                "observed_max_abs_normalized": float(np.abs(normalized).max()),
                                "rule": "Reviewed 49-episode press-button data: use train min/max for gripper width when training normalization exceeds magnitude 2; all other fields unchanged"}
        stats["state"].q01[7], stats["state"].q99[7] = value.min(), value.max()
        normalize.save(asset, stats)
        (asset / "normalization_guards.json").write_text(json.dumps(diagnostics, indent=2))
        print("Training-only gripper-width normalization guard: " + json.dumps(diagnostics), flush=True)
    env = os.environ.copy()
    env.update(PI05_REAL_ROOT=str(target), HF_LEROBOT_HOME=str(target / "datasets"), JAX_PLATFORMS="cpu")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_real_world.py", f"--junitxml={target / 'press49_test_results.xml'}"],
                   cwd=POLICY, env=env, check=True)
    identities = {}
    for task in ("shell_game", "put_back_block"):
        path = f"{task}/report.json"
        expected = hashlib.sha256((target / path).read_bytes()).hexdigest()
        assert hashlib.sha256((source / path).read_bytes()).hexdigest() == expected
        identities[path] = expected
    identities["press_button/report.json"] = hashlib.sha256((source / "press_button/report.json").read_bytes()).hexdigest()
    frozen = json.loads((POLICY / "checkpoints/.real_world_manifests/real_world-shell_game_lora-franka_single-ee-0_initial.json").read_text())
    for relative, expected in frozen["source_sha256"].items():
        assert hashlib.sha256((POLICY / relative).read_bytes()).hexdigest() == expected
    proof = {"status": "checked candidate; live prepared root unchanged",
             "task": "press_button", "source": str(source), "candidate": str(target),
             "additional_exclusions": json.loads((target / "press_button_additional_exclusions.json").read_text())["press_button"],
             "validation_source_episode_ids": [6, 14, 22, 35, 44],
             "counts": json.loads((target / "press_button/report.json").read_text())["counts"],
             "live_reports_unchanged": identities, "active_shell_sources_match_training_manifest": True,
             "normalization_guards": json.loads((asset / "normalization_guards.json").read_text()),
             "validation_variants": "three digits 1,2; two digits 2,1", "tests": "passed; see press49_test_results.xml"}
    (target / "press49_candidate.json").write_text(json.dumps(proof, indent=2))
    (target / "STATUS.md").write_text("# press_button 49 条候选数据\n\n独立候选目录，尚未替换正在使用的 prepared。\n\n"
                                    "剩余 49 条：44 条训练、5 条验证；验证源编号 6、14、22、35、44，覆盖数字排列 1/2 与 2/1 两种变化。"
                                    "四条累计排除见 review_exclusions.json。统计仅来自新训练集；测试通过。"
                                    "夹爪宽度维度的分位数归一化经测试发现异常放大，仅此维度回退训练 min/max；其他统计保持原值。"
                                    "shell_game 与 put_back_block 数据未改变。\n")
    print(json.dumps(proof, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--finish-existing", action="store_true", help="Recheck an inspected, already-built candidate")
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    filtered = target.with_name(target.name + ".filtering")
    if args.finish_existing:
        finish_candidate(source, target)
        if filtered.exists():
            shutil.rmtree(filtered)
        return
    if target.exists() or filtered.exists():
        raise FileExistsError("Inspect existing candidate before retrying")
    report = json.loads((source / "press_button/report.json").read_text())
    requested = ["press_button/press_button_2026-10-19_11-51-50_202920_275f34a4_episode_000000.hdf5",
                 "press_button/press_button_2026-10-19_09-41-42_716765_7e48f5fe_episode_000000.hdf5"]
    active = [r for r in report["episodes"] if r["split"] in {"train", "val"}]
    if len(active) != 51 or not set(requested) <= {r["source"] for r in active}:
        raise ValueError("Expected the reviewed 51 episodes before applying two additional exclusions")
    frozen = json.loads((POLICY / "checkpoints/.real_world_manifests/real_world-shell_game_lora-franka_single-ee-0_initial.json").read_text())
    identities = {f"{task}/report.json": hashlib.sha256((source / task / "report.json").read_bytes()).hexdigest()
                  for task in ("shell_game", "put_back_block")}
    identities["press_button/report.json"] = hashlib.sha256((source / "press_button/report.json").read_bytes()).hexdigest()
    shutil.copytree(source, filtered, copy_function=copy_immutable)
    exclusion_file = filtered / "press_button_additional_exclusions.json"
    exclusion_file.write_text(json.dumps({"press_button": requested}, indent=2))
    env = os.environ.copy()
    env.update(PI05_REAL_ROOT=str(filtered), HF_LEROBOT_HOME=str(filtered / "datasets"), JAX_PLATFORMS="cpu")
    subprocess.run([sys.executable, "apply_review.py", "--exclude-file", str(exclusion_file)], cwd=POLICY, env=env, check=True)
    exclusions = json.loads((filtered / "review_exclusions.json").read_text())
    exclusions["press_button"] = sorted(set(exclusions["press_button"]) | set(requested))
    (filtered / "review_exclusions.json").write_text(json.dumps(exclusions, indent=2))
    report = json.loads((filtered / "press_button/report.json").read_text())
    labels, val = {}, []
    for row in report["episodes"]:
        if row["split"] == "excluded":
            continue
        name = Path(row["source"]).stem
        if "11-55-08_440508_8ba8026a" not in name:
            raise ValueError("Unexpected retained recording session")
        number = int(name.rsplit("_", 1)[1])
        labels[row["source"]] = "digits_left1_middle2" if number < 25 else "digits_left2_middle1"
        if number in {6, 14, 22, 35, 44}:
            val.append(row["source"])
    plan = {"variation_definition": "Manual camera-1 first-frame review: digits 1,2 vs 2,1, changes at source episode 25",
            "variants": labels, "validation_sources": sorted(val)}

    def ignore(path, names):
        if Path(path) == filtered / "datasets":
            return [repo_id("press_button", s) for s in ("train", "val")]
        if Path(path) == filtered / "assets":
            return ["pi05_real_press_button_train"]
        return []

    shutil.copytree(filtered, target, copy_function=copy_immutable, ignore=ignore)
    rebuild_task(filtered, target, "press_button", plan)
    shutil.rmtree(target / ".resplit_work")
    full_plan = json.loads((target / "split_plan.json").read_text())
    full_plan["tasks"]["press_button"] = plan
    full_plan.pop("deferred_task", None)
    (target / "split_plan.json").write_text(json.dumps(full_plan, indent=2))
    env.update(PI05_REAL_ROOT=str(target), HF_LEROBOT_HOME=str(target / "datasets"))
    subprocess.run([sys.executable, "real_world_entry.py", "norm", "press_button"], cwd=POLICY, env=env, check=True)
    for path, expected in identities.items():
        assert hashlib.sha256((source / path).read_bytes()).hexdigest() == expected
        if path != "press_button/report.json":
            assert hashlib.sha256((target / path).read_bytes()).hexdigest() == expected
    for relative, expected in frozen["source_sha256"].items():
        assert hashlib.sha256((POLICY / relative).read_bytes()).hexdigest() == expected
    finish_candidate(source, target)
    shutil.rmtree(filtered)


if __name__ == "__main__":
    main()
