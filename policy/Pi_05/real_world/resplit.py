"""Rebuild explicit, reviewed episode splits using official LeRobot editing tools.

Build a separate prepared root first. Source archives, videos and action labels
are retained; the caller promotes the root only after normalization and tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from real_world.config import TASKS, repo_id
from real_world.norm import numeric_columns
from real_world.prepare import write_index


def copy_immutable(source, target):
    # Only large immutable media/data share storage. Reports and markers must be
    # separate files so writing staging metadata cannot change the live root.
    if Path(source).suffix in {".mp4", ".jpg", ".parquet"}:
        os.link(source, target)
        return target
    return shutil.copy2(source, target)


def episode_fingerprints(root):
    """Hash every retained frame, including embedded camera bytes, without decode."""
    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action",
               "observation.images.camera_1", "observation.images.camera_2"]
    hashes, counts = {}, Counter()
    for path in sorted((root / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                episode = row["episode_index"]
                digest = hashes.setdefault(episode, hashlib.sha256())
                assert row["frame_index"] == counts[episode]
                digest.update(np.asarray([row["frame_index"]], dtype="<i8").tobytes())
                for key in ("timestamp", "observation.state", "action"):
                    digest.update(np.asarray(row[key], dtype="<f4").tobytes())
                for key in columns[-2:]:
                    payload = row[key]["bytes"]
                    if payload is None:
                        raise ValueError("Expected embedded image bytes")
                    digest.update(len(payload).to_bytes(8, "little"))
                    digest.update(payload)
                counts[episode] += 1
    return {i: {"sha256": digest.hexdigest(), "frames": counts[i]} for i, digest in hashes.items()}


def validate_plan(report, plan):
    active = {r["source"] for r in report["episodes"] if r["split"] in {"train", "val"}}
    labels = plan["variants"]
    validation = plan["validation_sources"]
    if set(labels) != active:
        raise ValueError("Every active source must have exactly one reviewed variation label")
    if len(validation) != len(set(validation)) or not set(validation) <= active:
        raise ValueError("Invalid validation source list")
    if len(validation) != max(1, round(len(active) * 0.1)):
        raise ValueError("Validation count must be nearest whole episode to 10%")
    train = active - set(validation)
    if {labels[s] for s in validation} != set(labels.values()) or {labels[s] for s in train} != set(labels.values()):
        raise ValueError("Both training and validation must cover all reviewed variations")


def rebuild_task(source, target, task, plan):
    from lerobot.datasets.dataset_tools import merge_datasets, split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    report = json.loads((source / task / "report.json").read_text())
    validate_plan(report, plan)
    old_datasets, ordered, before = [], [], {}
    for split in ("train", "val"):
        rows = [r for r in report["episodes"] if r["split"] == split]
        old_root = source / "datasets" / repo_id(task, split)
        states, _, _ = numeric_columns(old_root)
        ds = LeRobotDataset(repo_id(task, split), root=old_root)
        assert ds.meta.total_episodes == len(rows)
        assert len(states) == sum(r["frames"] for r in rows)
        fingerprints = episode_fingerprints(old_root)
        for i, row in enumerate(rows):
            assert fingerprints[i]["frames"] == row["frames"]
            before[row["source"]] = fingerprints[i]
            row["previous_split"] = split
            row["previous_dataset_episode_index"] = i
        old_datasets.append(ds)
        ordered.extend(rows)

    temporary = target / ".resplit_work" / task
    merged = merge_datasets(old_datasets, f"real_world-{task}-reviewed", temporary / "merged")
    val = set(plan["validation_sources"])
    indices = {split: [i for i, r in enumerate(ordered) if (r["source"] in val) == (split == "val")]
               for split in ("train", "val")}
    results = split_dataset(merged, indices, temporary / "splits")
    proof = {}
    new_order = []
    for split in ("train", "val"):
        rows = [ordered[i] for i in indices[split]]
        dataset_root = temporary / "splits" / split
        fingerprints = episode_fingerprints(dataset_root)
        states, actions, ends = numeric_columns(dataset_root)
        assert results[split].meta.total_episodes == len(rows)
        assert len(states) == sum(r["frames"] for r in rows)
        offset = 0
        for i, row in enumerate(rows):
            assert fingerprints[i] == before[row["source"]], row["source"]
            np.testing.assert_array_equal(states[offset], np.asarray(row["state_first"], np.float32))
            np.testing.assert_array_equal(actions[offset], np.asarray(row["action_first"], np.float32))
            assert ends[offset] - offset == row["frames"]
            np.testing.assert_array_equal(actions[ends[offset] - 1], np.asarray(row["action_last"], np.float32))
            offset += row["frames"]
            row.update(split=split, dataset_episode_index=i, variation=plan["variants"][row["source"]])
            proof[row["source"]] = {**fingerprints[i], "split": split, "dataset_episode_index": i,
                                    "all_frames_equal_before_after": True}
        dataset_root.rename(target / "datasets" / repo_id(task, split))
        new_order.extend(rows)
    excluded = [r for r in report["episodes"] if r["split"] == "excluded"]
    report["episodes"] = new_order + excluded
    report["counts"] = {s: {"episodes": sum(r["split"] == s for r in report["episodes"]),
                              "frames": sum(r["frames"] for r in report["episodes"] if r["split"] == s)}
                        for s in ("train", "val", "excluded")}
    report["review_split_policy"] = "Manual scene-stratified whole episodes; nearest whole episode to 10%; all variants in both splits"
    report["variation_counts"] = {s: dict(Counter(r["variation"] for r in new_order if r["split"] == s))
                                  for s in ("train", "val")}
    report["split_plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    (target / task / "report.json").write_text(json.dumps(report, indent=2))
    (target / task / "resplit_integrity.json").write_text(json.dumps(proof, indent=2))
    write_index(target / task, new_order)
    del results, merged, old_datasets
    shutil.rmtree(temporary)
    print(f"{task}: {report['counts']}; variations {report['variation_counts']}; full frame hashes match", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    tasks = tuple(plan["tasks"])
    if not tasks or not set(tasks) <= set(TASKS) or args.target.exists():
        raise ValueError("Expected supported tasks and a new staging root")
    for task in tasks:
        validate_plan(json.loads((args.source / task / "report.json").read_text()), plan["tasks"][task])

    def ignore(path, names):
        directory = Path(path)
        if directory == args.source / "datasets":
            return [repo_id(t, s) for t in tasks for s in ("train", "val")]
        if directory == args.source / "assets":
            return [f"pi05_real_{t}_train" for t in tasks]
        return []

    shutil.copytree(args.source, args.target, copy_function=copy_immutable, ignore=ignore)
    for task in tasks:
        rebuild_task(args.source, args.target, task, plan["tasks"][task])
    shutil.rmtree(args.target / ".resplit_work")
    (args.target / "split_plan.json").write_text(json.dumps(plan, indent=2))
    # These checks described the former split; preserve them as history rather
    # than presenting them as verification of newly selected validation data.
    history = args.target / "history" / f"before_resplit_{os.environ.get('SLURM_JOB_ID', 'local')}"
    history.mkdir(parents=True)
    moved_history = []
    for name in ("official_shell_norm_reference.json", "official_shell_norm_reference_source.json", "verification.json"):
        if (args.target / name).exists():
            (args.target / name).rename(history / name)
            moved_history.append(name)
    (args.target / "verification.json").write_text(json.dumps({
        "formal_training_submitted": False, "gpu_smoke": "Historical results only; split changed; not rerun",
        "historical_results": str(history.relative_to(args.target) / "verification.json")
        if "verification.json" in moved_history else None,
        "tasks_rebuilt": tasks, "deferred_task": "press_button", "data_checks": "pending"}, indent=2))


if __name__ == "__main__":
    main()
