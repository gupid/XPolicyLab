#!/usr/bin/env python3
"""Promote the checked press-button candidate without replacing other tasks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    source, candidate = args.source.resolve(), args.candidate.resolve()
    proof = json.loads((candidate / "press49_candidate.json").read_text())
    tests = ET.parse(candidate / "press49_test_results.xml").getroot().find("testsuite").attrib
    if int(tests["failures"]) or int(tests["errors"]) or int(tests["skipped"]) or int(tests["tests"]) != 13:
        raise ValueError("Expected all 13 candidate checks to pass without skips")
    for relative, expected in proof["live_reports_unchanged"].items():
        if sha(source / relative) != expected:
            raise ValueError(f"Live source changed since preparation: {relative}")
    report = json.loads((candidate / "press_button/report.json").read_text())
    if report["counts"]["train"]["episodes"] != 44 or report["counts"]["val"]["episodes"] != 5:
        raise ValueError("Expected 44 train / 5 validation episodes")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = source / "history" / f"press_button_before_49_{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    paths = ["press_button", "datasets/real_world-press_button-franka_single-ee-train",
             "datasets/real_world-press_button-franka_single-ee-val", "assets/pi05_real_press_button_train"]
    moved, installed = [], []
    try:
        for relative in paths:
            old, new, saved = source / relative, candidate / relative, backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            old.rename(saved)
            moved.append(relative)
            new.rename(old)
            installed.append(relative)
    except BaseException:
        for relative in reversed(installed):
            (source / relative).rename(candidate / relative)
        for relative in reversed(moved):
            (backup / relative).rename(source / relative)
        raise
    for filename in ("review_exclusions.json", "split_plan.json"):
        (backup / filename).write_bytes((source / filename).read_bytes())
    (source / "review_exclusions.json").write_bytes((candidate / "review_exclusions.json").read_bytes())
    plan = json.loads((source / "split_plan.json").read_text())
    selected = json.loads((candidate / "split_plan.json").read_text())["tasks"]["press_button"]
    plan["tasks"]["press_button"] = selected
    plan.pop("deferred_task", None)
    (source / "split_plan.json").write_text(json.dumps(plan, indent=2))
    for name in ("shell_game", "put_back_block"):
        assert sha(source / name / "report.json") == proof["live_reports_unchanged"][f"{name}/report.json"]
    proof.update(status="promoted after explicit user task-name confirmation", live_root=str(source),
                 promoted_at_utc=timestamp, rollback=str(backup), pytest=tests)
    (source / "press49_preparation.json").write_text(json.dumps(proof, indent=2))
    (source / "press49_test_results.xml").write_bytes((candidate / "press49_test_results.xml").read_bytes())
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
