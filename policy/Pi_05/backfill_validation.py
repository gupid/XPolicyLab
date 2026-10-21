#!/usr/bin/env python3
"""Evaluate all retained current-run checkpoints without retraining any weights."""
import argparse
import dataclasses
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys

POLICY = Path(__file__).resolve().parent
sys.path.insert(0, str(POLICY.parents[2]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("shell_game", "press_button", "put_back_block"), required=True)
    parser.add_argument("--run-tag", default="lora_crop640_20261021")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import jax
    from flax import nnx
    import numpy as np
    from openpi.models import model as model_lib
    from openpi.shared import normalize
    from real_world_entry import resolved_config, register_configs, repo_id
    from real_world.validation import HeldOutValidation, atomic_json, read_records
    from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

    run_name = build_run_dir_name({"bench_name": "real_world", "ckpt_name": f"{args.task}_{args.run_tag}",
                                  "env_cfg_type": "franka_single", "action_type": "ee", "seed": 0})
    run = POLICY / "checkpoints" / run_name
    manifest_path = run.parent / ".real_world_manifests" / f"{run.name}_initial.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["slurm_job_id"] == "15398"
    root = register_configs()
    assert sha(root / args.task / "report.json") == manifest["dataset_report_sha256"]
    # Original source was archived and hash-verified before adding validation.
    snapshot = POLICY / "outputs/validation_backfill_15398/source_snapshot"
    for relative, digest in manifest["source_sha256"].items():
        assert sha(snapshot / relative) == digest, relative
        if relative not in ("real_world/train.py", "real_world_entry.py"):
            assert sha(POLICY / relative) == digest, relative
    cfg = dataclasses.replace(resolved_config(args.task), checkpoint_dir_override=str(run))
    evaluator = None
    checkpoints = sorted((p for p in run.iterdir() if p.name.isdigit() and (p / "params").exists()),
                         key=lambda p: int(p.name))
    assert [p.name for p in checkpoints] == ["2500", "5000", "7500", "9999"]
    for checkpoint in checkpoints:
        assert sha(checkpoint / "assets" / repo_id(args.task) / "norm_stats.json") == manifest["normalization_sha256"]
        existing = read_records(run / "validation.jsonl")
        if any(r["checkpoint"] == str(checkpoint) for r in existing):
            logging.info("Already evaluated: %s", checkpoint)
            continue
        params = model_lib.restore_params(checkpoint / "params")  # Preserve original parameter precision.
        model = cfg.model.load(params)
        del params
        graphdef, state = nnx.split(model)
        if evaluator is None:
            # Use the checkpoint's saved TRAINING statistics, never validation stats.
            stats = normalize.load(checkpoint / "assets" / repo_id(args.task))
            data = cfg.data.create(cfg.assets_dirs, cfg.model)
            for key in stats:
                np.testing.assert_array_equal(stats[key].q01, data.norm_stats[key].q01)
                np.testing.assert_array_equal(stats[key].q99, data.norm_stats[key].q99)
            evaluator = HeldOutValidation(cfg, num_steps=10)
        row = evaluator.evaluate(graphdef, state, updates=int(checkpoint.name) + 1,
                                 checkpoint=checkpoint, origin="posthoc_backfill_job_15398")
        row.update(manifest_sha256=sha(manifest_path), dataset_report_sha256=manifest["dataset_report_sha256"],
                   normalization_sha256=manifest["normalization_sha256"], evaluator_sha256=sha(Path(__file__)),
                   validation_module_sha256=sha(POLICY / "real_world/validation.py"))
        atomic_json(checkpoint / "held_out_validation.json", row)
        np.savez_compressed(checkpoint / "held_out_frame_metrics.npz", **evaluator.last_frame_metrics)
        del model, state
        gc.collect()
    rows = read_records(run / "validation.jsonl")
    assert len(rows) == 4
    best = min(rows, key=lambda r: r["val_flow_loss"])
    selection = {"selection_metric": "val_flow_loss", "best_checkpoint": best["checkpoint"],
                 "optimizer_updates": best["optimizer_updates"], "val_flow_loss": best["val_flow_loss"],
                 "scope": "minimum among four retained checkpoints, posthoc full held-out validation",
                 "best_by_metric": {k: min(rows, key=lambda r: r[k]) for k in
                                    ("val_flow_loss", "val_action_l1", "val_action_l2")}, "results": rows}
    atomic_json(run / "best_checkpoint.json", selection)
    logging.info("BEST CHECKPOINT: %s", json.dumps(selection))


if __name__ == "__main__":
    main()
