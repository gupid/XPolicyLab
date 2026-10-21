#!/usr/bin/env python3
"""Real Franka π0.5 entry; official OpenPI LoRA trainer, no simulation."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import importlib
import os
from pathlib import Path
import subprocess
import sys

POLICY = Path(__file__).resolve().parent
sys.path.insert(0, str(POLICY.parents[2]))  # parent of XPolicyLab checkout

from real_world.config import TASKS, register_configs, repo_id  # noqa: E402


def upstream_script(name):
    # Spawn workers must be able to import classes such as RemoveStrings by
    # their real module name; runpy's <run_path> cannot be unpickled.
    scripts = str(POLICY / "openpi/scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


def resolved_config(task, split="train"):
    from openpi.training import config
    root = register_configs()
    os.environ.setdefault("HF_LEROBOT_HOME", str(root / "datasets"))
    # Must be called before importing the LeRobot loader (its cache path is a module constant).
    return config.get_config(f"pi05_real_{task}_{split}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "norm", "train", "infer-check", "evaluate", "compare-checkpoints"))
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default="lora")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Separate run; save every step; tiny step count only")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-samples", type=int, default=32, help="Fixed evenly spaced frames; 0 evaluates every validation frame")
    parser.add_argument("--action-metric-interval", type=int, default=100,
                        help="Record sampled training-action MAE/MSE every N updates, also first/final")
    parser.add_argument("--action-metric-denoising-steps", type=int, default=10)
    args = parser.parse_args()
    if args.action_metric_interval < 1 or args.action_metric_denoising_steps < 1:
        parser.error("Action metric interval and denoising steps must be positive")
    cfg = resolved_config(args.task)
    from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
    name = build_run_dir_name({"bench_name": "real_world", "ckpt_name": f"{args.task}_{args.run_tag}",
                               "env_cfg_type": "franka_single", "action_type": "ee", "seed": args.seed})
    checkpoint_dir = POLICY / "checkpoints" / name
    cfg = dataclasses.replace(cfg, exp_name=name, checkpoint_dir_override=str(checkpoint_dir), seed=args.seed,
                              num_train_steps=args.steps, overwrite=False, resume=args.resume,
                              wandb_enabled=True)  # WANDB_MODE=offline preserves upstream logging calls.
    if args.steps < 1 or (args.smoke and (args.steps > 10 or "smoke" not in args.run_tag)):
        parser.error("Positive steps required; smoke requires <=10 steps and a distinct run-tag containing smoke")
    if args.smoke:
        cfg = dataclasses.replace(cfg, save_interval=1, keep_period=1, log_interval=1, num_workers=0)
    from real_world.validation import protocol as validation_protocol
    record = {"name": cfg.name, "run": name, "dataset": cfg.data.repo_id, "base_weights": str(cfg.weight_loader),
              "checkpoint_dir": str(cfg.checkpoint_dir), "steps": cfg.num_train_steps, "seed": cfg.seed,
              "batch_size": cfg.batch_size, "action_horizon": cfg.model.action_horizon,
              "action_dim": cfg.model.action_dim, "model": dataclasses.asdict(cfg.model),
              "lr_schedule": dataclasses.asdict(cfg.lr_schedule), "optimizer": dataclasses.asdict(cfg.optimizer),
              "save_interval": cfg.save_interval, "keep_period": cfg.keep_period,
              "num_workers": cfg.num_workers, "log_interval": cfg.log_interval,
              "fsdp_devices": cfg.fsdp_devices,
              "action_metrics": {"interval": args.action_metric_interval,
                                 "denoising_steps": args.action_metric_denoising_steps,
                                 "split": "current training batch", "coordinates": "normalized real action dimensions",
                                 "l1": "MAE", "l2": "MSE", "exclude_action_padding": True},
              "validation": validation_protocol(cfg, args.action_metric_denoising_steps),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
              "resume": cfg.resume, "smoke": args.smoke, "action_representation": "delta_ee_pose_xyzw_and_gripper",
              "fps": 15, "norm_asset_id": repo_id(args.task), "normalization_source": "training episodes only",
              "dataloader_resume": "official OpenPI restores model/optimizer; does not restore exact sample order"}
    print(json.dumps(record, indent=2, default=str), flush=True)
    if args.mode == "dry-run":
        return
    if args.mode == "compare-checkpoints":
        compare_checkpoints(cfg, args)
        return
    if args.mode == "norm":
        from real_world.norm import save_for_config
        save_for_config(cfg, register_configs())
        return
    if args.mode == "train":
        if not Path(cfg.weight_loader.params_path).exists():
            raise FileNotFoundError(cfg.weight_loader.params_path)
        provenance = checkpoint_dir.parent / ".real_world_manifests"
        provenance.mkdir(parents=True, exist_ok=True)
        report = register_configs() / args.task / "report.json"
        if not (report.parent / "COMPLETE").exists():
            raise FileNotFoundError(f"Prepared dataset incomplete: {report.parent}")
        record["dataset_report"] = json.loads(report.read_text())
        if record["dataset_report"]["review_only"]:
            raise ValueError("Review-only output cannot be used for training")
        stats = Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id / "norm_stats.json"
        record["dataset_report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
        record["normalization_sha256"] = hashlib.sha256(stats.read_bytes()).hexdigest()
        guards = stats.with_name("normalization_guards.json")
        record["normalization_guards"] = json.loads(guards.read_text()) if guards.exists() else {}
        initial = provenance / f"{name}_initial.json"
        if checkpoint_dir.exists() and not args.resume:
            raise FileExistsError(f"Run exists; use an explicit new run-tag or --resume: {checkpoint_dir}")
        if args.resume:
            previous = json.loads(initial.read_text())
            check_resume_identity(previous, record)
        record["source_sha256"] = {str(p.relative_to(POLICY)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), *sorted((POLICY / "real_world").glob("*.py"))]}
        suffix = f"resume_{args.steps}" if args.resume else "initial"
        (provenance / f"{name}_{suffix}.json").write_text(json.dumps(record, indent=2, default=str))
        script = upstream_script("train")
        from real_world.train import main as train_with_action_metrics
        train_with_action_metrics(cfg, script, metric_interval=args.action_metric_interval,
                                  denoising_steps=args.action_metric_denoising_steps)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required for infer-check/evaluate")
    check_checkpoint(cfg, args)


def check_resume_identity(previous, current):
    for key in ("dataset_report_sha256", "normalization_sha256", "model", "lr_schedule", "optimizer",
                "batch_size", "seed", "base_weights", "smoke", "action_metrics"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Resume identity differs for {key}; start a new run instead")
    if previous.get("validation") is not None and previous["validation"] != current.get("validation"):
        raise ValueError("Resume identity differs for validation; use the recorded validation protocol")


def check_checkpoint(cfg, args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi import transforms
    from openpi.models import model
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.shared import nnx_utils
    from openpi.training import data_loader

    val_cfg = resolved_config(args.task, "val")
    dc = val_cfg.data.create(val_cfg.assets_dirs, val_cfg.model)
    stats = normalize.load(args.checkpoint / "assets" / repo_id(args.task))
    dc = dataclasses.replace(dc, norm_stats=stats)
    dataset = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    # Explicit dataset-only repack before inference; runtime does not see LeRobot keys.
    raw = transforms.compose(dc.repack_transforms.inputs)(dataset[0])
    raw = jax.tree.map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, raw)
    print(f"Reloading trained checkpoint for {args.task} validation split", flush=True)
    policy = policy_config.create_trained_policy(val_cfg, str(args.checkpoint), norm_stats=stats,
                                                sample_kwargs={"num_steps": 2})
    print("Checkpoint loaded; compiling/running offline inference", flush=True)
    output = policy.infer(raw)["actions"]
    if output.shape != (50, 8) or not np.isfinite(output).all():
        raise AssertionError((output.shape, output))
    result = {"task": args.task, "checkpoint": str(args.checkpoint), "inference_shape": list(output.shape),
              "inference_finite": True, "quaternion_norm_range":
                  [float(np.linalg.norm(output[:, 3:7], axis=-1).min()),
                   float(np.linalg.norm(output[:, 3:7], axis=-1).max())],
              "scope": "offline inference, 2 denoising steps; no robot execution"}
    print("Offline inference passed: finite 50x8 actions", flush=True)
    if args.mode == "evaluate":
        print("Compiling/running held-out flow-loss checks", flush=True)
        transformed = data_loader.transform_dataset(dataset, dc)
        if args.eval_samples < 0:
            raise ValueError("eval-samples must be nonnegative; 0 means all validation frames")
        indices = (np.arange(len(transformed)) if args.eval_samples == 0 else
                   np.linspace(0, len(transformed) - 1, min(args.eval_samples, len(transformed)), dtype=int))
        loaded_model = policy._model
        compute_loss = nnx_utils.module_jit(loaded_model.compute_loss, static_argnames=("train",))
        losses = []
        for i in indices:
            sample = jax.tree.map(lambda x: np.asarray(x)[None], transformed[int(i)])
            obs = model.Observation.from_dict(sample)
            loss = compute_loss(jax.random.key(int(i)), obs, jnp.asarray(sample["actions"]), train=False)
            losses.append(float(jnp.mean(loss)))
            if len(losses) % 100 == 0:
                print(f"Validation: {len(losses)}/{len(indices)} frames", flush=True)
        if not np.isfinite(losses).all():
            raise AssertionError(losses)
        result.update(validation_samples=len(losses), validation_flow_loss=float(np.mean(losses)),
                      validation_flow_loss_std=float(np.std(losses)),
                      validation_frame_indices=indices.tolist(),
                      validation_rng="jax.random.key(frame_index); same noise/time seed at each checkpoint",
                      dataset_report_sha256=hashlib.sha256((register_configs() / args.task / "report.json").read_bytes()).hexdigest())
    (args.checkpoint / f"real_world_{args.mode}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def compare_checkpoints(cfg, args):
    """Evaluate retained checkpoints on the same complete held-out frame set."""
    manifest_path = cfg.checkpoint_dir.parent / ".real_world_manifests" / f"{cfg.exp_name}_initial.json"
    manifest = json.loads(manifest_path.read_text())
    root = register_configs()
    current_report = hashlib.sha256((root / args.task / "report.json").read_bytes()).hexdigest()
    stats = root / "assets" / cfg.name / repo_id(args.task) / "norm_stats.json"
    if manifest["dataset_report_sha256"] != current_report or manifest["normalization_sha256"] != hashlib.sha256(stats.read_bytes()).hexdigest():
        raise ValueError("Dataset or normalization changed since baseline training started")
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((POLICY / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Recorded training/evaluation source changed: {relative}")
    checkpoints = sorted((p for p in cfg.checkpoint_dir.iterdir()
                          if p.is_dir() and p.name.isdigit() and (p / "params").exists()), key=lambda p: int(p.name))
    if not checkpoints or int(checkpoints[-1].name) != cfg.num_train_steps - 1:
        raise ValueError("Final training checkpoint must exist before baseline comparison")
    results = []
    for checkpoint in checkpoints:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "evaluate", args.task,
                        "--checkpoint", str(checkpoint), "--eval-samples", "0"], check=True)
        result = json.loads((checkpoint / "real_world_evaluate.json").read_text())
        result["optimizer_updates"] = int(checkpoint.name) + 1
        results.append(result)
    if len({r["dataset_report_sha256"] for r in results}) != 1 or any(
            r["validation_frame_indices"] != results[0]["validation_frame_indices"] for r in results):
        raise ValueError("Checkpoint comparisons must use identical validation data")
    best = min(results, key=lambda r: r["validation_flow_loss"])
    report = {"task": args.task, "selection_metric": "mean flow loss over all validation frames; fixed per-frame RNG",
              "best_checkpoint": best["checkpoint"], "final_checkpoint": results[-1]["checkpoint"],
              "real_robot_success": "not evaluated", "results": results}
    target = cfg.checkpoint_dir / "baseline_validation.json"
    target.write_text(json.dumps(report, indent=2))
    print(f"Baseline comparison saved to {target}; minimum validation loss at {best['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
