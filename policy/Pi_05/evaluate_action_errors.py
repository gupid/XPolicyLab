#!/usr/bin/env python3
"""Offline sampled-action errors; independent of the recorded flow-loss evaluator."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np

POLICY = Path(__file__).resolve().parent


def metrics(error, mask):
    selected = np.asarray(error, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if selected.ndim != 2 or not len(selected) or not np.isfinite(selected).all():
        raise ValueError("Expected finite valid action vectors")
    return {"l1_mae": float(np.abs(selected).mean()),
            "l2_mse": float(np.square(selected).mean()),
            "rmse": float(np.sqrt(np.square(selected).mean())),
            "mean_l2_vector_norm": float(np.linalg.norm(selected, axis=-1).mean()),
            "valid_action_vectors": len(selected)}


def evaluate(args):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi import transforms
    from openpi.models import model
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import data_loader
    from real_world_entry import resolved_config, repo_id, register_configs

    start = time.monotonic()
    cfg = resolved_config(args.task, "val")
    stats = normalize.load(args.checkpoint / "assets" / repo_id(args.task))
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), norm_stats=stats)
    dataset = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    transform = transforms.compose([
        *dc.repack_transforms.inputs, *dc.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs,
    ])
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, norm_stats=stats)
    params = nnx.state(policy._model)
    trained = params.filter(cfg.trainable_filter)
    count = lambda state: sum(int(x.size) for x in jax.tree.leaves(state))
    param_counts = {"total": count(params), "trainable_under_recorded_filter": count(trained),
                    "lora": count(params.filter(nnx.All(nnx.Param, lambda path, _: "lora" in "/".join(map(str, path)))))}
    param_counts["trainable_fraction"] = param_counts["trainable_under_recorded_filter"] / param_counts["total"]
    for label, key in (("image_encoder", "img"), ("language_and_action_transformers", "llm")):
        param_counts[f"trainable_{label}"] = count(trained.filter(lambda path, _: key in path))
    print("PARAMETER_COUNTS", json.dumps(param_counts), flush=True)

    frame_indices, episodes, masks, predictions, targets, physical_predictions, physical_targets = [], [], [], [], [], [], []
    for offset in range(0, len(dataset), args.batch_size):
        indices = list(range(offset, min(offset + args.batch_size, len(dataset))))
        raw = [dataset[i] for i in indices]
        samples = [transform(item) for item in raw]
        actual = len(indices)
        samples.extend([samples[-1]] * (args.batch_size - actual))
        batch = jax.tree.map(lambda *values: jnp.asarray(np.stack(values)), *samples)
        noise = np.stack([np.random.default_rng(np.random.SeedSequence([args.seed, i])).standard_normal(
            (cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32) for i in indices])
        if actual < args.batch_size:
            noise = np.concatenate([noise, np.repeat(noise[-1:], args.batch_size - actual, axis=0)])
        pred = np.asarray(policy._sample_actions(jax.random.key(args.seed), model.Observation.from_dict(batch),
                                                num_steps=args.denoising_steps, noise=jnp.asarray(noise)))[:actual]
        expected = np.asarray(batch["actions"])[:actual]
        for k, (i, item) in enumerate(zip(indices, raw, strict=True)):
            padding = np.asarray(item["action_is_pad"], dtype=bool)
            if padding.shape != (cfg.model.action_horizon,):
                raise ValueError((i, padding.shape))
            output = policy._output_transform({"state": np.asarray(batch["state"])[k], "actions": pred[k]})["actions"]
            gt = np.asarray(item["action"])
            assert output.shape == gt.shape == (50, 8)
            frame_indices.append(i)
            episodes.append(int(np.asarray(item["episode_index"]).item()))
            masks.append(~padding)
            predictions.append(pred[k, :, :8])
            targets.append(expected[k, :, :8])
            physical_predictions.append(output)
            physical_targets.append(gt)
        if offset % (args.batch_size * 25) == 0:
            print(f"{args.task}/{args.checkpoint.name}: {offset + actual}/{len(dataset)} frames", flush=True)

    mask = np.asarray(masks)
    error = np.asarray(predictions) - np.asarray(targets)
    physical_error = np.asarray(physical_predictions) - np.asarray(physical_targets)
    by_episode = {str(e): metrics(error[np.asarray(episodes) == e], mask[np.asarray(episodes) == e])
                  for e in sorted(set(episodes))}
    report_path = register_configs() / args.task / "report.json"
    manifest_path = POLICY / "checkpoints/.real_world_manifests" / f"{args.checkpoint.parent.name}_initial.json"
    recorded = json.loads(manifest_path.read_text())
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == recorded["dataset_report_sha256"]
    checkpoint_norm = args.checkpoint / "assets" / repo_id(args.task) / "norm_stats.json"
    assert hashlib.sha256(checkpoint_norm.read_bytes()).hexdigest() == recorded["normalization_sha256"]
    for relative, digest in recorded["source_sha256"].items():
        assert hashlib.sha256((POLICY / relative).read_bytes()).hexdigest() == digest, relative
    result = {"task": args.task, "checkpoint": str(args.checkpoint),
              "checkpoint_label": int(args.checkpoint.name), "optimizer_updates": int(args.checkpoint.name) + 1,
              "validation_frames": len(dataset), "validation_episodes": len(by_episode),
              "normalized_real8": metrics(error, mask), "normalized_per_episode": by_episode,
              "normalized_per_dimension": [metrics(error[..., d:d+1], mask) for d in range(8)],
              "physical_groups": {"translation_m": metrics(physical_error[..., :3], mask),
                                  "quaternion_components": metrics(physical_error[..., 3:7], mask),
                                  "gripper_command": metrics(physical_error[..., 7:8], mask)},
              "parameter_counts": param_counts, "denoising_steps": args.denoising_steps,
              "noise": f"NumPy SeedSequence([{args.seed}, validation_global_frame_index]); identical across checkpoints",
              "masking": "Only real8 action dimensions and nonpadding horizon positions; all validation start frames",
              "l2_definition": "Mean squared component error (MSE); Euclidean vector norm and RMSE also exported",
              "weighting": "micro-average over valid overlapping action-chunk elements; not independent samples",
              "quaternion_note": "Component errors use recorded xyzw signs; not SO(3) angular error",
              "scope": "Offline open-loop sampled actions, not historical training L1/L2 or real robot success",
              "manifest_identity_verified": True, "seconds": time.monotonic() - start,
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"{args.task}_{args.checkpoint.name}"
    np.savez_compressed(str(stem) + ".npz", indices=frame_indices, episodes=episodes, valid=mask,
                        prediction_normalized=predictions, target_normalized=targets,
                        prediction_physical=physical_predictions, target_physical=physical_targets)
    Path(str(stem) + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def plot(output):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(path.read_text()) for path in output.glob("*.json")
               if path.stem.split("_")[-1].isdigit()]
    tasks = ("shell_game", "put_back_block", "press_button")
    colors = ("#087e8b", "#b73b57", "#4b7b35")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    rows = []
    for task, color in zip(tasks, colors, strict=True):
        data = sorted([r for r in records if r["task"] == task], key=lambda r: r["optimizer_updates"])
        if not data:
            raise ValueError(f"Missing {task} results")
        for ax, metric, title in zip(axes, ("l1_mae", "l2_mse"), ("Action L1 (MAE)", "Action L2 (MSE)"), strict=True):
            ax.plot([r["optimizer_updates"] for r in data], [r["normalized_real8"][metric] for r in data],
                    "o--", color=color, label=task, markersize=6)
            ax.set(title=title, xlabel="Optimizer updates", ylabel="Normalized real-action error")
            ax.grid(alpha=0.2)
        for r in data:
            rows.append({"task": task, "optimizer_updates": r["optimizer_updates"],
                         "validation_frames": r["validation_frames"], **r["normalized_real8"]})
    axes[0].legend(frameon=False)
    fig.suptitle("Held-out sampled-action error: best and final checkpoints (10 denoising steps)", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"action_l1_l2.{suffix}", dpi=200)
    plt.close(fig)
    with (output / "action_l1_l2.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Plots and CSV:", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("shell_game", "press_button", "put_back_block"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=POLICY / "outputs/action_errors")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot:
        plot(args.output)
    else:
        if args.task is None or args.checkpoint is None or args.batch_size < 1 or args.denoising_steps < 1:
            parser.error("task, checkpoint and positive batch/denoising sizes are required")
        evaluate(args)


if __name__ == "__main__":
    main()
