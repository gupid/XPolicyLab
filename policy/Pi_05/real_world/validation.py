"""Held-out native flow loss and sampled real-action MAE/MSE.

Every validation frame is used, including the last incomplete batch. Per-frame
RNG is independent of training, batching and checkpoint, with no augmentation.
"""
import dataclasses
import json
import logging
from pathlib import Path
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.training import data_loader as loaders
from real_world.action_metrics import PreserveActionPadding
from XPolicyLab.utils.process_data import get_robot_action_dim_info


def protocol(config, num_steps=10):
    return {"version": 1, "split": "val", "frames": "all validation start frames",
            "interval": "first update, each checkpoint save, final update",
            "flow": "unmodified compute_loss(train=False), all native model dimensions/time positions",
            "action_l1": "normalized real-action component MAE, valid horizon only",
            "action_l2": "normalized real-action component MSE, valid horizon only",
            "denoising_steps": num_steps, "selection_metric": "val_flow_loss",
            "flow_rng": "jax.random.key(global_validation_frame_index)",
            "action_rng": "NumPy SeedSequence([seed, global_validation_frame_index])",
            "seed": config.seed, "augmentation": False,
            "normalization": "training assets, never recomputed on validation",
            "precision": "preserve checkpoint/training parameter dtypes"}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_records(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def write_record(path, row):
    """Atomic, idempotent updates when an interrupted evaluation is repeated."""
    path = Path(path)
    records = {r["optimizer_updates"]: r for r in read_records(path)}
    records[row["optimizer_updates"]] = row
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(records[k], allow_nan=False) + "\n" for k in sorted(records)))
    tmp.replace(path)


def compile_evaluators(graphdef, *, action_dim, num_steps):
    def flow(params, observation, actions, indices):
        model = nnx.merge(graphdef, params)
        model.eval()

        def one(obs, action, index):
            obs = jax.tree.map(lambda value: value[None], obs)
            loss = model.compute_loss(jax.random.key(index), obs, action[None], train=False)
            return jnp.mean(loss)

        return jax.vmap(one)(observation, actions, indices)

    def sample(params, observation, actions, valid, noise):
        model = nnx.merge(graphdef, params)
        model.eval()
        prediction = model.sample_actions(jax.random.key(0), observation, num_steps=num_steps, noise=noise)
        error = prediction[..., :action_dim].astype(jnp.float32) - actions[..., :action_dim].astype(jnp.float32)
        error = jnp.where(valid[..., None], error, 0.)
        return (jnp.abs(error).sum(axis=(1, 2)), jnp.square(error).sum(axis=(1, 2)),
                valid.sum(axis=1) * action_dim)

    return jax.jit(flow), jax.jit(sample)


class HeldOutValidation:
    def __init__(self, config, *, num_steps=10, dataset=None, batch_size=None):
        self.config, self.num_steps = config, num_steps
        self.batch_size = batch_size or config.batch_size
        if self.batch_size < 1 or num_steps < 1:
            raise ValueError("Positive validation batch and denoising steps required")
        info = get_robot_action_dim_info("franka_single")
        self.action_dim = sum(info["arm_dim"]) + sum(info["ee_dim"])
        if dataset is None:
            if not config.data.repo_id.endswith("-train"):
                raise ValueError("Validation requires a training config with training normalization assets")
            data = dataclasses.replace(config.data, repo_id=config.data.repo_id.removesuffix("-train") + "-val")
            dc = data.create(config.assets_dirs, config.model)
            if dc.norm_stats is None or dc.rlds_data_dir is not None:
                raise ValueError("Held-out LeRobot data and training normalization required")
            raw = loaders.create_torch_dataset(dc, config.model.action_horizon, config.model)
            transform = transforms.compose([
                *dc.repack_transforms.inputs, *dc.data_transforms.inputs,
                transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
                *dc.model_transforms.inputs,
            ])
            dataset = loaders.TransformedDataset(raw, [PreserveActionPadding(transform)])
        # Cache CPU samples once: repeated full validation does not decode again.
        self.samples = [dataset[i] for i in range(len(dataset))]
        if not self.samples:
            raise ValueError("Empty held-out validation dataset")
        self.path = Path(config.checkpoint_dir) / "validation.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        expected = protocol(config, num_steps)
        protocol_path = self.path.with_name("validation_protocol.json")
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != expected:
            raise ValueError("Validation protocol changed; use a separate run")
        atomic_json(protocol_path, expected)
        self._compiled = None
        self.last_frame_metrics = None
        logging.info("Cached %d held-out validation frames", len(self.samples))

    def evaluate(self, graphdef, params, *, updates, checkpoint=None, origin="training"):
        start = time.monotonic()
        if self._compiled is None:
            self._compiled = compile_evaluators(graphdef, action_dim=self.action_dim, num_steps=self.num_steps)
        flow_fn, sample_fn = self._compiled
        all_flow, all_l1, all_l2, all_count = [], [], [], []
        for offset in range(0, len(self.samples), self.batch_size):
            indices = list(range(offset, min(offset + self.batch_size, len(self.samples))))
            actual = len(indices)
            samples = [self.samples[i] for i in indices]
            samples += [samples[-1]] * (self.batch_size - actual)
            batch = jax.tree.map(lambda *values: np.stack(values), *samples)
            observation = model_lib.Observation.from_dict(batch)
            frame_ids = np.asarray(indices + [indices[-1]] * (self.batch_size - actual), dtype=np.uint32)
            noise = np.stack([np.random.default_rng(np.random.SeedSequence([self.config.seed, int(i)]))
                              .standard_normal(batch["actions"].shape[1:]).astype(np.float32) for i in frame_ids])
            flow = jax.device_get(flow_fn(params, observation, batch["actions"], frame_ids))[:actual]
            l1, l2, count = jax.device_get(sample_fn(params, observation, batch["actions"],
                                                     ~batch["action_is_pad"], noise))
            all_flow.extend(flow.tolist())
            all_l1.extend(l1[:actual].tolist())
            all_l2.extend(l2[:actual].tolist())
            all_count.extend(count[:actual].tolist())
            if offset == 0 or offset // self.batch_size % 50 == 0:
                logging.info("Validation at update %d: %d/%d frames", updates, offset + actual, len(self.samples))
        count = int(sum(all_count))
        if not count or not np.isfinite([*all_flow, *all_l1, *all_l2]).all():
            raise ValueError("Nonfinite validation metrics or no valid real-action elements")
        row = {"optimizer_updates": int(updates), "val_flow_loss": float(np.mean(all_flow)),
               "val_action_l1": float(sum(all_l1) / count), "val_action_l2": float(sum(all_l2) / count),
               "validation_frames": len(all_flow), "valid_action_elements": count,
               "denoising_steps": self.num_steps, "checkpoint": str(checkpoint) if checkpoint else None,
               "origin": origin, "seconds": time.monotonic() - start}
        self.last_frame_metrics = {"flow_loss": np.asarray(all_flow), "action_l1_sum": np.asarray(all_l1),
                                   "action_l2_sum": np.asarray(all_l2), "valid_action_elements": np.asarray(all_count)}
        write_record(self.path, row)
        logging.info("Held-out validation: %s", json.dumps(row))
        return row
