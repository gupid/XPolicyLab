"""Sampled training-action diagnostics; never part of the optimization objective."""
import csv
import dataclasses
import json
import logging
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.training import data_loader as loaders
from XPolicyLab.utils.process_data import get_robot_action_dim_info


@dataclasses.dataclass(frozen=True)
class PreserveActionPadding:
    transform: object

    def __call__(self, data):
        # RepackTransform intentionally removes metadata. Save it before repacking.
        padding = np.asarray(data["action_is_pad"], dtype=bool).copy()
        result = self.transform(data)
        if padding.shape != result["actions"].shape[:-1]:
            raise ValueError("Action padding mask must match the action horizon")
        result["action_is_pad"] = padding
        return result


class ActionMetricDataLoader(loaders.DataLoaderImpl):
    """Same model batch and shuffle order; retain the current batch's padding mask."""

    def __iter__(self):
        for batch in self._data_loader:
            self.current_action_valid = ~batch["action_is_pad"]
            yield model_lib.Observation.from_dict(batch), batch["actions"]


def create_data_loader(config, *, sharding):
    dc = config.data.create(config.assets_dirs, config.model)
    if dc.norm_stats is None or dc.rlds_data_dir is not None:
        raise ValueError("Training-action metrics require normalized LeRobot data")
    raw = loaders.create_torch_dataset(dc, config.model.action_horizon, config.model)
    transform = transforms.compose([
        *dc.repack_transforms.inputs, *dc.data_transforms.inputs,
        transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ])
    dataset = loaders.TransformedDataset(raw, [PreserveActionPadding(transform)])
    loader = loaders.TorchDataLoader(
        dataset, local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding, shuffle=True, num_workers=config.num_workers, seed=config.seed,
    )
    return ActionMetricDataLoader(dc, loader)


def masked_action_errors(prediction, target, valid, *, action_dim):
    """Normalized real-action MAE/MSE; exclude padded dimensions and time positions."""
    error = prediction[..., :action_dim].astype(jnp.float32) - target[..., :action_dim].astype(jnp.float32)
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    count = jnp.sum(valid) * action_dim
    # where, rather than multiplication, also excludes nonfinite masked entries.
    absolute_sum = jnp.sum(jnp.where(valid[..., None], jnp.abs(error), 0))
    square_sum = jnp.sum(jnp.where(valid[..., None], jnp.square(error), 0))
    return {"action_l1": absolute_sum / jnp.maximum(count, 1),
            "action_l2": square_sum / jnp.maximum(count, 1),
            "valid_action_elements": count}


def sample_action_errors(model, rng, observation, actions, valid, *, action_dim, num_steps):
    model.eval()
    prediction = model.sample_actions(rng, observation, num_steps=num_steps)
    return masked_action_errors(prediction, actions, valid, action_dim=action_dim)


class TrainingActionMetrics:
    def __init__(self, config, *, interval=100, num_steps=10):
        if interval < 1 or num_steps < 1:
            raise ValueError("Positive action metric interval and denoising steps required")
        info = get_robot_action_dim_info("franka_single")
        self.action_dim = sum(info["arm_dim"]) + sum(info["ee_dim"])
        self.interval, self.num_steps = interval, num_steps
        self.total_steps = config.num_train_steps
        self.seed = config.seed
        self.path = Path(config.checkpoint_dir) / "action_metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = ["optimizer_updates", "action_l1", "action_l2", "valid_action_elements", "denoising_steps"]
        if self.path.exists():
            with self.path.open(newline="") as stream:
                if csv.DictReader(stream).fieldnames != self.fields:
                    raise ValueError(f"Unexpected action-metrics CSV header: {self.path}")
        metadata = {"split": "current training batch", "interval": interval, "denoising_steps": num_steps,
                    "action_dim": self.action_dim, "coordinates": "training-normalized",
                    "action_l1": "mean absolute component error (MAE)",
                    "action_l2": "mean squared component error (MSE), not Euclidean norm",
                    "padding": "exclude action_is_pad time positions and model-only dimensions",
                    "weights": "after current optimizer update, no EMA",
                    "observation": "no training image augmentation for this inference diagnostic",
                    "rng": "fold_in(key(seed + 1), optimizer_updates); separate from training RNG",
                    "objective": "unchanged official flow matching loss"}
        self.path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        self._compiled = None

    def should_record(self, updates):
        return updates == 1 or updates % self.interval == 0 or updates == self.total_steps

    def record(self, state, batch, valid):
        updates = int(state.step)
        if not self.should_record(updates):
            return None
        if self._compiled is None:
            # The graph is static; pass only parameter arrays on subsequent calls.
            graphdef = state.model_def

            def evaluate(params, rng, observation, actions, mask):
                model = nnx.merge(graphdef, params)
                return sample_action_errors(model, rng, observation, actions, mask,
                                            action_dim=self.action_dim, num_steps=self.num_steps)

            self._compiled = jax.jit(evaluate)
        rng = jax.random.fold_in(jax.random.key(self.seed + 1), updates)
        values = jax.device_get(self._compiled(state.params, rng, *batch, valid))
        if int(values["valid_action_elements"]) == 0 or not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Nonfinite action metric or no valid action elements")
        row = {"optimizer_updates": updates, "action_l1": float(values["action_l1"]),
               "action_l2": float(values["action_l2"]),
               "valid_action_elements": int(values["valid_action_elements"]), "denoising_steps": self.num_steps}
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logging.info("Training actions at update %d: L1=%.6f, L2(MSE)=%.6f", updates,
                     row["action_l1"], row["action_l2"])
        return {"train/action_l1": row["action_l1"], "train/action_l2": row["action_l2"],
                "train/action_valid_elements": row["valid_action_elements"]}
