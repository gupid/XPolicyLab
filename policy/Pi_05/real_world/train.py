"""OpenPI JAX loop with sampled-action logging, reusing its unmodified train_step."""
import functools
import json
import logging
import os
from pathlib import Path
import platform

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

from openpi.training import checkpoints, sharding, utils
from real_world.action_metrics import TrainingActionMetrics, create_data_loader
from real_world.validation import HeldOutValidation, atomic_json


def main(config, upstream, *, metric_interval=100, denoising_steps=10):
    upstream.init_logging()
    logging.info("Running on: %s", platform.node())
    if config.batch_size % jax.device_count() != 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    cache = epath.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")).expanduser()
    jax.config.update("jax_compilation_cache_dir", str(cache))
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume)
    upstream.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    loader = create_data_loader(config, sharding=data_sharding)
    iterator = iter(loader)
    batch = next(iterator)
    logging.info("Initialized data loader:\n%s", utils.array_tree_to_info(batch))
    images = [wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
              for i in range(min(5, len(next(iter(batch[0].images.values())))))]
    wandb.log({"camera_views": images}, step=0)
    state, state_sharding = upstream.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(state)
    logging.info("Initialized train state:\n%s", utils.array_tree_to_info(state.params))
    if resuming:
        state = checkpoints.restore_state(manager, state, loader)
    train_step = jax.jit(functools.partial(upstream.train_step, config),
                         in_shardings=(replicated, state_sharding, data_sharding),
                         out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    metrics = TrainingActionMetrics(config, interval=metric_interval, num_steps=denoising_steps)
    validation = HeldOutValidation(config, num_steps=denoising_steps)
    best_dir = Path(config.checkpoint_dir) / "best_checkpoint"
    best_manager, _ = checkpoints.initialize_checkpoint_dir(
        best_dir, keep_period=None, overwrite=False, resume=best_dir.exists())
    best_path = Path(config.checkpoint_dir) / "best_checkpoint.json"
    previous_best = json.loads(best_path.read_text()) if best_path.exists() else None
    best_loss = previous_best["val_flow_loss"] if previous_best else float("inf")
    start_step = int(state.step)
    progress = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step,
                         total=config.num_train_steps, dynamic_ncols=True)
    infos = []
    for step in progress:
        checkpoint_due = ((step % config.save_interval == 0 and step > start_step)
                          or step == config.num_train_steps - 1)
        with sharding.set_mesh(mesh):
            state, info = train_step(train_rng, state, batch)
            action_info = metrics.record(state, batch, loader.current_action_valid) if metrics.should_record(step + 1) else None
            validation_info = None
            if step == start_step or checkpoint_due:
                validation_info = validation.evaluate(
                    state.model_def, state.params, updates=step + 1,
                    checkpoint=Path(config.checkpoint_dir) / str(step) if checkpoint_due else None)
                if validation_info["val_flow_loss"] < best_loss:
                    # A separate upstream manager retains the best state even when
                    # the regular manager deletes intermediate checkpoints.
                    checkpoints.save_state(best_manager, state, loader, step)
                    best_manager.wait_until_finished()
                    best_loss = validation_info["val_flow_loss"]
                    atomic_json(best_path, {"selection_metric": "val_flow_loss",
                                           "best_checkpoint": str(best_dir / str(step)),
                                           "optimizer_updates": step + 1,
                                           "val_flow_loss": best_loss,
                                           "metrics": validation_info})
        infos.append(info)
        logged = {}
        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            stacked = common_utils.stack_forest(infos)
            logged.update(jax.device_get(jax.tree.map(jnp.mean, stacked)))
            progress.write(f"Step {step}: " + ", ".join(f"{key}={value:.4f}" for key, value in logged.items()))
            infos = []
        if action_info is not None:
            logged.update(action_info)
        if validation_info is not None:
            logged.update({f"val/{key.removeprefix('val_')}": validation_info[key]
                           for key in ("val_flow_loss", "val_action_l1", "val_action_l2")})
        if logged:
            # Same zero-based W&B step as OpenPI; CSV uses completed update count.
            wandb.log(logged, step=step)
        batch = next(iterator)
        if checkpoint_due:
            checkpoints.save_state(manager, state, loader, step)
    logging.info("Waiting for checkpoint manager to finish")
    manager.wait_until_finished()
    best_manager.wait_until_finished()
