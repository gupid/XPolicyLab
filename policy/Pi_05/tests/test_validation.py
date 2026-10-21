"""Verify fixed per-frame validation RNG, full coverage, masks and no mutation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

POLICY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(POLICY), str(POLICY.parents[2])]
from real_world.validation import HeldOutValidation, read_records, write_record  # noqa: E402


class TinyModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array(2.))
        self.deterministic = False

    def compute_loss(self, rng, observation, actions, *, train):
        assert not train and self.deterministic
        return jnp.mean(jnp.square(jax.random.normal(rng, actions.shape) + self.weight.value - actions), axis=-1)

    def sample_actions(self, rng, observation, *, num_steps, noise):
        assert self.deterministic and num_steps == 10
        return noise + self.weight.value


def samples():
    return [{"image": {}, "image_mask": {}, "state": np.zeros(32, np.float32),
             "actions": np.full((3, 32), i / 10, np.float32),
             "action_is_pad": np.arange(3) >= (i % 3 + 1)} for i in range(5)]


def test_all_frames_masks_fixed_rng_and_batch_invariance(tmp_path):
    model = TinyModel()
    graph, state = nnx.split(model)
    before = np.asarray(state["weight"].value).copy()
    rows = []
    for size in (2, 3):
        directory = tmp_path / str(size)
        cfg = SimpleNamespace(seed=0, batch_size=size, checkpoint_dir=directory)
        evaluator = HeldOutValidation(cfg, dataset=samples())
        row = evaluator.evaluate(graph, state, updates=100)
        rows.append(row)
        assert row["validation_frames"] == 5  # Do not drop or double-count the remainder.
        assert row["valid_action_elements"] == sum((i % 3 + 1) * 8 for i in range(5))
        # Native scalar calls must match the batched/vmapped flow estimator.
        eval_model = nnx.merge(graph, state)
        eval_model.eval()
        reference = [float(jnp.mean(eval_model.compute_loss(jax.random.key(i), None,
                     jnp.asarray(s["actions"])[None], train=False))) for i, s in enumerate(samples())]
        np.testing.assert_allclose(evaluator.last_frame_metrics["flow_loss"], reference, rtol=1e-6)
        # Repeating evaluation cannot advance global RNG or append duplicate steps.
        repeated = evaluator.evaluate(graph, state, updates=100)
        for key in ("val_flow_loss", "val_action_l1", "val_action_l2"):
            assert repeated[key] == row[key]
        assert len(read_records(directory / "validation.jsonl")) == 1
    for key in ("val_flow_loss", "val_action_l1", "val_action_l2"):
        np.testing.assert_allclose(rows[0][key], rows[1][key], rtol=1e-6)
    np.testing.assert_array_equal(state["weight"].value, before)
    assert not model.deterministic  # Evaluation mode stays local to the merged model.


def test_validation_protocol_and_empty_split_fail_loudly(tmp_path):
    cfg = SimpleNamespace(seed=0, batch_size=2, checkpoint_dir=tmp_path)
    with pytest.raises(ValueError, match="Empty"):
        HeldOutValidation(cfg, dataset=[])
    HeldOutValidation(cfg, dataset=samples())
    with pytest.raises(ValueError, match="protocol changed"):
        HeldOutValidation(cfg, num_steps=5, dataset=samples())


def test_atomic_records_sort_and_replace_step(tmp_path):
    path = tmp_path / "validation.jsonl"
    for step, value in ((500, 3.), (1, 7.), (500, 2.)):
        write_record(path, {"optimizer_updates": step, "val_flow_loss": value})
    assert read_records(path) == [{"optimizer_updates": 1, "val_flow_loss": 7.},
                                  {"optimizer_updates": 500, "val_flow_loss": 2.}]


def test_training_loop_runs_validation_and_preserves_best(tmp_path, monkeypatch):
    """Catch the original omission at the training-loop integration boundary."""
    import dataclasses
    from real_world import train

    observations = SimpleNamespace(images={"camera": np.zeros((1, 2, 2, 3))})
    batch = (observations, np.zeros((1, 3, 32)))
    batches_seen, validations, saved, logged = [], [], [], []

    class Loader:
        current_action_valid = np.ones((1, 3), bool)

        def __iter__(self):
            while True:
                yield batch

    class Manager:
        def __init__(self, directory):
            self.directory = Path(directory)

        def wait_until_finished(self):
            pass

    class Validation:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, graph, params, *, updates, checkpoint):
            validations.append(updates)
            return {"val_flow_loss": {1: 3., 3: 1., 5: 2.}[updates],
                    "val_action_l1": .1, "val_action_l2": .02}

    class Actions:
        def __init__(self, *args, **kwargs):
            pass

        def should_record(self, updates):
            return False

    def update(cfg, rng, state, current_batch):
        assert current_batch is batch
        batches_seen.append(int(state.step) + 1)
        return dataclasses.replace(state, step=state.step + 1), {"loss": jnp.array(.5)}

    @dataclasses.dataclass
    class State:
        step: int = 0
        model_def: str = "tiny"
        params: object = None

    cfg = SimpleNamespace(batch_size=1, seed=0, fsdp_devices=1, checkpoint_dir=tmp_path,
                          keep_period=2500, overwrite=False, resume=False, wandb_enabled=False,
                          num_train_steps=5, save_interval=2, log_interval=1)
    upstream = SimpleNamespace(init_logging=lambda: None, init_wandb=lambda *a, **kw: None,
                               init_train_state=lambda *a, **kw: (State(), None), train_step=update)
    monkeypatch.setattr(train, "create_data_loader", lambda *a, **kw: Loader())
    monkeypatch.setattr(train, "HeldOutValidation", Validation)
    monkeypatch.setattr(train, "TrainingActionMetrics", Actions)
    monkeypatch.setattr(train.jax, "jit", lambda fn, **kw: fn)
    monkeypatch.setattr(train.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(train.utils, "array_tree_to_info", lambda value: "tiny")
    monkeypatch.setattr(train.wandb, "Image", lambda value: value)
    monkeypatch.setattr(train.wandb, "log", lambda value, **kw: logged.append(value))
    monkeypatch.setattr(train.checkpoints, "initialize_checkpoint_dir",
                        lambda directory, **kw: (Manager(directory), False))
    monkeypatch.setattr(train.checkpoints, "save_state",
                        lambda manager, state, loader, step: saved.append((manager.directory.name, step)))
    train.main(cfg, upstream)
    assert batches_seen == [1, 2, 3, 4, 5]
    assert validations == [1, 3, 5]  # First, checkpoint boundary, final; no missing phase.
    assert ("best_checkpoint", 0) in saved and ("best_checkpoint", 2) in saved
    assert ("best_checkpoint", 4) not in saved
    best = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert best["optimizer_updates"] == 3 and best["val_flow_loss"] == 1.
    assert Path(best["best_checkpoint"]).parent.name == "best_checkpoint"
    assert sum("val/flow_loss" in row for row in logged) == 3
