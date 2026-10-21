"""Check sampled-action definitions, masks, unchanged batches and logging on CPU."""
import csv
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

from real_world.action_metrics import (  # noqa: E402
    PreserveActionPadding, TrainingActionMetrics, create_data_loader, masked_action_errors,
)


def test_mae_mse_exclude_padding_and_model_only_dimensions():
    # Real errors [1, -3, 2, -4] => MAE=2.5 and MSE=7.5.
    target = jnp.zeros((1, 3, 5))
    prediction = jnp.array([[[1., -3., 999., 999., 999.],
                             [2., -4., 999., 999., 999.],
                             [jnp.nan, jnp.nan, 999., 999., 999.]]])
    result = jax.jit(lambda p: masked_action_errors(p, target, jnp.array([[True, True, False]]),
                                                  action_dim=2))(prediction)
    assert float(result["action_l1"]) == 2.5
    assert float(result["action_l2"]) == 7.5
    assert int(result["valid_action_elements"]) == 4


def test_padding_survives_repack_without_changing_actions():
    raw = {"action": np.ones((3, 8)), "action_is_pad": np.array([False, False, True])}
    transform = PreserveActionPadding(lambda data: {"actions": data["action"]})
    output = transform(raw)
    np.testing.assert_array_equal(output["actions"], raw["action"])
    np.testing.assert_array_equal(output["action_is_pad"], raw["action_is_pad"])
    with pytest.raises(ValueError, match="horizon"):
        transform({**raw, "action_is_pad": np.array([False])})


class TinySampler(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array(2.))
        self.deterministic = False

    def sample_actions(self, rng, observation, *, num_steps):
        assert self.deterministic
        assert num_steps == 10
        return observation + self.weight.value


def test_compiled_sampling_logs_csv_without_mutating_weights(tmp_path):
    config = SimpleNamespace(seed=0, num_train_steps=201, checkpoint_dir=tmp_path)
    metrics = TrainingActionMetrics(config, interval=100)
    assert [i for i in range(1, 202) if metrics.should_record(i)] == [1, 100, 200, 201]
    model = TinySampler()
    graph, params = nnx.split(model)
    state = SimpleNamespace(step=1, model_def=graph, params=params)
    before = np.asarray(params["weight"].value).copy()
    obs, actions = jnp.zeros((1, 3, 32)), jnp.zeros((1, 3, 32))
    output = metrics.record(state, (obs, actions), jnp.array([[True, False, False]]))
    assert output == {"train/action_l1": 2., "train/action_l2": 4., "train/action_valid_elements": 8}
    np.testing.assert_array_equal(params["weight"].value, before)
    assert model.deterministic is False
    state.step = 2
    assert metrics.record(state, (obs, actions), jnp.ones((1, 3), bool)) is None
    state.step = 100
    metrics.record(state, (obs, actions), jnp.ones((1, 3), bool))
    with (tmp_path / "action_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["optimizer_updates"]) for row in rows] == [1, 100]
    assert [int(row["valid_action_elements"]) for row in rows] == [8, 24]


def test_all_padding_rejected(tmp_path):
    metrics = TrainingActionMetrics(SimpleNamespace(seed=0, num_train_steps=1, checkpoint_dir=tmp_path))
    graph, params = nnx.split(TinySampler())
    state = SimpleNamespace(step=1, model_def=graph, params=params)
    actions = jnp.zeros((1, 3, 32))
    with pytest.raises(ValueError, match="no valid"):
        metrics.record(state, (actions, actions), jnp.zeros((1, 3), bool))
    assert not (tmp_path / "action_metrics.csv").exists()


@pytest.mark.parametrize("task", ["shell_game", "press_button", "put_back_block"])
def test_real_batches_equal_original_loader(task):
    import dataclasses
    from openpi.training import data_loader
    from real_world_entry import resolved_config

    config = dataclasses.replace(resolved_config(task), num_workers=0)
    # Both loaders receive the same shuffle seed and batch size.
    original = data_loader.create_data_loader(config, shuffle=True, num_batches=1)
    instrumented = create_data_loader(config, sharding=None)
    expected = next(iter(original))
    actual = next(iter(instrumented))
    assert jax.tree.structure(expected) == jax.tree.structure(actual)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(left, right)
    valid = np.asarray(instrumented.current_action_valid)
    assert valid.shape == actual[1].shape[:-1]
    assert valid.dtype == bool and valid.any()


def test_resume_rejects_changed_metric_definition():
    from real_world_entry import check_resume_identity
    with pytest.raises(ValueError, match="action_metrics"):
        check_resume_identity({"action_metrics": {"denoising_steps": 10}},
                              {"action_metrics": {"denoising_steps": 5}})
