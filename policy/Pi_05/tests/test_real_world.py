"""Check source semantics, RGB import, trajectory split and actual OpenPI batches."""
import io
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
import pytest

POLICY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(POLICY), str(POLICY.parents[2])]
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit, get_robot_action_dim_info  # noqa: E402
from real_world.config import TASKS, FrankaDataConfig, register_configs, repo_id  # noqa: E402
from real_world.prepare import episode_arrays, merge_waiting_prefix, split_members  # noqa: E402
from real_world.transforms import FrankaInputs, FrankaOutputs, center_crop_resize  # noqa: E402
from real_world_entry import check_resume_identity  # noqa: E402


def test_external_rgb_and_existing_decoder():
    red = np.zeros((16, 16, 3), np.uint8)
    red[..., 0] = 245
    stream = io.BytesIO()
    Image.fromarray(red).save(stream, format="JPEG", quality=100)
    foreign = stream.getvalue()
    rgb = decode_image_bit(foreign, source_format="standard")
    assert rgb[..., 0].mean() > 240 and rgb[..., 2].mean() < 5
    np.testing.assert_array_equal(decode_image_bit([foreign, foreign], source_format="standard")[0], rgb)
    marked = encode_image_bit(red)
    assert decode_image_bit(marked)[..., 0].mean() > 235
    assert decode_image_bit(foreign)[..., 2].mean() > 240  # default legacy behavior unchanged
    with pytest.raises(ValueError):
        decode_image_bit(foreign, source_format="guess")


def test_whole_episode_split():
    names = [f"task/episode_{i:03}.hdf5" for i in range(50)]
    a = split_members(names)
    assert a == split_members(names[::-1])
    assert list(a.values()).count("val") == 5
    assert set(a) == set(names)


def test_reviewed_splits_cover_variations_and_preserve_all_frames():
    from real_world.resplit import validate_plan
    root = register_configs()
    path = root / "split_plan.json"
    if not path.exists():
        pytest.skip("Reviewed split not applied yet")
    plan = json.loads(path.read_text())
    for task, selected in plan["tasks"].items():
        report = json.loads((root / task / "report.json").read_text())
        validate_plan(report, selected)
        proof = json.loads((root / task / "resplit_integrity.json").read_text())
        assert set(proof) == set(selected["variants"])
        assert {r["source"] for r in report["episodes"] if r["split"] == "val"} == set(selected["validation_sources"])
        for split in ("train", "val"):
            rows = [r for r in report["episodes"] if r["split"] == split]
            assert [r["dataset_episode_index"] for r in rows] == list(range(len(rows)))
            for row in rows:
                assert row["variation"] == selected["variants"][row["source"]]
                assert proof[row["source"]]["all_frames_equal_before_after"]
                assert proof[row["source"]]["frames"] == row["frames"]


def test_resume_rejects_changed_data_or_normalization():
    baseline = {"dataset_report_sha256": "source-a", "normalization_sha256": "norm-a"}
    check_resume_identity(baseline, dict(baseline))
    for key in baseline:
        changed = {**baseline, key: "changed"}
        with pytest.raises(ValueError, match=key):
            check_resume_identity(baseline, changed)


def make_episode(file, n=4):
    g = file.create_group("data/demo_0")
    for k, v in {"action_representation": "delta_ee_pose_xyzw_and_gripper", "write_complete": True,
                 "terminal_valid": True, "configured_control_hz": 15.0}.items():
        g.attrs[k] = v
    action = np.zeros((n, 8), np.float32)
    action[:, 6], action[:, 7] = 1, -1
    action[:, 0] = np.arange(n) * .001
    pose = np.zeros((n, 7), np.float32)
    pose[:, 6] = 1
    g.create_dataset("actions", data=action)
    g.create_dataset("obs/ee_pose", data=pose)
    g.create_dataset("obs/gripper_width", data=np.full((n, 1), .065, np.float32))
    g.create_dataset("timestamps", data=np.arange(n) / 15)
    for k in (1, 2):
        g.create_dataset(f"obs/color_image{k}_jpeg", shape=(n,), dtype=h5py.vlen_dtype(np.uint8))
    return g, action


def test_action_labels_not_future_joint_states(tmp_path):
    with h5py.File(tmp_path / "fixture.hdf5", "w") as f:
        g, original = make_episode(f)
        state, action = episode_arrays(g)
        np.testing.assert_array_equal(action, original)
        assert state.shape == (4, 8)
        assert state[0, -1] == np.float32(.065)
        g["actions"][0, 0] = np.nan
        with pytest.raises(ValueError, match="Nonfinite"):
            episode_arrays(g)


def test_center_crop_resize_and_input_transform_masks_third_camera():
    image = np.zeros((720, 1280, 3), np.uint8)
    image[40:680, 320:960] = (20, 120, 240)
    cropped = center_crop_resize(image)
    assert cropped.shape == (224, 224, 3)
    assert np.allclose(cropped.mean(axis=(0, 1)), (20, 120, 240), atol=1)
    with pytest.raises(ValueError, match="smaller than required"):
        center_crop_resize(np.zeros((480, 640, 3), np.uint8))


def test_delta_transform_and_missing_camera():
    data = {"camera_1": np.zeros((3, 224, 224), np.float32),
            "camera_2": np.zeros((3, 224, 224), np.float32),
            "state": np.arange(8, dtype=np.float32), "actions": np.ones((50, 8), np.float32),
            "prompt": "task"}
    out = FrankaInputs()(data)
    np.testing.assert_array_equal(out["actions"], data["actions"])
    assert not out["image_mask"]["right_wrist_0_rgb"]
    assert not out["image"]["right_wrist_0_rgb"].any()
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert FrankaOutputs()({"actions": np.zeros((50, 32))})["actions"].shape == (50, 8)


def test_waiting_prefix_has_requested_synthetic_actions():
    state = np.zeros((3, 8), np.float32)
    state[:, 6] = 1
    action = np.zeros((3, 8), np.float32)
    action[:, 6] = 1
    merged_state, merged_action = merge_waiting_prefix(state, action, 2)
    assert merged_state.shape == (5, 8) and merged_action.shape == (5, 8)
    np.testing.assert_array_equal(merged_state[:2], np.repeat(state[:1], 2, axis=0))
    np.testing.assert_array_equal(merged_action[:2], np.tile([0, 0, 0, 0, 0, 0, 1, -1], (2, 1)))
    np.testing.assert_array_equal(merged_state[2:], state)
    np.testing.assert_array_equal(merged_action[2:], action)


def test_configs_preserve_existing_lora_defaults():
    from openpi.training import config
    register_configs()
    base = config.get_config("pi05_rmbench_lora_observe_and_pickup")
    for task in TASKS:
        cfg = config.get_config(f"pi05_real_{task}_train")
        assert cfg.model == base.model
        assert cfg.optimizer == base.optimizer and cfg.lr_schedule == base.lr_schedule
        assert cfg.batch_size == 8 and cfg.num_train_steps == 10000
        assert cfg.save_interval == 500 and cfg.keep_period == 2500
        assert isinstance(cfg.data, FrankaDataConfig)
    assert get_robot_action_dim_info("franka_single") == {"arm_dim": [7], "ee_dim": [1]}


@pytest.mark.parametrize("task", TASKS)
def test_prepared_real_dataset(task):
    from openpi.training import config, data_loader
    root = register_configs()
    report_path = root / task / "report.json"
    if not report_path.exists():
        pytest.skip("Run prepare_real_world.py first")
    report = json.loads(report_path.read_text())
    if task == "shell_game":
        assert all(r["waiting_frames_merged"] >= 0 for r in report["episodes"])
        assert all(r["frames"] == r["teleoperation_frames"] + r["waiting_frames_merged"]
                   for r in report["episodes"])
    train_sources = {r["source"] for r in report["episodes"] if r["split"] == "train"}
    val_sources = {r["source"] for r in report["episodes"] if r["split"] == "val"}
    assert train_sources and val_sources and not train_sources & val_sources
    for split in ("train", "val"):
        cfg = config.get_config(f"pi05_real_{task}_{split}")
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        ds = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
        assert len(ds) == report["counts"][split]["frames"]
        metadata = ds._dataset.meta if hasattr(ds, "_dataset") else ds.meta
        assert metadata.fps == 15
        row = next(r for r in report["episodes"] if r["split"] == split)
        sample = ds[0]
        np.testing.assert_allclose(sample["action"][0], row["action_first"])
        np.testing.assert_allclose(sample["observation.state"], row["state_first"])
        assert sample["action"].shape == (50, 8)
        repacked = dc.repack_transforms.inputs[0](sample)
        robot_sample = dc.data_transforms.inputs[0](repacked)
        assert not bool(robot_sample["image_mask"]["right_wrist_0_rgb"])
        assert not robot_sample["image"]["right_wrist_0_rgb"].any()
        if task == "shell_game":
            offset = 0
            for episode in (r for r in report["episodes"] if r["split"] == split):
                first = ds[offset]
                assert int(first["episode_index"]) == episode["dataset_episode_index"]
                np.testing.assert_allclose(first["action"][0], episode["action_first"])
                np.testing.assert_allclose(first["observation.state"], episode["state_first"])
                first_input = dc.data_transforms.inputs[0](dc.repack_transforms.inputs[0](first))
                assert not first_input["image_mask"]["right_wrist_0_rgb"]
                offset += episode["frames"]
            assert offset == len(ds)
        from real_world.norm import numeric_columns, action_windows
        states, actions, ends = numeric_columns(root / "datasets" / repo_id(task, split))
        indices = np.unique(np.r_[0, len(ds) - 1, np.flatnonzero(np.diff(ends))])
        # Check the projected numeric stream against the official temporal loader
        # at every episode end (where padding must not cross into the next one).
        for index in indices:
            official = ds[int(index)]
            np.testing.assert_array_equal(states[index], official["observation.state"])
            np.testing.assert_array_equal(action_windows(actions, ends, np.array([index]), 50)[0], official["action"])
        # At episode end, the official loader pads within that episode, never into the next one.
        end = ds[row["frames"] - 1]
        np.testing.assert_allclose(end["action"][-1], row["action_last"])
        if dc.norm_stats is not None:
            transformed = data_loader.transform_dataset(ds, dc)[0]
            assert transformed["actions"].shape == (50, 32)
            assert transformed["state"].shape == (32,)
            assert np.isfinite(transformed["actions"]).all()
    # Validation uses training statistics, never its own estimated normalization.
    a = config.get_config(f"pi05_real_{task}_train").data
    b = config.get_config(f"pi05_real_{task}_val").data
    assert a.assets == b.assets and a.assets.asset_id == repo_id(task)


def test_numeric_stats_match_full_official_reference():
    from openpi.shared import normalize
    from real_world.config import register_configs
    root = register_configs()
    reference = root / "official_shell_norm_reference.json"
    if not reference.exists():
        pytest.skip("Full official reference not generated yet")
    expected = normalize.deserialize_json(reference.read_text())
    actual = normalize.deserialize_json(
        (root / "assets" / "pi05_real_shell_game_train" / repo_id("shell_game") / "norm_stats.json").read_text()
    )
    for key in expected:
        for field in ("mean", "std", "q01", "q99"):
            np.testing.assert_array_equal(getattr(actual[key], field), getattr(expected[key], field))


def test_training_gripper_normalization_has_no_collapsed_range():
    from openpi.shared import normalize
    from real_world.norm import numeric_columns
    root = register_configs()
    for task in TASKS:
        target = root / 'assets' / f'pi05_real_{task}_train' / repo_id(task) / 'norm_stats.json'
        if not target.exists():
            pytest.skip('Prepared normalization required')
        stats = normalize.load(target.parent)
        states, actions, _ = numeric_columns(root / 'datasets' / repo_id(task))
        for key, values in [('state', states), ('actions', actions)]:
            quantile_span = float(stats[key].q99[7] - stats[key].q01[7])
            full_span = float(values[:, 7].max() - values[:, 7].min())
            assert (quantile_span == 0 and full_span == 0) or (
                quantile_span > 0 and quantile_span >= 0.01 * full_span
            ), (
                task, key, quantile_span, full_span
            )
