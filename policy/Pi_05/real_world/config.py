"""Register adapter-owned configs without editing the vendored model/trainer."""
import dataclasses
import os
from pathlib import Path

from openpi import transforms
from openpi.training import config as config_lib
from openpi.training import weight_loaders

from real_world.transforms import FrankaInputs, FrankaOutputs

POLICY = Path(__file__).resolve().parents[1]
WORKSPACE = POLICY.parents[2]
TASKS = ("shell_game", "press_button", "put_back_block")
TASK_PROMPTS = {
    "shell_game": "Complete the shell game task.",
    "press_button": "Observe the two numbers on the table. Press the left button the number of times "
                    "corresponding to the number on the left, and press the middle button the number of times "
                    "corresponding to the number on the right. Then press the right button once to confirm.",
    "put_back_block": "There are four mats, one block, and a button on the table. One block is on one of the mats. "
                      "First, put the block to the center, then press the button. Then, put the block back in "
                      "its original position.",
}


def repo_id(task, split="train"):
    if task not in TASKS or split not in {"train", "val"}:
        raise ValueError((task, split))
    return f"real_world-{task}-franka_single-ee-{split}"


@dataclasses.dataclass(frozen=True)
class FrankaDataConfig(config_lib.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        mapping = {"camera_1": "observation.images.camera_1", "camera_2": "observation.images.camera_2",
                   "state": "observation.state", "actions": "action", "prompt": "prompt"}
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(mapping)]),
            data_transforms=transforms.Group(inputs=[FrankaInputs()], outputs=[FrankaOutputs()]),
            model_transforms=config_lib.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def register_configs():
    base = config_lib.get_config("pi05_rmbench_lora_observe_and_pickup")
    root = Path(os.getenv("PI05_REAL_ROOT", WORKSPACE / "data/real_world/prepared"))
    checkpoint = Path(os.getenv("OPENPI_PI05_CHECKPOINT",
                               WORKSPACE / ".cache/openpi/openpi-assets/checkpoints/pi05_base/params"))
    for task in TASKS:
        for split in ("train", "val"):
            name = f"pi05_real_{task}_{split}"
            data = FrankaDataConfig(repo_id=repo_id(task, split),
                                   assets=config_lib.AssetsConfig(assets_dir=str(root / "assets" / f"pi05_real_{task}_train"),
                                                                  asset_id=repo_id(task)),
                                   base_config=config_lib.DataConfig(prompt_from_task=True))
            config_lib._CONFIGS_DICT[name] = dataclasses.replace(
                base, name=name, data=data, assets_base_dir=str(root / "assets"),
                weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint)),
                policy_metadata={"action_representation": "delta_ee_pose_xyzw_and_gripper",
                                 "control_hz": 15, "task": task, "camera_2_slot": "left_wrist_0_rgb",
                                 "third_image": "masked"})
    return root
