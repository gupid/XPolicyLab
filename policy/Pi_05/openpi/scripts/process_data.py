import os
import json
import numpy as np
import shutil
import argparse
import cv2
import h5py
import dataclasses
from pathlib import Path
from typing import Any, Literal
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_yaml, load_json
from XPolicyLab.utils.process_data import decode_image_bit, pack_robot_state

try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError as exc:
    if exc.name != "lerobot.datasets":
        raise
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROOT_PATH = Path(__file__).resolve().parents[5]

CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 1
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    mode: Literal["video", "image"] = "image",
    *,
    camera_shapes: dict[str, tuple[int, int, int]],
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    robot_action_dim_info: dict = None,
) -> LeRobotDataset:
    
    MOTORS = [
        *[f"left_{i}" for i in range(robot_action_dim_info["arm_dim"][0])],
        *[f"left_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][0])],
        *[f"right_{i}" for i in range(robot_action_dim_info["arm_dim"][1])],
        *[f"right_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][1])]
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": MOTORS,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }

    for camera_name, camera_shape in camera_shapes.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": camera_shape,
            "names": ["channels", "height", "width"],
        }

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def _load_compressed_images(group: h5py.Group, key: str) -> np.ndarray:
    return np.asarray(decode_image_bit(group[key][:]))


def _load_camera_shapes(ep_path: Path) -> dict[str, tuple[int, int, int]]:
    shapes = {}
    with h5py.File(ep_path, "r") as ep:
        if "vision" in ep:
            for source_name, output_name in CAMERA_ALIASES.items():
                source_shape = tuple(int(dim) for dim in ep[f"vision/{source_name}/shape"][:])
                if len(source_shape) != 3 or source_shape[-1] != 3:
                    raise ValueError(f"Unsupported camera shape for {source_name}: {source_shape}")
                height, width, channels = source_shape
                shapes[output_name] = (channels, height, width)
        elif "observation" in ep:
            native_aliases = {
                "head_camera": "cam_high",
                "left_camera": "cam_left_wrist",
                "right_camera": "cam_right_wrist",
            }
            for source_name, output_name in native_aliases.items():
                sample = decode_image_bit(ep[f"observation/{source_name}/rgb"][0])
                if sample.ndim != 3 or sample.shape[-1] != 3:
                    raise ValueError(f"Unsupported camera shape for {source_name}: {sample.shape}")
                height, width, channels = sample.shape
                shapes[output_name] = (channels, height, width)
        else:
            raise ValueError(f"Unsupported HDF5 layout in {ep_path}")
    return shapes

def _load_instructions(ep: h5py.File) -> list[str] | None:
    if "instructions" not in ep:
        return None

    value = ep["instructions"][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("instructions must decode to a string list")
    return value or None


def load_data(ep_path, action_type: str, robot_action_dim_info: dict) -> dict[str, Any]:
    with h5py.File(ep_path, "r") as ep:
        if "state" in ep and "action" in ep:
            state = pack_robot_state(
                {"state": ep["state"]},
                action_type,
                robot_action_dim_info,
                source_type="dataset",
                state_type="state",
            ).astype(np.float32)
            action = pack_robot_state(
                {"action": ep["action"]},
                action_type,
                robot_action_dim_info,
                source_type="dataset",
                state_type="action",
            ).astype(np.float32)

            images = {}
            for source_name, output_name in CAMERA_ALIASES.items():
                if source_name in ep["vision"]:
                    images[output_name] = _load_compressed_images(ep["vision"][source_name], "colors")
            instructions = _load_instructions(ep)
        elif "joint_action" in ep and "observation" in ep:
            # Native RMBench recordings store observations only.  Use adjacent
            # qpos frames as the supervised state/action pair without changing
            # the source HDF5 file.
            qpos = np.asarray(ep["joint_action/vector"], dtype=np.float32)
            if qpos.ndim != 2 or qpos.shape[-1] != 14 or qpos.shape[0] < 2:
                raise ValueError(f"Unsupported native RMBench qpos shape: {qpos.shape}")
            state, action = qpos[:-1], qpos[1:]
            native_aliases = {
                "head_camera": "cam_high",
                "left_camera": "cam_left_wrist",
                "right_camera": "cam_right_wrist",
            }
            images = {
                output_name: decode_image_bit(ep[f"observation/{source_name}/rgb"][:])[:-1]
                for source_name, output_name in native_aliases.items()
            }
            instructions = None
        else:
            raise ValueError(f"Unsupported HDF5 layout in {ep_path}")

    if instructions is None:
        instruction_path = ep_path.parent.parent / "instructions" / f"{ep_path.stem}.json"
        if instruction_path.exists():
            with instruction_path.open(encoding="utf-8") as instruction_file:
                value = json.load(instruction_file)
            instructions = value if isinstance(value, list) else [str(value)]

    return {
        "images": images,
        "state": state,
        "action": action,
        "velocity": None,
        "effort": None,
        "timestamps": None,
        "instructions": instructions,
    }

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("bench_name", type=str, help="Dataset bench name (e.g., RoboDojo)")
    parser.add_argument("ckpt_name", type=str, help="Run name; also selects raw task dir under data/<bench>/")
    parser.add_argument("env_cfg_type", type=str, help="Environment config type (e.g., arx_x5)")
    parser.add_argument("action_type", type=str, help="Action type for artifact naming (e.g., joint)")
    parser.add_argument(
        "expert_data_num",
        type=str,
        nargs="?",
        default=None,
        help="Optional number of episodes to process; non-numeric values are treated as raw_task_dirs.",
    )
    parser.add_argument(
        "raw_task_dirs",
        type=str,
        nargs="?",
        default=None,
        help="Optional raw task dir or comma-separated dirs under data/<bench>/; defaults to ckpt_name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="image",
        help="Whether to store images as videos or individual image files",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Do your job.",
        help="Default instruction when not present in HDF5",
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        default=None,
        help="Directory containing raw task directories; defaults to data/<bench_name>",
    )
    args = parser.parse_args()

    bench_name = args.bench_name
    ckpt_name = args.ckpt_name
    env_cfg_type = args.env_cfg_type
    action_type = args.action_type
    repo_id = f"{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"
    mode = args.mode
    instruction = args.instruction
    expert_data_num = None
    raw_task_dirs_arg = args.raw_task_dirs
    if args.expert_data_num is not None:
        try:
            expert_data_num = int(args.expert_data_num)
        except ValueError:
            if args.raw_task_dirs is not None:
                raise ValueError("raw_task_dirs was provided twice.") from None
            raw_task_dirs_arg = args.expert_data_num
    raw_task_dirs = [item.strip() for item in (raw_task_dirs_arg or ckpt_name).split(",") if item.strip()]

    env_cfg = load_yaml(os.path.join(ROOT_PATH, "./env_cfg", f"{env_cfg_type}.yml"))
    robot_type = env_cfg['config']['robot']

    robot_action_dim_info = load_json(
        os.path.join(ROOT_PATH, "env_cfg/robot", "_robot_info.json")
    )[robot_type]

    raw_data_root = args.raw_data_root or Path(ROOT_PATH) / "data" / str(bench_name)
    episode_files = []
    for raw_task_dir in raw_task_dirs:
        # XPolicyLab data uses <task>/<robot>/data; native RMBench exports use
        # <task>/demo_clean/data.  Accept both layouts while keeping the source
        # directory read-only.
        candidate_dirs = [
            raw_data_root / raw_task_dir / str(env_cfg_type),
            raw_data_root / raw_task_dir / str(env_cfg_type) / "data",
            raw_data_root / raw_task_dir / "demo_clean",
            raw_data_root / raw_task_dir / "demo_clean" / "data",
            raw_data_root / raw_task_dir,
            raw_data_root / raw_task_dir / "data",
        ]
        for load_data_dir in candidate_dirs:
            task_episode_files = sorted(load_data_dir.glob("data/episode*.hdf5"))
            if not task_episode_files:
                task_episode_files = sorted(load_data_dir.glob("episode*.hdf5"))
            if task_episode_files:
                episode_files.extend(task_episode_files)
                break
    if expert_data_num is not None:
        episode_files = episode_files[:expert_data_num]
    if not episode_files:
        raise FileNotFoundError(
            f"No HDF5 episodes found under {raw_data_root} for tasks {raw_task_dirs} "
            f"and environment {env_cfg_type}."
        )

    print(f"Loading {len(episode_files)} episodes from {raw_data_root}")
    camera_shapes = _load_camera_shapes(episode_files[0])
    print(f"Camera shapes: {camera_shapes}")
    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=50, # pi default to 50
        mode=mode,
        camera_shapes=camera_shapes,
        dataset_config=DEFAULT_DATASET_CONFIG,
        robot_action_dim_info=robot_action_dim_info,
    )

    for ep_file in tqdm(episode_files, desc="Processing episodes", unit="episode"):
        try:
            data = load_data(ep_file, action_type, robot_action_dim_info)
            num_frames = data["state"].shape[0]
            task = instruction if data["instructions"] is None else data["instructions"][0]

            for i in range(num_frames):
                frame = {
                    "observation.state": data["state"][i],
                    "action": data["action"][i],
                    "task": task,
                }
                for camera_name, images in data["images"].items():
                    frame[f"observation.images.{camera_name}"] = images[i]

                dataset.add_frame(frame)
            
            dataset.save_episode()
            dataset.hf_dataset = dataset.create_hf_dataset()
            tqdm.write(f"Finished {ep_file.name} with {num_frames} frames")
        except Exception as e:
            raise RuntimeError(f"Error processing episode {ep_file}: {e}") from e

if __name__ == "__main__":
    main()
