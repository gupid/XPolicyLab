"""Stream original archives into review MP4s and independent LeRobot v3 splits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import io
import json
from pathlib import Path
import tarfile

import av
import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
from openpi_client import image_tools

from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info
from real_world.config import TASKS, TASK_PROMPTS, WORKSPACE, repo_id
from real_world.transforms import center_crop_resize


WAIT_ACTION = np.asarray([0, 0, 0, 0, 0, 0, 1, -1], dtype=np.float32)


def split_members(names, seed=0, val_fraction=0.1):
    """Split whole episodes deterministically, independent of tar member ordering."""
    names = sorted(names)
    if len(names) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Need at least two episodes and 0 < val_fraction < 1")
    order = np.random.default_rng(seed).permutation(len(names))
    validation = {names[i] for i in order[:max(1, round(len(names) * val_fraction))]}
    return {name: "val" if name in validation else "train" for name in names}


def episode_arrays(group):
    attrs = group.attrs
    if str(attrs.get("action_representation")) != "delta_ee_pose_xyzw_and_gripper":
        raise ValueError("Unknown action convention; refusing to infer labels")
    if not attrs.get("write_complete", False) or not attrs.get("terminal_valid", False):
        raise ValueError("Incomplete episode")
    if attrs.get("terminal_error", "") or attrs.get("unlogged_command_possible", False):
        raise ValueError("Episode has terminal/command logging errors")
    if abs(float(attrs["configured_control_hz"]) - 15) > 1e-4:
        raise ValueError("Expected 15 Hz source; explicit adaptation required")
    action = group["actions"][:].astype(np.float32)
    state = np.concatenate([group["obs/ee_pose"][:], group["obs/gripper_width"][:]], axis=-1).astype(np.float32)
    n = len(action)
    if action.shape != (n, 8) or state.shape != (n, 8) or n < 2:
        raise ValueError((state.shape, action.shape))
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Nonfinite training fields")
    for label in (state[:, 3:7], action[:, 3:7]):
        if not np.allclose(np.linalg.norm(label, axis=-1), 1, atol=2e-3):
            raise ValueError("Non-unit xyzw quaternion")
    for key in ("color_image1_jpeg", "color_image2_jpeg"):
        if len(group[f"obs/{key}"]) != n:
            raise ValueError(f"Camera length mismatch: {key}")
    stamps = group["timestamps"][:]
    if len(stamps) != n or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        raise ValueError("Invalid source timestamps")
    return state, action


def merge_waiting_prefix(state, action, waiting_frames):
    """Prepend observation-only frames with explicit stationary/open actions."""
    if waiting_frames <= 0:
        return state, action
    prefix_state = np.repeat(state[:1], waiting_frames, axis=0)
    prefix_action = np.repeat(WAIT_ACTION[None], waiting_frames, axis=0)
    return np.concatenate([prefix_state, state]), np.concatenate([prefix_action, action])


class ReviewVideo:
    def __init__(self, path):
        self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=15)
        self.stream.width, self.stream.height, self.stream.pix_fmt = 960, 320, "yuv420p"
        self.stream.options = {"crf": "23", "preset": "fast"}
        self.stream.codec_context.thread_count = 1

    def write(self, images, text):
        canvas = Image.new("RGB", (960, 320))
        for i, image in enumerate(images):
            resized = image_tools.resize_with_pad(image, 270, 480)
            canvas.paste(Image.fromarray(resized), (i * 480, 0))
        ImageDraw.Draw(canvas).text((8, 274), text, fill="white")
        self.write_frame(np.asarray(canvas))

    def write_frame(self, image):
        frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def create_dataset(task, split, root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dim_info = get_robot_action_dim_info("franka_single")
    dim = sum(dim_info["arm_dim"]) + sum(dim_info["ee_dim"])
    if dim != 8:
        raise ValueError("Franka single-arm registry must describe 7 joints + 1 gripper")
    features = {
        "observation.state": {"dtype": "float32", "shape": (dim,),
                              "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width_m"]},
        "action": {"dtype": "float32", "shape": (dim,),
                   "names": ["dx", "dy", "dz", "dqx", "dqy", "dqz", "dqw", "gripper_command"]},
        **{f"observation.images.camera_{i}": {"dtype": "image", "shape": (3, 224, 224),
                                             "names": ["channels", "height", "width"]} for i in (1, 2)},
    }
    return LeRobotDataset.create(repo_id(task, split), root=root / "datasets" / repo_id(task, split),
                                 fps=15, robot_type="franka", features=features, use_videos=False,
                                 image_writer_threads=2, video_backend="pyav", metadata_buffer_size=1)


def process_task(task, args):
    cv2.setNumThreads(1)
    manifest = json.loads((args.source / f"{task}.source_manifest.json").read_text())
    hashes = {item["path"]: item["sha256"] for item in manifest}
    excluded = set(args.exclude.get(task, []))
    if excluded - hashes.keys():
        raise ValueError(f"Unknown excluded source members: {excluded - hashes.keys()}")
    splits = split_members([name for name in hashes if name not in excluded], args.seed)
    output = args.output / task
    output.mkdir(parents=True, exist_ok=False)
    (output / "videos").mkdir()
    datasets = {} if args.review_only else {s: create_dataset(task, s, args.output) for s in ("train", "val")}
    rows = []
    try:
        with tarfile.open(args.source / f"{task}.tar.gz", "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if member.name not in hashes:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                raw = tar.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest() != hashes[member.name]:
                    raise ValueError(f"SHA256 mismatch: {member.name}")
                with h5py.File(io.BytesIO(raw), "r") as h:
                    if list(h["data"]) != ["demo_0"]:
                        raise ValueError("Expected exactly one demo per source file")
                    group = h["data/demo_0"]
                    state, action = episode_arrays(group)
                    initial = group.get("initial_observations") if task == "shell_game" else None
                    initial_count = 0
                    if initial is not None:
                        if "no action labels" not in str(initial.attrs.get("semantics", "")):
                            raise ValueError("Unknown shell-game waiting-segment semantics")
                        initial_count = len(initial["obs/color_image1_jpeg"])
                        if initial_count < 1 or any(len(initial[f"obs/color_image{k}_jpeg"]) != initial_count
                                                    for k in (1, 2)):
                            raise ValueError("Invalid waiting-segment camera frame counts")
                        state, action = merge_waiting_prefix(state, action, initial_count)
                    video_name = Path(member.name).stem + ".mp4"
                    row = {"source": member.name, "sha256": hashes[member.name], "frames": len(action),
                           "waiting_frames_merged": initial_count, "teleoperation_frames": len(action) - initial_count,
                           "waiting_action": WAIT_ACTION.tolist(),
                           "state_policy": "repeat first teleoperation state over observation-only waiting frames"
                           if initial_count else "recorded state",
                           "schema_version": int(group.attrs["schema_version"]),
                           "split": splits.get(member.name, "excluded"), "video": "videos/" + video_name,
                           "measured_hz": float(group.attrs["measured_step_hz"]),
                           "state_first": state[0].tolist(), "action_first": action[0].tolist(),
                           "action_last": action[-1].tolist(),
                           "task_success": "not annotated", "prompt": TASK_PROMPTS[task]}
                    video = ReviewVideo(output / row["video"])
                    try:
                        if initial_count:
                            for i in range(initial_count):
                                images = [decode_image_bit(initial[f"obs/color_image{k}_jpeg"][i],
                                                           source_format=args.jpeg_format) for k in (1, 2)]
                                video.write(images, f"{task} | WAITING OBSERVATION {i+1}/{initial_count} | "
                                            f"synthetic action {WAIT_ACTION.tolist()}")
                                if datasets and member.name not in excluded:
                                    datasets[splits[member.name]].add_frame({
                                        "observation.state": state[i], "action": action[i], "task": TASK_PROMPTS[task],
                                        **{f"observation.images.camera_{k+1}": center_crop_resize(im)
                                           for k, im in enumerate(images)}})
                        for i in range(initial_count, len(action)):
                            # This external corpus declares JPEG; explicit standard import does not
                            # reinterpret unmarked foreign bytes as XPolicyLab legacy image buffers.
                            source_i = i - initial_count
                            images = [decode_image_bit(group[f"obs/color_image{k}_jpeg"][source_i],
                                                       source_format=args.jpeg_format) for k in (1, 2)]
                            command = action[i]
                            video.write(images, f"{task} | {Path(member.name).stem[-14:]} | {i+1}/{len(action)} | {i/15:.2f}s\n"
                                        f"delta xyz {command[:3].round(4)} | gripper {command[7]:+.2f} (-open/+close)")
                            if datasets and member.name not in excluded:
                                datasets[splits[member.name]].add_frame({
                                    "observation.state": state[i], "action": action[i], "task": TASK_PROMPTS[task],
                                    **{f"observation.images.camera_{k+1}": center_crop_resize(im)
                                       for k, im in enumerate(images)}})
                    finally:
                        video.close()
                    if datasets and member.name not in excluded:
                        datasets[splits[member.name]].save_episode()
                    rows.append(row)
                    print(f"{task}: {len(rows)}/{len(hashes)} {row['frames']} frames {row['split']}", flush=True)
                del raw
        if set(hashes) != {r["source"] for r in rows}:
            raise ValueError("Archive membership does not match source manifest")
    finally:
        for dataset in datasets.values():
            dataset.finalize()
            dataset.stop_image_writer()
    report = {"task": task, "seed": args.seed, "fps": 15, "jpeg_format": args.jpeg_format,
              "action_representation": "delta_ee_pose_xyzw_and_gripper", "review_only": args.review_only,
              "image_preprocessing": "center crop 640x640 then resize to 224x224; third image slot black and masked",
              "shell_waiting_segment": "merged at episode start with zero-motion/open-gripper actions",
              "lerobot_version": "v3.0 (lerobot 0.4.4)", "episodes": rows,
              "counts": {s: {"episodes": sum(r["split"] == s for r in rows),
                              "frames": sum(r["frames"] for r in rows if r["split"] == s)}
                         for s in ("train", "val", "excluded")}}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_index(output, rows)
    (output / "COMPLETE").write_text("SHA256 verified; all frames exported. Human task-success review pending.\n")
    return report


def write_index(output, rows):
    # Lazy loading prevents a browser from fetching all trajectory movies at once.
    cards = []
    for r in sorted(rows, key=lambda r: r["source"]):
        if r.get("video_removed"):
            continue
        name, video = html.escape(Path(r["source"]).name), html.escape(r["video"])
        context = (f'<br>Initial reveal: {r["initial_frames"]} frames; unrecorded gap '
                   f'{r["initial_pause_seconds"]:.2f}s; 1s labeled review hold'
                   if "initial_frames" in r else "")
        variation = f'<br>Variation: {html.escape(r["variation"])}' if "variation" in r else ""
        index = f' / dataset episode {r["dataset_episode_index"]}' if "dataset_episode_index" in r else ""
        cards.append(f'<article><p>{name}<br>{r["frames"]} frames / {r["frames"]/15:.1f}s action stream / {r["split"]}{index}{variation}{context}</p>'
                     f'<video controls preload="none" src="{video}"></video><p><a href="{video}">Download MP4</a></p></article>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Trajectory review</title>'
        '<style>body{font:16px sans-serif;background:#171717;color:white;margin:24px}article{margin:24px 0}'
        'video{max-width:100%;width:960px}a{color:#90caf9}</style><h1>Trajectory review</h1>'
        '<p>Left: source camera 1. Right: source camera 2. 15 fps; no task-success filtering.</p>' + "\n".join(cards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORKSPACE / "data/real_world")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "data/real_world/prepared")
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-file", type=Path, help='JSON {task: [exact archive member paths]}')
    parser.add_argument("--jpeg-format", choices=("standard", "auto"), default="standard")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    args.exclude = json.loads(args.exclude_file.read_text()) if args.exclude_file else {}
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if args.task == "all" else (args.task,)
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        list(pool.map(lambda t: process_task(t, args), tasks))
    links = ''.join(f'<li><a href="{t}/index.html">{t}</a></li>' for t in tasks)
    (args.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Real-world demos</h1><ul>' + links + '</ul>')


if __name__ == "__main__":
    main()
