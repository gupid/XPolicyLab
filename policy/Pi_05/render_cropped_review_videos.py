"""Render review MP4s directly from the stored training camera observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import pyarrow.compute as pc
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from real_world.config import TASKS, repo_id


class ReviewWriter:
    def __init__(self, path: Path):
        self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=15)
        self.stream.width, self.stream.height = 960, 360
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "fast"}
        self.stream.codec_context.thread_count = 1

    def write(self, images, label):
        canvas = Image.new("RGB", (960, 360), "black")
        canvas.paste(images[0].resize((320, 320), Image.Resampling.BILINEAR), (80, 0))
        canvas.paste(images[1].resize((320, 320), Image.Resampling.BILINEAR), (560, 0))
        ImageDraw.Draw(canvas).text((8, 326), label, fill="white")
        frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def decode(payload):
    import io
    return Image.open(io.BytesIO(payload)).convert("RGB")


def old_first_frame(path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            return Image.fromarray(frame.to_ndarray(format="rgb24"))
    raise ValueError(f"No frames in {path}")


def comparison_image(old_path, cropped_images, output, label):
    original = old_first_frame(old_path).resize((960, 360), Image.Resampling.BILINEAR)
    cropped = Image.new("RGB", (960, 360), "black")
    cropped.paste(cropped_images[0].resize((320, 320)), (80, 0))
    cropped.paste(cropped_images[1].resize((320, 320)), (560, 0))
    draw = ImageDraw.Draw(cropped)
    draw.text((8, 326), label, fill="white")
    combined = Image.new("RGB", (1920, 400), "#202020")
    combined.paste(original, (0, 40))
    combined.paste(cropped, (960, 40))
    draw = ImageDraw.Draw(combined)
    draw.text((12, 12), "Previous review video: full camera views", fill="white")
    draw.text((972, 12), "Stored training observations: center crop 640, resized 224", fill="white")
    combined.save(output, quality=92)


def render_preview(root: Path, task: str, preview_dir: Path):
    report = json.loads((root / task / "report.json").read_text())
    row_info = min((row for row in report["episodes"] if row["split"] == "val"),
                   key=lambda row: row["dataset_episode_index"])
    split, episode = row_info["split"], row_info["dataset_episode_index"]
    dataset = root / "datasets" / repo_id(task, split)
    metadata = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet",
                             columns=["episode_index", "data/chunk_index", "data/file_index"])
    episode_rows = metadata.filter(pc.equal(metadata["episode_index"], episode)).to_pylist()
    if len(episode_rows) != 1:
        raise FileNotFoundError(f"Expected metadata for {task}/{split}/{episode}, got {episode_rows}")
    location = episode_rows[0]
    parquet_path = dataset / "data" / f"chunk-{location['data/chunk_index']:03d}" / \
        f"file-{location['data/file_index']:03d}.parquet"
    table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "observation.images.camera_1",
                                                 "observation.images.camera_2", "action"])
    writer = ReviewWriter(preview_dir / f"{task}_cropped_input.mp4")
    first_images = None
    count = 0
    for row in table.to_pylist():
        if int(row["episode_index"]) != episode:
            continue
        if int(row["frame_index"]) != count:
            raise ValueError(f"Noncontiguous frame index for {task}/{episode}")
        images = [decode(row[f"observation.images.camera_{camera}"]["bytes"])
                  for camera in (1, 2)]
        if first_images is None:
            first_images = images
        action = np.asarray(row["action"], dtype=np.float32)
        writer.write(images, f"{task} | {row_info['variation']} | {count + 1}/{row_info['frames']} | "
                     f"dxyz {action[:3].round(3).tolist()} | grip {action[7]:+.2f}")
        count += 1
    writer.close()
    assert count == row_info["frames"], (task, count, row_info["frames"])
    old_root = root.parent / "prepared.before_crop640_waitmerge_15349"
    comparison_image(old_root / task / row_info["video"], first_images,
                     preview_dir / f"{task}_crop_comparison.jpg",
                     f"{task} | {row_info['variation']} | frame 0")
    print(f"{task}: previewed {count} stored training observations", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    preview_dir = args.root / "crop_review"
    preview_dir.mkdir(exist_ok=True)
    for task in TASKS:
        render_preview(args.root, task, preview_dir)
    gallery = ['<!doctype html><meta charset="utf-8"><title>Crop comparison</title>',
               '<style>body{font:16px sans-serif;background:#181818;color:#eee;margin:24px}img{max-width:100%;display:block;margin:16px 0 40px}</style>',
               '<h1>Original review video vs stored training observations</h1>']
    for task in TASKS:
        gallery.append(f'<h2>{task}</h2><img src="{task}_crop_comparison.jpg">'
                       f'<video controls preload="none" src="{task}_cropped_input.mp4"></video>')
    (preview_dir / "index.html").write_text("\n".join(gallery))


if __name__ == "__main__":
    main()
