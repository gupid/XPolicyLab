#!/usr/bin/env python3
"""Cache the reviewed LeRobot v3 train/val splits for LingBot-VA LoRA."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

POLICY = Path(__file__).resolve().parent
sys.path.insert(0, str(POLICY / 'lingbot_va'))
from wan_va.real_world_dataset import ACTION_CHANNELS, CAMERAS, TASK_STEPS, action_statistics, align_actions


def read_split(root, task, split):
    source = root / 'datasets' / f'real_world-{task}-franka_single-ee-{split}'
    info = json.loads((source / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0' or info['fps'] != 15 or info['features']['action']['shape'] != [8]:
        raise ValueError(f'Expected prepared v3, 15 Hz, 8D actions: {source}')
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((source / 'meta/episodes').rglob('*.parquet'))])
    actions = pd.read_parquet(source / 'data', columns=['action', 'episode_index', 'frame_index'])
    if len(episodes) != info['total_episodes'] or len(actions) != info['total_frames']:
        raise ValueError(f'Dataset metadata/count mismatch: {source}')
    return source, episodes.sort_values('episode_index'), actions


def window_starts(length, hop=48):
    # Stop once the final window covers the episode; never cross episode boundaries.
    starts = [0]
    while starts[-1] + 96 < length:
        starts.append(starts[-1] + hop)
    return starts


def inspect(root, task):
    report = json.loads((root / task / 'report.json').read_text())
    sources = {}
    for split in ('train', 'val'):
        source, episodes, actions = read_split(root, task, split)
        expected = report['counts'][split]
        if len(episodes) != expected['episodes'] or len(actions) != expected['frames']:
            raise ValueError(f'Reviewed split differs: {task}/{split}')
        sources[split] = (source, episodes, actions)
    train = np.stack(sources['train'][2]['action']).astype(np.float32)
    return sources, action_statistics(train), report


def decode_frame(value, source):
    # HF image features contain standard PNG/JPEG bytes; use the shared decoder.
    from XPolicyLab.utils.process_data import decode_image_bit
    if isinstance(value, dict):
        value = value.get('bytes') or (source / value['path']).read_bytes()
    frame = decode_image_bit(value, source_format='standard')
    if frame.shape != (224, 224, 3):
        raise ValueError(f'Expected already cropped prepared RGB image, got {frame.shape}')
    return frame


@torch.no_grad()
def prepare(args, sources, norm_stat, report):
    # Reuse the existing upstream VAE normalization and UMT5 embedding path.
    sys.path.insert(0, str(POLICY.parents[2]))
    from process_data import encode_text, encode_video_latent
    from modules.utils import load_text_encoder, load_tokenizer, load_vae, WanVAEStreamingWrapper

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompts = {entry['prompt'] for entry in report['episodes'] if entry['split'] in ('train', 'val')}
    if len(prompts) != 1:
        raise ValueError('Expected a single reviewed task prompt')
    prompt = prompts.pop()
    text_encoder = load_text_encoder(str(args.base_model / 'text_encoder'), torch.bfloat16, args.device)
    text_encoder.eval().requires_grad_(False)
    tokenizer = load_tokenizer(str(args.base_model / 'tokenizer'))
    torch.save(encode_text(prompt, tokenizer, text_encoder), output / 'text_emb.pt')
    torch.save(encode_text('', tokenizer, text_encoder), output / 'empty_emb.pt')
    del text_encoder, tokenizer
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    vae = load_vae(str(args.base_model / 'vae'), torch.bfloat16, args.device)
    vae.eval().requires_grad_(False)
    wrapper = WanVAEStreamingWrapper(vae)
    metadata = dict(format='lingbot_real_world_v1', task=args.task, source=str(args.prepared_root.resolve()),
                    base_model=str(args.base_model.resolve()), norm_stat=norm_stat,
                    used_action_channel_ids=ACTION_CHANNELS, image_size=256, stride=3, video_frames=33,
                    action_representation='delta_ee_pose_xyzw_and_gripper', control_hz=15,
                    obs_cam_keys=CAMERAS, counts=report['counts'], samples={'train': [], 'val': []},
                    split_report_sha256=hashlib.sha256((args.prepared_root / args.task / 'report.json').read_bytes()).hexdigest())
    for split, (source, episodes, _) in sources.items():
        (output / split).mkdir()
        for episode in episodes.itertuples(index=False):
            ep_id = int(episode.episode_index)
            data = pd.read_parquet(source / 'data', filters=[('episode_index', '==', ep_id)]).sort_values('frame_index')
            if len(data) != episode.length or data['frame_index'].tolist() != list(range(len(data))):
                raise ValueError(f'Episode frame continuity mismatch: {source}/{ep_id}')
            actions = np.stack(data['action']).astype(np.float32)
            for start in window_starts(len(data)):
                ids = np.minimum(start + np.arange(33) * 3, len(data) - 1)
                latents = []
                for camera in CAMERAS:
                    frames = np.stack([decode_frame(value, source) for value in data[camera].iloc[ids]])
                    flat, f, h, w = encode_video_latent(frames, vae, wrapper, args.device, torch.bfloat16, 256)
                    if f != 9:
                        raise ValueError(f'Expected 9 causal latent frames, got {f}')
                    latents.append(flat.reshape(f, h, w, -1).permute(3, 0, 1, 2))
                aligned, mask = align_actions(actions[start:], norm_stat)
                item = dict(latents=torch.cat(latents, dim=-1).contiguous(), actions=aligned, actions_mask=mask)
                relative = f'{split}/episode_{ep_id:06d}_{start:06d}.pt'
                torch.save(item, output / relative)
                metadata['samples'][split].append(relative)
            print(f'{split} episode={ep_id} frames={len(data)} windows={len(window_starts(len(data)))}', flush=True)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'COMPLETE').write_text('Frozen VAE/UMT5 caches; reviewed train/val splits; train-only statistics.\n')
    print(f'CACHE_COMPLETE {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_STEPS, required=True)
    parser.add_argument('--prepared-root', type=Path, default=POLICY.parents[2] / 'data/real_world/prepared')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inspect-only', action='store_true', help='Validate real source splits/statistics without GPU or writes')
    args = parser.parse_args()
    sources, stats, report = inspect(args.prepared_root, args.task)
    summary = dict(task=args.task, norm_stat=stats, splits={split: dict(episodes=len(eps), frames=len(actions),
                   windows=sum(len(window_starts(int(n))) for n in eps['length']))
                   for split, (_, eps, actions) in sources.items()})
    print(json.dumps(summary, indent=2), flush=True)
    if not args.inspect_only:
        if args.output is None or args.base_model is None:
            parser.error('--output and --base-model are required for latent extraction')
        prepare(args, sources, stats, report)


if __name__ == '__main__':
    main()
