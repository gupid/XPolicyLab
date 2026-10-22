"""Prepared Franka actions and frozen VAE/UMT5 caches; no LeRobot version dependency."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ACTION_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 28]
INVERSE_CHANNELS = [ACTION_CHANNELS.index(i) if i in ACTION_CHANNELS else 8 for i in range(30)]
CAMERAS = ['observation.images.camera_1', 'observation.images.camera_2']
TASK_STEPS = {'shell_game': 10000, 'put_back_block': 10000, 'press_button': 10000}


def action_statistics(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
        raise ValueError('Expected finite (N, 8) delta xyz/xyzw/gripper actions')
    stats = {}
    for name, quantile in [('q01', .01), ('q99', .99)]:
        values = np.zeros(30, dtype=np.float32)
        values[ACTION_CHANNELS] = np.quantile(actions, quantile, axis=0)
        stats[name] = values.tolist()
    return stats


def align_actions(actions, norm_stat, *, video_frames=33, stride=3):
    """Causal alignment: latent 0 has no actions; each next latent covers 4*stride steps."""
    action_per_frame = 4 * stride
    latent_frames = (video_frames - 1) // 4 + 1
    count = min(len(actions), (latent_frames - 1) * action_per_frame)
    result = np.zeros((latent_frames * action_per_frame, 30), dtype=np.float32)
    mask = np.zeros_like(result, dtype=bool)
    q01 = np.asarray(norm_stat['q01'])[ACTION_CHANNELS]
    q99 = np.asarray(norm_stat['q99'])[ACTION_CHANNELS]
    normalized = np.clip((np.asarray(actions[:count]) - q01) / (q99 - q01 + 1e-6) * 2 - 1, -1.5, 1.5)
    result[action_per_frame:action_per_frame + count, ACTION_CHANNELS] = normalized
    mask[action_per_frame:action_per_frame + count, ACTION_CHANNELS] = True
    shape = (latent_frames, action_per_frame, 30)
    return (torch.from_numpy(result.reshape(shape).transpose(2, 0, 1).copy()).unsqueeze(-1),
            torch.from_numpy(mask.reshape(shape).transpose(2, 0, 1).copy()).unsqueeze(-1))


def load_cache_metadata(root):
    root = Path(root)
    if not (root / 'COMPLETE').is_file():
        raise ValueError(f'Incomplete latent cache: {root}; run prepare_lora_data.py')
    metadata = json.loads((root / 'metadata.json').read_text())
    if metadata['used_action_channel_ids'] != ACTION_CHANNELS:
        raise ValueError('Latent cache action mapping differs from the real-world recipe')
    return metadata


def configure_real_world(config, root):
    metadata = load_cache_metadata(root)
    config.norm_stat = metadata['norm_stat']
    config.num_steps = TASK_STEPS[metadata['task']]
    config.height = config.width = metadata['image_size']
    config.action_per_frame = metadata['stride'] * 4
    config.empty_emb_path = str(Path(root) / 'empty_emb.pt')
    return metadata


class RealWorldLatentDataset(Dataset):
    def __init__(self, config, split='train'):
        self.root = Path(config.dataset_path)
        self.metadata = load_cache_metadata(self.root)
        if config.norm_stat != self.metadata['norm_stat']:
            raise ValueError('Training normalization must match the cache')
        self.samples = self.metadata['samples'][split]
        if not self.samples:
            raise ValueError(f'No cached samples in {split}')
        self.cfg_prob = config.cfg_prob if split == 'train' else 0.
        self.empty_emb = torch.load(self.root / 'empty_emb.pt', map_location='cpu', weights_only=True)
        self.text_emb = torch.load(self.root / 'text_emb.pt', map_location='cpu', weights_only=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = torch.load(self.root / self.samples[index], map_location='cpu', weights_only=True)
        item['text_emb'] = self.empty_emb if torch.rand(()).item() < self.cfg_prob else self.text_emb
        return item
