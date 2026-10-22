"""Sampled action diagnostics, independent of the flow-matching objective."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn


def masked_action_sums(prediction, target, mask, norm_stat):
    """Component MAE/MSE sums; padding (including nonfinite padding) is excluded."""
    error = prediction.float() - target.float()
    error = torch.where(mask, error, 0.)
    scale = (torch.tensor(norm_stat['q99'], device=error.device) -
             torch.tensor(norm_stat['q01'], device=error.device) + 1e-6) / 2
    physical = error * scale[None, :, None, None, None]
    return torch.stack([error.abs().sum(), error.square().sum(),
                        physical.abs().sum(), physical.square().sum(), mask.sum()])


def summarize_action_sums(sums):
    l1, l2, raw_l1, raw_l2, count = sums.tolist()
    if count <= 0 or not torch.isfinite(sums).all():
        raise ValueError('Nonfinite sampled-action errors or no valid action elements')
    return dict(action_l1=l1 / count, action_l2=l2 / count,
                action_l1_physical=raw_l1 / count, action_l2_physical=raw_l2 / count,
                valid_action_elements=int(count))


def single_window_batches(batch):
    """Keep native inference single-window, including for larger training batches."""
    for index in range(batch['latents'].shape[0]):
        yield {key: value[index:index + 1] for key, value in batch.items()}


class InferenceAttention(nn.Module):
    def forward(self, query, key, value):
        from wan_va.modules.model import custom_sdpa
        return custom_sdpa(query, key, value)


@contextmanager
def inference_context(model, device, seed):
    """Restore attention operators, dropout mode and all Torch RNG state."""
    operators = [(module, module.attn_op) for module in model.modules() if hasattr(module, 'attn_op')]
    was_training = model.training
    try:
        model.eval()
        for module, _ in operators:
            module.attn_op = InferenceAttention()
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield
    finally:
        model.clear_cache('action_metrics')
        for module, operator in operators:
            module.attn_op = operator
        model.train(was_training)


@torch.no_grad()
def sample_actions(model, config, batch, text_emb, empty_emb, *, device, dtype, num_steps=10):
    """Native initial-chunk video->action inference; observes only the first latent.

    No future video or target action is used. The first (dummy) latent action
    frame is zero, leaving 12 real action steps with the default chunk size=2.
    """
    from wan_va.train import FlowMatchScheduler
    from wan_va.wan_va_server import VA_Server
    from wan_va.utils import data_seq_to_patch

    server = VA_Server.__new__(VA_Server)
    server.device, server.dtype = device, dtype
    server.use_cfg = config.guidance_scale > 1 or config.action_guidance_scale > 1
    server.prompt_embeds = text_emb[None].to(device=device, dtype=dtype)
    server.negative_prompt_embeds = empty_emb[None].to(device=device, dtype=dtype)
    server.action_mask = torch.zeros(config.action_dim, dtype=torch.bool, device=device)
    server.action_mask[config.used_action_channel_ids] = True
    frames = config.frame_chunk_size
    initial = batch['latents'][:, :, :1].to(device=device, dtype=dtype)
    height, width = initial.shape[-2:]
    latents = torch.randn(1, initial.shape[1], frames, height, width, device=device, dtype=dtype)
    actions = torch.randn(1, config.action_dim, frames, config.action_per_frame, 1,
                          device=device, dtype=dtype)
    action_cond = torch.zeros_like(actions[:, :, :1])
    video_scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0., extra_one_step=True)
    action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0., extra_one_step=True)
    video_scheduler.set_timesteps(config.num_inference_steps)
    action_scheduler.set_timesteps(num_steps)
    # An initial chunk needs capacity for its video + action tokens, no history.
    model.create_empty_cache('action_metrics', 2, frames * height * width // 4,
                             frames * config.action_per_frame, device=device, dtype=dtype,
                             batch_size=2 if server.use_cfg else 1)
    for index, t in enumerate(torch.cat([video_scheduler.timesteps, torch.zeros(1)])):
        last = index == len(video_scheduler.timesteps)
        inputs = server._prepare_latent_input(latents, None, t, t, initial, None)['latent_res_lst']
        prediction = model(server._repeat_input_for_cfg(inputs), update_cache=int(last),
                           cache_name='action_metrics', action_mode=False)
        if not last:
            prediction = data_seq_to_patch(config.patch_size, prediction, frames, height, width,
                                            batch_size=2 if server.use_cfg else 1)
            prediction = (prediction[1:] + config.guidance_scale * (prediction[:1] - prediction[1:])
                          if config.guidance_scale > 1 else prediction[:1])
            latents = video_scheduler.step(prediction, t, latents)
        latents[:, :, :1] = initial
    for t in action_scheduler.timesteps:
        inputs = server._prepare_latent_input(None, actions, t, t, None, action_cond)['action_res_lst']
        prediction = model(server._repeat_input_for_cfg(inputs), cache_name='action_metrics', action_mode=True)
        prediction = prediction.reshape(prediction.shape[0], frames, config.action_per_frame,
                                          config.action_dim).permute(0, 3, 1, 2).unsqueeze(-1)
        prediction = (prediction[1:] + config.action_guidance_scale * (prediction[:1] - prediction[1:])
                      if config.action_guidance_scale > 1 else prediction[:1])
        actions = action_scheduler.step(prediction, t, actions)
        actions[:, :, :1] = action_cond
        actions[:, ~server.action_mask] = 0
    target = batch['actions'][:, :, :frames].to(device)
    mask = batch['actions_mask'][:, :, :frames].to(device)
    return masked_action_sums(actions, target, mask, config.norm_stat)


def record_action_metrics(trainer, batch, *, validation=False):
    from torch.utils.data import DataLoader, Subset
    from wan_va.real_world_dataset import RealWorldLatentDataset

    dataset = RealWorldLatentDataset(trainer.config, split='val' if validation else 'train')
    batches = [batch]
    if validation:
        # Fixed windows spread across the split; full held-out flow loss is separate.
        count = min(len(dataset), trainer.config.action_metric_val_windows)
        indices = torch.linspace(0, len(dataset) - 1, count).long().tolist()
        batches = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0)
    sums = torch.zeros(5, dtype=torch.float64, device=trainer.device)
    with inference_context(trainer.transformer, trainer.device, trainer.seed + 1000003):
        for item in batches:
            for window in single_window_batches(item):
                sums += sample_actions(trainer.transformer, trainer.config, window, dataset.text_emb,
                                       dataset.empty_emb, device=trainer.device, dtype=trainer.dtype,
                                       num_steps=trainer.config.action_metric_steps)
                trainer.transformer.clear_cache('action_metrics')
    values = summarize_action_sums(sums)
    prefix = 'val_' if validation else ''
    row = dict(step=trainer.step, **{prefix + key: value for key, value in values.items()},
               denoising_steps=trainer.config.action_metric_steps)
    path = Path(trainer.config.save_root) / ('val_action_metrics.jsonl' if validation else 'action_metrics.jsonl')
    with path.open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print('ACTION_METRICS', json.dumps(row), flush=True)
    return row
