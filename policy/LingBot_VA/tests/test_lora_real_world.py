"""CPU checks for trainable scope, dropout recomputation, padding and prepared data."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))
sys.path.insert(0, str(POLICY / 'lingbot_va'))
sys.path.insert(0, str(POLICY.parents[2]))
from wan_va.configs import VA_CONFIGS
from wan_va.lora import apply_lora, apply_lora_checkpointing, merged_state_dict, merge_lora_state_dict
from wan_va.real_world_dataset import (ACTION_CHANNELS, INVERSE_CHANNELS, RealWorldLatentDataset,
                                      action_statistics, align_actions, configure_real_world)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q, self.to_k, self.to_v = [nn.Linear(4, 4) for _ in range(3)]
        self.to_out = nn.Sequential(nn.Linear(4, 4))

    def forward(self, x):
        return self.to_out(self.to_q(x) + self.to_k(x) + self.to_v(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1, self.attn2 = Attention(), Attention()
        self.ffn = nn.Linear(4, 4)

    def forward(self, x):
        return x + self.attn1(x) * .01 + self.attn2(x) * .01 + self.ffn(x) * .01


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(30)])
        self.action_embedder, self.action_proj_out = nn.Linear(30, 4), nn.Linear(4, 30)
        self.condition_embedder_action = nn.Module()
        self.condition_embedder_action.time_embedder = nn.Linear(4, 4)
        self.condition_embedder_action.time_proj = nn.Linear(4, 4)
        self.condition_embedder_action.text_embedder = nn.Linear(4, 4)

    def forward(self, x):
        x = self.action_embedder(x)
        x = x + self.condition_embedder_action.time_proj(self.condition_embedder_action.time_embedder(x)) * .01
        for block in self.blocks:
            x = block(x)
        return self.action_proj_out(x)


def inject(model):
    cfg = VA_CONFIGS['real_world_lora_train']
    return apply_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules,
                      cfg.lora_dropout, cfg.lora_train_modules)


class LoRATests(unittest.TestCase):
    def test_scope_and_update(self):
        model = TinyModel()
        targets = inject(model)
        self.assertEqual(len(targets), 120)
        self.assertTrue(all('.attn1.' in name for name in targets))
        frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
        heads = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n}
        self.assertTrue(heads)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(torch.randn(2, 30)).square().mean().backward()
        opt.step()
        self.assertTrue(any(not torch.equal(p, heads[n]) for n, p in model.named_parameters() if n in heads))
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
        self.assertFalse(model.condition_embedder_action.text_embedder.weight.requires_grad)

    def test_merge_and_dtype(self):
        model = TinyModel().to(torch.bfloat16)
        inject(model)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in model.parameters()))
        # FP32 comparison avoids BF16 rounding differences between merged and split GEMMs.
        model = model.float().eval()
        for name, parameter in model.named_parameters():
            if name.endswith('lora_B.weight'):
                nn.init.normal_(parameter, std=.01)
        merged = TinyModel().eval()
        state = merged_state_dict(model)
        merged.load_state_dict(state, strict=True)
        raw_merged = merge_lora_state_dict(model.state_dict(), 16, 16)
        for name in state:
            torch.testing.assert_close(state[name], raw_merged[name])
        x = torch.randn(2, 30)
        torch.testing.assert_close(model(x), merged(x), atol=2e-6, rtol=2e-5)

    def test_merge_with_gradient_checkpoint_wrappers(self):
        model = TinyModel().eval()
        inject(model)
        for name, parameter in model.named_parameters():
            if name.endswith('lora_B.weight'):
                nn.init.normal_(parameter, std=.01)
        apply_lora_checkpointing(model)
        merged = TinyModel().eval()
        state = merged_state_dict(model)
        merged.load_state_dict(state, strict=True)
        self.assertFalse(any('lora_' in name or 'base_layer' in name for name in state))
        x = torch.randn(2, 30)
        torch.testing.assert_close(model(x), merged(x), atol=1e-3, rtol=2e-2)

    def test_dropout_checkpoint_gradients(self):
        model = TinyModel()
        inject(model)
        for name, parameter in model.named_parameters():
            if name.endswith('lora_B.weight'):
                nn.init.normal_(parameter, std=.1)
        checkpointed = copy.deepcopy(model)
        apply_lora_checkpointing(checkpointed)
        x = torch.randn(2, 30)
        torch.manual_seed(17)
        model(x).square().mean().backward()
        torch.manual_seed(17)
        checkpointed(x).square().mean().backward()
        for (_, a), (_, b) in zip(model.named_parameters(), checkpointed.named_parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_action_alignment(self):
        actions = np.arange(17 * 8, dtype=np.float32).reshape(17, 8)
        stats = action_statistics(actions)
        aligned, mask = align_actions(actions, stats)
        self.assertEqual(aligned.shape, (30, 9, 12, 1))
        self.assertEqual(int(mask.sum()), 17 * 8)
        self.assertFalse(mask[:, 0].any())
        padding = [i for i in range(30) if i not in ACTION_CHANNELS]
        self.assertFalse(aligned[padding].any())
        self.assertFalse(mask[padding].any())
        self.assertEqual([INVERSE_CHANNELS[i] for i in ACTION_CHANNELS], list(range(8)))
        self.assertTrue(all(INVERSE_CHANNELS[i] == 8 for i in padding))
        np.testing.assert_allclose(aligned[ACTION_CHANNELS, 1, 0, 0], -1.020408, atol=1e-5)

    def test_masked_noise_and_loss(self):
        from wan_va.train import Trainer, FlowMatchScheduler
        trainer = Trainer.__new__(Trainer)
        trainer.device, trainer.patch_size, trainer.gradient_accumulation_steps = torch.device('cpu'), (1, 2, 2), 8
        trainer.train_scheduler_latent = FlowMatchScheduler(shift=5, sigma_min=0., extra_one_step=True)
        trainer.train_scheduler_action = FlowMatchScheduler(shift=1, sigma_min=0., extra_one_step=True)
        for scheduler in (trainer.train_scheduler_latent, trainer.train_scheduler_action):
            scheduler.set_timesteps(1000, training=True)
        action, mask = align_actions(np.ones((96, 8), np.float32), action_statistics(np.ones((96, 8), np.float32)))
        inputs = trainer._prepare_input_dict(dict(latents=torch.randn(1, 48, 9, 4, 4),
                    actions=action[None], actions_mask=mask[None], text_emb=torch.zeros(1, 512, 4096)))
        for key in ('noisy_latents', 'targets', 'latent'):
            self.assertFalse(inputs['action_dict'][key][~mask[None]].any())
        video_pred = torch.randn(1, 36, 192, requires_grad=True)
        action_pred = torch.randn(1, 108, 30, requires_grad=True)
        loss = sum(trainer.compute_loss(inputs, (video_pred, action_pred)))
        loss.backward()
        gradient = action_pred.grad.reshape(1, 9, 12, 30).permute(0, 3, 1, 2).unsqueeze(-1)
        self.assertFalse(gradient[~mask[None]].any())
        self.assertTrue(gradient[mask[None]].abs().sum() > 0)

    def test_cache_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stats = action_statistics(np.ones((17, 8), np.float32))
            metadata = dict(task='shell_game', norm_stat=stats, used_action_channel_ids=ACTION_CHANNELS,
                            image_size=256, stride=3, samples=dict(train=['sample.pt'], val=['sample.pt']))
            (root / 'metadata.json').write_text(json.dumps(metadata))
            (root / 'COMPLETE').touch()
            torch.save(torch.zeros(512, 4096), root / 'empty_emb.pt')
            torch.save(torch.ones(512, 4096), root / 'text_emb.pt')
            action, mask = align_actions(np.ones((17, 8)), stats)
            torch.save(dict(latents=torch.zeros(48, 9, 16, 32), actions=action, actions_mask=mask), root / 'sample.pt')
            config = copy.deepcopy(VA_CONFIGS['real_world_lora_train'])
            config.dataset_path = str(root)
            configure_real_world(config, root)
            config.cfg_prob = 1.
            self.assertFalse(RealWorldLatentDataset(config)[0]['text_emb'].any())
            self.assertTrue(RealWorldLatentDataset(config, 'val')[0]['text_emb'].all())
            self.assertEqual(config.num_steps, 10000)

    def test_prepared_splits_and_decode(self):
        from prepare_lora_data import inspect, decode_frame, window_starts
        root = POLICY.parents[2] / 'data/real_world/prepared'
        if not root.exists():
            self.skipTest('Local prepared corpus unavailable')
        for task in ('shell_game', 'put_back_block', 'press_button'):
            sources, stats, _ = inspect(root, task)
            expected = action_statistics(np.stack(sources['train'][2]['action']))
            self.assertEqual(stats, expected)
            source = sources['train'][0]
            import pandas as pd
            frame = pd.read_parquet(next((source / 'data').rglob('*.parquet'))).iloc[0]
            rgb = decode_frame(frame['observation.images.camera_1'], source)
            self.assertEqual(rgb.shape, (224, 224, 3))
        for length in (2, 96, 97, 150, 300):
            covered = set()
            for start in window_starts(length):
                covered.update(range(start, min(start + 96, length)))
            self.assertEqual(covered, set(range(length)))

    def test_inference_mapping(self):
        from wan_va.wan_va_server import VA_Server
        server = VA_Server.__new__(VA_Server)
        server.job_config = copy.deepcopy(VA_CONFIGS['real_world_lora_train'])
        stats = action_statistics(np.stack([np.zeros(8), np.ones(8)]))
        server.actions_q01 = torch.tensor(stats['q01']).reshape(30, 1, 1)
        server.actions_q99 = torch.tensor(stats['q99']).reshape(30, 1, 1)
        server.action_norm_method = 'quantiles'
        server.action_mask = torch.zeros(30, dtype=torch.bool)
        server.action_mask[ACTION_CHANNELS] = True
        normalized = server.preprocess_action(np.ones((8, 2, 12), dtype=np.float32))
        self.assertFalse(normalized[:, ~server.action_mask].any())
        reconstructed = server.postprocess_action(normalized)
        np.testing.assert_allclose(reconstructed, np.ones((8, 2, 12)), atol=1e-5)
        bridge = importlib.import_module('XPolicyLab.policy.LingBot_VA.model')
        model = bridge.Model.__new__(bridge.Model)
        model.is_real_world, model.action_dim, model.job_config = True, 30, server.job_config
        formatted = model._format_action_chunk(reconstructed)
        self.assertEqual(model._convert_to_joint_control_chunk(formatted).shape, (24, 8))
        observation = dict(vision={f'camera_{i}': {'color': np.full((224, 224, 3), i, np.uint8)} for i in (1, 2)})
        encoded = bridge.encode_obs(observation, 'ee', None, 'prompt', real_world=True)
        self.assertEqual(set(encoded), {'observation.images.camera_1', 'observation.images.camera_2', 'task'})

    def test_checkpoint_adapter_and_metadata(self):
        from train_cpu_offload import CPUOffloadTrainer
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'metadata.json').write_text('{"task": "shell_game"}')
            trainer = CPUOffloadTrainer.__new__(CPUOffloadTrainer)
            trainer.config = copy.deepcopy(VA_CONFIGS['real_world_lora_train'])
            trainer.config.dataset_path = str(root)
            trainer.save_dir, trainer.step = root, 500
            trainer.transformer = TinyModel()
            inject(trainer.transformer)
            apply_lora_checkpointing(trainer.transformer)
            trainer.lora_enabled, trainer.synthetic_smoke = True, False
            trainer.sampler = Mock(state_dict=Mock(return_value={'cursor': 8}))
            trainer.loader_generator = torch.Generator()
            trainer._rng_state = Mock(return_value={})
            trainer._resume_signature = Mock(return_value={'recipe': 'real-world'})
            trainer.data_identity = {'source': str(root)}
            trainer.engine = Mock(save_checkpoint=Mock(return_value=True))
            trainer._export_transformer = lambda directory: directory.mkdir()
            checkpoint = trainer.save_checkpoint()
            self.assertTrue((checkpoint / 'COMPLETE').is_file())
            self.assertEqual(json.loads((checkpoint / 'training_metadata.json').read_text())['optimizer_step'], 500)
            self.assertEqual(json.loads((checkpoint / 'real_world_metadata.json').read_text())['task'], 'shell_game')
            adapter = load_file(checkpoint / 'adapter.safetensors')
            self.assertIn('action_embedder.weight', adapter)
            self.assertIn('blocks.0.attn1.to_q.lora_A.weight', adapter)
            self.assertFalse(any('base_layer' in key or 'checkpoint_wrapped' in key for key in adapter))
            self.assertEqual(json.loads((checkpoint / 'adapter_config.json').read_text())['lora_rank'], 16)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
