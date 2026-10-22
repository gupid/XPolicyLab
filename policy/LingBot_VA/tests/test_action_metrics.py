"""CPU checks of sampled-action masks, inference inputs and RNG isolation."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

POLICY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(POLICY), str(POLICY / 'lingbot_va')]

from action_metrics import inference_context, masked_action_sums, sample_actions, summarize_action_sums
from wan_va.configs import VA_CONFIGS


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_op = nn.Identity()
        self.calls = []
        self.cleared = False

    def clear_cache(self, name):
        self.cleared = True

    def create_empty_cache(self, *args, **kwargs):
        self.capacity = args[1:]

    def forward(self, inputs, *, action_mode=False, **kwargs):
        self.calls.append((action_mode, inputs['noisy_latents'].clone()))
        value = inputs['noisy_latents']
        b, c, f, h, w = value.shape
        if action_mode:
            assert not value[:, :, 0].any()
        else:
            torch.testing.assert_close(value[:, :, 0], torch.full_like(value[:, :, 0], 3.))
        return torch.zeros(b, f * h * w, c, device=value.device, dtype=value.dtype)


class ActionMetricTests(unittest.TestCase):
    def test_component_errors_ignore_nan_padding(self):
        prediction = torch.tensor([1., -3., 2., -4., float('nan')]).reshape(1, 5, 1, 1, 1)
        mask = torch.tensor([True, True, True, True, False]).reshape_as(prediction)
        stats = dict(q01=[0.] * 5, q99=[4. - 1e-6] * 5)
        result = summarize_action_sums(masked_action_sums(prediction, torch.zeros_like(prediction), mask, stats))
        self.assertEqual(result['action_l1'], 2.5)
        self.assertEqual(result['action_l2'], 7.5)
        self.assertAlmostEqual(result['action_l1_physical'], 5., places=5)
        self.assertAlmostEqual(result['action_l2_physical'], 30., places=5)
        self.assertEqual(result['valid_action_elements'], 4)
        with self.assertRaises(ValueError):
            summarize_action_sums(torch.zeros(5))

    def test_native_sampling_no_future_leak_and_restores_state(self):
        config = copy.deepcopy(VA_CONFIGS['real_world_lora_train'])
        config.norm_stat = dict(q01=[0.] * 30, q99=[1.] * 30)
        config.num_inference_steps = 2
        batch = dict(latents=torch.full((1, 48, 9, 4, 4), 999.),
                     actions=torch.full((1, 30, 9, 12, 1), 999.),
                     actions_mask=torch.zeros(1, 30, 9, 12, 1, dtype=torch.bool))
        batch['latents'][:, :, 0] = 3.
        batch['actions_mask'][:, config.used_action_channel_ids, 1] = True
        before = {key: value.clone() for key, value in batch.items()}
        model = FakePolicy().train()
        operator = model.attn_op
        state = torch.get_rng_state().clone()
        with inference_context(model, torch.device('cpu'), 1234):
            result = sample_actions(model, config, batch, torch.ones(512, 4096), torch.zeros(512, 4096),
                                    device=torch.device('cpu'), dtype=torch.bfloat16, num_steps=3)
            self.assertFalse(model.training)
        self.assertTrue(model.training)
        self.assertIs(model.attn_op, operator)
        self.assertTrue(model.cleared)
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        self.assertEqual(int(result[-1]), 12 * 8)
        self.assertEqual(len(model.calls), 3 + 3)
        for action_mode, noisy in model.calls:
            self.assertFalse((noisy == 999.).any())
        for key in before:
            torch.testing.assert_close(batch[key], before[key])
        try:
            with inference_context(model, torch.device('cpu'), 99):
                raise RuntimeError('test failure')
        except RuntimeError:
            pass
        self.assertIs(model.attn_op, operator)
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)

    def test_native_transformer_with_lora_and_checkpoint_wrappers(self):
        from wan_va.modules.model import WanTransformer3DModel
        from wan_va.lora import apply_lora, apply_lora_checkpointing
        config = copy.deepcopy(VA_CONFIGS['real_world_lora_train'])
        config.norm_stat = dict(q01=[0.] * 30, q99=[1.] * 30)
        config.num_inference_steps = 1
        model = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=128,
                                       num_layers=1, ffn_dim=512, attn_mode='flex').to(torch.bfloat16)
        apply_lora(model, config.lora_rank, config.lora_alpha, config.lora_target_modules,
                   config.lora_dropout, config.lora_train_modules)
        apply_lora_checkpointing(model)
        model.train()
        operators = [module.attn_op for module in model.modules() if hasattr(module, 'attn_op')]
        batch = dict(latents=torch.randn(1, 48, 9, 4, 4), actions=torch.zeros(1, 30, 9, 12, 1),
                     actions_mask=torch.zeros(1, 30, 9, 12, 1, dtype=torch.bool))
        batch['actions_mask'][:, config.used_action_channel_ids, 1] = True
        with inference_context(model, torch.device('cpu'), 42):
            sums = sample_actions(model, config, batch, torch.ones(512, 4096), torch.zeros(512, 4096),
                                  device=torch.device('cpu'), dtype=torch.bfloat16, num_steps=2)
        self.assertEqual(summarize_action_sums(sums)['valid_action_elements'], 96)
        self.assertTrue(model.training)
        self.assertEqual(operators, [module.attn_op for module in model.modules() if hasattr(module, 'attn_op')])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
