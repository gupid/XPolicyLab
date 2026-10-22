"""Larger microbatches: full batches, committed resume and native loss scaling."""
from pathlib import Path
import sys
import unittest

import torch

POLICY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(POLICY), str(POLICY / 'lingbot_va')]
from train_cpu_offload import CommittedBatchSampler, CPUOffloadTrainer
from action_metrics import single_window_batches, masked_action_sums
from wan_va.train import FlowMatchScheduler


class BatchTrainingTests(unittest.TestCase):
    def test_prefetch_tail_and_resume(self):
        sampler = CommittedBatchSampler(413, 4, 0)
        iterator = iter(sampler)
        first, second = next(iterator), next(iterator)  # prefetch is not consumption
        self.assertEqual(sampler.cursor, 0)
        sampler.commit()
        restored = CommittedBatchSampler(413, 4, 0)
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(next(iter(restored)), second)
        remaining = list(restored)
        for batch in remaining:
            self.assertEqual(len(batch), 4)
            restored.commit()
        self.assertEqual(restored.cursor, 413)
        self.assertEqual(len({i for b in [first] + remaining for i in b}), 412)
        old_order = restored.order.clone()
        restored.next_epoch()
        self.assertEqual(len(list(restored)), 103)
        self.assertFalse(torch.equal(old_order, restored.order))
        with self.assertRaises(ValueError):
            CommittedBatchSampler(3, 4, 0)

    def test_native_loss_and_gradients_equal_for_fixed_inputs(self):
        trainer = CPUOffloadTrainer.__new__(CPUOffloadTrainer)
        trainer.patch_size = (1, 2, 2)
        trainer.train_scheduler_latent = FlowMatchScheduler(shift=5, sigma_min=0., extra_one_step=True)
        trainer.train_scheduler_action = FlowMatchScheduler(shift=1, sigma_min=0., extra_one_step=True)
        for scheduler in (trainer.train_scheduler_latent, trainer.train_scheduler_action):
            scheduler.set_timesteps(1000, training=True)
        generator = torch.Generator().manual_seed(42)
        latent_target = torch.randn(8, 48, 3, 4, 8, generator=generator)
        action_target = torch.randn(8, 30, 3, 12, 1, generator=generator)
        mask = torch.zeros_like(action_target, dtype=torch.bool)
        mask[:, [0, 1, 2, 3, 4, 5, 6, 28], 1:] = True
        predictions = (torch.randn(8, 96, 48, generator=generator),
                       torch.randn(8, 36, 30, generator=generator))
        results = []
        for batch_size, accumulation in ((1, 8), (4, 2), (8, 1)):
            parameter = torch.tensor(.7, requires_grad=True)
            trainer.gradient_accumulation_steps = accumulation
            total = 0.
            for start in range(0, 8, batch_size):
                selection = slice(start, start + batch_size)
                inputs = dict(latent_dict=dict(targets=latent_target[selection],
                                               timesteps=torch.full((batch_size, 3), 500.)),
                              action_dict=dict(targets=action_target[selection], actions_mask=mask[selection],
                                               timesteps=torch.full((batch_size, 3), 500.)))
                loss = sum(trainer.compute_loss(inputs, tuple(p[selection] * parameter for p in predictions)))
                loss.backward()
                total += loss.detach()
            results.append((total, parameter.grad))
        for result in results[1:]:
            for actual, expected in zip(result, results[0]):
                torch.testing.assert_close(actual, expected)

    def test_action_diagnostics_cover_all_four_windows(self):
        batch = dict(latents=torch.randn(4, 48, 3, 2, 2),
                     actions=torch.arange(4.).reshape(4, 1, 1, 1, 1).expand(4, 30, 3, 12, 1),
                     actions_mask=torch.ones(4, 30, 3, 12, 1, dtype=torch.bool))
        stats = dict(q01=[0.] * 30, q99=[2.] * 30)
        def sums(item):
            return masked_action_sums(torch.zeros_like(item['actions']), item['actions'],
                                      item['actions_mask'], stats)
        windows = list(single_window_batches(batch))
        self.assertEqual(len(windows), 4)
        torch.testing.assert_close(sum(sums(item) for item in windows), sums(batch))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
