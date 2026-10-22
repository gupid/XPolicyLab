"""CPU checks for GPU AdamW selection, update semantics and checkpoint identity."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

POLICY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(POLICY), str(POLICY / 'lingbot_va')]
from wan_va.configs import VA_CONFIGS
from train_cpu_offload import CPUOffloadTrainer, create_optimizer, deepspeed_config


class BackendTests(unittest.TestCase):
    def test_gpu_optimizer_update_without_cpu_adam_import(self):
        config = copy.deepcopy(VA_CONFIGS['real_world_lora_train'])
        parameter = torch.nn.Parameter(torch.tensor([.25, -.5]))
        reference = torch.nn.Parameter(parameter.detach().clone())
        # The GPU path must work even when the CPU Adam extension is unavailable.
        with patch.dict(sys.modules, {'deepspeed.ops.adam': None}):
            optimizer = create_optimizer(config, [parameter], 'gpu')
        expected = torch.optim.AdamW([reference], lr=config.learning_rate,
                                    betas=(config.beta1, config.beta2), eps=1e-8,
                                    weight_decay=config.weight_decay)
        for p in (parameter, reference):
            p.grad = torch.tensor([.1, -.2])
        optimizer.step()
        expected.step()
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
        self.assertTrue(optimizer.state[parameter])
        gpu = deepspeed_config(config, 'gpu')
        offload = deepspeed_config(config, 'cpu_offload')
        self.assertNotIn('offload_optimizer', gpu['zero_optimization'])
        self.assertEqual(offload['zero_optimization'].pop('offload_optimizer')['device'], 'cpu')
        self.assertEqual(gpu, offload)
        self.assertEqual(gpu['data_types']['grad_accum_dtype'], 'fp32')
        with self.assertRaises(ValueError):
            deepspeed_config(config, 'invalid')

    def test_resume_rejects_backend_change_before_loading_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / 'COMPLETE').touch()
            (checkpoint / 'training_metadata.json').write_text(json.dumps({'backend': 'cpu_offload'}))
            trainer = CPUOffloadTrainer.__new__(CPUOffloadTrainer)
            trainer.backend, trainer.resume_from = 'gpu', checkpoint
            trainer.engine = Mock()
            with self.assertRaisesRegex(ValueError, 'saved backend: cpu_offload'):
                trainer._restore_checkpoint()
            trainer.engine.load_checkpoint.assert_not_called()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
