"""GPU integration: loss scaling, warmup, save/resume and exported model reload."""
import copy
import gc
import json
import os
import random
import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))
from audit_upstream import VENDOR, check_sources
check_sources()
sys.path.insert(0, str(VENDOR))
import numpy as np
import torch
import torch.distributed as dist
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from wan_va.configs import VA_CONFIGS
from wan_va.modules.model import WanTransformer3DModel
from train_cpu_offload import CPUOffloadTrainer, deepspeed_config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def verify_scaling():
    config = copy.deepcopy(VA_CONFIGS['demo_train'])
    config.batch_size, config.gradient_accumulation_steps = 1, 2
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.25)
    reference = torch.nn.Linear(2, 1, bias=False)
    reference.load_state_dict(model.state_dict())
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, adamw_mode=True)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,
                                          config=deepspeed_config(config), dist_init_required=False)
    for scalar in (0.125, 0.25):
        x = torch.full((1, 2), scalar)
        (reference(x).square().mean() / 2).backward()
        loss = engine(x.to('cuda', dtype=torch.bfloat16)).float().square().mean() / 2
        engine.backward(loss, scale_wrt_gas=False)
        engine.step()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 2.).item()
    assert abs(float(engine.get_global_grad_norm()) - expected_norm) < 1e-7
    ref_opt.step()
    master = engine.optimizer.single_partition_of_fp32_groups[0].detach().cpu()
    torch.testing.assert_close(master[:2], reference.weight.detach().flatten(), atol=2e-7, rtol=0)
    print('ACCUMULATION_SCALED_EXACTLY_ONCE_CPU_ADAMW_UPDATE_OK', flush=True)


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '1')))
    dist.init_process_group('nccl')
    try:
        verify_scaling()
        clear()
        seed_all(123)
        tiny = output / 'tiny_base/transformer'
        WanTransformer3DModel(num_attention_heads=2, attention_head_dim=128,
                              ffn_dim=1024, num_layers=2, attn_mode='flex').save_pretrained(tiny)
        base = copy.deepcopy(VA_CONFIGS['demo_train'])
        base.update(wan22_pretrained_model_name_or_path=str(tiny.parent), enable_wandb=False,
                    batch_size=1, gradient_accumulation_steps=2, load_worker=0,
                    warmup_steps=2, num_steps=4, save_interval=4, learning_rate=1e-4)
        base.save_root = str(output / 'uninterrupted')
        seed_all(1234)
        uninterrupted = CPUOffloadTrainer(base, seed=1234, synthetic_smoke=True)
        uninterrupted.train()
        expected = {k: v.detach().cpu().clone() for k, v in uninterrupted.transformer.state_dict().items()}
        expected_masters = [p.detach().cpu().clone() for p in uninterrupted.engine.optimizer.single_partition_of_fp32_groups]
        expected_lr = uninterrupted.lr_scheduler.state_dict()
        expected_sampler = uninterrupted.sampler.state_dict()
        del uninterrupted
        clear()

        split = copy.deepcopy(base)
        split.update(save_root=str(output / 'split'), num_steps=2, save_interval=2)
        seed_all(1234)
        first = CPUOffloadTrainer(split, seed=1234, synthetic_smoke=True)
        first.train()
        checkpoint = Path(split.save_root) / 'checkpoints/checkpoint_step_2'
        del first
        clear()

        resumed_config = copy.deepcopy(split)
        resumed_config.update(num_steps=4, save_interval=4)
        # Use a different initial RNG: loading must restore checkpoint RNG/state.
        seed_all(9999)
        resumed = CPUOffloadTrainer(resumed_config, seed=1234, synthetic_smoke=True, resume_from=checkpoint)
        assert resumed.step == 2 and resumed.engine.global_steps == 2
        resumed.train()
        max_master_difference = max((a - b.detach().cpu()).abs().max().item() for a, b in
                                    zip(expected_masters, resumed.engine.optimizer.single_partition_of_fp32_groups))
        assert max_master_difference < 2e-7, max_master_difference
        for name, value in resumed.transformer.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[name], atol=0, rtol=0)
        assert resumed.lr_scheduler.state_dict() == expected_lr
        assert resumed.sampler.cursor == expected_sampler['cursor']
        assert resumed.sampler.epoch == expected_sampler['epoch']
        assert torch.equal(resumed.sampler.order, expected_sampler['order'])
        export = Path(resumed_config.save_root) / 'checkpoints/checkpoint_step_4/transformer'
        reloaded = WanTransformer3DModel.from_pretrained(str(export), torch_dtype=torch.bfloat16).to(torch.bfloat16)
        for name, value in reloaded.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[name], atol=0, rtol=0)
        result = {'success': True, 'job_id': os.getenv('SLURM_JOB_ID'),
                  'tests': ['accumulation_scale_and_CPU_AdamW', 'warmup_scheduler',
                            'full_state_save_and_resume', 'sample_order_and_rng_resume',
                            'resumed_BF16_weights_equal_uninterrupted', 'diffusers_export_reload'],
                  'gradient_accumulation_steps': 2, 'load_worker': 0,
                  'max_FP32_master_difference': max_master_difference,
                  'model': 'tiny two-block WanTransformer3DModel; synthetic cached latents'}
        (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
        print('CPU_OFFLOAD_INTEGRATION_VERIFIED', json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
