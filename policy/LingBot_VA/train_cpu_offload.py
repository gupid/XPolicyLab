"""Single-GPU full-finetune or LoRA backend; reuse the pinned model/noise/loss.

DeepSpeed owns backward, gradient accumulation, clipping and AdamW updates.
The official loss already divides by accumulation, so backward MUST disable
DeepSpeed's additional loss division. GPU weights are BF16; master weights,
Adam states and gradient accumulation are FP32, on GPU or optionally CPU. This is an explicit backend
choice, not a claim of bitwise equivalence with the official FSDP implementation.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from audit_upstream import POLICY, sha256

# train_entry.py pins sys.path to the vendored checkout before this import.
from wan_va.train import (Trainer, MultiLatentLeRobotDataset, FlowMatchScheduler,
                          load_transformer, apply_ac, warmup_constant_lambda)
from wan_va.lora import apply_lora, apply_lora_checkpointing, merged_state_dict, lora_summary


def deepspeed_config(config, backend='cpu_offload'):
    if backend not in ('gpu', 'cpu_offload'):
        raise ValueError(f'Unknown single-GPU backend: {backend}')
    result = {
        'train_micro_batch_size_per_gpu': config.batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'bf16': {'enabled': True},
        'data_types': {'grad_accum_dtype': 'fp32'},
        'zero_allow_untested_optimizer': True,
        'gradient_clipping': 2.0,
        'zero_optimization': {
            'stage': 2,
            'overlap_comm': False, 'contiguous_gradients': True,
            'reduce_bucket_size': 10000000, 'allgather_bucket_size': 10000000,
        },
        'steps_per_print': 1000000,
    }
    if backend == 'cpu_offload':
        result['zero_optimization']['offload_optimizer'] = {'device': 'cpu', 'pin_memory': True}
    return result


def create_optimizer(config, parameters, backend):
    options = dict(lr=config.learning_rate, betas=(config.beta1, config.beta2),
                   eps=1e-8, weight_decay=config.weight_decay)
    if backend == 'gpu':
        # DeepSpeed creates FP32 master parameters on the GPU before AdamW steps.
        return torch.optim.AdamW(parameters, **options)
    if backend == 'cpu_offload':
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam(parameters, **options, adamw_mode=True, fp32_optimizer_states=True)
    raise ValueError(f'Unknown single-GPU backend: {backend}')


def precision_description(backend):
    location = 'GPU' if backend == 'gpu' else 'CPU'
    return f'GPU BF16; {location} master weights/Adam states/gradient accumulation FP32'


class CommittedBatchSampler:
    """Commit consumed full batches independently of worker prefetch; drop tails."""
    def __init__(self, size, batch_size, seed):
        if batch_size < 1 or size < batch_size:
            raise ValueError('Dataset must contain at least one full positive-size batch')
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.cursor = 0, 0
        self.order = self._order()

    def _order(self):
        return torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))

    def __iter__(self):
        for start in range(self.cursor, self.size - self.batch_size + 1, self.batch_size):
            yield self.order[start:start + self.batch_size].tolist()

    def __len__(self):
        return (self.size - self.cursor) // self.batch_size

    def commit(self):
        self.cursor = min(self.size, self.cursor + self.batch_size)
        if self.size - self.cursor < self.batch_size:
            self.cursor = self.size

    def next_epoch(self):
        if self.cursor != self.size:
            raise RuntimeError('Iterator ended before all batches were consumed')
        self.epoch += 1
        self.cursor = 0
        self.order = self._order()

    def state_dict(self):
        return dict(size=self.size, batch_size=self.batch_size, seed=self.seed,
                    epoch=self.epoch, cursor=self.cursor, order=self.order)

    def load_state_dict(self, state):
        for key in ('size', 'batch_size', 'seed'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Resume sampler {key} changed')
        if not 0 <= state['cursor'] <= self.size:
            raise ValueError('Invalid resume sampler cursor')
        self.epoch, self.cursor, self.order = state['epoch'], state['cursor'], state['order'].cpu()
        if not torch.equal(self.order, self._order()):
            raise ValueError('Resume sample order is inconsistent')


class SyntheticLatents(Dataset):
    """Explicit smoke data only. Never selected automatically for missing data."""
    def __init__(self, config, frames=3, height=16, width=32):
        self.config, self.frames, self.height, self.width = config, frames, height, width

    def __len__(self):
        return 4

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(9317 + index)
        action_shape = (30, self.frames, self.config.action_per_frame, 1)
        mask = torch.zeros(action_shape, dtype=torch.bool)
        mask[self.config.used_action_channel_ids] = True
        text = torch.randn(512, 4096, generator=generator, dtype=torch.bfloat16)
        if torch.rand(()).item() < self.config.cfg_prob:
            text.zero_()
        return {
            'latents': torch.randn(48, self.frames, self.height, self.width,
                                   generator=generator, dtype=torch.bfloat16),
            'actions': torch.randn(action_shape, generator=generator),
            'actions_mask': mask, 'text_emb': text,
        }


class CPUOffloadTrainer(Trainer):
    # Inherit _prepare_input_dict/_add_noise/convert_input_format/compute_loss
    # verbatim. Do not call the upstream constructor: it creates a GPU optimizer.
    def __init__(self, config, *, seed=0, resume_from=None, dataset_init_workers=1,
                 synthetic_smoke=False, smoke_shape=(3, 16, 32), save_final=True, backend='cpu_offload'):
        import deepspeed

        if backend not in ('gpu', 'cpu_offload'):
            raise ValueError(f'Unknown single-GPU backend: {backend}')
        self.backend = backend

        if not dist.is_initialized() or dist.get_world_size() != 1:
            raise ValueError('Single-GPU backend requires exactly one GPU/process')
        if config.param_dtype != torch.bfloat16:
            raise ValueError('Single-GPU backend requires BF16 param_dtype')
        for name in ('batch_size', 'gradient_accumulation_steps', 'num_steps', 'save_interval', 'gc_interval'):
            if int(getattr(config, name)) < 1:
                raise ValueError(f'{name} must be positive')
        self.config, self.seed, self.step = config, seed, 0
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.dtype, self.patch_size = config.param_dtype, tuple(config.patch_size)
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.synthetic_smoke, self.save_final = synthetic_smoke, save_final
        self.resume_from = Path(resume_from).resolve() if resume_from else None
        self.started = time.monotonic()
        self.save_dir = Path(config.save_root) / 'checkpoints'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.report = {'backend': backend, 'synthetic_smoke': synthetic_smoke,
                       'job_id': os.getenv('SLURM_JOB_ID'), 'seed': seed, 'success': False,
                       'precision': precision_description(backend),
                       'resume_from': str(self.resume_from) if self.resume_from else None}
        self.wandb = None
        if config.enable_wandb:
            import wandb
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            wandb.init(entity=os.environ['WANDB_TEAM_NAME'], project=os.getenv('WANDB_PROJECT', 'va_robotwin'),
                       config=dict(config), name=Path(config.save_root).name)
            self.wandb = wandb

        model_path = (self.resume_from if self.resume_from else Path(config.wan22_pretrained_model_name_or_path)) / 'transformer'
        if getattr(config, 'lora_enabled', False) and self.resume_from:
            # The exported transformer is merged for inference. Resume the
            # optimizer state from the original base plus the saved LoRA
            # parameters instead of applying LoRA twice to a merged model.
            model_path = Path(config.wan22_pretrained_model_name_or_path) / 'transformer'
        model = load_transformer(str(model_path), torch_dtype=torch.bfloat16, torch_device='cpu', attn_mode='flex')
        if tuple(model.patch_size) != self.patch_size:
            raise ValueError('Model and training patch sizes differ')
        self.lora_enabled = bool(getattr(config, 'lora_enabled', False))
        if self.lora_enabled:
            self.lora_targets = apply_lora(
                model,
                rank=int(config.lora_rank),
                alpha=float(config.lora_alpha),
                target_modules=config.lora_target_modules,
                dropout=float(getattr(config, 'lora_dropout', 0.0)),
                train_modules=getattr(config, 'lora_train_modules', ()),
            )
            model.train()
        else:
            model.train().requires_grad_(True)
        self.report['parameters'] = sum(p.numel() for p in model.parameters())
        self.report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.report['lora'] = {
            'enabled': self.lora_enabled,
            'rank': int(getattr(config, 'lora_rank', 0)) if self.lora_enabled else None,
            'alpha': float(getattr(config, 'lora_alpha', 0)) if self.lora_enabled else None,
            'target_modules': getattr(config, 'lora_target_modules', None) if self.lora_enabled else None,
            'targets': getattr(self, 'lora_targets', []),
        }
        (apply_lora_checkpointing if self.lora_enabled else apply_ac)(model)
        optimizer = create_optimizer(config, [p for p in model.parameters() if p.requires_grad], backend)
        ds_config = deepspeed_config(config, backend)
        self.engine, _, _, self.lr_scheduler = deepspeed.initialize(
            model=model, optimizer=optimizer, config=ds_config, dist_init_required=False,
            lr_scheduler=lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps)))
        self.transformer = self.engine.module
        self.report['deepspeed_config'] = ds_config
        masters = self.engine.optimizer.single_partition_of_fp32_groups
        master_device = 'cuda' if backend == 'gpu' else 'cpu'
        if not all(p.device.type == master_device and p.dtype == torch.float32 for p in masters):
            raise RuntimeError(f'Expected FP32 optimizer master weights on {master_device}')
        self.report['optimizer_master_dtype_device'] = f'torch.float32/{master_device}'
        for p in self.transformer.parameters():
            if p.dtype != torch.bfloat16:
                raise RuntimeError('Expected BF16 GPU model parameters')

        self.train_scheduler_latent = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0., extra_one_step=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0., extra_one_step=True)
        for scheduler in (self.train_scheduler_latent, self.train_scheduler_action):
            scheduler.set_timesteps(1000, training=True)
        if synthetic_smoke:
            dataset = SyntheticLatents(config, *smoke_shape)
        elif getattr(config, 'dataset_type', None) == 'real_world_cached':
            from wan_va.real_world_dataset import RealWorldLatentDataset
            dataset = RealWorldLatentDataset(config)
        else:
            dataset = MultiLatentLeRobotDataset(config, num_init_worker=dataset_init_workers)
        if len(dataset) == 0:
            raise ValueError('No training samples: check action_config, cameras and latent files')
        self.data_identity = {'source': 'synthetic_cached_latents_v1' if synthetic_smoke else str(Path(config.dataset_path).resolve()),
                              'samples': len(dataset), 'smoke_shape': list(smoke_shape) if synthetic_smoke else None}
        if not synthetic_smoke and getattr(config, 'dataset_type', None) == 'real_world_cached':
            self.data_identity['metadata_sha256'] = sha256(Path(config.dataset_path) / 'metadata.json')
        self.sampler = CommittedBatchSampler(len(dataset), config.batch_size, seed)
        self.loader_generator = torch.Generator().manual_seed(seed + 8191)
        self.train_loader = DataLoader(dataset, batch_sampler=self.sampler,
                                       num_workers=config.load_worker, generator=self.loader_generator)
        self.train_loader_iter = None
        if self.resume_from:
            self._restore_checkpoint()
        self.report['initial_step'] = self.step
        self.report['data'] = self.data_identity
        self._log_resources('initialized')

    def _get_next_batch(self):
        if self.sampler.cursor == self.sampler.size:
            self.sampler.next_epoch()
            self.train_loader_iter = None
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        return next(self.train_loader_iter)

    def _log_resources(self, stage):
        torch.cuda.synchronize()
        values = {'elapsed_seconds': time.monotonic() - self.started,
                  'gpu_peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                  'gpu_reserved_GiB': torch.cuda.memory_reserved() / 2**30,
                  'host_peak_RSS_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
        self.report[stage] = values
        print(stage, json.dumps(values), flush=True)

    def _train_step(self, batch, batch_idx):
        inputs = self._prepare_input_dict(self.convert_input_format(batch))
        predictions = self.engine(inputs, train_mode=True)
        video_loss, action_loss = self.compute_loss(inputs, predictions)
        loss = video_loss + action_loss
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Nonfinite loss; aborting without overwriting checkpoints')
        boundary = self.engine.is_gradient_accumulation_boundary()
        # compute_loss already scales both losses by 1/G, exactly once.
        self.engine.backward(loss, scale_wrt_gas=False)
        self.engine.step()
        self.sampler.commit()
        result = {'latent_loss': float(video_loss.detach()), 'action_loss': float(action_loss.detach()),
                  'should_log': boundary}
        if boundary:
            result['total_norm'] = float(self.engine.get_global_grad_norm())
            if not math.isfinite(result['total_norm']):
                raise FloatingPointError('Nonfinite gradient norm')
        return result

    def _resume_signature(self):
        ignored = {'__name__', 'save_root', 'num_steps', 'save_interval', 'gc_interval', 'load_worker',
                   'enable_wandb', 'rank', 'local_rank', 'world_size', 'wan22_pretrained_model_name_or_path'}
        return json.loads(json.dumps({k: v for k, v in self.config.items() if k not in ignored}, default=str))

    def _rng_state(self):
        return {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}

    def _restore_checkpoint(self):
        if not any((self.resume_from / marker).is_file()
                   for marker in ('COMPLETE', 'DEEPSPEED_COMPLETE')):
            raise ValueError('Resume requires a completed DeepSpeed optimizer checkpoint')
        metadata = json.loads((self.resume_from / 'training_metadata.json').read_text())
        if metadata['backend'] != self.backend:
            raise ValueError(f"Resume requires the saved backend: {metadata['backend']}")
        if metadata['signature'] != self._resume_signature() or metadata['data'] != self.data_identity:
            raise ValueError('Resume training configuration/data changed; use transformer as a new finetune instead')
        loaded_path, state = self.engine.load_checkpoint(str(self.resume_from / 'deepspeed'), tag='state',
                                                       load_optimizer_states=True, load_lr_scheduler_states=True)
        if not loaded_path or not state:
            raise RuntimeError('DeepSpeed failed to restore the complete checkpoint')
        self.step = state['optimizer_step']
        if self.engine.global_steps != self.step:
            raise RuntimeError('Checkpoint optimizer step mismatch')
        self.sampler.load_state_dict(state['sampler'])
        self.loader_generator.set_state(state['loader_generator'].cpu())
        rng = state['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch_cpu'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in rng['torch_cuda']])
        print(f'RESUMED_FULL_TRAINING_STATE step={self.step}', flush=True)

    def _export_transformer(self, directory):
        """Bound CPU staging memory while exporting normal Diffusers BF16 shards."""
        directory.mkdir(parents=True)
        self.transformer.save_config(str(directory))
        shard, shard_bytes, shard_paths, weight_map = {}, 0, [], {}
        total_bytes = 0

        def flush():
            nonlocal shard, shard_bytes
            path = directory / f'shard-{len(shard_paths) + 1:05d}.safetensors'
            save_file(shard, str(path), metadata={'format': 'pt'})
            for name in shard:
                weight_map[name] = path.name
            shard_paths.append(path)
            shard, shard_bytes = {}, 0

        state_source = merged_state_dict(self.transformer) if self.lora_enabled else self.transformer.state_dict()
        for name, tensor in state_source.items():
            name = name.replace('._checkpoint_wrapped_module', '')
            value = tensor.detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
            size = value.numel() * value.element_size()
            if shard and shard_bytes + size > 2_000_000_000:
                flush()
            shard[name] = value
            shard_bytes += size
            total_bytes += size
        if shard:
            flush()
        renamed = {}
        for index, path in enumerate(shard_paths, 1):
            name = ('diffusion_pytorch_model.safetensors' if len(shard_paths) == 1 else
                    f'diffusion_pytorch_model-{index:05d}-of-{len(shard_paths):05d}.safetensors')
            renamed[path.name] = name
            path.rename(directory / name)
        if len(shard_paths) > 1:
            (directory / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(
                {'metadata': {'total_size': total_bytes}, 'weight_map': {k: renamed[v] for k, v in weight_map.items()}}, indent=2))

    def save_checkpoint(self):
        # Only save at an optimizer boundary. Publish the directory atomically;
        # an interrupted save cannot masquerade as a usable checkpoint.
        final = self.save_dir / f'checkpoint_step_{self.step}'
        if final.exists():
            raise FileExistsError(f'Refusing to overwrite {final}')
        pending = self.save_dir / f'.checkpoint_step_{self.step}.incomplete'
        if pending.exists():
            raise FileExistsError(f'Inspect previous incomplete save: {pending}')
        pending.mkdir()
        state = {'optimizer_step': self.step, 'sampler': self.sampler.state_dict(),
                 'loader_generator': self.loader_generator.get_state(), 'rng': self._rng_state()}
        metadata = {'backend': getattr(self, 'backend', 'cpu_offload'), 'optimizer_step': self.step,
                    'signature': self._resume_signature(), 'data': self.data_identity,
                    'backend_source_sha256': sha256(POLICY / 'train_cpu_offload.py'),
                    'loader_workers': self.config.load_worker,
                    'worker_rng_note': 'Exact worker RNG restoration is only supported with load_worker=0'}
        (pending / 'training_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        saved = self.engine.save_checkpoint(str(pending / 'deepspeed'), tag='state', client_state=state, save_latest=False)
        if not saved:
            raise RuntimeError('DeepSpeed checkpoint save failed')
        (pending / 'DEEPSPEED_COMPLETE').write_text('DeepSpeed model, optimizer, scheduler, RNG and sample order saved.\n')
        self._export_transformer(pending / 'transformer')
        if self.lora_enabled:
            # Small standalone adapter includes full trainable action heads.
            adapter = {name.replace('._checkpoint_wrapped_module', ''): value.detach().cpu().clone()
                       for name, value in self.transformer.named_parameters() if value.requires_grad}
            save_file(adapter, str(pending / 'adapter.safetensors'))
            recipe = {key: value for key, value in self.config.items() if key.startswith('lora_')}
            recipe['base_model'] = self.config.wan22_pretrained_model_name_or_path
            (pending / 'adapter_config.json').write_text(json.dumps(recipe, indent=2) + '\n')
        if getattr(self.config, 'dataset_type', None) == 'real_world_cached' and not self.synthetic_smoke:
            cache_metadata_path = Path(self.config.dataset_path) / 'metadata.json'
            (pending / 'real_world_metadata.json').write_text(cache_metadata_path.read_text())
        (pending / 'COMPLETE').write_text('Model, optimizer, LR scheduler, step, RNG and sample order saved.\n')
        pending.rename(final)
        print(f'CHECKPOINT_COMPLETE {final}', flush=True)
        return final

    def train(self):
        samples = {name: p.detach().flatten()[:4096].clone() for name, p in self.transformer.named_parameters()
                   if self.synthetic_smoke and any(key in name for key in
                       ('patch_embedding_mlp.weight', 'attn1.to_q.weight', 'proj_out.weight'))}
        last_saved = self.step if self.resume_from else -1
        try:
            while self.step < self.config.num_steps:
                update_started = time.monotonic()
                video_loss = action_loss = 0.
                for micro_step in range(self.gradient_accumulation_steps):
                    batch = self._get_next_batch()
                    result = self._train_step(batch, micro_step)
                    video_loss += result['latent_loss']
                    action_loss += result['action_loss']
                    if result['should_log'] != (micro_step == self.gradient_accumulation_steps - 1):
                        raise RuntimeError('DeepSpeed and trainer accumulation boundaries disagree')
                self.step += 1
                if self.engine.global_steps != self.step:
                    raise RuntimeError('DeepSpeed and trainer optimizer step counters disagree')
                metrics = {'step': self.step, 'video_loss': video_loss, 'action_loss': action_loss,
                           'update_seconds': time.monotonic() - update_started,
                           'gpu_peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                           'flow_loss': video_loss + action_loss,
                           'gradient_norm': result['total_norm'], 'lr_next': self.lr_scheduler.get_last_lr()[0]}
                metric_interval = getattr(self.config, 'action_metric_interval', 0)
                if (not self.synthetic_smoke and metric_interval and
                        (self.step == 1 or self.step % metric_interval == 0 or self.step == self.config.num_steps)):
                    from action_metrics import record_action_metrics
                    metrics.update(record_action_metrics(self, batch))
                if (not self.synthetic_smoke and getattr(self.config, 'validation_interval', 0)
                        and (self.step == 1 or self.step % self.config.validation_interval == 0
                             or self.step == self.config.num_steps)):
                    validation = self.validate_real_world()
                    validation['val_flow_loss'] = validation['val_video_loss'] + validation['val_action_loss']
                    if metric_interval:
                        validation.update(record_action_metrics(self, batch, validation=True))
                    if not all(math.isfinite(validation[key]) for key in ('val_video_loss', 'val_action_loss')):
                        raise FloatingPointError('Nonfinite validation loss')
                    print('VALIDATION', json.dumps(validation), flush=True)
                    with (Path(self.config.save_root) / 'validation.jsonl').open('a') as stream:
                        stream.write(json.dumps(validation) + '\n')
                    metrics.update(validation)
                self.report['last_metrics'] = metrics
                print('TRAIN_STEP', json.dumps(metrics), flush=True)
                with (Path(self.config.save_root) / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(metrics) + '\n')
                if self.wandb:
                    self.wandb.log(metrics, step=self.step)
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()
                    last_saved = self.step
            if self.save_final and last_saved != self.step:
                self.save_checkpoint()
            if samples:
                changes = {}
                for name, p in self.transformer.named_parameters():
                    if name in samples:
                        changes[name] = int(torch.count_nonzero(p.detach().flatten()[:4096] != samples[name]).item())
                self.report['sampled_weight_changes'] = changes
            self.report['success'] = True
            print('GPU_TRAIN_OK' if self.backend == 'gpu' else 'CPU_OFFLOAD_TRAIN_OK', flush=True)
        finally:
            self.report['optimizer_steps'] = self.step
            self._log_resources('finished')
            (Path(self.config.save_root) / 'training_result.json').write_text(json.dumps(self.report, indent=2) + '\n')
            if self.wandb:
                self.wandb.finish()


def run(config, args):
    if int(os.getenv('WORLD_SIZE', '1')) != 1:
        raise ValueError('Select one GPU for gpu/cpu_offload; use --backend fsdp for the official multi-GPU backend')
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '1')))
    dist.init_process_group('nccl')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        trainer = CPUOffloadTrainer(config, seed=args.seed, resume_from=args.resume_from,
                                    dataset_init_workers=args.dataset_init_workers,
                                    synthetic_smoke=args.synthetic_smoke,
                                    save_final=not args.no_final_save, backend=args.backend)
        trainer.train()
    finally:
        dist.destroy_process_group()
