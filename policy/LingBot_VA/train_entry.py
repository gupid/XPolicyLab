"""XPolicyLab training adapter: single GPU, optional CPU offload or official FSDP."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from audit_upstream import POLICY, VENDOR, check_sources, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', default='robotwin_train')
    parser.add_argument('--save-root', type=Path, required=True)
    parser.add_argument('--backend', choices=('gpu', 'cpu_offload', 'fsdp'),
                        default=os.getenv('LINGBOT_VA_TRAIN_BACKEND', 'cpu_offload'))
    parser.add_argument('--overrides', type=Path, help='Explicit JSON overrides of existing selected config keys')
    parser.add_argument('--seed', type=int, default=int(os.getenv('LINGBOT_VA_TRAIN_SEED', '0')))
    parser.add_argument('--resume-from', type=Path, help='Complete gpu/cpu_offload checkpoint directory')
    parser.add_argument('--dataset-init-workers', type=int,
                        default=min(8, int(os.getenv('SLURM_CPUS_PER_TASK', '1'))))
    parser.add_argument('--synthetic-smoke', action='store_true', help='Use labeled synthetic cached latents; never real data')
    parser.add_argument('--no-final-save', action='store_true', help='Skip final save (periodic saves still apply); smoke tests only')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved configuration; no GPU/training/writes')
    args = parser.parse_args()
    manifest = check_sources()
    # The active environment also points to the independent official checkout.
    # Select the vendored source explicitly so this launch really uses XPolicyLab.
    sys.path.insert(0, str(VENDOR))
    from wan_va import train
    if not Path(train.__file__).resolve().is_relative_to(VENDOR):
        raise RuntimeError(f'Wrong training source: {train.__file__}')
    if args.config_name not in train.VA_CONFIGS:
        parser.error(f'Unknown config {args.config_name!r}; choose from {sorted(train.VA_CONFIGS)}')
    original = train.VA_CONFIGS[args.config_name]
    resolved = copy.deepcopy(original)
    overrides = {'save_root': str(args.save_root.resolve())}
    if args.overrides:
        explicit = json.loads(args.overrides.read_text())
        if not isinstance(explicit, dict) or set(explicit) - set(resolved):
            parser.error('JSON overrides must be an object containing only keys in the selected config')
        if any(key in explicit for key in ('param_dtype', 'rank', 'local_rank', 'world_size')):
            parser.error('Precision/distributed keys cannot be overridden with JSON')
        overrides.update(explicit)
        overrides['save_root'] = str(args.save_root.resolve())
    base_model = os.getenv('LINGBOT_VA_BASE_MODEL_PATH')
    dataset = os.getenv('LINGBOT_VA_DATASET_PATH')
    if base_model:
        overrides['wan22_pretrained_model_name_or_path'] = str(Path(base_model).expanduser().resolve())
    if dataset:
        overrides['dataset_path'] = str(Path(dataset).expanduser().resolve())
        overrides['empty_emb_path'] = str(Path(overrides['dataset_path']) / 'empty_emb.pt')
    # Resolve task statistics/default budget before explicit hyperparameter overrides.
    cache_metadata = None
    if resolved.get('dataset_type') == 'real_world_cached' and not args.synthetic_smoke:
        from wan_va.real_world_dataset import configure_real_world
        cache_root = overrides.get('dataset_path', resolved.dataset_path)
        try:
            cache_metadata = configure_real_world(resolved, cache_root)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        protected = {'norm_stat', 'used_action_channel_ids', 'inverse_used_action_channel_ids',
                     'obs_cam_keys', 'height', 'width', 'action_dim', 'action_per_frame', 'dataset_type', 'empty_emb_path'}
        if args.overrides and protected.intersection(explicit):
            parser.error('Real-world layout/statistics must come from the latent cache')
        if cache_metadata.get('base_model') != str(Path(overrides.get(
                'wan22_pretrained_model_name_or_path', resolved.wan22_pretrained_model_name_or_path)).resolve()):
            parser.error('Use the same base model for latent extraction and training')
    wandb = os.getenv('LINGBOT_VA_ENABLE_WANDB')
    if wandb is not None:
        if wandb not in ('0', '1'):
            parser.error('LINGBOT_VA_ENABLE_WANDB must be 0 or 1')
        overrides['enable_wandb'] = wandb == '1'
    resolved.update(overrides)
    if args.backend == 'fsdp' and (args.resume_from or args.synthetic_smoke or args.no_final_save):
        parser.error('Resume/synthetic-smoke/no-final-save options are for gpu/cpu_offload only')
    if args.no_final_save and not args.synthetic_smoke:
        parser.error('--no-final-save is restricted to explicitly synthetic smoke tests')
    if args.dataset_init_workers < 1:
        parser.error('--dataset-init-workers must be positive')
    if args.backend != 'fsdp' and int(os.getenv('WORLD_SIZE', '1')) != 1:
        parser.error('gpu/cpu_offload supports exactly one GPU/process')
    record = {'official_revision': manifest['official_revision'],
              'config_name': args.config_name,
              'configuration_origin': 'XPolicyLab extension' if args.config_name in {
                  'robotwin30_train', 'robotwin30_lora_train', 'real_world_lora_train'} else 'official',
              'training_source': str(Path(train.__file__).resolve()),
              'backend': args.backend,
              'adapter_overrides': overrides, 'save_root': str(args.save_root.resolve()),
              'seed_semantics': ('Python/numpy/torch and independent sampler seeds; restored on single-GPU resume'
                                 if args.backend != 'fsdp' else
                                 'Run label and PYTHONHASHSEED only; official FSDP RNG behavior unchanged'),
              'seed': args.seed if args.backend != 'fsdp' else None,
              'resume_from': str(args.resume_from.resolve()) if args.resume_from else None,
              'synthetic_smoke': args.synthetic_smoke,
              'dataset_init_workers': args.dataset_init_workers,
              'adapter_source_sha256': {name: sha256(POLICY / name) for name in
                                        ('train_entry.py', 'train_cpu_offload.py', 'action_metrics.py',
                                         'train.sh', 'train_lora.sh')},
              'resolved_config': dict(resolved)}
    if args.backend != 'fsdp':
        from train_cpu_offload import deepspeed_config, precision_description
        record['deepspeed_config'] = deepspeed_config(resolved, args.backend)
        record['precision'] = precision_description(args.backend)
    print(json.dumps(record, indent=2, default=str), flush=True)
    if args.dry_run:
        return
    model_root = args.resume_from if args.resume_from else Path(resolved.wan22_pretrained_model_name_or_path)
    required = [model_root / 'transformer/config.json']
    if not args.synthetic_smoke:
        required.extend([Path(resolved.dataset_path), Path(resolved.empty_emb_path)])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error('Missing model/prepared dataset paths: ' + ', '.join(missing))
    if args.backend != 'fsdp' and not args.resume_from and (args.save_root / 'metrics.jsonl').exists():
        parser.error('This run already has training output; choose a new run name or --resume-from a complete checkpoint')
    # Each rank imports the same config. Rank zero records it before training.
    if int(os.getenv('RANK', '0')) == 0:
        args.save_root.mkdir(parents=True, exist_ok=True)
        (args.save_root / 'resolved_training_config.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
        if cache_metadata is not None:
            (args.save_root / 'real_world_metadata.json').write_text(json.dumps(cache_metadata, indent=2) + '\n')
    if args.backend != 'fsdp':
        from train_cpu_offload import run
        run(resolved, args)
    else:
        train.VA_CONFIGS[args.config_name] = resolved
        train.init_logger()
        train.run(args)


if __name__ == '__main__':
    main()
