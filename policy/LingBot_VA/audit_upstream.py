"""Check the pinned upstream training source, configs, and CPU noise/loss parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parent
VENDOR = POLICY / 'lingbot_va'
MANIFEST = POLICY / 'upstream_training_manifest.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(reference=None):
    manifest = json.loads(MANIFEST.read_text())
    errors = []
    for name, digest in manifest['official_files'].items():
        for root in ([VENDOR, Path(reference)] if reference else [VENDOR]):
            path = root / name
            if not path.is_file() or sha256(path) != digest:
                errors.append(f'Upstream source mismatch: {path}')
    for name, entry in manifest['adapter_exceptions'].items():
        if not (VENDOR / name).is_file() or sha256(VENDOR / name) != entry['vendor_sha256']:
            errors.append(f'Unreviewed adapter exception: {name}')
        if reference and entry.get('official_sha256'):
            path = Path(reference) / name
            if not path.is_file() or sha256(path) != entry['official_sha256']:
                errors.append(f'Reference exception changed: {name}')
    expected = set(manifest['official_files']) | set(manifest['adapter_exceptions'])
    actual = {str(p.relative_to(VENDOR)) for p in (VENDOR / 'wan_va').rglob('*.py')}
    errors.extend(f'Unreviewed source file: {name}' for name in sorted(actual - expected))
    if errors:
        raise RuntimeError('\n'.join(errors))
    return manifest


def probe(root):
    # Separate subprocesses prevent the official .pth and duplicate package names
    # from making both sides accidentally import the same checkout.
    sys.path.insert(0, str(Path(root).resolve()))
    import torch
    from wan_va.configs import VA_CONFIGS
    import wan_va.train as train
    assert Path(train.__file__).resolve().is_relative_to(Path(root).resolve())
    for function in (train.load_transformer, train.apply_ac, train.shard_model,
                     train.sample_timestep_id, train.data_seq_to_patch):
        module_path = Path(sys.modules[function.__module__].__file__).resolve()
        assert module_path.is_relative_to(Path(root).resolve()), module_path
    torch.set_num_threads(1)
    configs = {name: dict(cfg) for name, cfg in VA_CONFIGS.items()
               if name not in {'robotwin30_train', 'robotwin30_lora_train', 'real_world_lora_train'}}
    trainer = train.Trainer.__new__(train.Trainer)
    trainer.device = torch.device('cpu')
    trainer.patch_size = (1, 2, 2)
    trainer.gradient_accumulation_steps = 2
    trainer.train_scheduler_latent = train.FlowMatchScheduler(shift=5, sigma_min=0., extra_one_step=True)
    trainer.train_scheduler_action = train.FlowMatchScheduler(shift=1, sigma_min=0., extra_one_step=True)
    for scheduler in (trainer.train_scheduler_latent, trainer.train_scheduler_action):
        scheduler.set_timesteps(1000, training=True)
    probes = []
    for seed in (0, 42, 20261019):
        torch.manual_seed(seed)
        shape = (1, 30, 3, 8, 1)
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[:, [0, 1, 2, 3, 4, 28]] = True
        batch = {'latents': torch.randn(1, 48, 3, 4, 8), 'actions': torch.randn(shape),
                 'actions_mask': mask, 'text_emb': torch.randn(1, 512, 4096)}
        inputs = trainer._prepare_input_dict(trainer.convert_input_format(batch))
        predictions = (torch.randn(1, 96, 48, requires_grad=True),
                       torch.randn(1, 24, 30, requires_grad=True))
        losses = trainer.compute_loss(inputs, predictions)
        sum(losses).backward()

        def fingerprint(value):
            if isinstance(value, dict):
                return {key: fingerprint(val) for key, val in value.items()}
            if torch.is_tensor(value):
                tensor = value.detach().contiguous()
                return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                        'sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
            return value

        probes.append({'seed': seed, 'inputs': fingerprint(inputs),
                       'losses': [loss.item() for loss in losses],
                       'prediction_gradients': [fingerprint(pred.grad) for pred in predictions]})
    return json.loads(json.dumps({'configs': configs, 'noise_loss_probes': probes}, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, help='Official checkout pinned in the manifest')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--probe-root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_root:
        print(json.dumps(probe(args.probe_root), default=str))
        return
    manifest = check_sources(args.reference)
    result = {'official_revision': manifest['official_revision'],
              'identical_source_files': len(manifest['official_files']),
              'adapter_exceptions': manifest['adapter_exceptions'],
              'source_check': 'passed', 'runtime_parity': 'not_run'}
    if args.reference:
        outputs = []
        for root in (args.reference, VENDOR):
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        '--probe-root', str(root)], check=True, capture_output=True, text=True)
            outputs.append(json.loads(completed.stdout))
        if outputs[0] != outputs[1]:
            raise AssertionError('Official configurations or noise/loss/gradient probes differ')
        result.update(runtime_parity='passed', official_configurations=len(outputs[0]['configs']),
                      probe_seeds=[p['seed'] for p in outputs[0]['noise_loss_probes']],
                      scope='Exact CPU config/noise/weighted-loss/prediction-gradient parity. '
                            'Not GPU forward/update, preprocessing or FSDP/DeepSpeed numerical parity.')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
