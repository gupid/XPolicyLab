"""State/action-only projection of this adapter's official normalization stream.

FrankaInputs leaves these two numeric fields unchanged. Avoid decoding RGB just
for statistics; retain upstream RunningStats, batch order, horizon and drop_last.
"""
from pathlib import Path
import json

import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize


def numeric_columns(dataset_root):
    files = sorted((Path(dataset_root) / 'data').rglob('*.parquet'))
    if not files:
        raise FileNotFoundError(dataset_root)
    keys = ('observation.state', 'action', 'episode_index', 'index', 'frame_index')
    tables = [pq.read_table(p, columns=list(keys)) for p in files]
    arrays = {key: np.concatenate([np.asarray(t[key].to_pylist()) for t in tables]) for key in keys}
    states = arrays['observation.state'].astype(np.float32)
    actions = arrays['action'].astype(np.float32)
    episodes = arrays['episode_index']
    if states.shape != actions.shape or states.shape[1:] != (8,):
        raise ValueError('This projection only supports this adapter\'s unmodified 8D Franka fields')
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError('Nonfinite numeric fields')
    if not np.array_equal(arrays['index'], np.arange(len(states))):
        raise ValueError('Dataset rows are not in official global index order')
    starts = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1]
    counts = np.diff(np.r_[starts, len(episodes)])
    if not np.array_equal(episodes[starts], np.arange(len(starts))):
        raise ValueError('Noncontiguous episode indices')
    if not np.array_equal(arrays['frame_index'], np.arange(len(states)) - np.repeat(starts, counts)):
        raise ValueError('Noncontiguous episode frames')
    ends = np.repeat(starts + counts, counts)
    return states, actions, ends


def action_windows(actions, ends, indices, horizon):
    query = np.minimum(np.asarray(indices)[:, None] + np.arange(horizon), ends[indices, None] - 1)
    return actions[query]


def compute(dataset_root, batch_size=8, horizon=50, diagnostics=None):
    states, actions, ends = numeric_columns(dataset_root)
    stats = {k: normalize.RunningStats() for k in ('state', 'actions')}
    total = len(states) // batch_size
    for batch in range(total):
        indices = np.arange(batch * batch_size, (batch + 1) * batch_size)
        stats['state'].update(states[indices])
        stats['actions'].update(action_windows(actions, ends, indices, horizon))
    result = {k: value.get_statistics() for k, value in stats.items()}
    # A task can hold its gripper closed for >99% of all samples. Histogram
    # quantiles then collapse and upstream divides even ordinary commands by
    # 1e-6. Guard only the semantically known gripper dimension, using train data.
    numeric = {'state': states, 'actions': actions}
    for key, value in result.items():
        lower, upper = float(numeric[key][:, 7].min()), float(numeric[key][:, 7].max())
        span = float(value.q99[7] - value.q01[7])
        if upper > lower and span < 0.01 * (upper - lower):
            if diagnostics is not None:
                diagnostics[key] = {'dimension': 7, 'original_q01': float(value.q01[7]),
                                    'original_q99': float(value.q99[7]), 'replacement_q01': lower,
                                    'replacement_q99': upper, 'rule': 'train min/max if quantile span <1% of full train span'}
            value.q01[7], value.q99[7] = lower, upper
    return result


def save_for_config(cfg, root):
    from real_world.config import FrankaDataConfig
    if not isinstance(cfg.data, FrankaDataConfig):
        raise TypeError('Numeric projection is only valid for the identity Franka state/action transform')
    diagnostics = {}
    stats = compute(root / 'datasets' / cfg.data.repo_id, cfg.batch_size, cfg.model.action_horizon, diagnostics)
    target = cfg.assets_dirs / cfg.data.repo_id
    normalize.save(target, stats)
    (target / 'normalization_guards.json').write_text(json.dumps(diagnostics, indent=2))
    if diagnostics:
        print('Gripper quantile range fallback: ' + json.dumps(diagnostics), flush=True)
    print(f'Writing stats to: {target} (official RunningStats; numeric columns only; batch/horizon/drop_last unchanged)', flush=True)
