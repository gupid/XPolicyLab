"""Apply confirmed episode exclusions using LeRobot's official editing tool."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from real_world.config import TASKS, register_configs, repo_id
from real_world.norm import numeric_columns
from real_world.prepare import write_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exclude-file', type=Path, required=True)
    args = parser.parse_args()
    root = register_configs()
    exclusions = json.loads(args.exclude_file.read_text())
    if exclusions.get("shell_game"):
        raise ValueError("Shell-game exclusions require a fresh conversion to reindex initial context")
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    for task in TASKS:
        requested = set(exclusions.get(task, []))
        if not requested:
            continue
        report_path = root / task / 'report.json'
        report = json.loads(report_path.read_text())
        rows = report['episodes']
        if requested - {r['source'] for r in rows}:
            raise ValueError('Unknown source member in exclusions')
        for split in ('train', 'val'):
            split_rows = [r for r in rows if r['split'] == split]
            remove = [i for i, row in enumerate(split_rows) if row['source'] in requested]
            if not remove:
                continue
            dataset_root = root / 'datasets' / repo_id(task, split)
            temporary = dataset_root.with_name(dataset_root.name + '.review_filtered')
            old_root = dataset_root.with_name(dataset_root.name + '.before_review')
            if temporary.exists() or old_root.exists():
                raise FileExistsError('Previous review operation needs inspection before retry')
            states, actions, ends = numeric_columns(dataset_root)
            starts = np.r_[0, np.flatnonzero(np.diff(ends)) + 1]
            keep = np.ones(len(states), dtype=bool)
            for i in remove:
                keep[starts[i]:ends[starts[i]]] = False
            source = LeRobotDataset(repo_id(task, split), root=dataset_root)
            filtered = delete_episodes(source, remove, output_dir=temporary, repo_id=repo_id(task, split))
            actual_state, actual_action, _ = numeric_columns(temporary)
            np.testing.assert_array_equal(actual_state, states[keep])
            np.testing.assert_array_equal(actual_action, actions[keep])
            assert filtered.meta.total_episodes == len(split_rows) - len(remove)
            assert len(filtered) == int(keep.sum())
            del filtered, source
            dataset_root.rename(old_root)
            temporary.rename(dataset_root)
            # The retained data are byte-preserving reindexed copies, checked above.
            shutil.rmtree(old_root)
            print(f'{task}/{split}: removed episode indices {remove}; kept {int(keep.sum())} action frames', flush=True)
        for row in rows:
            if row['source'] in requested:
                row['original_split'] = row.get('original_split', row['split'])
                row['split'] = 'excluded'
                row['review_excluded_reason'] = 'User confirmed unusable'
                (root / task / row['video']).unlink(missing_ok=True)
                row['video_removed'] = True
        report['counts'] = {s: {'episodes': sum(r['split'] == s for r in rows),
                                'frames': sum(r['frames'] for r in rows if r['split'] == s)}
                            for s in ('train', 'val', 'excluded')}
        report['review_split_policy'] = 'Retain existing validation trajectories; remove confirmed unusable episodes'
        report_path.write_text(json.dumps(report, indent=2))
        write_index(root / task, [r for r in rows if r['split'] != 'excluded'])
        # The old statistics included the excluded episodes and must not be reused.
        (root / 'assets' / f'pi05_real_{task}_train' / repo_id(task) / 'norm_stats.json').unlink(missing_ok=True)
        print(json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    main()
