"""Run the planned fixed-pose 4000/8000-Gaussian comparison on three seeds.

Runs are sequential to avoid GPU contention. Existing completed runs are checked
and reused; the original 8000/seed-0 baseline is explicitly identified as reused.
"""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
TRAINER = ROOT / 'scripts' / 'train_tiny_gs.py'

def main():
    destination = ROOT / 'outputs' / 'gaussian-count-study-001'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for points in (4000, 8000):
        for seed in (0, 1, 2):
            run = 'tiny-nerf-gsplat-001' if (points, seed) == (8000, 0) else f'tiny-nerf-g{points}-s{seed}'
            output = ROOT / 'outputs' / run
            metric_file = output / 'metrics.json'
            reused = metric_file.exists()
            if output.exists() and not reused:
                raise RuntimeError(f'Incomplete prior run requires investigation: {run}')
            if not reused:
                command = [str(PYTHON), str(ROOT / 'scripts' / 'run_local.py'),
                           str(TRAINER), '--points', str(points), '--seed', str(seed),
                           '--steps', '1500', '--run', run]
                log = ROOT / 'logs' / f'{run}.log'
                print(f'Running {run}', flush=True)
                with log.open('w', encoding='utf-8') as stream:
                    result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                            stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f'{run} failed; inspect {log.name}')
            metrics = json.loads(metric_file.read_text(encoding='utf-8'))
            if metrics['status'] != 'completed' or metrics['config']['points'] != points or metrics['config']['seed'] != seed or metrics['config']['steps'] != 1500:
                raise ValueError(f'Configuration mismatch: {run}')
            if metrics['train_views'] != list(range(100)) or metrics['test_views'] != list(range(100, 106)):
                raise ValueError(f'Evaluation split mismatch: {run}')
            record = {
                'run': run, 'points': points, 'seed': seed,
                'reused_existing_result': reused,
                'test_mean_psnr_db': metrics['test_mean_psnr_db'],
                'training_seconds': metrics['training_seconds_excludes_compile_and_final_evaluation'],
                'peak_torch_allocated_mib': metrics['peak_torch_allocated_mib'],
                'metrics': metric_file.relative_to(ROOT).as_posix(),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    groups = []
    for points in (4000, 8000):
        entries = [r for r in records if r['points'] == points]
        group = {'points': points, 'seeds': [r['seed'] for r in entries]}
        for key in ('test_mean_psnr_db', 'training_seconds', 'peak_torch_allocated_mib'):
            values = [r[key] for r in entries]
            group[key + '_mean'] = statistics.mean(values)
            group[key + '_sample_std'] = statistics.stdev(values)
        groups.append(group)
    differences = []
    for seed in (0, 1, 2):
        pair = {r['points']: r for r in records if r['seed'] == seed}
        differences.append({'seed': seed, 'psnr_8000_minus_4000_db': pair[8000]['test_mean_psnr_db'] - pair[4000]['test_mean_psnr_db']})
    report = {
        'status': 'completed', 'trainer_sha256': hashlib.sha256(TRAINER.read_bytes()).hexdigest(),
        'dataset': 'official Tiny NeRF Lego; 100x100; 100 train + 6 held-out views',
        'steps': 1500, 'records': records, 'groups': groups, 'paired_differences': differences,
        'scope': 'Exploratory fixed-pose renderer/map-capacity comparison; no tracking or SLAM',
        'limitations': [
            'Only three seeds and one small synthetic scene',
            'Same seed does not give a one-to-one matched Gaussian initialization across point counts',
            'Single-run optimization timings are short and affected by GPU scheduling/clock variability',
            'Peak PyTorch allocated memory excludes context and non-PyTorch allocations',
            'The original 8000/seed-0 result is reused rather than rerun',
            'Held-out images serve as validation for this exploratory comparison; future tuning needs a separate final test',
        ],
    }
    (destination / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'groups': groups, 'paired_differences': differences}, indent=2), flush=True)

if __name__ == '__main__':
    main()
