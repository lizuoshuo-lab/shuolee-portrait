"""Frozen single-scene gsplat baseline; no scene-specific algorithm modifications."""
import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'repos' / 'gsplat'
COMMIT = '937e29912570c372bed6747a5c9bf85fed877bae'
sys.path.insert(0, str(REPO / 'examples'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    if not a.run.replace('-', '').replace('_', '').isalnum() or a.steps < 1:
        raise ValueError('Invalid run identifier or step count')
    run = ROOT / 'outputs' / 'large-scene-gs-001' / 'training' / a.run
    run.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    outcome = {'status': 'starting', 'started_utc': started, 'preflight': a.preflight}
    save(run / 'outcome.json', outcome)
    try:
        import numpy as np
        import torch
        import simple_trainer as trainer

        actual_commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
        if actual_commit != COMMIT:
            raise RuntimeError(f'Unexpected gsplat commit {actual_commit}')
        source = REPO / 'examples' / 'simple_trainer.py'
        canonical = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:examples/simple_trainer.py'])
        if source.read_bytes().replace(b'\r\n', b'\n') != canonical.replace(b'\r\n', b'\n'):
            raise RuntimeError('Official trainer source differs from fixed commit')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        prepared = ROOT / 'outputs' / 'large-scene-gs-001' / 'data-preparation'
        manifest = json.loads((prepared / 'input-manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'ready_for_isolated_official_gsplat_single_scene_baseline':
            raise RuntimeError('Complete dataset audit not ready')
        split = json.loads((prepared / 'split.json').read_text(encoding='utf-8'))
        cfg = trainer.Config(
            disable_viewer=True, disable_video=True,
            data_dir=str(ROOT / 'data' / 'large-scene-gs' / 'tandt' / 'truck'),
            data_factor=4, result_dir=str(run), test_every=8,
            max_steps=a.steps,
            eval_steps=[] if a.preflight else sorted(set([min(7000, a.steps), a.steps])),
            save_steps=[a.steps], save_ply=not a.preflight, ply_steps=[a.steps],
            packed=True,
        )
        # Deliberately do not rescale the official SH or densification schedule.
        # A preflight is an execution check, not a quality/densification experiment.
        configuration = dataclasses.asdict(cfg)
        save(run / 'configuration.json', {
            'scope': 'Official gsplat v1.5.3 trainer on full Truck; not Hierarchical3DGS reproduction',
            'gsplat_commit': COMMIT, 'trainer_sha256': digest(source),
            'config': configuration, 'seed': 42,
            'source_changes': 'None in trainer. Existing Windows JIT compile flag patch in gsplat backend; four fixed-width little-endian read fields in isolated pure-Python pycolmap, receipt in environment/pycolmap-windows-binary.json.',
            'schedule': 'Original schedule; max_steps truncates only the total budget; no adjust_steps call',
            'heldout_scope': split['used_for_camera_sfm'],
            'split_sha256': digest(prepared / 'split.json'),
            'input_manifest_sha256': digest(prepared / 'input-manifest.json'),
            'checkpoint_timing': 'Official snapshots are saved before final optimizer/strategy update. final_state.pt below is saved after all updates, matching final evaluation state.',
        })
        packages = {}
        for name in ['torch', 'torchvision', 'numpy', 'gsplat', 'pycolmap', 'nerfview', 'viser', 'torchmetrics', 'fused_ssim', 'Pillow', 'scikit-learn']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        save(run / 'runtime.json', {'python': sys.version, 'executable': sys.executable,
            'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
            'capability': torch.cuda.get_device_capability(), 'packages': packages})
        torch.cuda.reset_peak_memory_stats()
        runner = trainer.Runner(0, 0, 1, cfg)
        train_names = [runner.parser.image_names[i] for i in runner.trainset.indices]
        test_names = [runner.parser.image_names[i] for i in runner.valset.indices]
        if train_names != split['train_names'] or test_names != split['test_names']:
            raise RuntimeError('Parser split differs from frozen independent input audit')
        np.savez_compressed(run / 'parser_geometry.npz',
            image_names=np.asarray(runner.parser.image_names),
            camtoworlds=runner.parser.camtoworlds,
            global_transform=runner.parser.transform,
            points=runner.parser.points,
            camera_ids=np.asarray(runner.parser.camera_ids),
            K=np.asarray([runner.parser.Ks_dict[k] for k in runner.parser.camera_ids]),
            train_indices=runner.trainset.indices, test_indices=runner.valset.indices)
        save(run / 'split.json', {'train_names': train_names, 'test_names': test_names})
        runner.train()
        torch.cuda.synchronize()
        final_path = run / 'final_state.pt'
        torch.save({'step': a.steps-1, 'completed_updates': a.steps,
                    'splats': runner.splats.state_dict(),
                    'state_timing': 'after final optimizer and densification updates'}, final_path)
        runner.writer.close()
        outcome.update(status='preflight_passed' if a.preflight else 'training_completed',
            completed_updates=a.steps, gaussian_count=len(runner.splats['means']),
            elapsed_seconds=time.perf_counter()-timer,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            final_state_sha256=digest(final_path))
    except Exception as error:
        outcome.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                       elapsed_seconds=time.perf_counter()-timer)
        save(run / 'outcome.json', outcome)
        raise
    save(run / 'outcome.json', outcome)
    print(json.dumps(outcome, ensure_ascii=False))

if __name__ == '__main__':
    main()
