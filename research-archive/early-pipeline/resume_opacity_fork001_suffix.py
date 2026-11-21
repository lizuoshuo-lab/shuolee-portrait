"""Recover only the 001 suffixes stopped by the pinned NumPy float64 checkpoint scalar."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

import run_opacity_fork001 as fork
from review_bound_alpha_ab import pin, read, require, sha


CHECKPOINT = 'shared-after-densify-step9000.pt'
CHECKPOINT_SHA = '97613a21e06d694db4cb448d336018649c015ac8f0dfbfe0b3a87e56db921818'
PINS = {
    'preregistration.json': '88f4344ecfeebc60bfa589812527c666ff0a6000c5464f5d6b36cf52d545d434',
    'status.json': '2e9fe7d5cefc7869058814349915f1512727fe5217d853b08de6c7b61bab6ff5',
    'stock/run/status.json': 'df2ccb88b0bcc26bd2aeba758a91c2df6b0f0e1f2df9d212e11df7625087eb12',
    'stock/run/gaussian-run/result.json': 'b5fb15363ef6f7aa10e9553eee08c23daad9c675a55842bb48514b93b0212e69',
    'stock/run/gaussian-run/step-010000/bound-replay.json': '2453e38769c315f921a40abe5cf21b819b68eb31de163366ce98153f34dd342e',
    'stock-repeat/run/gaussian-run/loss.jsonl': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    CHECKPOINT: CHECKPOINT_SHA,
}
UNSAFE_GLOBALS = {'numpy.core.multiarray.scalar', 'numpy.dtype'}
ARMS = ('stock-repeat', 'omit-reset')


def safe_globals():
    import torch

    return torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))])


class RecoveredFork(fork.OpacityFork):
    def restore(self, *args):
        with safe_globals():
            return super().restore(*args)


def prepare(original, project, output):
    import torch

    require(output.parent == original.parent and output.name == original.name+'-recovered'
            and not (original/'result.json').exists(), 'Expected a fresh sibling recovery and no prior completed fork.')
    for name, digest in PINS.items():
        pin(original/name, digest)
    failure = read(original/'status.json')
    require(failure['state'] == 'failed' and failure['arm'] == 'stock-repeat'
            and failure['error_type'] == 'UnpicklingError' and 'numpy.core.multiarray.scalar' in failure['error'],
            'The pinned failure no longer matches the scalar decoder issue.')
    plan = read(original/'preregistration.json')
    baseline = Path(plan['baseline']).resolve(strict=True)
    rt, profile, order, bins, current = fork.prepare(baseline, project, output)
    require(plan == current, 'Current shared-state inputs, source or configuration differ.')
    checkpoint = original/CHECKPOINT
    require(set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)) == UNSAFE_GLOBALS,
            'Different checkpoint deserialization contract.')
    with safe_globals():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = saved['model']
    groups = model['optimizer']['param_groups']
    require(saved['step'] == fork.STEP and saved['frame_order'] == order and saved['row_bins'] == bins
            and len(groups) == len(model['optimizer']['state']) == 7 and groups[0]['name'] == 'xyz'
            and type(groups[0]['lr']) is np.float64 and float(groups[0]['lr']) == .0005000000000000001
            and set(saved['surface']) == set(fork.FIXED_SURFACE)
            and list(name for name, grad in model['gradients'].items() if grad is not None) == ['static_offset']
            and model['parameters']['_xyz'].shape[0] == 62103 and model['rng']['cuda'].numel() > 0,
            'The exact stock step-9000 state is required.')
    stock = read(original/'stock/run/status.json')
    terminal = original/'stock/run/gaussian-run/step-010000'
    for name, key in (('world.ply', 'world_ply_sha256'), ('bound-local.ply', 'bound_ply_sha256'),
                      ('reference-render.png', 'reference_render_sha256')):
        pin(terminal/name, stock[key])
    pin(original/'stock/run/gaussian-run/training-cameras.json', stock['training_camera_sha256'])
    receipt = dict(state='verified_resume_suffix_inputs', original_fork=str(original), recovered_root=str(output),
                   original_pins_sha256=PINS, checkpoint_sha256=CHECKPOINT_SHA,
                   recovery_driver_sha256=sha(Path(__file__)), allowed_numpy_globals=sorted(UNSAFE_GLOBALS),
                   allowed_numpy_dtype='float64', weights_only=True, recovered_arms=list(ARMS),
                   expected_optimizer_updates_per_suffix=1000, reused_stock_prefix=True,
                   original_failure_preserved=True, quality_accepted=False)
    return rt, baseline, profile, order, bins, plan, stock, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'project', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    original, project, output = (p.resolve() for p in (args.original, args.project, args.output))
    rt, baseline, profile, order, bins, plan, stock, receipt = prepare(original, project, output)
    if args.preflight:
        print(json.dumps(dict(**receipt, optimizer_updates=0, gpu_calls=0)))
        return
    import torch
    from shuolee_portrait.cli import gpu_lock

    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock():
            require(torch.cuda.device_count() == 1, 'Expose one CUDA device for the recovered suffixes.')
            output.mkdir(parents=True, exist_ok=False)
            stock_summary = {key: value for key, value in stock.items() if key not in ('stage', 'updated', 'quality_accepted')}
            summaries = [stock_summary]
            for arm in ARMS:
                rt.save(output/'status.json', dict(state='running', arm=arm, updated=rt.now()))
                state = RecoveredFork(original/CHECKPOINT, arm, order, bins)
                state.sha256 = CHECKPOINT_SHA
                try:
                    summaries.append(fork.run_arm(baseline, output, rt, profile, plan, state))
                except BaseException as error:
                    rt.save(output/'status.json', dict(state='failed', arm=arm, updated=rt.now(),
                                                       error_type=type(error).__name__, error=str(error)))
                    raise
            for name, digest in PINS.items():
                pin(original/name, digest)
            expected = sha(baseline/'gaussian-run/training-cameras.json')
            require(all(row['training_camera_sha256'] == expected and row['checkpoint_sha256'] == CHECKPOINT_SHA
                        for row in summaries), 'Camera or shared checkpoint identity changed.')
            recovery = output/'recovery.json'
            arm_status = {arm: sha(output/arm/'run/status.json') for arm in ARMS}
            rt.save(recovery, dict(receipt, state='suffixes_recovered', arms=summaries[1:],
                                   recovered_arm_status_sha256=arm_status))
            result = dict(state='completed_pending_controlled_render_review', checkpoint_sha256=CHECKPOINT_SHA,
                          training_camera_sha256=expected, arms=summaries, quality_accepted=False,
                          recovery_receipt=str(recovery), recovery_receipt_sha256=sha(recovery))
            require(not (original/'result.json').exists(), 'Original completion receipt appeared concurrently.')
            rt.save(original/'result.json', result)
            rt.save(output/'status.json', dict(**result, updated=rt.now()))
            print(json.dumps(result))


if __name__ == '__main__':
    main()
