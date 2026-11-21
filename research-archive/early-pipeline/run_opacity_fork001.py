"""Run an isolated reset/no-reset/stock-repeat comparison from one live SH0 state."""
import argparse
import fcntl
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np

from review_bound_alpha_ab import pin, read, require, sha
from run_same_fit_opacity_tail001 import BASELINE_RESULT_SHA, INPUTS, RESET_STEPS


BASE_COMMIT = 'cf19d6677415fd23f12929820e0ccc4af35340f6'
TRAINING = 'src/shuolee_portrait/training.py'
BACKWARD = 'src/avatar_core/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu'
STEP = 9000
STOCK_RESETS = [*RESET_STEPS, STEP]
PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation', 'static_offset')
BUFFERS = ('binding', 'binding_counter', 'max_radii2D', 'xyz_gradient_accum', 'denom')
FIXED_SURFACE = ('base', 'linear', 'clothing', 'faces')
ARMS = {'stock': True, 'stock-repeat': True, 'omit-reset': False}


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def same_tree(expected, actual):
    import torch

    if type(expected) is not type(actual):
        return False
    if isinstance(expected, torch.Tensor):
        return (expected.dtype == actual.dtype and expected.shape == actual.shape and
                torch.equal(expected.contiguous().reshape(-1).view(torch.uint8),
                            actual.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(same_tree(expected[key], actual[key]) for key in expected)
    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(same_tree(a, b) for a, b in zip(expected, actual))
    return expected == actual


def snapshot(model):
    import torch

    params = {name: getattr(model, name) for name in PARAMETERS}
    groups = model.optimizer.param_groups
    require(len(groups) == len(params) and all(group['params'][0] is param and len(group['params']) == 1
            for group, param in zip(groups, params.values(), strict=True)), 'Unexpected SH0 Adam group layout.')
    device = model._xyz.device
    rng = np.random.get_state()
    return cpu_tree(dict(parameters=params, gradients={key: val.grad for key, val in params.items()},
        buffers={name: getattr(model, name) for name in BUFFERS}, optimizer=model.optimizer.state_dict(),
        timestep=model.timestep, active_sh_degree=model.active_sh_degree, spatial_lr_scale=model.spatial_lr_scale,
        rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                 python=random.getstate(), numpy=(rng[0], rng[1].tolist(), *rng[2:]))))


def surface_hashes(model):
    arrays = {name: getattr(model, name).detach().cpu().contiguous().numpy() for name in FIXED_SURFACE}
    return {name: (array.dtype.str, tuple(array.shape), sha256(array.tobytes()).hexdigest())
            for name, array in arrays.items()}


def restore_snapshot(model, options, state):
    import torch

    device = model._xyz.device
    model.active_sh_degree = state['active_sh_degree']
    model.spatial_lr_scale = state['spatial_lr_scale']
    for name, value in state['parameters'].items():
        setattr(model, name, torch.nn.Parameter(value.to(device=device, copy=True)))
    model.binding = state['buffers']['binding'].to(device=device, copy=True)
    model.select_mesh_by_timestep(state['timestep'])
    model.training_setup(options)
    # Stock training_setup creates the first six groups. Use the saved seventh group and its Adam state.
    model.optimizer.add_param_group(dict(params=[model.static_offset], name='static_offset'))
    model.optimizer.load_state_dict(cpu_tree(state['optimizer']))
    for name, value in state['buffers'].items():
        setattr(model, name, value.to(device=device, copy=True))
    for name, value in state['gradients'].items():
        getattr(model, name).grad = None if value is None else value.to(device=device, copy=True)
    rng = state['rng']
    torch.set_rng_state(rng['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(rng['cuda'], device)
    random.setstate(rng['python'])
    name, keys, *tail = rng['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *tail))
    require(same_tree(state, snapshot(model)), 'Restored Gaussian, Adam, pending gradient or RNG bytes differ.')


class OpacityFork:
    step = STEP

    def __init__(self, checkpoint, arm, camera_order, row_bins):
        self.checkpoint = checkpoint
        self.arm = arm
        self.camera_order = camera_order
        self.row_bins = row_bins
        self.sha256 = None
        self.restore_verified = False

    def capture(self, model, counts, bin_draws, slot):
        import torch

        require(self.arm == 'stock' and not self.checkpoint.exists() and model.timestep == slot,
                'Only the common stock prefix can save the live checkpoint.')
        self._check_counts(counts, bin_draws)
        state = snapshot(model)
        require(state['gradients']['static_offset'] is not None, 'The pending shared-mesh gradient was lost.')
        with self.checkpoint.open('xb') as stream:
            torch.save(dict(model=state, surface=surface_hashes(model), step=STEP,
                            frame_order=self.camera_order, row_bins=self.row_bins), stream)
        self.sha256 = sha(self.checkpoint)

    def _check_counts(self, counts, bin_draws):
        prefix = self.camera_order[:STEP]
        require(counts.tolist() == np.bincount(prefix, minlength=len(self.row_bins)).tolist()
                and bin_draws == {key: sum(self.row_bins[i] == key for i in prefix) for key in bin_draws},
                'Training did not use the frozen camera prefix.')

    def restore(self, model, options, order, counts, row_bins, bin_draws):
        import torch

        require(row_bins == self.row_bins and options.iterations == len(self.camera_order) == 10000,
                'Diagnostic options or view schedule changed.')
        if self.arm == 'stock':
            return 1
        pin(self.checkpoint, self.sha256)
        record = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        require(record['step'] == STEP and record['frame_order'] == self.camera_order and record['row_bins'] == row_bins,
                'Wrong checkpoint or camera order.')
        require(record['surface'] == surface_hashes(model), 'The fixed per-frame SMPL-X surface or LBS differs.')
        seen = list(islice(order, STEP))
        require(seen == self.camera_order[:STEP], 'Cannot replay the same camera prefix.')
        counts[:] = np.bincount(seen, minlength=len(row_bins))
        for slot in seen:
            bin_draws[row_bins[slot]] += 1
        self._check_counts(counts, bin_draws)
        with torch.no_grad():
            restore_snapshot(model, options, record['model'])
            if ARMS[self.arm]:
                model.reset_opacity()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        self.restore_verified = True
        return STEP + 1


def prepare(baseline, project, output):
    require(not output.exists() and not output.is_relative_to(baseline.parent)
            and not output.is_relative_to(project), 'Use a new output outside the baseline and checkout.')
    # Windows Git writes an absolute drive path into the linked worktree's .git file.
    worktree_git = (project/'.git').read_text(encoding='utf-8').strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['wslpath', '-u', worktree_git], text=True).strip()
    git = ['git', '-c', 'core.filemode=false', '-c', 'core.safecrlf=false',
           '--git-dir', git_dir, '--work-tree', str(project)]
    rev = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output([*git, 'diff', '--name-only', 'HEAD'], text=True).splitlines()
    require(rev == BASE_COMMIT and changes == [TRAINING], 'Use the dedicated detached opacity diagnostic checkout.')
    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.stable_selection import source_bin_training_order
    from shuolee_portrait.training import smpl_sh0_effective_config, verify_official_fit

    pin(baseline/'result.json', BASELINE_RESULT_SHA)
    result = read(baseline/'result.json')
    require(result['state'] == 'completed' and result['frames'] == 24 and result['steps'] == 10000
            and result['points'] == 58451 and result['reference_frame_id'] == 'frame_000056',
            'The exact delivered 001 baseline is required.')
    verify_official_fit(baseline)
    replay_path = baseline/result['bound_replay']
    pin(replay_path, result['bound_replay_sha256'])
    bound_path = replay_path.parent.parent/'bound-replay.json'
    bound = read(bound_path)
    pin(bound_path, read(replay_path)['bound_replay_sha256'])
    for name, digest in bound['source_files_sha256'].items():
        if name != TRAINING:
            pin(project/name, digest)
    asset = Path(read(baseline/'run.json')['assets'])/'smplx/SMPLX_NEUTRAL.npz'
    pin(asset, bound['asset_sha256'])
    profile, original = smpl_sh0_effective_config(), read(baseline/'gaussian-run/config.json')
    require(profile['iterations'] == 10000 and profile['seed'] == 0 and profile['opacity_reset_steps'] == RESET_STEPS
            and STEP % profile['optimization']['densification_interval'] == 0
            and original['optimizer_initial_state_entries'] == 0 and original['initial_points'] == 20908
            and all(value == original[key] for key, value in profile.items()
                    if key not in ('initial_points', 'image_cache_frames', 'opacity_reset_steps')),
            'The effective SH0 profile differs from the exact baseline.')
    pins = {name: sha(baseline/name) for name in INPUTS}
    rows = read(baseline/'data/training-inputs.json')['records']
    mask_pins = {row['mask']: pin(Path(row['mask']), row['mask_sha256']) for row in rows}
    selection = read(baseline/'fit/selection.json')
    row_bins, order = source_bin_training_order(rows, selection, seed=profile['seed'])
    order = list(islice(order, profile['iterations']))
    order_sha = sha256(np.asarray(order, dtype='<i4').tobytes()).hexdigest()
    plan = dict(state='frozen_common_state_diagnostic', baseline=str(baseline),
                baseline_result_sha256=BASELINE_RESULT_SHA, baseline_bound_sha256=sha(bound_path),
                input_sha256=pins, mask_sha256=mask_pins, source_commit=rev, diagnostic_training_sha256=sha(project/TRAINING),
                backward_sha256=sha(project/BACKWARD), driver_sha256=sha(Path(__file__)),
                step=STEP, seam='after_densify_before_opacity_reset_and_optimizer_step',
                terminal=10000, stock_reset_steps=STOCK_RESETS, omit_reset_steps=RESET_STEPS,
                arms=ARMS, camera_order_sha256=order_sha, cold_video_clock=False, quality_accepted=False)
    return rt, profile, order, row_bins, plan


def run_arm(baseline, output, rt, profile, plan, fork):
    from shuolee_portrait.training import _train_smpl_sh0

    root = output/fork.arm/'run'
    pin(rt.PROJECT/TRAINING, plan['diagnostic_training_sha256'])
    for name, digest in plan['mask_sha256'].items():
        pin(Path(name), digest)
    for name, digest in plan['input_sha256'].items():
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline/name, target)
        pin(target, digest)
    run = read(root/'run.json')
    run.update(created=rt.now(), copied_fitting_input_from=str(baseline), fit_reused=True)
    rt.save(root/'run.json', run)
    effective = dict(profile, opacity_reset_steps=STOCK_RESETS if ARMS[fork.arm] else RESET_STEPS)
    rt.save(root/'config.json', dict(read(root/'config.json'), **effective))
    rt.configure(run['assets'], root=root)
    rt.save(root/'status.json', dict(state='running', stage='diagnostic_training', arm=fork.arm, updated=rt.now()))
    bound = _train_smpl_sh0(root, effective, opacity_fork=fork)
    require(bound is not None and (fork.arm == 'stock' or fork.restore_verified), 'Diagnostic training or restore failed.')
    terminal = root/'gaussian-run/step-010000'
    summary = dict(arm=fork.arm, state='completed', root=str(root),
                   checkpoint_sha256=fork.sha256, restored_exact=fork.restore_verified,
                   new_optimizer_updates=9999 if fork.arm == 'stock' else 1000,
                   points=read(root/'gaussian-run/result.json')['points'],
                   training_camera_sha256=sha(root/'gaussian-run/training-cameras.json'),
                   world_ply_sha256=sha(terminal/'world.ply'), bound_ply_sha256=sha(terminal/'bound-local.ply'),
                   reference_render_sha256=sha(terminal/'reference-render.png'))
    rt.save(root/'status.json', dict(**summary, stage='terminal_diagnostic', updated=rt.now(), quality_accepted=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'project', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    baseline, project, output = (path.resolve() for path in (args.baseline, args.project, args.output))
    rt, profile, order, row_bins, plan = prepare(baseline, project, output)
    if args.preflight:
        print(json.dumps(dict(**plan, fit_calls=0, gpu_calls=0)))
        return
    import torch
    from shuolee_portrait.cli import gpu_lock

    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock():
            require(torch.cuda.device_count() == 1, 'Expose one CUDA device for the isolated diagnostic.')
            output.mkdir(parents=True, exist_ok=False)
            rt.save(output/'preregistration.json', plan)
            checkpoint = output/'shared-after-densify-step9000.pt'
            checkpoint_sha = None
            receipts = []
            for arm in ARMS:
                rt.save(output/'status.json', dict(state='running', arm=arm, updated=rt.now()))
                fork = OpacityFork(checkpoint, arm, order, row_bins)
                fork.sha256 = checkpoint_sha
                try:
                    receipts.append(run_arm(baseline, output, rt, profile, plan, fork))
                    checkpoint_sha = pin(checkpoint, fork.sha256)
                except BaseException as error:
                    rt.save(output/'status.json', dict(state='failed', arm=arm, updated=rt.now(),
                                                       error_type=type(error).__name__, error=str(error)))
                    raise
            expected_cameras = sha(baseline/'gaussian-run/training-cameras.json')
            require(all(arm['training_camera_sha256'] == expected_cameras for arm in receipts), 'Camera receipt changed.')
            result = dict(state='completed_pending_controlled_render_review', checkpoint_sha256=checkpoint_sha,
                          training_camera_sha256=expected_cameras, arms=receipts, quality_accepted=False)
            rt.save(output/'result.json', result)
            rt.save(output/'status.json', dict(**result, updated=rt.now()))
            print(json.dumps(result))


if __name__ == '__main__':
    main()
