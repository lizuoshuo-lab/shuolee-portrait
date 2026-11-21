"""Transfer the frozen 001 late-opacity fork to the original fitted 005 input."""

import argparse
import fcntl
from inspect import getsource
import json
import os
from pathlib import Path

import run_opacity_fork001 as fork
import resume_opacity_fork001_suffix as recovery
from review_bound_alpha_ab import pin, read, require, sha


LAB = Path('/opt/gs-slam-lab')
BASE = LAB / 'shuolee-smpl-coverage-005/run'
FORMAL = LAB / 'shuolee-smpl-coverage-005-formal'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-opacity-fork')
OUT = LAB / 'shuolee-smpl-opacity-fork-005'
REGISTRATION = OUT.with_name(OUT.name + '-registration.json')
SNAPSHOT = BASE / 'gaussian-run/step-010000'
HELD = ['frame_000011', 'frame_000027', 'frame_000042', 'frame_000055']
BASE_RESULT = 'f55b70846760238b595cf1cb00803254991d98bffd6caaa730c9007d38e71946'
FIXED = {
    Path(fork.__file__).resolve(): 'd33c78316a4ebc5befccfbf38c5a5a1deb4056ff5bfc2adefe089eaa200fee28',
    Path(recovery.__file__).resolve(): '57cf4900d7899b1e704b95be2959cf13f1519c9302c7ba2cd9ca675f9fb31ada',
    PROJECT / fork.TRAINING: '7bb1c3040575658a33e84be0be63b646cc8635148658e1938576e70495f82f9a',
    PROJECT / fork.BACKWARD: '068eb4b0c20af25d22f274443e3fa6e24368418f691f099183e84109faccd6f7',
    BASE / 'result.json': BASE_RESULT,
    BASE / 'fit/selection.json': '56d9af357c8442c23a6b0f8cfba2d1909cc0eacd855782e7cd33445d21dd3251',
    BASE / 'gaussian-run/training-cameras.json': 'bae82c15ebc310615a36deda46307b0e0dd9f4bd55cc71de8d43fe34ff820d87',
    SNAPSHOT / 'bound-replay.json': '02711fbfba9e483370057d77a880ed50060e5b0abc595f56303a2383497077f3',
    SNAPSHOT / 'bound-local.ply': 'ad38c5f020a5c7406b3e090d8b531af688bdd00982edac943536043bc7958fab',
    BASE / 'export/scene.zip': '133de48a6656f7c121deb9153e4ac4450133d1883121e46cbb7715e39ec65ab7',
    FORMAL / 'evaluation-fitted.npz': '56ee461a693d176708bad5593eac4ee96f854328e8acc2d303ce5eac0d733a3b',
    FORMAL / 'renders/result.json': 'b4fad17398d85a6809712caa11e66f8059c5436ae3968109d9e7a3d9d3300dc6',
    LAB / 'shuolee-scale-coverage-005/result.json': '8e1c060e495a67b2c8f0b42cf5cbfeb1c3fa1713e1e92f62c47485c588ff3ece',
}


def preflight():
    pins = {str(path): pin(path, digest) for path, digest in FIXED.items()}
    # Only the archive-specific identity gate differs from the known 001 setup.
    source = getsource(fork.prepare)
    for old, new in (
        ("result['frames'] == 24", "result['frames'] == 23"),
        ("result['points'] == 58451", "result['points'] == 87792"),
        ("result['reference_frame_id'] == 'frame_000056'", "result['reference_frame_id'] == 'frame_000020'"),
        ('The exact delivered 001 baseline is required.', 'The exact delivered 005 baseline is required.'),
    ):
        require(source.count(old) == 1, 'Archived helper identity gate changed.')
        source = source.replace(old, new)
    scope = dict(vars(fork), BASELINE_RESULT_SHA=BASE_RESULT)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), scope)
    rt, profile, order, bins, plan = scope['prepare'](BASE, PROJECT, OUT)
    selection = read(BASE / 'fit/selection.json')
    archived = read(FORMAL / 'renders/result.json')
    rows = read(BASE / 'data/training-inputs.json')['records']
    ordered = [row['frame'] for row in rows]
    require(selection['retained_holdouts'] == list(archived['protected_frames']) == HELD
            and not set(ordered) & set(HELD) and set(ordered) == set(selection['selected_frames'])
            and ordered[0] == 'frame_000020'
            and plan['stock_reset_steps'] == [1800, 3600, 5400, 7200, 9000]
            and plan['omit_reset_steps'] == [1800, 3600, 5400, 7200]
            and plan['terminal'] == len(order) == 10000 and profile['seed'] == 0
            and archived['gaussian_optimizer_updates'] == 0,
            'Original source roster, frozen heldouts or reset schedule changed.')
    source_rgb = {row['image']: pin(Path(row['image']), row['source_sha256']) for row in rows}
    plan.update(wrapper_sha256=sha(Path(__file__).resolve()), exact_source_rgb_sha256=source_rgb,
                external_fixed_pins=pins, selected_training_frames=ordered, excluded_heldouts=HELD,
                archived_formal_pose_pass_frames=[r['frame'] for r in archived['records']
                    if r['pose_screen_passed'] and r['frame'] in HELD[:3]],
                original_stock_camera_sha256=FIXED[BASE / 'gaussian-run/training-cameras.json'],
                training_fits_reused=True, common_gaussian_optimizer_initial_state_entries=0,
                total_adam_update_budget=11999, gpu_calls_at_registration=0, evaluation_refits=0)
    return rt, profile, order, bins, plan


class VerifiedFork(recovery.RecoveredFork):
    def restore(self, *args):
        if self.arm != 'stock':
            import torch

            pin(self.checkpoint, self.sha256)
            require(set(torch.serialization.get_unsafe_globals_in_checkpoint(self.checkpoint)) == recovery.UNSAFE_GLOBALS,
                    'Unexpected shared checkpoint decoder contract.')
        return super().restore(*args)


def run(rt, profile, order, bins, plan):
    import torch
    from shuolee_portrait.cli import gpu_lock

    require(read(REGISTRATION) == plan, 'Current 005 inputs differ from preregistration.')
    with Path('/tmp/shuolee-native122-gpu.lock').open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock():
            require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            OUT.mkdir()
            rt.save(OUT / 'preregistration.json', plan)
            checkpoint = OUT / 'shared-after-densify-step9000.pt'
            digest, arms = None, []
            try:
                for name in fork.ARMS:
                    rt.save(OUT / 'status.json', dict(state='running', arm=name, updated=rt.now()))
                    branch = VerifiedFork(checkpoint, name, order, bins)
                    branch.sha256 = digest
                    receipt = fork.run_arm(BASE, OUT, rt, profile, plan, branch)
                    digest = pin(checkpoint, branch.sha256)
                    require(receipt['training_camera_sha256'] == plan['original_stock_camera_sha256'],
                            'An arm used different native training cameras.')
                    arms.append(receipt)
                for path, expected in (plan['external_fixed_pins'] | plan['exact_source_rgb_sha256']
                                       | plan['mask_sha256']).items():
                    pin(Path(path), expected)
                require(sum(arm['new_optimizer_updates'] for arm in arms) == plan['total_adam_update_budget']
                        and all(arm['checkpoint_sha256'] == digest for arm in arms),
                        'Unexpected shared state or optimization budget.')
            except BaseException as error:
                rt.save(OUT / 'status.json', dict(state='failed', arm=name, updated=rt.now(),
                                                 error_type=type(error).__name__, error=str(error)))
                raise
            result = dict(state='completed_pending_controlled_render_review', checkpoint_sha256=digest,
                training_camera_sha256=plan['original_stock_camera_sha256'], arms=arms,
                preregistration_sha256=sha(OUT / 'preregistration.json'), quality_accepted=False)
            rt.save(OUT / 'result.json', result)
            rt.save(OUT / 'status.json', dict(result, updated=rt.now()))
            print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'check', 'run'))
    mode = parser.parse_args().mode
    if mode != 'run':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    inputs = preflight()
    plan = inputs[-1]
    if mode == 'run':
        run(*inputs)
    elif mode == 'prepare':
        with REGISTRATION.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(registration=str(REGISTRATION), sha256=sha(REGISTRATION), gpu_calls=0)))
    else:
        require(read(REGISTRATION) == plan, 'The pending run differs from registration.')
        print(json.dumps(dict(state='isolated_pending_run_verified', selected=plan['selected_training_frames'],
                              heldout=HELD, gpu_calls=0)))


if __name__ == '__main__':
    main()
