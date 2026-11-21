"""Stage a comparison-only delivery of the pinned 001 omit-reset Gaussian."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

from review_bound_alpha_ab import pin, read, require, sha
from review_opacity_fork001 import validate_fork


FORK_SHA = '6413cb90909129d292a14750e46a7ab99494f5d75b20c329b2f1ca7fdc0dca1c'
HELDOUT_SHA = '64c0691d1497ac7a294bee437f145425f75fed692228ab29ddf931a7616ceeec'
TRAINING_REVIEW_SHA = 'fbb17c51c48af2680f73dade316391c8ed73995a875fa77b03f686e0b77b80bb'
ARM = 'omit-reset'
REFERENCE = 'frame_000056'
TERMINAL = 'gaussian-run/step-010000'
EARLY = 'gaussian-run/step-001000'
EARLY_PINS = {
    'native-ply-parity.json': 'c4639e2d290f18a48f3b77db91d7b6aac14b5ef85c52f647ead851fa5bdaec71',
    'reference-mesh.npz': '759c84713621ba8b0e9bbebf9508c97df12344a91aee22a4734591e6cf62b4ed',
    'reference-render.png': 'bdd470397956ad74390b6698cce8514eeb52694ec67f27600a5cdec6a44f8bfa',
    'world.ply': 'ab7cd81229bb3afb6ca996a09318ed695bd3d31dcfb8064c124b3b79e78d39fc',
}


def prepare(fork, project, reference_audit, output):
    require(output == fork.with_name(fork.name+'-delivery') and not output.exists(),
            'Use the new sibling delivery directory.')
    pin(fork/'result.json', FORK_SHA)
    holdout_path = fork.with_name(fork.name+'-holdouts')/'result.json'
    training_review_path = fork.with_name(fork.name+'-review')/'result.json'
    pin(holdout_path, HELDOUT_SHA)
    pin(training_review_path, TRAINING_REVIEW_SHA)
    held = read(holdout_path)
    rt, _, _, arms, evidence = validate_fork(fork, project, reference_audit)
    arm = arms[ARM]
    source = arm.root
    require(held['state'] == 'primary_heldout_replay_complete' and held['optimizer_updates'] == 0
            and held['fork_result_sha256'] == FORK_SHA and held['checkpoint_sha256'] == evidence['checkpoint_sha256']
            and held['training_review_sha256'] == TRAINING_REVIEW_SHA and held['quality_accepted'] is False
            and held['terminal_models'][ARM] == arm.terminal and arm.terminal['points'] == 65914
            and read(training_review_path)['records'][ARM]['records'][REFERENCE]['reference_replay_max_abs_8bit'] == 0,
            'Expected the exact reviewed comparison arm with native reference parity.')
    run = read(source/'run.json')
    require(run['fit_reused'] is True and arm.facts['reference_frame_id'] == REFERENCE
            and read(source/'gaussian-run/result.json')['from_scratch'] is False,
            'The comparison fit or reference provenance differs.')
    paths = {name: source/name for name in ('run.json', 'config.json', 'fit/fitted.npz',
        'fit/official-fit-result.json', 'fit/selection.json', 'fit/shared-body.json', 'data/training-inputs.json',
        'gaussian-run/config.json', 'gaussian-run/result.json', 'gaussian-run/loss.jsonl', 'gaussian-run/training-cameras.json')}
    paths.update({f'{TERMINAL}/{name}': source/TERMINAL/name for name in ('world.ply', 'bound-local.ply',
        'bound-replay.json', 'reference-mesh.npz', 'reference-render.png', 'native-ply-parity.json')})
    stock = arms['stock'].root
    require(read(stock/'gaussian-run/result.json')['from_scratch'] is True, 'Missing common-prefix stock run.')
    for name, digest in EARLY_PINS.items():
        path = stock/EARLY/name
        pin(path, digest)
        paths[f'{EARLY}/{name}'] = path
    early = read(stock/EARLY/'native-ply-parity.json')
    require(early['passed'] and early['reference_frame_id'] == REFERENCE and early['reference_pose_index'] == 0,
            'The common-prefix 1k native comparison differs.')
    lineage = dict(schema='shuolee-bound-comparison-v1', from_scratch=False, fit_reused=True, arm=ARM,
        source_run=str(source), common_prefix_run=str(stock), source_fork=str(fork),
        checkpoint_sha256=evidence['checkpoint_sha256'], fork_result_sha256=FORK_SHA,
        recovery_receipt_sha256=evidence['recovery_receipt_sha256'], heldout_review_sha256=HELDOUT_SHA,
        training_review_sha256=TRAINING_REVIEW_SHA, reference_frame=REFERENCE, points=arm.terminal['points'],
        training_camera_sha256=arm.terminal['training_camera_sha256'], original_arm_optimizer_updates=1000,
        optimizer_updates=0, cold_video_clock=False, quality_accepted=False, comparison_only=True,
        stage_driver_sha256=sha(Path(__file__)), copied_files_sha256={name: sha(path) for name, path in paths.items()})
    return rt, paths, lineage


def stage(output, rt, paths, lineage):
    from shuolee_portrait.bound_review_panels import write_panels
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.export import export
    from shuolee_portrait.training import replay_official_bound

    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock():
            root = output/'run'
            root.mkdir(parents=True, exist_ok=False)
            for name, path in paths.items():
                digest = lineage['copied_files_sha256'][name]
                pin(path, digest)
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                pin(target, digest)
            receipt = root/'comparison-provenance.json'
            rt.save(receipt, lineage)
            receipt_sha = sha(receipt)
            rt.configure(read(root/'run.json')['assets'], root=root)
            rt.save(root/'status.json', dict(state='running', stage='comparison_export', updated=rt.now(),
                                            optimizer_updates=0, quality_accepted=False))
            try:
                export(root, read(root/'config.json'))
                result = read(root/'result.json')
                require(result['state'] == 'completed' and result['format_passed'] is True
                        and result['points'] == lineage['points'] and result['reference_frame_id'] == REFERENCE,
                        'Native comparison export differs.')
                # The standard exporter assumes a full fresh run; its generic claim is false for this staged arm.
                result.update(from_scratch=False, fit_reused=True, comparison_provenance_sha256=receipt_sha,
                              optimizer_updates=0, comparison_only=True)
                rt.save(root/'result.json', result)
                replay_official_bound(root)
                replay = root/TERMINAL/'posed-replay/result.json'
                write_panels(root, replay)
                panels = root/'audit/bound-panels/result.json'
                result.update(bound_replay=replay.relative_to(root).as_posix(), bound_replay_sha256=sha(replay),
                              bound_panels=panels.relative_to(root).as_posix(), bound_panels_sha256=sha(panels))
                for name, digest in lineage['copied_files_sha256'].items():
                    pin(root/name, digest)
                    pin(paths[name], digest)
                rt.save(root/'result.json', result)
                rt.save(root/'status.json', dict(state='completed', stage='complete', static_export_passed=True,
                    sha256=result['sha256'], reference_frame=REFERENCE, format_passed=True, updated=rt.now(),
                    optimizer_updates=0, comparison_provenance_sha256=receipt_sha, quality_accepted=False))
                print(json.dumps(dict(result, run=str(root), result_sha256=sha(root/'result.json'))))
            except BaseException as error:
                rt.save(root/'status.json', dict(state='failed', stage='comparison_export', updated=rt.now(),
                    optimizer_updates=0, error_type=type(error).__name__, error=str(error), quality_accepted=False))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fork', 'project', 'reference-audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    fork, project, audit, output = (p.resolve() for p in (args.fork, args.project, args.reference_audit, args.output))
    rt, paths, lineage = prepare(fork, project, audit, output)
    if args.preflight:
        print(json.dumps(dict(lineage, state='comparison_delivery_inputs_verified', gpu_calls=0)))
        return
    stage(output, rt, paths, lineage)


if __name__ == '__main__':
    main()
