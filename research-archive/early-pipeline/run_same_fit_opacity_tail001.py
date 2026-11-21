"""Fresh Gaussian replay on the delivered 001 fit, omitting only the final reset."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

from review_bound_alpha_ab import pin, read, require, sha


INPUTS = ('fit/fitted.npz', 'fit/official-fit-result.json', 'fit/selection.json',
          'fit/shared-body.json', 'data/training-inputs.json', 'run.json', 'config.json')
BASELINE_RESULT_SHA = 'f02291d953b2dc3df627c613c2e80a7a968330d9c41cce674015776068815e67'
RESET_STEPS = [1800, 3600, 5400, 7200]


def comparable_progress(path):
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    return [{key: value for key, value in row.items() if key not in ('updated', 'seconds')}
            for row in rows if row['step'] <= 9000]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'project', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    baseline, project, output = (p.resolve() for p in (args.baseline, args.project, args.output))
    require(not output.exists() and not output.is_relative_to(baseline.parent)
            and not output.is_relative_to(project), 'Use a new output separate from the baseline and checkout.')
    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.training import smpl_sh0_effective_config, verify_official_fit

    pin(baseline/'result.json', BASELINE_RESULT_SHA)
    result = read(baseline/'result.json')
    require(result['state'] == 'completed' and result['frames'] == 24 and result['steps'] == 10000
            and result['points'] == 58451 and result['reference_frame_id'] == 'frame_000056',
            'The exact delivered 001 baseline is required.')
    fit = verify_official_fit(baseline)
    replay_path = baseline/result['bound_replay']
    pin(replay_path, result['bound_replay_sha256'])
    bound_path = replay_path.parent.parent/'bound-replay.json'
    pin(bound_path, read(replay_path)['bound_replay_sha256'])
    bound = read(bound_path)
    training_source = 'src/shuolee_portrait/training.py'
    for name, digest in bound['source_files_sha256'].items():
        if name != training_source:
            pin(project/name, digest)
    profile = smpl_sh0_effective_config()
    old = read(baseline/'gaussian-run/config.json')
    require(profile['opacity_reset_steps'] == RESET_STEPS and 'opacity_reset_steps' not in old
            and old['optimizer_initial_state_entries'] == 0 and old['initial_points'] == 20908
            and old['image_cache_frames'] == 24
            and all(value == old[key] for key, value in profile.items()
                    if key not in ('initial_points', 'image_cache_frames', 'opacity_reset_steps')),
            'The effective profile differs outside the late opacity reset.')
    pins = {name: sha(baseline/name) for name in INPUTS}
    asset = Path(read(baseline/'run.json')['assets'])/'smplx/SMPLX_NEUTRAL.npz'
    pin(asset, bound['asset_sha256'])
    plan = dict(baseline=str(baseline), baseline_result_sha256=BASELINE_RESULT_SHA,
                baseline_bound_sha256=sha(bound_path), fitted_sha256=fit['fit_sha256'], pins=pins,
                source_sha256=sha(project/training_source), driver_sha256=sha(Path(__file__)),
                baseline_reset_steps=[*RESET_STEPS, 9000], candidate_reset_steps=RESET_STEPS,
                seed=profile['seed'], terminal=profile['iterations'], same_fit=True, cold_video_clock=False)
    if args.preflight:
        print(json.dumps(dict(state='exact_same_fit_inputs_verified', **plan)))
        return
    from shuolee_portrait.bound_review_panels import write_panels
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.export import export
    from shuolee_portrait.training import replay_official_bound, train_official

    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock():
            output.mkdir(parents=True, exist_ok=False)
            root = output/'run'
            for name, digest in pins.items():
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(baseline/name, target)
                pin(target, digest)
            rt.save(output/'preregistration.json', plan)
            run = read(root/'run.json')
            run.update(created=rt.now(), copied_fitting_input_from=str(baseline), fit_reused=True)
            rt.save(root/'run.json', run)
            cfg = dict(read(root/'config.json'), **profile)
            rt.save(root/'config.json', cfg)
            rt.configure(run['assets'], root=root)
            status = dict(state='running', stage='gaussian_inputs', updated=rt.now(), training_started=False,
                          full_preprocessing_complete=True, fitting_complete=True, quality_accepted=False)
            rt.save(root/'status.json', status)
            try:
                trained = train_official(root, cfg)
                status = read(root/'status.json')
                status.update(stage='export', training_complete=True, updated=rt.now())
                rt.save(root/'status.json', status)
                export(root, cfg)
                status.update(stage='bound_review', static_export_passed=True, static_export_completed_at=rt.now())
                rt.save(root/'status.json', status)
                replay_official_bound(root, replay=trained)
                replay_path = root/'gaussian-run/step-010000/posed-replay/result.json'
                write_panels(root, replay_path)
                panel_path = root/'audit/bound-panels/result.json'
                candidate = read(root/'result.json')
                require(candidate['state'] == 'completed' and candidate['format_passed'] is True
                        and candidate['reference_frame_id'] == result['reference_frame_id'],
                        'Candidate export failed named-reference validation.')
                candidate.update(bound_replay=replay_path.relative_to(root).as_posix(), bound_replay_sha256=sha(replay_path),
                                 bound_panels=panel_path.relative_to(root).as_posix(), bound_panels_sha256=sha(panel_path))
                rt.save(root/'result.json', candidate)
                status.update(state='completed', stage='complete', updated=rt.now(), format_passed=True,
                              ply=str(root/candidate['ply']), sha256=candidate['sha256'])
                rt.save(root/'status.json', status)
            except Exception as error:
                status = read(root/'status.json')
                status.update(state='failed', updated=rt.now(), error=str(error))
                rt.save(root/'status.json', status)
                raise
    first = 'gaussian-run/step-001000'
    early = {name: sha(root/first/name) == sha(baseline/first/name)
             for name in ('world.ply', 'reference-mesh.npz', 'reference-render.png')}
    progress_equal = comparable_progress(root/'gaussian-run/loss.jsonl') == comparable_progress(
        baseline/'gaussian-run/loss.jsonl')
    report = dict(state='completed_candidate_review_pending', result_sha256=sha(root/'result.json'),
                  step_1000_exact=early, through_logged_step_9000_exact=progress_equal,
                  training_camera_exact=sha(root/'gaussian-run/training-cameras.json') ==
                                        sha(baseline/'gaussian-run/training-cameras.json'),
                  same_fit=True, from_scratch=True, quality_accepted=False)
    rt.save(output/'comparison.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
