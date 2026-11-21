"""Verify the completed 016 and run the same Gaussian pipeline on the frozen 006 fit."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_head_appearance_frozen_fit as prior


COMMIT = '762d6d0b2a5db804d8d4de94c53779b2a5eb80b7'
PRIOR_SHA = '03742754dae69c35608830827a06e3c749f1637857a87f7c5c0f4dbb83ce188d'
DELIVERY_016_SHA = '8a7a4c50be432974012638d3bf969588ebda136fb445f6b33dfcaa9a1ee650a7'
SOURCE_006 = {
    'result.json': '18333c3592caa278dfdd9cb8c11ee3ec2b561e94688178587824b31208c6faf7',
    'fit/fitted.npz': '6f3fc5095618e908e391f4dd4f7d7f763d32aadba0948c222b05394874a8960c',
    'fit/official-fit-result.json': '86a83a1e04cf669784c1f5037a9878bbae5563a7af257c0bbf1e1d10887f3e76',
    'fit/shared-body.json': '4150ef5150162ae618dde2e67c4d2dfb67b2111c03c6f05563715c41bb30b567',
    'fit/selection.json': '7760e10a8d1b2c17f731d0773f80b5605ecc2fdcef26e5d6fd40453912cf8c32',
    'data/training-inputs.json': 'e753246da0365a49258d8e17ba7eb1fd28728fdbb7a01b3d5c1e803204d2d2e2',
    'gaussian-run/training-cameras.json': '20116a3ca1f873e54a1e0a5b5b2bddb6cd4e3648d8164377008f787ab05d8248',
    'export/cameras.json': 'd24a129ac7036101ee5d9499b7eb2aecb0f53ec4df8c5fa54e4cc7d96a3f8474',
}
SOURCE_006_WRAPPER_SHA = '852fbc4d048005220b5ed758f33613d295b30ad88b03983648b7d8ca58fa4930'
HOLDOUT_006_SHA = 'b557e5484eec35117a52d67f04a8fc005d87435c9268e38e9b511c694b3793d8'
base = prior.base
need, read, sha, pin = prior.need, prior.read, prior.sha, prior.pin


def root_for(subject):
    if subject == '016':
        return prior.root_for(subject)
    return prior.common.LAB/'shuolee-smpl-delivery-round1-006'


common = SimpleNamespace(**(vars(prior.common) | dict(root_for=root_for)))
stage = prior.rebound(base.stage, common=common, imports=prior.imports)
launch = prior.rebound(base.execute, common=common, COMMIT=COMMIT, __file__=__file__)
worker = prior.rebound(prior.worker, common=common, root_for=root_for)


def verify_016():
    root = root_for('016')
    review_path = root/'delivery-review.json'
    pin(review_path, DELIVERY_016_SHA)
    review = read(review_path)
    result_path = root/'run/result.json'
    wrapper_path = root/'wrapper-terminal.json'
    scene_path = root/'run/export/scene.zip'
    pins = {str(review_path): DELIVERY_016_SHA}
    for label, path in (('terminal', result_path), ('wrapper', wrapper_path), ('scene', scene_path)):
        record = review[label]
        need(record['path'] == str(path), f'016 {label} path changed.')
        pins[str(path)] = prior.checked_sha(path, record['sha256'])
    result = read(result_path)
    wrapper = read(wrapper_path)
    need(result['state'] == wrapper['state'] == 'completed' and wrapper['candidate_commit'] == COMMIT
         and wrapper['result_sha256'] == pins[str(result_path)] and result['format_passed'] is True
         and result['final_snapshot'] == wrapper['final_snapshot'] == 'step-010000-head-000600'
         and result['head_appearance']['state'] == 'completed' and result['fit_calls'] == 0
         and result['frames'] == len(wrapper['selected_frames']) == 24,
         '016 reusable completion changed.')
    for key in ('bound_replay', 'bound_panels'):
        path = result_path.parent/result[key]
        pins[str(path)] = prior.checked_sha(path, result[f'{key}_sha256'])
    return pins, result


def prepare_006():
    pins, _ = verify_016()
    pins[prior.__file__] = prior.checked_sha(Path(prior.__file__), PRIOR_SHA)
    changed = prior.check_source(COMMIT)
    _, selector, profile, verify = prior.imports()
    source = base.common.root_for('006')
    old = source/'run'
    for name, digest in SOURCE_006.items():
        pins[str(old/name)] = prior.checked_sha(old/name, digest)
    wrapper = source/'wrapper-terminal.json'
    pins[str(wrapper)] = prior.checked_sha(wrapper, SOURCE_006_WRAPPER_SHA)
    result, terminal = read(old/'result.json'), read(wrapper)
    receipt, selection = read(old/'fit/official-fit-result.json'), read(old/'fit/selection.json')
    metadata, cfg = (read(old/name) for name in ('run.json', 'config.json'))
    rows = read(old/'data/training-inputs.json')['records']
    selected, held, reference = (selection[key] for key in ('selected_frames', 'retained_holdouts', 'reference_frame'))
    spec = base.common.PREVIOUS['006']
    need(result['state'] == terminal['state'] == 'completed' and terminal['candidate_commit'] == base.COMMIT
         and terminal['result_sha256'] == SOURCE_006['result.json'] and result['format_passed'] is True
         and result['fit_reused'] is True and result['fit_calls'] == 0 and result['steps'] == 10000
         and metadata['assets'] == str(common.ASSETS) and verify(old)['fit_sha256'] == SOURCE_006['fit/fitted.npz']
         and receipt['selection'] == selection and selection['arc'] == 'half'
         and selection['coverage_basis'] == selector.SOURCE_BIN_COVERAGE_BASIS
         and metadata['video_sha256'] == spec['video_sha256'] and metadata['camera_config_sha256'] == spec['camera_sha256']
         and reference == result['reference_frame_id'] == terminal['reference_frame'] == 'frame_000033'
         and held == spec['named_holdouts'] and set(held).isdisjoint(selected)
         and len(set(selected)) == len(rows) == result['frames'] == 24
         and [reference]+[frame for frame in selected if frame != reference] == [row['frame'] for row in rows],
         '006 original shared shape, fit, camera or selected roster changed.')
    gate_path = common.LAB/'shuolee-smpl-unified-006-protected-holdouts/result.json'
    pins[str(gate_path)] = prior.checked_sha(gate_path, HOLDOUT_006_SHA)
    gate = read(gate_path)
    formal = ['frame_000022', 'frame_000037']
    need(gate['terminal_result_sha256'] == SOURCE_006['result.json'] and gate['primary_frames'] == formal
         and all(row['pose_screen_passed'] is True for row in gate['primary_records'])
         and set(formal) <= set(held), '006 independent pose screens changed.')
    replay_path = old/result['bound_replay']
    pins[str(replay_path)] = prior.checked_sha(replay_path, result['bound_replay_sha256'])
    replay = read(replay_path)
    cameras = {name: SOURCE_006[name] for name in prior.CAMERA_FILES}
    need(cameras[prior.CAMERA_FILES[0]] == replay['training_camera_sha256']
         and all(all(record['camera'][key] == row[key] for key in ('frame', 'K', 'w2c', 'wh'))
                 for row, record in zip(rows, replay['records'], strict=True)),
         '006 frozen native cameras changed.')
    pins.update({str(old/name): sha(old/name) for name in ('run.json', 'config.json')})
    for row in rows:
        for field, digest_field in (('image', 'source_sha256'), ('mask', 'mask_sha256')):
            pins[row[field]] = prior.checked_sha(Path(row[field]), row[digest_field])
    effective, stock = profile(), read(old/'gaussian-run/config.json')
    need(effective['head_appearance_steps'] == 600 and effective['head_appearance_weight'] == .05
         and all(stock[key] == value for key, value in effective.items()
                 if key not in ('head_appearance_steps', 'head_appearance_weight', 'initial_points', 'image_cache_frames'))
         and stock['optimizer_initial_state_entries'] == 0 and stock['initial_points'] == 20908,
         '006 Gaussian body training profile changed.')
    plan = dict(schema='shuolee-head-appearance-frozen-fit-transfer-v1', subject='006',
        candidate_commit=COMMIT, baseline_commit=base.COMMIT, baseline_run=str(old),
        baseline_result_sha256=SOURCE_006['result.json'], changed_paths=changed, camera_pins=cameras,
        source_video_sha256=spec['video_sha256'], camera_config_sha256=spec['camera_sha256'],
        selected_frames=selected, training_slots=len(rows), retained_holdouts=held, formal_holdouts=formal,
        reference_frame=reference, coverage=selection['coverage'], full_bin_coverage=selection['full_bin_coverage'],
        fit_reused=True, fit_calls=0, cold_video_clock=False, gaussian_from_scratch=True,
        fresh_gaussian_optimizer=True, head_appearance_steps=600, head_appearance_weight=.05,
        input_sha256=pins, asset_sha256=common.official_pins(), runtime=common.cpu_runtime(common.clean_env()),
        code_sha256={name: sha(prior.PROJECT/name) for name in common.SOURCE_FILES},
        script_sha256=sha(Path(__file__)), reused_transfer_launcher_sha256=PRIOR_SHA,
        launcher_helper_sha256=prior.queue.HELPER, reused_driver_sha256=prior.queue.DRIVER)
    return plan, None, (old, receipt, selection, metadata, cfg), effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'stage', 'launch', 'worker'))
    parser.add_argument('subject', choices=('016', '006'))
    args = parser.parse_args()
    if args.subject == '016':
        need(args.action == 'check', '016 is a verified existing delivery.')
        _, result = verify_016()
        print(json.dumps(dict(state='reuse_verified', subject='016', frames=result['frames'],
                              points=result['points'], final_snapshot=result['final_snapshot'])))
    elif args.action == 'worker':
        worker('006')
    elif args.action == 'launch':
        prior.check_source(COMMIT)
        raise SystemExit(launch('006'))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        state = prepare_006()
        if args.action == 'stage':
            stage('006', state)
        else:
            print(json.dumps(dict(state='fit_reuse_verified', subject='006', frames=state[0]['training_slots'],
                                  formal_holdouts=state[0]['formal_holdouts'], gpu_calls=0)))


if __name__ == '__main__':
    main()
