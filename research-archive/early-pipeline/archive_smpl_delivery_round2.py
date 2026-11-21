"""Package a finished fixed-mesh Gaussian delivery and archive its matched views."""

import argparse
import json
from pathlib import Path
from shutil import copyfileobj
import sys
from tempfile import TemporaryDirectory

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_smpl_delivery_round1 import bundle, common, read, verify_scene


def fresh_proof(root, run, snapshot, result, wrapper, plan):
    import numpy as np

    sha, require = common.sha, common.require
    fit = run / 'fit/fitted.npz'
    frozen = run / plan['frozen_surface']['file']
    frozen_sha = plan['frozen_surface']['sha256']
    require(plan['frozen_surface']['file'] == 'fit/frozen-surface.npz'
            and sha(frozen) == sha(snapshot / 'reference-mesh.npz') == frozen_sha
            and sha(fit) == plan['staged_sha256']['fit/fitted.npz'],
            'Fresh training changed the fixed mesh or fitted pose.')
    plan_sha = sha(root / 'preregistration.json')
    initialization = run / 'gaussian-run/initialization.json'
    integration = run / 'gaussian-run/integration.json'
    launch = root / 'launch.json'
    init, main, command = read(initialization), read(integration), read(launch)
    require(wrapper['preregistration_sha256'] == command['preregistration_sha256'] == plan_sha
            and wrapper['candidate_commit'] == command['candidate_commit'] == plan['candidate_commit']
            and wrapper['subject'] == command['subject'] == plan['subject']
            and command['command'][-5:] == ['-m', 'shuolee_portrait.cli', 'train-fitted', '--run', str(run)]
            and command['fit_calls'] == 0 and command['gaussian_from_scratch'] is True,
            'Fresh training is not bound to the public CLI launch.')
    training = read(run / 'gaussian-run/result.json')
    integration_sha = sha(integration)
    require(wrapper['integration_sha256'] == integration_sha
            and wrapper['unified_main_training'] == result['unified_main_training'] == training['unified_main_training'] == main
            and snapshot.name == wrapper['final_snapshot'] == training['final_snapshot'] == 'step-010000'
            and wrapper['points'] == result['points'] == training['points']
            and init['policy'] == 'one_per_triangle' and init['initial_points'] == 20908
            and init['optimizer_updates'] == init['optimizer_initial_state_entries'] == 0
            and init['no_prior_gaussian_loaded'] is True and init['local_parameter_sha256']
            and all(main[key] == value for key, value in init.items() if key != 'optimizer_updates')
            and main['state'] == 'completed' and main['optimizer_updates'] == result['steps'] == training['steps'] == 10000
            and main['raw_native_rgb_target'] is True and main['prior_render_used_as_target'] is False
            and main['frozen_surface_sha256'] == frozen_sha
            and main['schedule'] == plan['effective_profile']['unified_main']
            and [row['step'] for row in main['coverage_events']] == main['schedule']['coverage_steps'],
            'Fresh initialization, unified schedule or terminal receipt differs.')
    for name, digest in plan['camera_pins'].items():
        require(sha(run / name) == digest, f'Fresh training changed the paired camera: {name}')
    replay_path = run / result['bound_replay']
    require(replay_path == snapshot / 'posed-replay/result.json'
            and sha(replay_path) == result['bound_replay_sha256'], 'Posed replay receipt differs.')
    replay = read(replay_path)
    posed = replay_path.parent / replay['mesh_file']
    baseline_pose = Path(plan['baseline_posed_mesh'])
    require(sha(posed) == replay['mesh_sha256']
            and sha(baseline_pose) == plan['input_sha256'][str(baseline_pose)], 'A posed mesh receipt differs.')
    with np.load(baseline_pose, allow_pickle=False) as old, np.load(posed, allow_pickle=False) as new:
        require(old.files == new.files and all(old[key].dtype == new[key].dtype and old[key].shape == new[key].shape
                and old[key].tobytes() == new[key].tobytes() for key in old.files),
                'Fresh training changed a posed mesh array.')
    baseline = Path(plan['baseline_run']) / 'result.json'
    require(sha(baseline) == plan['baseline_result_sha256'] == wrapper['baseline_result_sha256'],
            'Reference terminal differs.')
    baseline_points = read(baseline)['points']
    require(baseline_points == plan['baseline_points'], 'Reference point count differs.')
    proofs = {'initialization': (initialization, sha(initialization)), 'integration': (integration, integration_sha),
              'launch': (launch, sha(launch))}
    return baseline_points, main, proofs


def copy_proof(delivery, proofs):
    archived = {}
    for key, (source, digest) in proofs.items():
        target = delivery / source.name
        common.require(common.sha(source) == digest, f'Proof changed: {source}')
        if not target.exists():
            with source.open('rb') as original, target.open('xb') as copy:
                copyfileobj(original, copy)
        common.require(common.sha(target) == digest, f'Archived proof differs: {target}')
        archived[key] = dict(path=str(target), sha256=digest)
    return archived


def collect(subject, output, round_number=2):
    root = common.LAB / f'shuolee-smpl-delivery-round{round_number}-{subject}'
    run = root / 'run'
    result_path = run / 'result.json'
    result = read(result_path)
    wrapper = read(root / 'wrapper-terminal.json')
    sha, require = common.sha, common.require
    commit = wrapper['candidate_commit']
    fresh = round_number in (5, 6, 7, 8, 9)
    terminal = (wrapper['frozen_mesh_verified'] is True and wrapper['fit_reused'] is result['fit_reused'] is True
                and wrapper['fit_calls'] == result['fit_calls'] == 0
                and wrapper['gaussian_from_scratch'] is result['gaussian_from_scratch'] is True) if fresh else (
                result[f'warm_started_from_round{round_number-1}'] and result['frozen_mesh_verified'])
    require(wrapper['state'] == result['state'] == 'completed' and wrapper['result_sha256'] == sha(result_path)
            and result['format_passed'] and terminal,
            'No verified fixed-mesh terminal.')
    mesh = verify_scene(run, result)
    snapshot = run / 'gaussian-run' / result['final_snapshot']
    plan = read(root / 'preregistration.json')
    require(sha(mesh) == plan['baseline_portable_mesh_sha256'], 'The export changed the fixed reference OBJ.')
    if fresh:
        baseline_points, refinement, proofs = fresh_proof(root, run, snapshot, result, wrapper, plan)
        if round_number == 6:
            from launch_smpl_delivery_round6 import world_scale_proof

            proofs['world_scale'] = world_scale_proof(run, refinement, plan)
        elif round_number == 7:
            from launch_smpl_delivery_round7 import needle_proof

            proofs['world_scale'] = needle_proof(run, refinement, plan)
        elif round_number == 8:
            from launch_smpl_delivery_round8 import support_proof

            proofs['world_scale'] = support_proof(run, refinement, plan)
        elif round_number == 9:
            from launch_smpl_delivery_round9 import support_proof

            proofs['world_scale'] = support_proof(run, refinement, plan)
            endpoint = plan['replacement_endpoint_check']
            endpoint_path = Path(endpoint['file'])
            require(sha(endpoint_path) == endpoint['sha256'], 'The registered replacement endpoint check changed.')
            endpoint_receipt = read(endpoint_path)
            detail = endpoint_receipt['subjects'][subject]
            detail_path = Path(detail['file'])
            require(endpoint_receipt['state'] == 'completed' and endpoint_receipt['candidate_commit'] == commit
                    and detail['state'] == 'verified' and detail_path.parent == endpoint_path.parent
                    and detail_path.name == f'{subject}.json' and sha(detail_path) == detail['sha256'],
                    'The subject endpoint check does not match the registered candidate.')
    else:
        require(sha(snapshot / 'reference-mesh.npz') == plan['baseline_reference_mesh_sha256'],
                'The export changed the fixed reference mesh.')
        baseline_points = result['baseline_points']

    view_path = root / 'view-comparison/result.json'
    views = read(view_path)
    panel_counts = {2: 10, 3: 12, 4: 13, 5: 13, 6: 13, 7: 13, 8: 13, 9: 13}
    require(views['state'] == f'round{round_number}_delivery_views_complete' and views['subject'] == subject
            and len(views['panels']) == panel_counts[round_number], 'Required matched views are incomplete.')
    if round_number in (6, 7, 8, 9):
        path, digest = proofs['world_scale']
        require(views['world_scale_receipt'] == dict(file=str(path), sha256=digest),
                'World-scale audit differs from the reviewed terminal.')
    if round_number == 7:
        causal = plan['causal_diagnostic']
        causal_files = {key: (Path(item['path']), item['sha256']) for key, item in causal['files'].items()}
        require(all(sha(path) == digest for path, digest in causal_files.values()),
                'The registered contribution diagnosis changed.')
    elif round_number == 8:
        diagnostic = plan['render_support_diagnostic']
        diagnostic_files = {key: (Path(item['path']), item['sha256']) for key, item in diagnostic['files'].items()}
        require(all(sha(path) == digest for path, digest in diagnostic_files.values()),
                'The registered render-support diagnosis changed.')
    compare_frames = {view['frame'] for view in views['panels'] if view['view'] in ('front', 'side')}
    bound_path = run / result['bound_panels']
    require(sha(bound_path) == result['bound_panels_sha256'], 'Saved four-panel receipt changed.')
    bound_receipt = read(bound_path)
    all_bound_panels = bound_receipt['panels']
    bound_panels = [p for p in all_bound_panels if p['frame'] in compare_frames]
    require(len(compare_frames) == len(bound_panels) == 2, 'Missing same-frame source, mesh, or keypoint panel.')
    extra = {}
    if fresh:
        selected = plan['selected_frames']
        frame_ids = [row['frame'] for row in all_bound_panels]
        require(wrapper['selected_frames'] == selected and len(selected) == len(set(frame_ids)) == len(frame_ids) == 24
                and set(selected) == set(frame_ids) and bound_receipt['native_crops'] is True
                and bound_receipt['same_frame_bound_gaussians'] is True
                and bound_receipt['replay_receipt_sha256'] == result['bound_replay_sha256']
                and views['integration_receipt']['sha256'] == proofs['integration'][1],
                'Fresh delivery does not bind the complete frame roster and reviewed terminal.')
        selected_panels = {}
        for row in all_bound_panels:
            name = row['panel']
            require(Path(name).name == name == row['frame'] + '.png', 'Unexpected selected panel path.')
            panel = bound_path.parent / name
            require(sha(panel) == row['panel_sha256'], f'Selected panel differs: {panel}')
            selected_panels[row['frame']] = dict(frame=row['frame'], path=str(panel), sha256=row['panel_sha256'],
                crop_xyxy=row['crop_xyxy'], source_frame_index=row['source_frame_index'])
        extra.update(selected_frame_ids=selected, selected_frame_panels=[selected_panels[frame] for frame in selected],
                     selected_panel_receipt=dict(path=str(bound_path), sha256=result['bound_panels_sha256']))
        if subject == '016':
            bound_panels = all_bound_panels
    records = common.RECORDS / subject
    require(records.resolve(strict=True) == records, 'Use the existing subject record folder.')
    tag = f'交付{round_number:02d}_{commit[:7]}'
    images = []
    with TemporaryDirectory(prefix='shuolee-round2-record-') as scratch:
        files, pins = [], {}

        def add(source, name, expected, crop=None):
            require(sha(source) == expected, 'An image differs from its receipt.')
            pins[source] = expected
            target = records / name
            stage = Path(scratch) / name
            if crop is None:
                with Image.open(source) as image:
                    image.convert('RGB').save(stage, quality=95, subsampling=0)
            else:
                common.record_jpeg(source, stage, crop)
            if target.exists():
                require(sha(target) == sha(stage), f'Preserve different existing record: {target}')
            else:
                files.append((stage, target))
            return dict(path=str(target), sha256=sha(stage))

        for panel in views['panels']:
            filename = Path(panel['file'])
            require(filename.name == panel['file'] and filename.suffix == '.png', 'Unexpected view path.')
            archived = add(view_path.parent / filename, f'分析_{subject}_{tag}_{filename.stem}.jpg', panel['sha256'])
            images.append(dict(panel, archive=archived))
        for panel in bound_panels:
            archived = add(bound_path.parent / panel['panel'],
                           f'3DGS_{subject}_{tag}_原图-高斯-Mesh-特征点_{panel["frame"]}.jpg',
                           panel['panel_sha256'], panel['crop_xyxy'])
            images.append(dict(frame=panel['frame'], view='driven_four_panel', archive=archived))
        common.publish_images(files, pins)

    delivery = output / subject
    packages = bundle(run, snapshot, result, delivery)
    config = read(run / 'gaussian-run/config.json')
    stage_keys = {2: 'surface_refinement', 3: 'coverage_refinement', 4: 'shape_refinement'}
    if fresh:
        stages = dict(unified_main_training=refinement['schedule'])
        extra['training_proof'] = copy_proof(delivery, proofs)
        if round_number == 7:
            causal_dir = delivery / 'causal-diagnostic'
            causal_dir.mkdir(exist_ok=True)
            extra['causal_diagnostic'] = dict(view=causal['view'], point_count=causal['point_count'],
                                             files=copy_proof(causal_dir, causal_files))
        elif round_number == 8:
            diagnostic_dir = delivery / 'render-support-diagnostic'
            diagnostic_dir.mkdir(exist_ok=True)
            extra['render_support_diagnostic'] = dict(selected_frames=diagnostic['selected_frames'],
                                                      files=copy_proof(diagnostic_dir, diagnostic_files))
        elif round_number == 9:
            endpoint_dir = delivery / 'endpoint-check'
            endpoint_dir.mkdir(exist_ok=True)
            extra['replacement_endpoint_check'] = copy_proof(
                endpoint_dir, {'receipt': (endpoint_path, endpoint['sha256']), 'subject': (detail_path, detail['sha256'])})
    else:
        stages = {key:result[key]['options'] for key in stage_keys.values() if key in result}
        refinement = result[stage_keys[round_number]]
    effective_path = delivery / 'effective-config.json'
    effective_path.write_text(json.dumps(dict(base=config, **stages),
                                             ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return dict(subject=subject, commit=commit, points=result['points'], baseline_points=baseline_points,
        reference=result['reference_frame_id'], frames=result['frames'], coverage=wrapper['coverage'],
        result=str(result_path), result_sha256=sha(result_path), mesh=str(mesh), packages=packages,
        bound_terminal=str(snapshot / 'bound-replay.json'), posed_replay=str(run / result['bound_replay']),
        review=str(view_path), review_sha256=sha(view_path), images=images,
        effective_config=str(effective_path), effective_config_sha256=sha(effective_path),
        refinement=refinement, frozen_mesh_verified=True, quality_accepted=False, **extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', choices=('016', '006'))
    parser.add_argument('--delivery', required=True, type=Path)
    parser.add_argument('--round', choices=(2, 3, 4, 5, 6, 7, 8, 9), type=int, default=2, dest='round_number')
    args = parser.parse_args()
    receipt = collect(args.subject, args.delivery, args.round_number)
    path = args.delivery / args.subject / 'receipt.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps(dict(receipt=str(path), sha256=common.sha(path), points=receipt['points'],
                          images=len(receipt['images']))))
