"""Verify R22 against the fixed R21 camera grid and exact original pose sources."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round22 as launch
import review_smpl_delivery_round21 as prior


pin, read, sha, require, descriptor, rebound = prior.pin, prior.read, prior.sha, prior.require, prior.descriptor, prior.rebound
static, stage, review = prior.static_replay, prior.stage_io, prior.review
COMMIT = '1b217e3d1149be41f254f1160c6af098e3461afd'
R22_016_VERIFIED_SHA = '5f52d3aea55c71d2d8e812a2e3c0aa0d233a29ef275cdf60280e99c9d539677a'
HISTORICAL = {'r15': 'old', 'r21_10': 'ten', 'r21': 'new'}
NEW = {'r22_10': 10000, 'r22': 18000}
ARMS = (*HISTORICAL, *NEW)
LABELS = ('R15 reference', 'R21 10,000', 'R21 18,000', 'R22 10,000', 'R22 18,000')
POSE_SUFFIXES = ('supplement', 'heldout', 'near')
PREFIXES = ('face_only_', 'heldout_', 'physical_near_', 'dark_')


def checked(record):
    return pin(record['file'], record['sha256'])


def identical_npz(first, second):
    with np.load(checked(first), allow_pickle=False) as a, np.load(checked(second), allow_pickle=False) as b:
        require(a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files),
                'The actual fitted pose vertices or topology changed.')


def grid_records(plan, name, path):
    grid = plan['static_grid']
    arm = grid['arms'][name]
    result = read(path)
    rows = result['records']
    require(result['side'] == name and result['ply_sha256'] == arm['ply_sha256']
            and result['view_count'] == len(rows) == len(grid['camera_roster']) == 50
            and set(rows) == set(grid['camera_roster']), f'{name} lost a camera or changed its portable PLY.')
    for key, spec in grid['camera_roster'].items():
        row = rows[key]
        expected = arm['views'][key]
        panel = row['panel']
        require(row['fixed_world_camera'] == spec['world_camera'] and row['center_receipt'] == expected['centers']
                and all(row['camera'][field] == expected['camera'][field] for field in ('K', 'wh', 'w2c'))
                and panel['crop_xyxy'] == (spec['normal_body_crop_xyxy'] or spec['head_crop_xyxy'])
                and panel['columns'][:3] == ['black', 'white', 'alpha']
                and row['actual_portable_ply'] and not row['real_rgb_ground_truth_available']
                and row['black_white_alpha_composition_max_abs'] <= 1e-5,
                f'{name}/{key} differs from the frozen camera or render contract.')
        checked(panel)
    return rows


def history(subject):
    index = checked(dict(file=str(launch.OLD_DELIVERY/'review-receipt.json'), sha256=launch.REVIEW_SHA))
    aggregate = read(index)
    row = aggregate['subjects'][subject]
    result_path = checked(row['review'])
    results = read(result_path)
    plan_path = pin(result_path.with_name('plan.json'), results['plan_sha256'])
    old = read(plan_path)
    archive = read(checked(row['archive']))
    grid_path = checked(results['static_grid'])
    signed_grid = read(grid_path)
    grid = old['static_grid']
    fixed_path = checked(grid['proof']['camera_manifest'])
    fixed = read(fixed_path)
    require(aggregate['commit'] == results['commit'] == old['commit'] == launch.PARENT
            and results['all_paired_native_same_source_mask_camera']
            and fixed['views'] == grid['camera_roster'] and len(fixed['views']) == 50
            and len(old['matched_frames']) == 24, 'The signed R21 reference is incompatible.')
    for item in grid['proof'].values():
        checked(item)
    copied = {}
    for name, side in HISTORICAL.items():
        arm = grid['arms'][name]
        pin(arm['ply'], arm['ply_sha256'])
        pin(arm['camera_path'], arm['camera_sha256'])
        checked(arm['local'])
        checked(arm['world'])
        endpoint = archive['secondary_endpoint'] if name == 'r21_10' else archive
        if name == 'r15':
            expected = fixed['baseline']['R15']['portable']
            for key in ('ply_sha256', 'camera_sha256'):
                require(expected[key] == arm[key], 'The independent R15 model or camera changed.')
        else:
            for key, field in (('portable_ply', 'ply_sha256'), ('portable_camera', 'camera_sha256')):
                item = endpoint[key]
                pin(item['path'], item['sha256'])
                require(item['sha256'] == arm[field], 'The archived R21 portable model or camera changed.')
        item = signed_grid['arms'][name]['receipt']
        require(grid_records(old, name, checked(item)) == signed_grid['arms'][name]['records'],
                'A frozen render differs from its signed R21 metric/crop receipt.')
        for key, panel in results[side]['panels'].items():
            if key.startswith(PREFIXES):
                pin(result_path.parent/panel['file'], panel['sha256'])
        copied[name] = item
    for field in ('close_face_proxy', 'supplement_poses', 'heldout_mesh'):
        checked(old['proof'][field])
    proof = dict(r21_index=descriptor(index), r21_review=descriptor(result_path), r21_review_plan=descriptor(plan_path),
                 r21_static_grid=descriptor(grid_path), r21_archive=row['archive'])
    return old, results, grid, proof, copied


def project_static(arm, grid):
    import review_round21_camera_grid as cameras

    xyz = PlyData.read(pin(arm['ply'], arm['ply_sha256']))['vertex'].data
    transform = arm['world_to_viewer']
    centers = np.column_stack([xyz[k] for k in ('x', 'y', 'z')]) @ np.asarray(transform['rotation']) + np.asarray(transform['origin_world'])
    views = {}
    for key, spec in grid['camera_roster'].items():
        world = spec['world_camera']
        cam = dict(world, w2c=cameras.to_portable(world['w2c'], transform).tolist())
        restored = static.r10.baseline_world_camera(cam, transform)
        np.testing.assert_allclose(restored['w2c'], world['w2c'], rtol=0, atol=1e-10)
        views[key] = dict(camera=cam, centers=cameras.center_receipt(centers, world['K'], world['w2c'], world['wh']))
    return dict(arm, points=len(centers), views=views)


def prepare_static(model, grid):
    return project_static(static.static_model(model.provenance),grid)


def review_plan(subject, historical):
    old, _, old_grid, proof, archived = historical
    root = launch.root_for(subject)
    run = root/'run'
    wrapper_path = root/'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    if subject == '016' and wrapper['state'] == 'incomplete':
        original = wrapper
        proof['original_wrapper_terminal'] = descriptor(wrapper_path)
        wrapper_path = pin(root/'wrapper-terminal-verified.json',R22_016_VERIFIED_SHA)
        wrapper = read(wrapper_path)
        require(original['returncode'] == 0 and original['validation_error'] == "KeyError: 'head_fit'"
                and all(wrapper[key] == value for key,value in original.items()
                        if key not in ('state','validation_error')),
                'The read-only wrapper correction must reference the original successful training invocation.')
    result_path = pin(run/'result.json', wrapper['result_sha256'])
    terminal = read(result_path)
    require(wrapper['state'] == terminal['state'] == 'completed'
            and wrapper['candidate_commit'] == COMMIT and wrapper['baseline_commit'] == launch.PARENT
            and wrapper['same_round21_fit_and_training_selection']
            and wrapper['fit_method'] == 'reused_exact_round21_fit_and_garment'
            and terminal['steps'] == 18000 and terminal['gaussian_from_scratch'] and terminal['format_passed'],
            'R22 exact frozen-fit and from-scratch terminal proof is incomplete.')
    duration = prior.duration_evidence(run, wrapper, terminal)
    attachments = prior.attachment_evidence(run, wrapper, terminal, duration['comparison'])
    proof.update(duration['proof'], **attachments['proof'], wrapper=descriptor(wrapper_path))
    split = wrapper['surface_attachment']['surface_distribution_splits']
    split_path = pin(run/'gaussian-run'/split['file'],split['sha256'])
    proof['distribution_splits'] = descriptor(split_path)
    grid = dict(old_grid, arms={name: old_grid['arms'][name] for name in HISTORICAL}, proof=dict(old_grid['proof']))
    pose_plans, raw_equivalence = {}, {}
    for name, steps in NEW.items():
        early = steps == 10000
        branch = run/'duration-10000' if early else run
        end_item = proof['duration_secondary_terminal'] if early else descriptor(result_path)
        end = read(checked(end_item))
        verification = dict(wrapper['duration_comparison']['intermediate'], head_fit=wrapper['head_fit']) if early else wrapper
        model = stage.planned_model(branch, launch.PROJECT, end_item['sha256'], steps)
        expected = old[HISTORICAL['r21_10' if early else 'r21']]
        require(model.provenance['fitted_sha256'] == expected['fitted_sha256']
                and model.provenance['training_inputs_sha256'] == expected['training_inputs_sha256']
                and model.provenance['reference_frame_id'] == expected['reference_frame_id']
                and model.provenance['protected_holdouts'] == expected['protected_holdouts']
                and list(model.rows) == old['matched_frames'],
                'The R22 native fit, selected frames, heldouts or reference differs from R21.')
        for file in (*prior.FROZEN_FILES, 'data/head-support-views.json'):
            pin(branch/file, sha(Path(expected['run'])/file))
        arm = prepare_static(model, grid)
        grid['arms'][name] = arm
        snapshot = branch/'gaussian-run'/end['final_snapshot']
        previous = Path(expected['run'])/'gaussian-run'/end['final_snapshot']
        mesh = pin(snapshot/'reference-mesh.npz', sha(previous/'reference-mesh.npz'))
        raw = duration['comparison']['raw_snapshots'][str(steps)]
        raw_data = PlyData.read(checked(raw['files']['world.ply']))['vertex'].data
        after = PlyData.read(checked(arm['world']))['vertex'].data
        require(np.array_equal(raw_data, after) and sha(mesh) == raw['files']['reference-mesh.npz']['sha256'],
                'The raw and portable endpoints differ before world-to-viewer conversion.')
        raw_equivalence[name] = dict(raw_world=raw['files']['world.ply'], final_world=arm['world'],
                                     all_vertex_fields_identical=True, frozen_reference_mesh=descriptor(mesh))
        live_proof = stage.active_pose_proofs(branch, verification, end)
        old_proof = old['ten_pose_evidence']['proof'] if early else old['proof']
        old_pose = old['ten_pose_evidence'] if early else old
        for key in ('supplement_poses', 'heldout_mesh'):
            identical_npz(live_proof[key], old_proof[key])
        keys = ('asset', 'pairs', 'matched_frames', 'near_views', 'independent_annotations', 'near_roi_sha256')
        pose = dict({key: old[key] for key in keys}, subject=subject, new=model.provenance,
                    proof=dict(proof, **live_proof))
        pose['supplement'], pose['holdout'] = stage.pose_plans(branch, pose, end)
        for group, fields in (
            ('supplement', ('frame', 'source_sha256', 'mask_sha256', 'camera', 'valid_mask_sha256', 'box')),
            ('holdout', ('frame', 'source_sha256', 'mask_sha256', 'camera', 'crop_xyxy')),
        ):
            require(all(all(a[k] == b[k] for k in fields)
                        for a, b in zip(pose[group]['frames'], old_pose[group]['frames'], strict=True)),
                    'An original auxiliary/heldout camera, crop, mask or source changed.')
        replay = prior.dolly.endpoint(subject,model.provenance,[row['frame'] for row in pose['pairs']])
        depths = {}
        for pair in pose['pairs']:
            role,frame = pair['role'],pair['frame']
            E = np.asarray(pose['near_views'][role]['near_camera']['w2c'],dtype=np.float64)
            depth = float((replay['centers'][frame]@E[2,:3]+E[2,3]).min())
            require(depth > prior.dolly.CUDA_NEAR,'The original-pose close camera clips some Gaussian centers.')
            depths[role] = depth
        pose['cpu_minimum_close_center_depth_m'] = depths
        pose_plans[name] = pose
    grid['proof']['staged_camera_manifest'] = descriptor(pin(run/'data/head-support-views.json',
                                                            grid['proof']['camera_manifest']['sha256']))
    return dict(schema='shuolee-r22-fixed-camera-quality-review/1', subject=subject, commit=COMMIT,
                asset=old['asset'], proof=proof, archived_static=archived, static_grid=grid, poses=pose_plans,
                raw_equivalence=raw_equivalence, duration=duration['comparison'],
                surface_attachment=attachments['branches'], reviewer_sha256=sha(Path(__file__)),
                quality_accepted=False)


def worker(plan_path, name, fd):
    plan = read(plan_path)
    pin(Path(__file__), plan['reviewer_sha256'])
    out = plan_path.parent
    pose = out/'posed'/name/'plan.json'
    require(read(pose) == plan['poses'][name], 'The native R22 pose plan changed.')
    sys.path.insert(0,str(launch.PROJECT/'src'))
    import shuolee_portrait

    require(Path(shuolee_portrait.__file__).resolve().is_relative_to((launch.PROJECT/'src').resolve()),
            'The renderer process loaded a different project checkout.')
    grid_result = out/'static-grid'/name/'result.json'
    if not grid_result.is_file():
        rebound(static.render_grid, PORTABLE=ARMS)(plan_path, name, fd)
    grid_records(plan, name, grid_result)
    for suffix, render in zip(POSE_SUFFIXES, (prior.auxiliary.render, prior.pose.render_heldouts, prior.dolly.render), strict=True):
        if not pose.with_name(f'new-{suffix}.json').is_file():
            render(pose, 'new', fd)


def stack(out, files, labels=LABELS):
    with Image.open(files[0]) as first:
        width, height = first.size
    label_height = 22
    canvas = Image.new('RGB', (width, len(files)*(height+label_height)), '#181818')
    draw = ImageDraw.Draw(canvas)
    for i, (file, label) in enumerate(zip(files, labels, strict=True)):
        y = i*(height+label_height)
        with Image.open(file) as image:
            require(image.size == (width, height), f'Native panel dimensions differ for {file}.')
            canvas.paste(image.convert('RGB'), (0, y+label_height))
        draw.text((6, y+5), label, fill='white')
    if not out.exists():
        with out.open('xb') as stream:
            canvas.save(stream, format='PNG', compress_level=1)
    else:
        with Image.open(out) as old:
            require(old.size == canvas.size and old.convert('RGB').tobytes() == canvas.tobytes(),
                    f'Preserve the signed panel at {out}.')
    return descriptor(out)


def heldout_face(out, key, full, row, fit):
    """Take the same unresized face crop from each full-body heldout strip."""
    with np.load(checked(fit), allow_pickle=False) as data:
        slot = list(data['frame_ids']).index(row['frame'])
        observed = data['observed_native_keypoints'][slot, 67:118]
    reliable = observed[np.isfinite(observed).all(axis=1) & (observed[:, 2] >= .5), :2]
    if len(reliable) < 6:
        return None
    lo, hi = reliable.min(axis=0), reliable.max(axis=0)
    pad = np.maximum((hi-lo)*.6, 12.)
    x0,y0,x1,y1 = row['crop_xyxy']
    bx0,by0 = np.maximum(np.floor(lo-pad).astype(int), [x0,y0])
    bx1,by1 = np.minimum(np.ceil(hi+pad).astype(int), [x1,y1])
    require(bx1 > bx0 and by1 > by0, 'The observed face is outside its fixed heldout crop.')
    width,height = int(bx1-bx0),int(by1-by0)
    tile_width,tile_height = x1-x0,y1-y0
    canvas = Image.new('RGB',(4*width,len(ARMS)*(height+22)),'#181818')
    draw = ImageDraw.Draw(canvas)
    with Image.open(checked(full)) as strips:
        require(strips.size == (6*tile_width,len(ARMS)*(tile_height+22)), 'Unexpected heldout strip shape.')
        for i,label in enumerate(LABELS):
            y = i*(height+22)
            draw.text((3,y+4),label,fill='white')
            sy = i*(tile_height+22)+22+by0-y0
            for col in range(4):
                sx = col*tile_width+bx0-x0
                canvas.paste(strips.crop((sx,sy,sx+width,sy+height)),(col*width,y+22))
    target = out/f'face_detail_{key}.png'
    if not target.exists():
        with target.open('xb') as stream:
            canvas.save(stream,format='PNG',compress_level=1)
    else:
        with Image.open(target) as old:
            require(old.size == canvas.size and old.convert('RGB').tobytes() == canvas.tobytes(),
                    'Preserve the registered native face detail.')
    return dict(descriptor(target),scope='heldout_native_face',frame=row['frame'],
                columns=['source_rgb','black','white','alpha'],crop_xyxy=[int(i) for i in (bx0,by0,bx1,by1)],
                reliable_detector_points=len(reliable),crop_detector_is_not_physical_correspondence=True)


def summarize(grid, native, roster):
    close = [key for key,spec in roster.items() if spec['distance_class']=='close' and spec['pitch_deg']]
    require(len(close)==20,'The pitch summary requires the full twenty nonlevel close cameras.')
    metrics = {}
    for name in ARMS:
        rows = [grid[name]['records'][key] for key in close]
        face = [row['opacity']['visible_face_proxy']['low_alpha_095_fraction'] for row in rows]
        outside = [row['anatomical_exterior_ring']['alpha_mass']/row['anatomical_exterior_ring']['pixels']
                   for row in rows]
        metrics[name] = dict(
            close_nonlevel_views=len(close),
            mean_visible_anatomical_face_proxy_low_alpha=float(np.mean([item for item in face if item is not None])),
            mean_anatomical_exterior_ring_alpha=float(np.mean(outside)),
            native_auxiliary_rgb_mae={frame:row['foreground_rgb_mae']
                                     for frame,row in native[name]['supplement_native'].items()},
            native_heldout_rgb_mae={frame:row['foreground_rgb_mae']
                                   for frame,row in native[name]['heldout_native'].items()},
            native_heldout_foreground_low_alpha={frame:row['alpha']['fraction_below_095']
                                                 for frame,row in native[name]['heldout_native'].items()})
    return dict(by_arm=metrics,opacity_scope='Regression only. The anatomical proxy is not a real surface annotation.')


def assemble(path, plan, historical):
    out = path.parent
    old, old_results, _, proof, archived = historical
    history_dir = Path(proof['r21_review']['file']).parent
    grid, panels = {}, {}
    for name in ARMS:
        result = checked(archived[name]) if name in archived else out/'static-grid'/name/'result.json'
        grid[name] = dict(receipt=descriptor(result), records=grid_records(plan, name, result))
    paired = out/'paired'
    paired.mkdir(exist_ok=True)
    for key, view in plan['static_grid']['camera_roster'].items():
        items = [grid[name]['records'][key]['panel'] for name in ARMS]
        require(all((item['crop_xyxy'], item['columns']) == (items[0]['crop_xyxy'], items[0]['columns']) for item in items),
                'Camera grid comparisons have a different crop or column layout.')
        panels[key] = dict(stack(paired/f'{key}.png', [checked(item) for item in items]),
                           scope='static_export', camera=view['world_camera'], columns=items[0]['columns'],
                           crop_xyxy=items[0]['crop_xyxy'], real_rgb_ground_truth=False)
    native = {}
    for name, side in HISTORICAL.items():
        native[name] = dict(old_results[side], directory=str(history_dir))
    for name in NEW:
        directory = out/'posed'/name
        merged = dict(panels={}, directory=str(directory))
        for suffix in POSE_SUFFIXES:
            payload = read(directory/f'new-{suffix}.json')
            merged['panels'].update(payload.pop('panels'))
            merged.update(payload)
        native[name] = merged
    common = plan['poses']['r22']
    for name, arm in native.items():
        for key, group, fields in (
            ('supplement_native', 'supplement', ('camera', 'source_sha256', 'valid_mask_sha256')),
            ('heldout_native', 'holdout', ('camera', 'source_sha256', 'mask_sha256')),
        ):
            for row in common[group]['frames']:
                frame = row['frame']
                got, ref = arm[key][frame], native['r21'][key][frame]
                require(all(got[field] == ref[field] for field in fields), f'Native {name}/{frame} changed.')
        for role, registered in common['near_views'].items():
            seen = arm['physical_near'][role]
            require(seen['camera'] == native['r21']['physical_near'][role]['camera']
                    and seen['pose_slot'] == registered['slot'], 'Actual-pose dolly camera changed.')
            if name in NEW:
                expected = plan['poses'][name]['cpu_minimum_close_center_depth_m'][role]
                require(abs(seen['min_gaussian_center_depth_m']-expected)<6e-6,
                        'The independent CPU and CUDA near camera depth checks differ.')
    reference_keys = {key for key in native['r21']['panels'] if key.startswith(PREFIXES)}
    require(all({key for key in arm['panels'] if key.startswith(PREFIXES)} == reference_keys for arm in native.values()),
            'A signed auxiliary, dark-region, dolly or heldout image is missing.')
    for key in sorted(reference_keys):
        items = [native[name]['panels'][key] for name in ARMS]
        require(all((item.get('crop_xyxy'), item.get('columns')) == (items[0].get('crop_xyxy'), items[0].get('columns')) for item in items),
                f'An original-pose comparison uses different crop columns: {key}.')
        files = [pin(Path(native[name]['directory'])/item['file'], item['sha256']) for name, item in zip(ARMS, items, strict=True)]
        panels[key] = dict(stack(paired/f'{key}.png', files), scope='original_pose',
                           frame=items[0].get('frame'), columns=items[0].get('columns'),
                           crop_xyxy=items[0].get('crop_xyxy'))
    for row in common['holdout']['frames']:
        key = f"heldout_{row['frame']}"
        detail = heldout_face(paired,key,panels[key],row,common['proof']['head_alignment_holdout_poses'])
        if detail:
            panels['face_detail_'+key] = detail
    for arm in native.values():
        for field in list(arm):
            if field not in ('panels', 'directory', 'supplement_native', 'heldout_native', 'physical_near'):
                del arm[field]
    split_file = checked(plan['proof']['distribution_splits'])
    splits = []
    for event in read(split_file)['events']:
        for trial in event['trials']:
            if trial['outcome'] != 'committed':
                continue
            parent = trial['parent_id']
            for row in trial['local_fit']['representatives']:
                path_img = split_file.parent/'splat-surface-diagnostics'/f'parent-{parent:07d}'/row['path']
                key = f"split_s{event['step']:06d}_parent_{parent:07d}_{row['frame']}"
                panels[key] = dict(descriptor(pin(path_img,row['sha256'])),scope='local_split',
                                   frame=row['frame'],crop_xyxy=row['box'])
            splits.append(dict(step=event['step'],parent_id=parent,child_ids=trial['child_ids'],
                               source_bins=trial['contributing_skin_source_bins'],
                               parent_expected_distance_m=trial['parent_expected_distance_m'],
                               child_expected_distance_m=trial['geometry']['child_expected_distance_m']))
    curve_path = out/'training-curve.json'
    if not curve_path.is_file():
        review.save(curve_path,prior.training_curve.build(out,plan['proof']['main_training_loss']))
    curve = read(curve_path)
    require(curve['raw']==plan['proof']['main_training_loss'],'The training curve changed its source log.')
    for field in ('csv','image'):
        checked(curve[field])
    panels['training_curve'] = dict(curve['image'],scope='training_curve')
    result = dict(state='paired_native_and_static_views_rendered', subject=plan['subject'], commit=COMMIT,
                  plan=descriptor(path), proof=plan['proof'], static=grid, native=native, panels=panels,
                  labels=LABELS, static_camera_count=50, raw_equivalence=plan['raw_equivalence'],
                  accepted_local_splits=splits,diagnostic_metrics=summarize(grid,native,plan['static_grid']['camera_roster']),
                  training_curve=curve,frozen_fit_reused=True, quality_accepted=False)
    target = out/'result.json'
    if target.exists():
        require(read(target) == json.loads(json.dumps(result, allow_nan=False)), 'Preserve the completed R22 review.')
    else:
        review.save(target, result)
    print(json.dumps(dict(result=descriptor(target), static_views=50, original_pose_panels=len(reference_keys))), flush=True)


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '_worker':
        worker(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check-history', 'check', 'render', 'assemble'))
    parser.add_argument('subject', choices=prior.SUBJECTS)
    args = parser.parse_args()
    if args.action != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    historical = history(args.subject)
    if args.action == 'check-history':
        print(json.dumps(dict(subject=args.subject, static_cameras=50, historical_arms=list(HISTORICAL),
                              source_frames=historical[0]['matched_frames'])), flush=True)
        return
    plan = review_plan(args.subject, historical)
    if args.action == 'check':
        print(json.dumps(dict(subject=args.subject, static_cameras=50, arms=list(ARMS), quality_accepted=False)), flush=True)
        return
    out = launch.root_for(args.subject)/'view-comparison'
    path = out/'plan.json'
    if args.action == 'render':
        with review.GPU_LEASE.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not path.exists():
                out.mkdir(exist_ok=False)
                review.save(path, plan)
            require(read(path) == json.loads(json.dumps(plan, allow_nan=False)), 'R22 review plan changed.')
            for name in NEW:
                directory = out/'posed'/name
                directory.mkdir(parents=True, exist_ok=True)
                roi = directory/'near-rois.npz'
                src = checked(plan['proof']['r21_review']).with_name('near-rois.npz')
                expected = plan['poses'][name]['near_roi_sha256']
                if not roi.exists():
                    shutil.copyfile(pin(src, expected), roi)
                pin(roi, expected)
                pose_path = directory/'plan.json'
                if not pose_path.exists():
                    review.save(pose_path, plan['poses'][name])
                require(read(pose_path) == plan['poses'][name], 'The saved native pose plan differs.')
                if (out/'static-grid'/name/'result.json').is_file() and all((directory/f'new-{suffix}.json').is_file() for suffix in POSE_SUFFIXES):
                    continue
                with (out/f'{name}.log').open('a', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-I', '-B', __file__, '_worker', str(path), name, str(lease.fileno())],
                                   pass_fds=(lease.fileno(),), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    require(read(path) == json.loads(json.dumps(plan, allow_nan=False)), 'R22 review plan changed.')
    assemble(path, plan, historical)


if __name__ == '__main__':
    main()
