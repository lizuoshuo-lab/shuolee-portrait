"""Evaluate 006 eye-balanced appearance against the accepted 600-step control."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

import review_head_appearance_pilot_006 as prior
from review_face_photometric_006_holdouts import render as render_protected


face, pilot = prior.face, prior.pilot
held = prior.pilot_holdouts.held
pin, read, require, sha = prior.pin, prior.read, prior.require, prior.sha
PLAN = face.PLAN.with_name('smpl-eye-balanced-quality-006-preregistration.json')
PLAN_SHA = '3b13d2da185d95c393c79714d125995e90e4d648d11510ed460392d1954e4472'
ROOT = face.LAB/'shuolee-smpl-006-eye-balanced-quality'
STAGE_SHA = 'e505ad9c3fa5e9f12a8f97a8ca671484cbe040bedb47366a12dffaf8793c8c27'
EYES = {'eye_36_41':slice(86,92), 'eye_42_47':slice(92,98)}


def polygon(points, wh):
    raster = Image.new('1', tuple(wh))
    ImageDraw.Draw(raster).polygon([tuple(xy) for xy in points], fill=1)
    return np.asarray(raster)


def project(world, row):
    K, E = np.asarray(row['K']), np.asarray(row['w2c'])
    camera = world @ E[:3,:3].T + E[:3,3]
    projected = camera @ K.T
    require(np.isfinite(camera).all() and (camera[:,2] > .2).all(), 'Eye points cross the raster depth cutoff.')
    return projected[:,:2]/projected[:,2:]-.5


def dolly_eyes(plan, rois, registered):
    row, near = plan['source_row'], plan['near_row']
    E = np.asarray(row['w2c'])
    geometry = {}
    with np.load(pilot.OUT/'fit/fitted.npz',allow_pickle=False) as fit:
        require(fit['frame_ids'][0] == registered['reference']['frame']
                and np.array_equal(fit['w2c'][0], E), 'Frozen reference fit or camera changed.')
        for name, slots in EYES.items():
            points = fit['joints_camera'][0,slots].astype(np.float64)
            observed = fit['observed_native_keypoints'][0,slots]
            xyz = (points-E[:3,3]) @ E[:3,:3]
            native, dolly = project(xyz,row), project(xyz,near)
            errors = np.linalg.norm(native-observed[:,:2],axis=1)
            require(len(errors) == 6 and np.isfinite(observed).all() and (observed[:,2] >= .5).all()
                    and float(errors.max()) <= registered['reference']['test_eye_reprojection_agreement_max_px']
                    and np.max(abs(native-fit['joints_projected_native'][0,slots])) < .001
                    and np.array_equal(polygon(observed[:,:2],row['wh']),rois[name]),
                    'The six fitted/observed native eye correspondences differ.')
            x0,y0,x1,y1 = plan['dolly_crop']
            require(((dolly >= [x0,y0]) & (dolly < [x1,y1])).all(),
                    'Dolly eye polygon extends outside the cached raster.')
            roi = polygon(dolly,near['wh'])
            require(roi.sum() >= 32, 'Dolly eye polygon is too small.')
            rois['dolly_'+name] = roi
            geometry[name] = dict(max_native_reprojection_px=float(errors.max()),
                native_errors_px=errors.tolist(), observed_native_xy=observed[:,:2].tolist(),
                projected_dolly_xy=dolly.tolist(), dolly_pixels=int(roi.sum()),
                dolly_roi_sha256=hashlib.sha256(roi.tobytes()).hexdigest())
    return geometry


def inputs():
    pin(PLAN, PLAN_SHA)
    registered = read(PLAN)
    require(registered['schema'] == 'shuolee-head-eye-balanced-quality-v1'
            and registered['state'] == 'registered_before_006_eye_balance_trial'
            and registered['subject'] == '006' and registered['output'] == str(ROOT)
            and registered['prior_pilot_preregistration_sha256'] == pilot.ACCEPTANCE_SHA
            and registered['protected_holdouts'] == list(held.FRAMES),
            'Wrong registered eye appearance trial.')
    for item in (registered['start'], *registered['accepted_control'].values()):
        pin(Path(item['path']),item['sha256'])
    plan, rois, _, control = prior.preflight()
    require(control is not None, 'Accepted prior 600-step terminal is unavailable.')
    accepted = registered['accepted_control']
    require(plan['candidate']['terminal_sha256'] == accepted['terminal']['sha256']
            and control.run == Path(accepted['terminal']['path']).parent
            and plan['frame'] == registered['reference']['frame'], 'Wrong accepted control.')
    endpoint = read(Path(accepted['face']['path']))
    result_path = prior.OUT/'candidate.json'
    accepted_face = endpoint['reports']['candidate']
    require(accepted_face == read(result_path)
            and accepted_face['pilot']['terminal_sha256'] == accepted['terminal']['sha256'],
            'Accepted reference report changed.')
    cache_path = prior.OUT/'candidate-panels.npz'
    pin(cache_path,accepted_face['panels_sha256'])
    with np.load(cache_path,allow_pickle=False) as z:
        cache = dict(z)
    geom = dolly_eyes(plan,rois,registered)
    with Image.open(plan['source_row']['image']) as im:
        source = np.asarray(im.convert('RGB'),dtype=np.float32)/255.
    proof = read(Path(accepted['protected']['path']))
    require(proof['protected_gate_passed'] is True and proof['pilot_result_sha256'] == accepted['terminal']['sha256']
            and tuple(proof['frames']) == held.FRAMES and proof['far_distance_strictly_over_px'] == 24,
            'Accepted protected baseline changed.')
    cache_alpha = Path(accepted['protected']['path']).with_name('candidate-native-alpha.npz')
    pin(cache_alpha,proof['candidate_alpha_sha256'])
    alternate = {}
    with np.load(cache_alpha,allow_pickle=False) as z:
        require(tuple(z.files) == held.FRAMES, 'Accepted protected alpha roster changed.')
        for view, result in zip(control.views,proof['records'],strict=True):
            frame = view.spec['frame']
            visible = z[frame] >= .5
            spill, far = int((visible & view.exterior).sum()), int((visible & view.far).sum())
            require(z[frame].dtype == np.float32 and visible.shape == view.visible.shape
                    and np.isfinite(z[frame]).all() and frame == result['frame']
                    and result['camera'] == view.spec['camera']
                    and result['source_sha256'] == view.spec['source_sha256']
                    and result['scoring_mask_sha256'] == view.spec['score_mask_sha256']
                    and spill == result['exterior_pixels']['candidate'] and far == result['far_pixels']['candidate'],
                    'Accepted protected raster or source identity changed.')
            alternate[frame] = dict(visible=visible, spill=spill, far=far,
                                    rgb_mae=result['foreground_rgb_mae']['candidate'])
    protected = registered['gates']['protected_vs_stock']
    limits = dict(rgb=protected['rgb_mae_regression_max'], spill=protected['exterior_alpha_half_relative_increase_max'],
                  far_added=protected['far_added_pixels_max'])
    require(limits == proof['limits'] and protected['far_distance_strictly_over_px'] == 24, 'Wrong protected gate.')
    return SimpleNamespace(registered=registered, plan=plan, rois=rois, eye_geometry=geom, source=source,
        control=control, accepted_face=accepted_face['views'], cache=cache, alternate=alternate,
        accepted_alpha_sha256=proof['candidate_alpha_sha256'], limits=limits)


def candidate(info):
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply

    result_path = ROOT/'result.json'
    registration = ROOT/'preregistration.json'
    if not registration.is_file():
        return None
    pin(registration,STAGE_SHA)
    stage = read(registration)
    pin(Path(__file__).with_name('run_eye_balanced_quality_006.py'),stage['wrapper_code_sha256'])
    require(stage['acceptance_plan_sha256'] == PLAN_SHA
            and stage['eye_balance'] == info.registered['single_change']
            and stage['eye_training_eligibility'] == info.registered['eye_training_eligibility']
            and stage['prior_uniform_face_acceptance_sha256'] == pilot.ACCEPTANCE_SHA
            and not set(stage['training_frames']) & set(info.registered['protected_holdouts']),
            'Eye candidate stage differs from the registered quality trial.')
    if not result_path.is_file():
        return None
    result = read(result_path)
    require(result['state'] == 'diagnostic_completed' and result['acceptance_plan_sha256'] == PLAN_SHA
            and result['preregistration_sha256'] == STAGE_SHA
            and result['head_optimizer_updates'] == stage['appearance_steps'] == 600
            and result['start_terminal_sha256'] == stage['source_terminal_sha256'] == info.registered['start']['sha256']
            and result['locked_metrics_plan_sha256'] == pilot.face_trial.ACCEPTANCE_SHA
            and result['fit_updates'] == stage['fit_updates'] == 0 and result['reused_terminal'] is True
            and result['snapshot'] == pilot.SNAPSHOT and result['exact_frozen_and_geometry_ply_bytes'] is True
            and stage['schema'] == 'shuolee-head-appearance-pilot-v1'
            and stage['no_densification_pruning_or_resets'] is True
            and result['eye_balanced_quality'] == dict(wrapper_code_sha256=stage['wrapper_code_sha256'],
                prior_uniform_face_acceptance_sha256=pilot.ACCEPTANCE_SHA,eligible_eye_count=stage['eligible_eye_count'],
                eligible_eye_frames=stage['eligible_eye_frames'],loss_blend=stage['eye_balance'],
                registered_eye_rois=True,cpu_differential_control=stage['cpu_differential_control']),
            'Wrong completed eye-balanced appearance candidate.')
    for name in (*pilot.FROZEN_FILES, 'export/cameras.json'):
        digest = stage['input_sha256'][str(pilot.STOCK/name)]
        pin(pilot.STOCK/name,digest)
        pin(ROOT/name,digest)
    snapshot = ROOT/pilot.SNAPSHOT
    pin(ROOT/result['viewer_ply'],result['viewer_ply_sha256'])
    pin(ROOT/result['camera'],result['camera_sha256'])
    pin(ROOT/result['scene'],result['scene_sha256'])
    manifest = snapshot/'bound-replay.json'
    pin(manifest,result['bound_manifest_sha256'])
    bound = read(manifest)
    pin(snapshot/'bound-local.ply',bound['bound_ply_sha256'])
    require(result['bound_ply_sha256'] == bound['bound_ply_sha256'], 'Candidate bound PLY changed.')
    _, mesh, facts = _inputs(ROOT,snapshot,pilot.ASSET,bound)
    stock = pilot.STOCK/'gaussian-run/step-010000'
    pin(snapshot/'reference-mesh.npz',sha(stock/'reference-mesh.npz'))
    binding, _ = inspect_bound_ply(stock/'bound-local.ply',len(info.control.poses['faces']))
    editable = pilot.head_mask(info.control.poses['faces'],binding)
    changed = {kind:pilot.verify_attributes(before,after,editable) for kind,before,after in (
        ('bound-local',stock/'bound-local.ply',snapshot/'bound-local.ply'),
        ('native-world',stock/'world.ply',snapshot/'world.ply'),
        ('viewer-world',pilot.STOCK/'export/point_cloud.ply',ROOT/'export/point_cloud.ply'))}
    require(changed == result['changed_head_rows_by_ply'] and min(changed.values()) > 0
            and facts['reference_frame_id'] == info.plan['frame'] and facts['reference_pose_index'] == 0
            and set(info.registered['protected_holdouts']).isdisjoint(row['frame'] for row in facts['frames']),
            'Candidate geometry or registered scope differs.')
    return SimpleNamespace(rt=info.control.rt, run=ROOT, snapshot=snapshot, asset=pilot.ASSET,
        poses=info.control.poses, views=info.control.views, mesh=mesh,
        result_sha256=sha(result_path), manifest_sha256=result['bound_manifest_sha256'], changed=changed)


def compare_reference(before, after, gates):
    n0,n1 = before['native'],after['native']
    d0,d1 = before['dolly'],after['dolly']
    native = n1['appearance']['black_rgb_mae']-n0['appearance']['black_rgb_mae']
    whole_dolly = d1['alpha']['fraction_below_095']-d0['alpha']['fraction_below_095']
    eyes = {}
    for name in EYES:
        a,b = n0['eyes'][name],n1['eyes'][name]
        mae = b['black_rgb_mae']-a['black_rgb_mae']
        contrast_drop = a['black_render_luma']['contrast_p90_p10']-b['black_render_luma']['contrast_p90_p10']
        alpha = d1['eyes'][name]['fraction_below_095']-d0['eyes'][name]['fraction_below_095']
        eyes[name] = dict(native_mae_delta=mae,native_luma_contrast_drop=contrast_drop,dolly_low_alpha_delta=alpha,
            native_mae_passed=mae <= gates['native_each_eye_rgb_mae_regression_max'],
            native_contrast_passed=contrast_drop <= gates['native_each_eye_luma_contrast_drop_max'],
            dolly_eye_passed=alpha <= gates['dolly_each_eye_alpha_below_095_regression_max'])
    mean_gain = -float(np.mean([e['dolly_low_alpha_delta'] for e in eyes.values()]))
    flags = dict(native_face_passed=native <= gates['native_face_rgb_mae_regression_max'],
        dolly_face_passed=whole_dolly <= gates['dolly_whole_face_alpha_below_095_regression_max'],
        dolly_mean_eye_passed=mean_gain >= gates['dolly_mean_eye_alpha_below_095_improvement_min'],
        each_eye_passed=all(e['native_mae_passed'] and e['native_contrast_passed'] and e['dolly_eye_passed']
                            for e in eyes.values()))
    return dict(native_face_mae_delta=native,dolly_whole_face_low_alpha_delta=whole_dolly,
        dolly_mean_eye_low_alpha_improvement=mean_gain,eyes=eyes,flags=flags,passed=all(flags.values()))


def render_reference(info, candidate_state):
    import torch
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    reports = {'accepted':{'native':info.accepted_face['native']}, 'candidate':{}}
    tiles = {'accepted':dict(info.cache),'candidate':{}}
    with face.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with gpu_lock(),torch.no_grad():
            require(torch.cuda.device_count() == 1,'Expose exactly one CUDA GPU.')
            candidate_state.rt.configure(candidate_state.asset.parents[1])
            from gaussian_renderer import render as raster

            black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
            pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            for arm,state,views in (('accepted',info.control,('dolly',)),('candidate',candidate_state,('native','dolly'))):
                model,manifest = load_bound_snapshot(state.run,state.snapshot,asset_path=state.asset)
                require(manifest['reference_frame_id'] == info.plan['frame'] and manifest['reference_pose_index'] == 0
                        and model.optimizer is None,'Reference pose or optimizer state changed.')
                model.select_mesh_by_timestep(0)
                for name in views:
                    row = info.plan['source_row' if name == 'native' else 'near_row']
                    cam,receipt = camera(row,0,pose_index=0)
                    require(receipt == info.accepted_face[name]['camera'],'Native or dolly camera changed.')
                    d = model.get_xyz.new_tensor(row['w2c'][2])
                    require(float((model.get_xyz @ d[:3]+d[3]).min()) > .2,'Camera crosses raster cutoff.')
                    (rgb,over_white,alpha),error = face.scale_pixels(raster,render_pose,cam,model,pipe,black,white,0,1.)
                    stats = dict(camera=receipt,alpha=face.face_alpha(alpha,info.rois[name]),composition_max_abs=error)
                    if name == 'native':
                        stats.update(appearance=face.appearance(rgb,info.source,info.rois[name]),
                            eyes={eye:face.appearance(rgb,info.source,info.rois[eye]) for eye in EYES})
                        with Image.open(state.snapshot/'reference-render.png') as image:
                            delta = abs(face.original.uint8(rgb).astype(np.int16)-np.asarray(image.convert('RGB')))
                        require(delta.max() <= 1,'Candidate reference replay differs from delivered RGB.')
                    else:
                        stats['eyes'] = {eye:face.face_alpha(alpha,info.rois['dolly_'+eye]) for eye in EYES}
                    x0,y0,x1,y1 = info.plan[name+'_crop']
                    quantize = lambda pixels:face.original.uint8(pixels[y0:y1,x0:x1])
                    tile = np.stack([quantize(rgb),quantize(over_white),np.repeat(quantize(alpha)[...,None],3,axis=2)])
                    if arm == 'accepted':
                        require(abs(tile.astype(np.int16)-info.cache['dolly']).max() <= 1
                                and abs(stats['alpha']['fraction_below_095']-info.accepted_face['dolly']['alpha']['fraction_below_095'])
                                    <= 1/info.rois['dolly'].sum(), 'Accepted dolly replay does not match the archived cache.')
                    tiles[arm][name] = tile
                    reports[arm][name] = stats
                del model
    return reports,tiles


def comparison_panel(path, name, tiles, plan, source):
    base,after = (tiles[arm][name] for arm in ('accepted','candidate'))
    _,h,w,_ = base.shape
    page = Image.new('RGB',((4 if name == 'native' else 3)*w,2*(h+21)),'#181818')
    draw = ImageDraw.Draw(page)
    for i,(arm,values) in enumerate((('accepted',base),('candidate',after))):
        y = i*(h+21)
        for j,(label,value) in enumerate(zip(('black','white','alpha'),values,strict=True)):
            page.paste(Image.fromarray(value),(j*w,y+21))
            draw.text((j*w+4,y+4),f'{arm} {label}',fill='white')
        if name == 'native':
            x0,y0,x1,y1 = plan['native_crop']
            page.paste(Image.fromarray(face.original.uint8(source[y0:y1,x0:x1])),(3*w,y+21))
            draw.text((3*w+4,y+4),'Source',fill='white')
    page.save(path)
    return dict(file=path.name,sha256=sha(path))


def render(info,state):
    output = ROOT/'quality-review'
    require(not output.exists(),'Preserve the previous quality review.')
    reports,tiles = render_reference(info,state)
    records,alphas = render_protected(state,info.limits)
    for row in records:
        frame = row['frame']
        view, = [item for item in state.views if item.spec['frame'] == frame]
        old = info.alternate[frame]
        visible = alphas[frame] >= .5
        added = int((visible & ~old['visible'] & view.far).sum())
        removed = int((~visible & old['visible'] & view.far).sum())
        far = row['far_pixels']['candidate']-old['far']
        require(added-removed == far,'Accepted far-spill accounting changed.')
        row['delta_vs_accepted'] = dict(rgb_mae=row['foreground_rgb_mae']['candidate']-old['rgb_mae'],
            exterior=row['exterior_pixels']['candidate']-old['spill'],far_added=added,far_removed=removed,far_net=far)
    reference = compare_reference(reports['accepted'],reports['candidate'],info.registered['gates']['vs_accepted_600'])
    protected = all(row['passed'] for row in records)
    output.mkdir(exist_ok=False)
    panels = {name:comparison_panel(output/f'{name}.png',name,tiles,info.plan,info.source) for name in ('native','dolly')}
    alpha_path = output/'protected-native-alpha.npz'
    np.savez_compressed(alpha_path,**alphas)
    result = output/'result.json'
    face.save(result,dict(state='eye_balanced_numeric_review_complete',plan_sha256=PLAN_SHA,
        evaluator_sha256=sha(Path(__file__)),candidate_terminal_sha256=state.result_sha256,
        candidate_bound_manifest_sha256=state.manifest_sha256,geometry_and_nonhead_exact=True,
        changed_head_rows=state.changed,eye_geometry=info.eye_geometry,reference_metrics=reports,reference_gates=reference,
        protected_vs_stock=records,protected_gate_passed=protected,protected_alpha_sha256=sha(alpha_path),
        accepted_protected_alpha_sha256=info.accepted_alpha_sha256,numeric_gates_passed=reference['passed'] and protected,
        baseline_dolly_only=True,baseline_native_and_protected_from_exact_cache=True,
        panels=panels,dolly_rgb_truth_available=False,visual_gray_or_eye_occlusion_review='pending',
        quality_accepted=False,optimizer_updates=0,refits=0))
    print(json.dumps(dict(result=str(result),sha256=sha(result),reference=reference,protected_passed=protected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','terminal-check','render'))
    args = parser.parse_args()
    if args.action != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    info = inputs()
    state = candidate(info)
    if args.action != 'check':
        require(state is not None,'Wait for the registered appearance candidate.')
    if args.action == 'render':
        render(info,state)
    else:
        print(json.dumps(dict(state='eye_balanced_inputs_verified',plan_sha256=PLAN_SHA,
            terminal_ready=state is not None,candidate_terminal_sha256=state.result_sha256 if state else None,
            eye_geometry=info.eye_geometry,protected=list(info.alternate),
            accepted_dolly_float_replay_required=True,gpu_calls=0)))


if __name__ == '__main__':
    main()
