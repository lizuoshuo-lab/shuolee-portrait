"""Compare the R14 color constraint with the exact R13 fitted endpoint."""

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_smpl_delivery_round13 as r13
from review_bound_scale_coverage import scale_pixels

r10, r9, review, round1 = r13.r10, r13.r9, r13.review, r13.round1
read, require, sha, pin = r13.read, r13.require, r13.sha, r13.pin
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-ldr-20261111')
DELIVERY = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round14'
BASE_COMMIT = 'a9dbfb03565ad89b29b1bd2b5b22fdcd2009d43a'
BASE_SHA = '55ccd67daf8607f75f069a8d17ff9783327097fd56938ab19b57475bffdfbbd8'
SUBJECTS = r13.SUBJECTS
FIXED_FILES = (
    'fit/fitted.npz', 'fit/selection.json', 'fit/selection-before-joint.json',
    'fit/joint-candidates.npz', 'fit/joint-body-refinement.json', 'fit/official-fit-result.json',
    'fit/shared-body.json', 'data/training-inputs.json', 'data/inputs.json', 'data/observations.npz',
)


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round14-{subject}'


def descriptor(path):
    return dict(file=str(path), sha256=sha(path))


def verify_base(subject):
    receipt_path = pin(DELIVERY.with_name('smpl-delivery-round13')/'receipt.json', BASE_SHA)
    receipt = read(receipt_path)
    arm = receipt['subjects'][subject]
    require(receipt['commit'] == arm['commit'] == BASE_COMMIT, 'Wrong R13 training baseline.')
    compare_path = pin(arm['review'], arm['review_sha256'])
    comparison = read(compare_path)
    plan_path = pin(compare_path.with_name('plan.json'), comparison['plan_sha256'])
    lock = read(plan_path)
    require(comparison['state'] == 'round13_delivery_views_complete'
            and comparison['commit'] == lock['commit'] == BASE_COMMIT
            and lock['subject'] == subject, 'Invalid R13 comparison archive.')
    old = review.planned_model(Path(arm['result']).parent, r13.PROJECT, arm['result_sha256'])
    require(old.provenance == lock['new'], 'R13 comparison does not bind the training endpoint.')
    proofs = dict(round13_receipt=descriptor(receipt_path), round13_review=descriptor(compare_path),
                  round13_review_plan=descriptor(plan_path),
                  fixed_native_rois=descriptor(pin(plan_path.with_name('rois.npz'), lock['roi_sha256'])))
    return old, arm, comparison, lock, proofs


def completed_training(subject, commit):
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'Pin the exact completed R14 candidate.')
    r13.r9.transfer.rebound(r13.r9.transfer.base.common.git_state, PROJECT=PROJECT)(commit)
    root = root_for(subject)
    run = root/'run'
    wrapper_path = root/'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    terminal = read(pin(run/'result.json', wrapper['result_sha256']))
    require(wrapper['state'] == terminal['state'] == 'completed' and wrapper['candidate_commit'] == commit
            and terminal['gaussian_from_scratch'] is terminal['format_passed'] is terminal['fit_reused'] is True
            and terminal['fit_calls'] == 0 and terminal['steps'] == 10000
            and terminal['final_snapshot'] == 'step-010000', 'Missing verified R14 fixed-fit terminal.')
    integration_path = run/'gaussian-run/integration.json'
    integration = read(integration_path)
    require(integration['optimizer_updates'] == 10000 and integration['initial_points'] == 20908
            and integration['optimizer_initial_state_entries'] == 0 and integration['no_prior_gaussian_loaded'] is True,
            'R14 Gaussian training did not start from the protected empty optimizer.')
    color = pin(run/'gaussian-run/color-constraint.json', integration['sh0_color_constraint']['sha256'])
    return terminal, wrapper, dict(wrapper=descriptor(wrapper_path), integration=descriptor(integration_path),
        color_constraint=descriptor(color), joint=descriptor(run/'fit/joint-body-refinement.json'),
        joint_candidates=descriptor(run/'fit/joint-candidates.npz'))


def dc_range(path):
    vertices = PlyData.read(path)['vertex'].data
    require(set(('f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity')) <= set(vertices.dtype.names)
            and not any(name.startswith('f_rest_') for name in vertices.dtype.names), 'Expected standard SH0 PLY.')
    dc = np.stack([vertices[f'f_dc_{i}'] for i in range(3)], axis=1).astype(np.float64)
    colors = .28209479177387814*dc+.5
    active = vertices['opacity'] > np.log(.1/.9)
    require(np.isfinite(colors).all() and np.isfinite(vertices['opacity']).all(), 'Nonfinite DC or opacity.')
    return dict(points=len(colors), minimum_per_channel=colors.min(axis=0).tolist(),
        maximum_per_channel=colors.max(axis=0).tolist(), points_above_one=int((colors > 1).any(axis=1).sum()),
        points_below_zero=int((colors < 0).any(axis=1).sum()), opacity_above_point_one=int(active.sum()),
        active_points_above_one=int(((colors > 1).any(axis=1) & active).sum()),
        active_points_below_zero=int(((colors < 0).any(axis=1) & active).sum()),
        points_outside_roundoff=int(((colors < -2e-7) | (colors > 1+2e-7)).any(axis=1).sum()))


def common_plans(old, new, locked, parts):
    reference = locked['common_reference']
    with np.load(pin(reference['file'], reference['sha256']), allow_pickle=False) as arrays:
        vertices, faces = arrays['vertices_world'], arrays['faces']
        require(str(arrays['reference_frame_id'].item()) == reference['frame'], 'R10 common mesh reference changed.')
    arms, centers, colors = {}, {}, {}
    for side, model in (('old', old), ('new', new)):
        snapshot, _, bound = parts[side]
        ref_frame = model.provenance['reference_frame_id']
        arm = r13.portable(model.provenance, model.rows[ref_frame], parts[side])
        own_mesh, _, own_faces = r13.mesh_parts(snapshot, bound)
        require(np.array_equal(faces, own_faces) and np.array_equal(faces, model.fit['faces']),
                'Ordered triangle topology differs from R10 common geometry.')
        local = pin(snapshot/'bound-local.ply', bound['bound_ply_sha256'])
        world = pin(snapshot/'world.ply', bound['inputs_sha256']['world_ply'])
        points = PlyData.read(world)['vertex'].data
        xyz = np.stack([points[axis] for axis in ('x','y','z')], axis=1)
        posed = r13.common_centers(vertices, faces, local)
        require(len(posed) == len(xyz) == arm['points'] and np.isfinite(posed).all(), 'Invalid common mesh centers.')
        centers[side] = (xyz, posed)
        arms[side] = dict(arm, reference_frame=ref_frame, reference_mesh_sha256=sha(own_mesh),
                          bound_ply=str(local), bound_ply_sha256=sha(local), world_ply=str(world), world_ply_sha256=sha(world))
        colors[side] = dc_range(local)
        require(colors[side] == dc_range(world) == dc_range(Path(arm['ply'])),
                'SH0 DC or opacity range differs across local, world and delivered PLY.')
    views = {}
    for name, view in locked['static_views'].items():
        raw = view['fixed_world_camera']
        E = np.asarray(raw['w2c'])
        depth = {side:[float((xyz@E[2,:3]+E[2,3]).min()) for xyz in points] for side,points in centers.items()}
        require(all(np.isfinite(z) and z > .2 for values in depth.values() for z in values),
                'A center crosses the matched camera near plane.')
        item = dict(fixed_world_camera=raw, crop_xyxy=view['crop_xyxy'], source_rgb_available=False,
                    common_mesh_sha256=reference['sha256'], near_movement_fraction=view['near_movement_fraction'],
                    common_min_center_depth_m={side:depth[side][1] for side in arms}, arms={})
        for side, arm in arms.items():
            converted = r13.exported_view(raw, arm['world_to_viewer'])
            np.testing.assert_allclose(r10.baseline_world_camera(converted,arm['world_to_viewer'])['w2c'],
                                       E,rtol=0,atol=1e-10)
            item['arms'][side] = dict(camera=converted, min_center_depth_m=depth[side][0])
        views[name] = item
    return views, arms, reference, colors


def preflight(subject, commit=None):
    old, arm, historical, lock, proofs = verify_base(subject)
    if commit is None:
        return dict(subject=subject, baseline=arm['commit'], reference=lock['common_reference'],
                    native_roles=[row['role'] for row in lock['pairs']], selected=len(old.rows))
    terminal, wrapper, proof = completed_training(subject, commit)
    proof.update(proofs)
    run, old_run = root_for(subject)/'run', Path(old.provenance['run'])
    new = review.planned_model(run, PROJECT, wrapper['result_sha256'])
    for name in FIXED_FILES:
        pin(run/name, sha(old_run/name))
    require(all(old.provenance[key] == new.provenance[key] for key in
                ('fitted_sha256', 'training_inputs_sha256', 'protected_holdouts', 'reference_frame_id'))
            and list(old.rows) == list(new.rows) and len(old.rows) == 24
            and all(old.rows[name] == new.rows[name] and review.pose_digest(old,name) == review.pose_digest(new,name)
                    for name in old.rows), 'R14 changed a native camera, mask, fitted pose or selected frame.')
    parts = dict(old=round1.delivery_paths(old_run,read(old_run/'result.json')),new=round1.delivery_paths(run,terminal))
    round1.check_source_equivalence(r13.PROJECT, PROJECT, parts['old'][2]['source_files_sha256'],
                                   parts['new'][2]['source_files_sha256'])
    for field in ('fitted', 'training_rows'):
        lhs = parts['old'][2]['inputs_sha256'].get(field)
        require(lhs is not None and lhs == parts['new'][2]['inputs_sha256'].get(field),
                f'R14 changed the saved binding input {field}.')
    views, arms, common, colors = common_plans(old, new, lock, parts)
    require(colors['new']['points_outside_roundoff'] == 0, 'Delivered trained SH0 colors lie outside LDR.')
    pairs = []
    for pair in lock['pairs']:
        frame = pair['frame']
        pose = review.pose_digest(old, frame)
        require(old.slots[frame] == new.slots[frame] == pair['new_slot'] and pose == pair['new_pose_sha256'],
                'The locked R13 native pose differs from the actual pair.')
        pairs.append(dict(pair, old_slot=old.slots[frame], new_slot=new.slots[frame],
                          old_pose_sha256=pose, new_pose_sha256=pose))
    special = {}
    for frame, row in lock['special_016'].items():
        special[frame] = dict(row, R14=row['R13'])
        require((frame in new.rows) == (row['R13'] == 'trained'), 'A special-frame identity changed.')
    if special:
        proof['common_special_posed_meshes'] = lock['proof']['common_special_posed_meshes']
    proof['common_reference_mesh'] = descriptor(Path(common['file']))
    proof['pre_joint_selection'] = descriptor(run/'fit/selection-before-joint.json')
    endpoint_path = root_for(subject)/'endpoint-color-isolation/result.json'
    proof['endpoint_isolation'] = descriptor(endpoint_path)
    selection = read(run/'fit/selection.json')
    return dict(schema='shuolee-round14-color-review-v1',subject=subject,commit=commit,old=old.provenance,
        new=new.provenance,proof=proof,reviewer_sha256=r13.R10_REVIEWER_SHA,
        round14_reviewer_sha256=sha(Path(__file__)), asset=str(review.ASSET), output=str(root_for(subject)/'view-comparison'),
        pairs=pairs,matched_frames=list(old.rows),static_views=views,static_arms=arms,common_reference=common,
        common_special=lock['common_special'],special_016=special,pose_relative_azimuth=lock['pose_relative_azimuth'],
        selections=dict(old=selection,new=selection,round13=selection,pre_joint=read(run/'fit/selection-before-joint.json')),
        native_regions=lock['native_regions'],roi_sha256=lock['roi_sha256'], brightness_contract=lock['brightness_contract'],
        historical_R9_native=historical['historical_R9_native'],
        historical_R10_native=historical['old']['native'],historical_R10_brightness={
            name:value['old'] for name,value in historical['brightness_diagnostic'].items()},
        same_native_score_mask=True, same_native_fitted_pose=True, terminal_sh0_rgb_range=colors,
        common_static_source_rgb_truth=False,quality_accepted=False)


def float_summary(pixels):
    rgb, white, alpha = pixels
    maximum = rgb.max(axis=-1)
    difference = maximum-alpha
    return dict(black_minimum=float(rgb.min()), black_maximum=float(rgb.max()),
        white_minimum=float(white.min()), white_maximum=float(white.max()),
        alpha_minimum=float(alpha.min()), alpha_maximum=float(alpha.max()),
        black_above_one_pixels=int((maximum>1).sum()), white_above_one_pixels=int((white.max(axis=-1)>1).sum()),
        max_black_minus_alpha=float(difference.max()),
        black_over_alpha_5_levels_pixels=int((difference>5/255).sum()),
        black_float_sha256=sha256(rgb.tobytes()).hexdigest(), alpha_float_sha256=sha256(alpha.tobytes()).hexdigest())


def native_float_worker(plan_path,side,lease_fd):
    """Collect the unclipped float range for each actual training camera."""
    require(os.path.samestat(os.fstat(lease_fd),review.GPU_LEASE.stat()), 'Missing inherited GPU lease.')
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = read(plan_path)
    model_info = plan[side]
    root = Path(model_info['run'])
    sys.path.insert(0,str(Path(model_info['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    asset = Path(plan['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render

    rows = read(pin(root/'data/training-inputs.json',model_info['training_inputs_sha256']))['records']
    saved = round1.delivered(root,asset,rows,model_info['protected_holdouts'])
    records = {}
    with gpu_lock(),torch.no_grad():
        require(torch.cuda.device_count() == 1, 'Expose exactly one CUDA device.')
        model,_ = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,saved.mesh,saved.facts,asset)
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        for slot, (row, archived) in enumerate(zip(rows,saved.replay['records'],strict=True)):
            cam, record = camera(row,0,pose_index=slot)
            require(archived['pose_index'] == slot and all(archived['camera'][key] == record[key]
                    for key in ('K','w2c','wh')), 'A float native render has the wrong fitted pose or camera.')
            pixels,composition = scale_pixels(render,render_pose,cam,model,pipe,black,white,slot,1.)
            with Image.open(pin(saved.snapshot/'posed-replay'/archived['image'],archived['image_sha256'])) as image:
                ref = np.asarray(image.convert('RGB'),dtype=np.int16)
            parity = int(np.abs(np.rint(np.clip(pixels[0],0,1)*255).astype(np.int16)-ref).max())
            require(parity <= 1, 'Raw float native render does not reproduce the delivered replay.')
            with Image.open(pin(row['image'],row['source_sha256'])) as image:
                source = np.asarray(image.convert('RGB'),dtype=np.float32)/255
            with Image.open(pin(row['mask'],row['mask_sha256'])) as image:
                matte = np.asarray(image.convert('L'),dtype=np.uint8)
            fg, bg = matte >= 253, matte <= 2
            require(fg.any() and bg.any(), 'Float view scoring requires reliable foreground and background.')
            rgb, _, alpha = pixels
            records[row['frame']] = dict(**float_summary(pixels), source_sha256=row['source_sha256'],
                mask_sha256=row['mask_sha256'], camera=record, archive_max_abs_8bit=parity,
                composition_max_abs=composition, reliable_foreground_pixels=int(fg.sum()),
                foreground_rgb_mae_raw=float(np.abs(rgb[fg]-source[fg]).mean()),
                foreground_rgb_mae_display=float(np.abs(np.clip(rgb[fg],0,1)-source[fg]).mean()),
                foreground_hole_fraction_at_half=float((alpha[fg]<.5).mean()),
                exterior_spill_pixels_at_half=int((alpha[bg]>=.5).sum()))
    review.save(plan_path.parent/f'{side}-float-native.json',records)


def actual_worker(plan_path, side, lease_fd):
    measured = []

    def capture(*args, **kwargs):
        pixels,error = scale_pixels(*args, **kwargs)
        measured.append(float_summary(pixels))
        return pixels,error

    r13.r9.transfer.rebound(r10.render_worker,scale_pixels=capture)(plan_path,side,lease_fd)
    plan = read(plan_path)
    offset = len(plan['pairs'])
    require(len(measured) == offset+len(plan['static_views']), 'Float recording missed a review render.')
    records = dict(zip(plan['static_views'],measured[offset:],strict=True))
    review.save(plan_path.parent/f'{side}-float-actual.json',records)


def render_common_worker(plan_path,side,lease_fd):
    """Replay only the trained local Gaussian parameters on the exact common mesh."""
    require(os.path.samestat(os.fstat(lease_fd),review.GPU_LEASE.stat()), 'Missing inherited GPU lease.')
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan,out = read(plan_path),plan_path.parent
    pin(Path(__file__),plan['round14_reviewer_sha256'])
    sys.path.insert(0,str(Path(plan[side]['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import inspect_bound_ply
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    from review_bound_scale_coverage import face_alpha, scale_pixels

    rt.configure(Path(plan['asset']).parents[1])
    from gaussian_renderer import render
    from scene.smplx_gaussian_model import SMPLXGaussianModel

    ref,arm = plan['common_reference'],plan['static_arms'][side]
    with np.load(pin(ref['file'],ref['sha256']),allow_pickle=False) as archive:
        vertices,faces = archive['vertices_world'],archive['faces']
        require(str(archive['reference_frame_id'].item()) == ref['frame'], 'Common mesh frame differs.')
    special = plan['common_special']
    posed = [vertices]
    if special:
        mesh = special[0]
        with np.load(pin(mesh['common_mesh_file'],mesh['common_mesh_file_sha256']),allow_pickle=False) as archive:
            require(np.array_equal(faces,archive['faces']),'Special R10 mesh topology differs.')
            indices = {frame:index for index,frame in enumerate(archive['frame_ids'].tolist())}
            positions = archive['vertices_world']
            for item in special:
                index = indices[item['frame']]
                require(archive['pose_indices'][index] == item['old_slot']
                        and archive['image_sha256'][index].item() == item['source_sha256']
                        and archive['mask_sha256'][index].item() == item['score_mask_sha256'],
                        'Special R10 mesh archive identity differs.')
                posed.append(positions[index])
        require(all(array.shape == vertices.shape and array.dtype == vertices.dtype and np.isfinite(array).all()
                    for array in posed), 'Invalid special R10 mesh sequence.')
    path = pin(arm['bound_ply'],arm['bound_ply_sha256'])
    ids,counts = inspect_bound_ply(path,len(faces))
    records,panels = {},{}
    with gpu_lock(),torch.no_grad():
        require(torch.cuda.device_count() == 1, 'Expose one CUDA device.')
        model = SMPLXGaussianModel(0)
        model.vertices_sequence = torch.as_tensor(np.stack(posed),device='cuda')
        model.faces = torch.as_tensor(faces,dtype=torch.long,device='cuda')
        model.num_timesteps = len(posed)
        model.load_ply(str(path))
        model.binding_counter = torch.as_tensor(counts,dtype=torch.int32,device='cuda')
        model.select_mesh_by_timestep(0)
        require(model.optimizer is None and model.active_sh_degree == 0 and len(model.get_xyz) == arm['points']
                and torch.equal(model.binding.long(),torch.as_tensor(ids,device='cuda')),
                'Loaded local parameters or ordered triangle IDs differ.')
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        positions = model.get_xyz.detach().cpu().numpy()
        for name,view in plan['static_views'].items():
            row,crop = view['fixed_world_camera'],view['crop_xyxy']
            cam,record = camera(row,0,pose_index=0)
            E = np.asarray(row['w2c'])
            depth = float((positions@E[2,:3]+E[2,3]).min())
            require(depth > .2 and abs(depth-view['common_min_center_depth_m'][side]) < 5e-6,
                    'CUDA center depth differs from the matched camera plan.')
            pixels,error = scale_pixels(render,render_pose,cam,model,pipe,black,white,0,1.)
            x0,y0,x1,y1 = crop
            alpha = pixels[2][y0:y1,x0:x1]
            occupied = pixels[2] >= .5
            panel_name = f'common_{name}'
            panel = dict(r10.strip(out/f'{panel_name}-{side}.png',crop,pixels),frame=ref['frame'],
                         columns=['black','white','alpha'])
            panels[panel_name] = panel
            records[name] = dict(camera=record,fixed_world_camera=row,common_mesh_sha256=ref['sha256'],
                panel=panel,actual_reference_frame=arm['reference_frame'],local_bound_ply_sha256=arm['bound_ply_sha256'],
                source_rgb_compared=False,composition_max_abs=error,min_center_depth_m=depth,
                alpha_ge_half_in_crop=int(occupied[y0:y1,x0:x1].sum()),
                alpha_ge_half_outside_crop=int(occupied.sum()-occupied[y0:y1,x0:x1].sum()),
                alpha_mass_in_crop=float(alpha.sum(dtype=np.float64)),float_rgb=float_summary(pixels))
        common_special = {}
        with np.load(pin(out/'rois.npz',plan['roi_sha256']),allow_pickle=False) as rois:
            for slot,item in enumerate(special,start=1):
                frame,role = item['frame'],item['role']
                model.select_mesh_by_timestep(slot)
                row = dict(item['camera'],frame=frame)
                cam,record = camera(row,0,pose_index=slot)
                pixels,error = scale_pixels(render,render_pose,cam,model,pipe,black,white,slot,1.)
                delta = None
                if side == 'old':
                    proof = plan['proof']['round13_review']
                    baseline_path = pin(proof['file'], proof['sha256'])
                    expected = read(baseline_path)['new']['panels'][role]
                    require(expected['crop_xyxy'] == item['crop_xyxy']
                            and expected['columns'] == ['source_rgb','black','white','alpha'],
                            'The historical R13 controlled crop differs.')
                    x0,y0,x1,y1 = expected['crop_xyxy']
                    width = x1-x0
                    with Image.open(pin(baseline_path.parent/expected['file'],expected['sha256'])) as saved:
                        original = np.asarray(saved.convert('RGB'),dtype=np.int16)[:,width:2*width]
                    observed = np.rint(np.clip(pixels[0][y0:y1,x0:x1],0,1)*255).astype(np.int16)
                    delta = int(np.abs(observed-original).max())
                    require(delta <= 1,'Common special replay differs from the actual R13 archive.')
                with Image.open(pin(item['source'],item['source_sha256'])) as source:
                    rgb = np.asarray(source.convert('RGB'))
                with Image.open(pin(item['score_mask'],item['score_mask_sha256'])) as mask:
                    matte = np.asarray(mask.convert('L'),dtype=np.float32)/255.
                stats = r10.mask_metrics(pixels[2],matte,np.zeros(matte.shape,dtype=bool))
                regions = {}
                for name,digest in plan['native_regions'][role].items():
                    roi = rois[f'{role}_{name}']
                    require(r9.native.hash_mask(roi) == digest,'Special native ROI changed.')
                    alpha = np.rint(np.clip(pixels[2][roi],0,1)*255)
                    regions[name] = dict(alpha=face_alpha(pixels[2],roi),alpha_u8_lt_243=int((alpha<243).sum()),
                        black_rgb_mae=float(np.abs(np.clip(pixels[0][roi],0,1)-rgb[roi].astype(np.float32)/255).mean()))
                for label,crop in ((role,item['crop_xyxy']),(f'{role}_head',item['head_crop_xyxy'])):
                    if crop:
                        panels[label] = dict(r10.strip(out/f'{label}-{side}.png',crop,(rgb,*pixels)),frame=frame,
                                             columns=['source_rgb','black','white','alpha'])
                common_special[frame] = dict(camera=record,crop_xyxy=item['crop_xyxy'],head_crop_xyxy=item['head_crop_xyxy'],
                    common_mesh_file_sha256=item['common_mesh_file_sha256'],common_mesh_pose_index=item['old_slot'],
                    source_sha256=item['source_sha256'],score_mask_sha256=item['score_mask_sha256'],
                    local_bound_ply_sha256=arm['bound_ply_sha256'],composition_max_abs=error,archive_max_abs_8bit=delta,
                    R13_fitted_pose_used=False,R14_fitted_pose_used=False,mask_metrics=stats,
                    region_metrics=regions,float_rgb=float_summary(pixels))
    review.save(out/f'{side}-common.json',dict(common=records,common_special=common_special,panels=panels,optimizer_updates=0))


def endpoint(plan):
    proof = plan['proof']['endpoint_isolation']
    data = read(pin(proof['file'],proof['sha256']))
    require(data['state'] == 'round14_r13_endpoint_color_isolation_complete'
            and data['subject'] == plan['subject'] and data['round13_commit'] == BASE_COMMIT
            and data['round13_terminal_ply_sha256'] == plan['static_arms']['old']['bound_ply_sha256']
            and data['round13_review_plan_sha256'] == plan['proof']['round13_review_plan']['sha256']
            and data['source_views'] == len(plan['matched_frames']) == 24
            and set(row['frame'] for row in data['native']) == set(plan['matched_frames'])
            and data['common_views'] == 3 and set(data['controlled']) == set(plan['special_016'])
            and data['optimizer_updates'] == 0 and data['alpha_bitwise_equal'] and data['restored_bitwise_equal'],
            'The isolated clamp audit did not preserve R13 geometry, alpha or endpoint identity.')
    return data


def fit_only_panel(out, plan):
    if not plan['special_016']:
        return []
    proof = plan['proof']['round13_review']
    historical = pin(proof['file'],proof['sha256'])
    panels = []
    for item in read(historical)['panels']:
        if item['view'] != 'available_record' or plan['special_016'][item['frame']]['R14'] != 'fit_only':
            continue
        source = pin(historical.parent/item['file'],item['sha256'])
        shutil.copyfile(source,out/item['file'])
        panels.append(dict(item,rows=['round10_historical', 'round13_round14_identical_fit']))
    return panels


def assemble(path, expected):
    plan = read(path)
    require(plan == json.loads(json.dumps(expected,allow_nan=False)), 'The saved comparison plan changed.')
    out = path.parent
    pin(out/'rois.npz',plan['roi_sha256'])
    diagnostic = endpoint(plan)
    isolation_native = {row['frame']:row for row in diagnostic['native']}
    results = []
    for side in ('old','new'):
        native = read(out/f'{side}-result.json')
        common = read(out/f'{side}-common.json')
        native['panels'].update(common.pop('panels'))
        native.update(common)
        native['float_native'] = read(out/f'{side}-float-native.json')
        native['float_actual'] = read(out/f'{side}-float-actual.json')
        require(set(native['float_native']) == set(plan['matched_frames'])
                and set(native['float_actual']) == set(plan['static_views']), 'Float audit missed a paired render.')
        results.append(native)
    old,new = results
    for frame, record in old['float_native'].items():
        other, isolation = new['float_native'][frame],isolation_native[frame]
        require(record['source_sha256'] == other['source_sha256'] == isolation['source_sha256']
                and record['mask_sha256'] == other['mask_sha256'] == isolation['score_mask_sha256']
                and record['camera'] == other['camera'] == isolation['camera']
                and record['archive_max_abs_8bit'] <= 1 and other['archive_max_abs_8bit'] <= 1,
                'The paired float native camera, image, mask or archive differs.')
        record['matches_isolated_original_float'] = (record['black_float_sha256'] == isolation['original_rgb_sha256']
                and record['alpha_float_sha256'] == isolation['alpha_sha256'])
    r13.check_render_receipts(plan,old,new)
    r13.check_source_pixels(out,plan,results)
    brightness = r13.brightness_diagnostic(out,plan,old,new)
    panels = r10.merge(out,plan,old,new)
    for panel in panels:
        panel['rows'] = [f'round13_{BASE_COMMIT[:7]}',f'round14_{plan["commit"][:7]}']
    panels += fit_only_panel(out,plan)
    path_out = out/'result.json'
    require(not path_out.exists(),'Preserve the completed color-review receipt.')
    review.save(path_out,dict(state='round14_delivery_views_complete',subject=plan['subject'],commit=plan['commit'],
        plan_sha256=sha(path),assembler_sha256=sha(Path(__file__)),panels=panels,old=old,new=new,proof=plan['proof'],
        special_016=plan['special_016'],common_reference=plan['common_reference'],
        pose_relative_azimuth=plan['pose_relative_azimuth'],terminal_sh0_rgb_range=plan['terminal_sh0_rgb_range'],
        historical_R9_native=plan['historical_R9_native'],historical_R10_native=plan['historical_R10_native'],
        historical_R10_brightness=plan['historical_R10_brightness'],brightness_diagnostic=brightness,
        all_paired_native_same_source_mask_camera_pose=True,quality_accepted=False))
    print(f'{path_out} {sha(path_out)}',flush=True)


def main():
    if len(sys.argv) == 5 and sys.argv[1] in ('_worker','_common'):
        path,side,fd = Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
        if sys.argv[1] == '_worker':
            actual_worker(path,side,fd)
            native_float_worker(path,side,fd)
        render_common_worker(path,side,fd)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render','assemble'))
    parser.add_argument('subject',choices=SUBJECTS)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action in ('check','assemble'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.action != 'check' and not args.commit:
        parser.error('Review requires a completed R14 candidate commit.')
    plan = preflight(args.subject,args.commit)
    if args.action == 'check':
        print(json.dumps({key:plan[key] for key in ('subject','baseline','reference','native_roles','selected',
            'matched_frames','terminal_sh0_rgb_range') if key in plan}),flush=True)
        return
    endpoint(plan)
    out = Path(plan['output'])
    path = out/'plan.json'
    if args.action == 'assemble':
        assemble(path,plan)
        return
    with review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        roi = plan['proof']['fixed_native_rois']
        shutil.copyfile(pin(roi['file'],roi['sha256']),out/'rois.npz')
        review.save(path,plan)
        for side in ('old','new'):
            with (out/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                    pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assemble(path,plan)


if __name__ == '__main__':
    main()
