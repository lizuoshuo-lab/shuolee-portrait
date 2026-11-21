"""Compare the R11 protected-face delivery on fixed native pixels and a common bound mesh."""

import argparse
import fcntl
from itertools import permutations
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_smpl_delivery_round10 as r10
from review_smpl_delivery_round3 import portable, exported_view

review, round1, r9 = r10.review, r10.round1, r10.r9
read, require, sha, pin = r10.read, r10.require, r10.sha, r10.pin
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-guard-20261111')
DELIVERY = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round10'
BASE_COMMIT = 'ed7755c6a293484c2c83fc642fde43981d88b611'
BASE_SHA = '359da8f7873f9c6f6a2b0ea3a0f022178da027cfc5693d55b7af55bff0b5c301'
R10_REVIEWER_SHA = '7f67ac766158880d6377cefc7a1022fcdd34d76be87754f1bf9bfdffa70bbb10'
SUBJECTS = r10.SUBJECTS


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round11-{subject}'


def verify_base(subject):
    pin(Path(r10.__file__),R10_REVIEWER_SHA)
    receipt = read(pin(DELIVERY/'receipt.json',BASE_SHA))
    arm = receipt['subjects'][subject]
    require(receipt['commit'] == arm['commit'] == BASE_COMMIT, 'Wrong R10 baseline release.')
    comparison = pin(arm['review'],arm['review_sha256'])
    result = read(comparison)
    plan = read(pin(comparison.with_name('plan.json'),result['plan_sha256']))
    require(result['state'] == 'round10_delivery_views_complete' and plan['commit'] == BASE_COMMIT
            and plan['reviewer_sha256'] == R10_REVIEWER_SHA, 'Wrong R10 render baseline.')
    model = review.planned_model(Path(arm['result']).parent,r10.PROJECT,arm['result_sha256'])
    require(model.provenance == plan['new'], 'The R10 review belongs to another endpoint.')
    return model, plan, result, comparison


def completed_training(subject, commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Pin the exact completed R11 candidate.')
    r9.transfer.rebound(r9.transfer.base.common.git_state, PROJECT=PROJECT)(commit)
    root, run = root_for(subject), root_for(subject)/'run'
    wrapper_path = root/'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    terminal = read(pin(run/'result.json',wrapper['result_sha256']))
    require(wrapper['state'] == terminal['state'] == 'completed' and wrapper['candidate_commit'] == commit
            and terminal['format_passed'] is True and terminal['steps'] == 10000
            and terminal['final_snapshot'] == 'step-010000', 'Missing verified R11 terminal.')
    integration = run/'gaussian-run/integration.json'
    fit = read(run/'fit/official-fit-result.json')
    joint = pin(run/'fit/joint-body-refinement.json',wrapper['joint_refinement_sha256'])
    metrics = read(joint)
    candidates = pin(run/'fit/joint-candidates.npz',metrics['candidate_sha256'])
    require(fit['fitted_candidate_frames'] == 52 and read(integration)['optimizer_updates'] == 10000
            and metrics['shared_betas_frozen'] is metrics['native_K_frozen'] is True
            and metrics['per_frame_scale_parameters'] == 0,
            'The protected fit or Gaussian budget differs.')
    return terminal,wrapper,{key:dict(file=str(path),sha256=sha(path)) for key,path in
        (('wrapper',wrapper_path),('integration',integration),('joint',joint),('joint_candidates',candidates))}


def paired_native(old,new,locked,subject):
    common = sorted(set(old.rows)&set(new.rows))
    require(all(r10.same_view(old.rows[frame],new.rows[frame])
                and old.rows[frame]['mask_sha256'] == new.rows[frame]['mask_sha256'] for frame in common),
            'A common native frame changed its source, calibration or score mask.')
    preferred = {pair['role']:pair['frame'] for pair in locked['pairs'] if pair['role'] in ('front','side')}
    roles = {}
    if len(common) >= 2:
        def rank(pair):
            return (-sum(frame == preferred[role] for role,frame in zip(preferred,pair,strict=True)),
                    sum(abs(old.rows[frame]['source_frame_index']-
                            old.rows[preferred[role]]['source_frame_index'])
                        for role,frame in zip(preferred,pair,strict=True)),pair)
        roles = dict(zip(preferred,min(permutations(common,2),key=rank),strict=True))
    elif common:
        frame, = common
        role = min(preferred,key=lambda label:abs(old.rows[frame]['source_frame_index']-
                                                 old.rows[preferred[label]]['source_frame_index']))
        roles[role] = frame
    if subject == '016':
        roles.update((frame.removeprefix('frame_'),frame) for frame in r10.SPECIAL_016
                     if frame in common and frame not in roles.values())
    prior = {pair['frame']:pair for pair in locked['pairs']}
    pairs = []
    for role,frame in roles.items():
        row = old.rows[frame]
        other = new.rows[frame]
        raw = pin(row['image'],row['source_sha256'])
        mask = pin(row['mask'],row['mask_sha256'])
        pin(other['image'],other['source_sha256'])
        pin(other['mask'],other['mask_sha256'])
        earlier = prior.get(frame)
        if earlier is None:
            a,b,c,d = row['initial_box_xyxy']
            crop = r10.native_box((a-64,b-64,c+64,d+64),row['wh'])
            head = None
            if row.get('face_box_xyxy'):
                x0,y0,x1,y1 = row['face_box_xyxy']
                head = r10.native_box((x0-24,y0-24,x1+24,y1+24),row['wh'])
        else:
            crop,head = earlier['crop_xyxy'],earlier['head_crop_xyxy']
        pairs.append(dict(role=role,frame=frame,source_frame_index=row['source_frame_index'],
            source=str(raw),source_sha256=row['source_sha256'],
            score_mask=str(mask),score_mask_sha256=row['mask_sha256'],
            fresh_score_mask=str(mask),fresh_score_mask_sha256=row['mask_sha256'],
            camera={key:row[key] for key in r10.CAMERA_KEYS},crop_xyxy=crop,head_crop_xyxy=head,
            old_slot=old.slots[frame],new_slot=new.slots[frame],
            old_pose_sha256=review.pose_digest(old,frame),new_pose_sha256=review.pose_digest(new,frame),
            preferred_frame=preferred.get(role,frame),pinned_r10_role=earlier['role'] if earlier else None))
    return pairs,common


def mesh_parts(snapshot,bound):
    path = pin(snapshot/'reference-mesh.npz',bound['inputs_sha256']['reference_mesh'])
    with np.load(path,allow_pickle=False) as arrays:
        xyz,faces = arrays['vertices_world'],arrays['faces']
        frame = str(arrays['reference_frame_id'].item())
    require(frame == bound['reference_frame_id'] and xyz.shape == (10475,3) and xyz.dtype == np.float32
            and faces.shape == (20908,3) and np.isfinite(xyz).all(), 'Invalid terminal reference mesh.')
    return path,xyz,faces


def common_centers(vertices,faces,path):
    """CPU planning only. The renderer applies its original mesh binding on CUDA."""
    triangle = vertices[faces].astype(np.float64)
    e,f = triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]

    def unit(vector):
        return vector/np.sqrt(np.maximum((vector*vector).sum(axis=1,keepdims=True),1e-20))

    x = unit(e)
    y = unit(np.cross(x,f))
    z = unit(np.cross(x,y))
    basis = np.stack((x,y,z),axis=-1)
    scale = (np.sqrt(np.maximum((e*e).sum(axis=1),1e-20))+np.abs((z*f).sum(axis=1)))*.5
    local = PlyData.read(path)['vertex'].data
    ids = local['binding_0'].astype(np.int64)
    require(np.array_equal(ids,local['binding_0']) and ids.min() >= 0 and ids.max() < len(faces),
            'Invalid local Gaussian triangle identity.')
    offsets = np.stack([local[key] for key in ('x','y','z')],axis=1)
    return np.einsum('nij,nj->ni',basis[ids],offsets)*np.maximum(scale[ids,None],1e-8)+triangle[ids].mean(axis=1)


def actual_and_common(old,new,locked,parts):
    old_snapshot,_,old_manifest = parts['old']
    mesh,vertices,faces = mesh_parts(old_snapshot,old_manifest)
    common = dict(file=str(mesh),sha256=sha(mesh),frame=old_manifest['reference_frame_id'],
                  pose_index=old_manifest['reference_pose_index'],source='R10 terminal fitted and clothed reference mesh',
                  vertices=len(vertices),ordered_faces=len(faces))
    arms, centers = {}, {}
    for side,model in (('old',old),('new',new)):
        snapshot,_,manifest = parts[side]
        reference = model.provenance['reference_frame_id']
        arm = portable(model.provenance,model.rows[reference],parts[side])
        own_mesh,_,own_faces = mesh_parts(snapshot,manifest)
        require(np.array_equal(faces,own_faces) and np.array_equal(faces,model.fit['faces'])
                and np.array_equal(old.fit['shared_betas'],model.fit['shared_betas']),
                'Common SMPL-X shape or ordered triangle topology differs between arms.')
        bound = pin(snapshot/'bound-local.ply',manifest['bound_ply_sha256'])
        world = pin(snapshot/'world.ply',manifest['inputs_sha256']['world_ply'])
        points = PlyData.read(world)['vertex'].data
        xyz = np.stack([points[axis] for axis in ('x','y','z')],axis=1)
        posed = common_centers(vertices,faces,bound)
        if side == 'old':
            np.testing.assert_allclose(xyz,posed,rtol=0,atol=3e-6)
        require(len(posed) == len(xyz) == arm['points'] and np.isfinite(posed).all(),
                'Common-reference point count or centers are invalid.')
        centers[side] = (xyz,posed)
        arms[side] = dict(arm,reference_frame=reference,reference_mesh_sha256=sha(own_mesh),
            bound_ply=str(bound),bound_ply_sha256=sha(bound),world_ply=str(world),world_ply_sha256=sha(world))
    views = {}
    front = np.asarray(locked['static_views']['static_reference_front']['fixed_world_camera']['w2c'])
    for name,view in locked['static_views'].items():
        camera = view['fixed_world_camera']
        for fraction in ((1.,.75,.5,.25,0.) if name == 'dolly' else (1.,)):
            E = np.asarray(camera['w2c']).copy()
            if fraction != 1.:
                E[:3,3] = front[:3,3]+fraction*(E[:3,3]-front[:3,3])
            raw = dict(camera,frame=common['frame'],w2c=E.tolist())
            depth = {side:[float((xyz@E[2,:3]+E[2,3]).min()) for xyz in values]
                     for side,values in centers.items()}
            if all(np.isfinite(z) and z > .201 for values in depth.values() for z in values):
                break
        else:
            raise ValueError('A shared camera crosses the center depth limit.')
        crop = view['crop_xyxy']
        if name == 'dolly':
            frame = common['frame']
            native_E = np.asarray(old.rows[frame]['w2c'])
            fitted = Path(old.provenance['run'])/'fit/fitted.npz'
            with np.load(pin(fitted,old.provenance['fitted_sha256']),allow_pickle=False) as archive:
                landmarks = archive['joints_camera'][old.slots[frame],67:118]
            world = (landmarks-native_E[:3,3])@native_E[:3,:3]
            camera_xyz = world@E[:3,:3].T+E[:3,3]
            require(np.isfinite(camera_xyz).all() and np.min(camera_xyz[:,2])>.2,
                    'The common head does not have positive close-up depth.')
            uv = camera_xyz@np.asarray(raw['K']).T
            uv = uv[:,:2]/uv[:,2:]
            lo,hi = uv.min(axis=0)-12.,uv.max(axis=0)+12.
            crop = r10.native_box([*np.minimum(crop[:2],lo),*np.maximum(crop[2:],hi)],raw['wh'])
        item = dict(fixed_world_camera=raw,crop_xyxy=crop,source_rgb_available=False,
                    common_mesh_sha256=common['sha256'],near_movement_fraction=fraction if name == 'dolly' else None,
                    common_min_center_depth_m={side:depth[side][1] for side in arms},arms={})
        for side,arm in arms.items():
            converted = exported_view(raw,arm['world_to_viewer'])
            np.testing.assert_allclose(r10.baseline_world_camera(converted,arm['world_to_viewer'])['w2c'],
                                       E,rtol=0,atol=1e-10)
            item['arms'][side] = dict(camera=converted,min_center_depth_m=depth[side][0])
        views[name] = item
    return views,arms,common


def special_identity(run,new,old,subject,proof):
    if subject != '016':
        return {}
    raw = {row['frame']:row for row in read(run/'data/inputs.json')['records']}
    desc = proof['joint_candidates']
    with np.load(pin(desc['file'],desc['sha256']),allow_pickle=False) as candidates:
        fit_ids = set(candidates['frame_ids'].tolist())
    require(len(fit_ids) == 52 and fit_ids.isdisjoint(new.provenance['protected_holdouts']),
            'R11 did not preserve the special candidate or holdout identities.')
    result = {}
    for frame in r10.SPECIAL_016:
        row,base = raw[frame],old.rows[frame]
        require(r10.same_view(row,base) and row['mask_sha256'] == base['mask_sha256'],
                'An exact f051/f055 source or score mask changed.')
        result[frame] = dict(source_frame_index=row['source_frame_index'], source_sha256=row['source_sha256'],
            R10='trained', R11='trained' if frame in new.rows else 'fit_only' if frame in fit_ids else 'source_only',
            old_mask_sha256=base['mask_sha256'],new_mask_sha256=row['mask_sha256'],
            source=row['image'],wh=row['wh'],K_integer_centers=row['K_integer_centers'])
    return result


def common_special_plan(old,locked,parts,identities,proof):
    if not identities:
        return []
    snapshot,replay_path,_ = parts
    replay = read(replay_path)
    mesh_digest = replay['mesh_sha256']
    mesh = pin(snapshot/'posed-replay'/replay['mesh_file'],mesh_digest)
    proof['common_special_posed_meshes'] = dict(file=str(mesh),sha256=mesh_digest)
    prior = {pair['frame']:pair for pair in locked['pairs']}
    renders = {item['frame']:item for item in replay['records']}
    result = []
    for frame,identity in identities.items():
        row = old.rows[frame]
        item,rendered = prior[frame],renders[frame]
        require(rendered['pose_index'] == old.slots[frame]
                and all(rendered['camera'][key] == row[key] for key in ('K','w2c','wh'))
                and item['source_sha256'] == row['source_sha256'],
                'The R10 archived special pose or camera changed.')
        label = f'controlled_{frame.removeprefix("frame_")}'
        result.append(dict(role=label,frame=frame,old_slot=old.slots[frame],pinned_r10_role=item['role'],
            source=row['image'],source_sha256=row['source_sha256'],score_mask=row['mask'],
            score_mask_sha256=row['mask_sha256'],camera={key:row[key] for key in r10.CAMERA_KEYS},
            crop_xyxy=item['crop_xyxy'],head_crop_xyxy=item['head_crop_xyxy'],
            common_mesh_file=str(mesh),common_mesh_file_sha256=mesh_digest,
            old_archive=str(snapshot/'posed-replay'/rendered['image']),old_archive_sha256=rendered['image_sha256'],
            R11=identity['R11'],R11_fitted_pose_used=False))
    return result


def preflight(subject,commit=None):
    old,locked,baseline,previous = verify_base(subject)
    old_run = Path(old.provenance['run'])
    old_parts = round1.delivery_paths(old_run,read(old_run/'result.json'))
    if commit is None:
        views,_,common = actual_and_common(old,old,locked,dict(old=old_parts,new=old_parts))
        return dict(subject=subject,baseline_reference=common['frame'],common_reference_mesh=common,
                    reference_cameras=list(views),baseline_selected=len(old.rows))
    terminal,wrapper,proof = completed_training(subject,commit)
    root,run = root_for(subject),root_for(subject)/'run'
    new = review.planned_model(run,PROJECT,wrapper['result_sha256'])
    new_parts = round1.delivery_paths(run,terminal)
    round1.check_source_equivalence(r10.PROJECT,PROJECT,old_parts[2]['source_files_sha256'],
                                   new_parts[2]['source_files_sha256'])
    require(old.provenance['protected_holdouts'] == new.provenance['protected_holdouts'],
            'The protected holdout identities changed.')
    selections = {key:read(path/'fit/selection.json') for key,path in (('old',old_run),('new',run))}
    prior = run/'fit/selection-before-joint.json'
    selections['pre_joint'] = read(prior)
    proof['pre_joint_selection'] = dict(file=str(prior),sha256=sha(prior))
    for side,model in (('old',old),('new',new)):
        selected = selections[side]
        require(set(selected['selected_frames']) == set(model.rows)
                and len(model.rows) == len(selected['selected_frames'])
                and selected['retained_holdouts'] == model.provenance['protected_holdouts'],
                'A selected frame or holdout differs from the bound manifest.')
    require(selections['pre_joint']['retained_holdouts'] == selections['new']['retained_holdouts'],
            'Protected pre-fit identities differ.')
    pairs,common = paired_native(old,new,locked,subject)
    views,arms,reference = actual_and_common(old,new,locked,dict(old=old_parts,new=new_parts))
    proof['common_reference_mesh'] = dict(file=reference['file'],sha256=reference['sha256'])
    special = special_identity(run,new,old,subject,proof)
    common_special = common_special_plan(old,locked,old_parts,special,proof)
    historical = [dict(frame=row['frame'],source_sha256=row['source_sha256'],
                       score_mask_sha256=row['score_mask_sha256'],regions=row['region_metrics'])
                  for row in baseline['old']['native']]
    return dict(schema='shuolee-round11-fixed-reference-review-v1',subject=subject,commit=commit,
        old=old.provenance,new=new.provenance,baseline_review=str(previous),baseline_review_sha256=sha(previous),
        reviewer_sha256=R10_REVIEWER_SHA,round11_reviewer_sha256=sha(Path(__file__)),
        asset=str(review.ASSET),output=str(root/'view-comparison'),pairs=pairs,matched_frames=common,
        static_views=views,static_arms=arms,common_reference=reference,selections=selections,
        special_016=special,common_special=common_special,proof=proof,historical_R9_native=historical,
        same_native_score_mask=True,common_static_source_rgb_truth=False,quality_accepted=False)


def setup_regions(plan,out):
    native,arrays = {},{}
    root = Path(plan['old']['run'])
    rows = {row['frame']:row for row in read(root/'data/training-inputs.json')['records']}
    locked_result = read(pin(plan['baseline_review'],plan['baseline_review_sha256']))
    locked_path = Path(plan['baseline_review']).with_name('plan.json')
    locked = read(pin(locked_path,locked_result['plan_sha256']))
    locked_rois = pin(locked_path.with_name('rois.npz'),locked['roi_sha256'])
    with np.load(pin(root/'fit/fitted.npz',plan['old']['fitted_sha256']),allow_pickle=False) as npz:
        fit = {key:npz[key] for key in ('joints_camera','joints_projected_native','observed_native_keypoints')}
    with np.load(locked_rois,allow_pickle=False) as baseline:
        for item in plan['pairs']+plan.get('common_special',[]):
            frame,role,prior = item['frame'],item['role'],item['pinned_r10_role']
            with Image.open(pin(item['score_mask'],item['score_mask_sha256'])) as mask:
                matte = np.asarray(mask.convert('L')) >= 253
            if prior:
                masks = {name:baseline[f'{prior}_{name}'] for name in locked['native_regions'][prior]}
                require(all(r9.native.hash_mask(value) == locked['native_regions'][prior][name]
                            and matte[value].all() for name,value in masks.items()),
                        'An archived native ROI is not reliable on the common R10 matte.')
                keypoints = baseline[f'{prior}_score_keypoints']
            else:
                crop = item['head_crop_xyxy'] or item['crop_xyxy']
                row = rows[frame]
                masks,_ = r9.native.source_regions(fit,item['old_slot'],row,row,matte,crop)
                keypoints = fit['observed_native_keypoints'][item['old_slot']]
            native[role] = {name:r9.native.hash_mask(value) for name,value in masks.items()}
            arrays.update({f'{role}_{name}':value for name,value in masks.items()})
            arrays[f'{role}_score_keypoints'] = keypoints
    path = out/'rois.npz'
    np.savez_compressed(path,**arrays)
    plan.update(native_regions=native,roi_sha256=sha(path))



def render_common_worker(plan_path,side,lease_fd):
    """Replay only the trained local Gaussian parameters on the exact common mesh."""
    require(os.path.samestat(os.fstat(lease_fd),review.GPU_LEASE.stat()), 'Missing inherited GPU lease.')
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan,out = read(plan_path),plan_path.parent
    pin(Path(__file__),plan['round11_reviewer_sha256'])
    sys.path.insert(0,str(Path(plan[side]['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _compare_geometry, _geometry, inspect_bound_ply
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    from review_bound_scale_coverage import face_alpha, scale_pixels

    rt.configure(Path(plan['asset']).parents[1])
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
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
    records,panels,parity = {},{},None
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
        if side == 'old':
            actual = GaussianModel(0)
            actual.load_ply(str(pin(arm['world_ply'],arm['world_ply_sha256'])))
            require(torch.equal(model.get_features,actual.get_features)
                    and torch.equal(model.get_opacity,actual.get_opacity),
                    'Common R10 replay changed the actual terminal colors or opacity.')
            parity = _compare_geometry(_geometry(model),_geometry(actual))
            del actual
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
                alpha_mass_in_crop=float(alpha.sum(dtype=np.float64)))
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
                    with Image.open(pin(item['old_archive'],item['old_archive_sha256'])) as saved:
                        original = np.asarray(saved.convert('RGB'),dtype=np.int16)
                    delta = int(np.abs(np.rint(np.clip(pixels[0],0,1)*255).astype(np.int16)-original).max())
                    require(delta <= 1,'Common special R10 mesh failed its native archive parity control.')
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
                    R11_fitted_pose_used=False,mask_metrics=stats,region_metrics=regions)
    review.save(out/f'{side}-common.json',dict(common=records,common_special=common_special,panels=panels,
                                              r10_actual_world_parity=parity,optimizer_updates=0))


def complete_special(out,plan):
    if not plan['special_016']:
        return []
    paired = {item['frame'] for item in plan['pairs']}
    missing = set(plan['special_016']) - paired
    if not missing:
        return []
    old = Path(plan['old']['run'])
    terminal = read(pin(old/'result.json',plan['old']['terminal_sha256']))
    path = pin(old/terminal['bound_panels'],terminal['bound_panels_sha256'])
    old_panels = {item['frame']:item for item in read(path)['panels']}
    desc = plan['proof']['joint_candidates']
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait.bound_review_panels import _features, _mesh_panel

    result = []
    with np.load(pin(desc['file'],desc['sha256']),allow_pickle=False) as data:
        slots = {name:i for i,name in enumerate(data['frame_ids'].tolist())}
        for frame in sorted(missing):
            identity = plan['special_016'][frame]
            item = old_panels[frame]
            require(item['source_frame_index'] == identity['source_frame_index']
                    and item['source_sha256'] == identity['source_sha256'],
                    'Historical special-frame panel identity differs.')
            crop = item['crop_xyxy']
            x0,y0,x1,y1 = crop
            w,h = x1-x0,y1-y0
            with Image.open(pin(path.parent/item['panel'],item['panel_sha256'])) as old_panel:
                require(old_panel.size == (4*w,h+26), 'Unexpected baseline geometry strip dimensions.')
                old_tiles = [old_panel.crop((i*w,26,(i+1)*w,26+h)) for i in (0,3,2,1)]
            with Image.open(pin(identity['source'],identity['source_sha256'])) as source:
                raw = source.convert('RGB').crop(crop)
            require(np.array_equal(np.asarray(raw),np.asarray(old_tiles[0])), 'Special frame RGB pixels changed.')
            blank = Image.new('RGB',raw.size,'#202020')
            ImageDraw.Draw(blank).text((12,12),'No trained R11 replay for this source frame',fill='white')
            keypoints,mesh = blank,blank
            if identity['R11'] == 'fit_only':
                i = slots[frame]
                require(int(data['source_frames'][i]) == identity['source_frame_index']
                        and data['image_sha256'][i].item() == identity['source_sha256']
                        and data['mask_sha256'][i].item() == identity['new_mask_sha256']
                        and np.array_equal(data['K_integer_centers'][i],identity['K_integer_centers']),
                        'The unselected fit does not bind the original special frame.')
                observed,projected = data['observed_native_keypoints'][i],data['joints_projected_native'][i]
                keypoints = _features(raw,crop,observed,projected)
                mesh = _mesh_panel(raw,crop,data['vertices_camera'][i],data['faces'],
                                   np.asarray(identity['K_integer_centers']))
            sheet = Image.new('RGB',(4*w,2*h))
            for y,tiles in enumerate((old_tiles,(raw,keypoints,mesh,blank))):
                for x,tile in enumerate(tiles):
                    sheet.paste(tile,(x*w,y*h))
            target = out/f'available-record-{frame}.png'
            sheet.save(target)
            result.append(dict(view='available_record',file=target.name,sha256=sha(target),frame=frame,new_frame=frame,
                crop_xyxy=crop,columns=['source_rgb','keypoints','mesh','same_frame_gaussian'],
                rows=['round10_ed7755c',f'round11_{plan["commit"][:7]}'],new_gaussian_available=False))
    return result


def check_render_receipts(plan,old,new):
    require(len(old['native']) == len(new['native']) == len(plan['pairs']), 'Native comparison missed a matched frame.')
    for pair,left,right in zip(plan['pairs'],old['native'],new['native'],strict=True):
        require(left['frame'] == right['frame'] == pair['frame']
                and left['source_sha256'] == right['source_sha256'] == pair['source_sha256']
                and left['score_mask_sha256'] == right['score_mask_sha256'] == pair['score_mask_sha256']
                and all(left['native_camera'][key] == right['native_camera'][key] == pair['camera'][key]
                        for key in ('K','w2c','wh')), 'Native source, mask or camera differed while rendering.')
    for name,view in plan['static_views'].items():
        a,b = old['common'][name],new['common'][name]
        require(a['camera'] == b['camera']
                and a['fixed_world_camera'] == b['fixed_world_camera'] == view['fixed_world_camera']
                and a['common_mesh_sha256'] == b['common_mesh_sha256'] == plan['common_reference']['sha256']
                and a['panel']['crop_xyxy'] == b['panel']['crop_xyxy'] == view['crop_xyxy'],
                'Common mesh, camera or crop differed while rendering.')
    for item in plan['common_special']:
        a,b = (arm['common_special'][item['frame']] for arm in (old,new))
        fields = ('crop_xyxy','head_crop_xyxy','common_mesh_file_sha256','source_sha256','score_mask_sha256')
        require(a['camera'] == b['camera'] and a['common_mesh_pose_index'] == b['common_mesh_pose_index'] == item['old_slot']
                and all(a[key] == b[key] == item[key] for key in fields)
                and a['archive_max_abs_8bit'] <= 1 and b['R11_fitted_pose_used'] is False,
                'Special mesh, camera, source, mask or crop differed while rendering.')


def assemble(path,expected):
    import json

    plan = read(path)
    expected = json.loads(json.dumps(expected,allow_nan=False))
    normalized = {key:value for key,value in plan.items()
                  if key not in ('native_regions','roi_sha256','round11_reviewer_sha256')}
    expected.pop('round11_reviewer_sha256')
    require(normalized == expected, 'Saved render plan differs from the current exact preflight.')
    out = path.parent
    pin(out/'rois.npz',plan['roi_sha256'])
    results = []
    for side in ('old','new'):
        native = read(out/f'{side}-result.json')
        static = read(out/f'{side}-common.json')
        native['panels'].update(static.pop('panels'))
        native.update(static)
        results.append(native)
    old,new = results
    check_render_receipts(plan,old,new)
    panels = r10.merge(out,plan,old,new)
    for panel in panels:
        panel['rows'] = ['round10_ed7755c',f'round11_{plan["commit"][:7]}']
    panels += complete_special(out,plan)
    result = out/'result.json'
    require(not result.exists(),'Preserve the completed review receipt.')
    review.save(result,dict(state='round11_delivery_views_complete',subject=plan['subject'],commit=plan['commit'],
        plan_sha256=sha(path),assembler_sha256=sha(Path(__file__)),panels=panels,old=old,new=new,proof=plan['proof'],
        special_016=plan['special_016'],common_reference=plan['common_reference'],
        historical_R9_native=plan['historical_R9_native'],quality_accepted=False))
    print(f'{result} {sha(result)}',flush=True)


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '_worker':
        path,side,fd = Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
        r10.render_worker(path,side,fd)
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
        parser.error('Review requires an exact completed R11 candidate commit.')
    plan = preflight(args.subject,args.commit)
    if args.action == 'check':
        import json
        print(json.dumps({key:plan[key] for key in ('subject','baseline_reference','baseline_selected',
            'common_reference_mesh','common_reference','matched_frames') if key in plan}),flush=True)
        return
    out = Path(plan['output'])
    path = out/'plan.json'
    if args.action == 'assemble':
        assemble(path,plan)
        return
    expected = plan.copy()
    with review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        setup_regions(plan,out)
        review.save(path,plan)
        for side in ('old','new'):
            with (out/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                               pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=360)
    assemble(path,expected)


if __name__ == '__main__':
    main()

