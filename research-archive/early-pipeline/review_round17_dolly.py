"""Plan and render fixed-focal physical camera moves from locked native poses."""

import fcntl
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

import review_smpl_delivery_round16 as prior
from review_bound_scale_coverage import face_alpha, interior, scale_pixels


r10, r13, review, round1 = prior.r10, prior.r13, prior.review, prior.round1
pin, read, require = prior.pin, prior.read, prior.require
CUDA_NEAR = .2
BASELINE_MARGIN_M = .40
TARGET_MAGNIFICATION = 2.5


def endpoint(subject, provenance, frames):
    run = Path(provenance['run'])
    result = read(pin(run/'result.json',provenance['terminal_sha256']))
    snapshot,replay_path,manifest = round1.delivery_paths(run,result)
    replay = read(replay_path)
    model = pin(snapshot/'bound-local.ply',manifest['bound_ply_sha256'])
    posed = pin(replay_path.parent/replay['mesh_file'],replay['mesh_sha256'])
    with np.load(posed,allow_pickle=False) as meshes:
        indices = {frame:i for i,frame in enumerate(meshes['frame_ids'].tolist())}
        data = meshes['vertices_world']
        faces = meshes['faces']
        require(set(frames) <= indices.keys(),f'{subject} pose replay is missing a planned dolly frame.')
        centers = {frame:r13.common_centers(data[indices[frame]],faces,model) for frame in frames}
    return dict(centers=centers, posed_mesh_sha256=replay['mesh_sha256'], bound_ply_sha256=manifest['bound_ply_sha256'])


def plan_native_dollies(subject, lock, old, mid):
    pairs = lock['pairs']
    frames = [row['frame'] for row in pairs]
    baselines = {side:endpoint(subject,arm.provenance,frames) for side,arm in (('R15',old),('R16',mid))}
    with np.load(pin(Path(old.provenance['run'])/'fit/fitted.npz',old.provenance['fitted_sha256']),allow_pickle=False) as fit:
        joints = fit['joints_camera']
        observed = fit['observed_native_keypoints']
    views, masks = {}, {}
    for item in pairs:
        frame,role = item['frame'],item['role']
        row = old.rows[frame]
        K,E = (np.asarray(row[key],dtype=np.float64) for key in ('K','w2c'))
        fit_slot = old.slots[frame]
        require(fit_slot == mid.slots[frame] == item['new_slot'], 'Dolly pose identity differs across the locked arms.')
        xyz = joints[fit_slot,67:118]
        ref = observed[fit_slot,67:118]
        image = xyz @ K.T
        valid = ((ref[:,2] >= .5) & (xyz[:,2] > CUDA_NEAR) &
                 (np.linalg.norm(image[:,:2]/image[:,2:]-.5-ref[:,:2],axis=1) <= 6.))
        match = xyz[valid]
        require(len(match) >= 8,f'{subject} {frame} lacks enough fixed inner-face points for a dolly.')
        face = np.median(match,axis=0).astype(np.float64)
        minimum = {side:float((arm['centers'][frame]@E[2,:3]+E[2,3]).min()) for side,arm in baselines.items()}
        advance_z = min(face[2]*(1-1/TARGET_MAGNIFICATION),min(minimum.values())-BASELINE_MARGIN_M)
        require(advance_z >= .2*face[2], 'The protected raster near plane prevents a meaningful move toward the face.')
        move_camera = face*(advance_z/face[2])
        new_E = E.copy()
        new_E[:3,3] -= move_camera
        native_center = -E[:3,:3].T @ E[:3,3]
        close_center = -new_E[:3,:3].T @ new_E[:3,3]
        near_xyz = match-move_camera
        projected = near_xyz@K.T
        xy = projected[:,:2]/projected[:,2:]-.5
        require((near_xyz[:,2] > CUDA_NEAR).all() and (xy >= 0).all() and (xy < np.asarray(row['wh'])-1).all(),
                'Matched face proxy would cross the near plane or image boundary.')
        masks[role] = interior(xy,row['wh'])
        require(masks[role] is not None and masks[role].sum() >= 32,'Dolly face proxy lacks interior samples.')
        crop = r10.native_box([*(xy.min(axis=0)-54),*(xy.max(axis=0)+54)],row['wh'])
        camera = {key:row[key] for key in ('frame','wh','K','K_integer_centers','w2c')}
        views[role] = dict(frame=frame,role=role,slot=fit_slot,source_camera=camera,
            near_camera=dict(camera,w2c=new_E.tolist()),crop_xyxy=list(crop),
            projected_inner_face_landmarks=len(match),face_proxy_sha256=prior.r9.native.hash_mask(masks[role]),
            native_center_world=native_center.tolist(),near_center_world=close_center.tolist(),
            camera_displacement_m=float(np.linalg.norm(close_center-native_center)),
            native_center_to_face_m=float(np.linalg.norm(face)),near_center_to_face_m=float(np.linalg.norm(face-move_camera)),
            projected_face_scale_ratio=float(face[2]/(face[2]-advance_z)),
            native_face_depth_m=float(face[2]),near_face_depth_m=float(face[2]-advance_z),
            cpu_baseline_min_near_center_depth_m={side:z-advance_z for side,z in minimum.items()},
            near_raster_cutoff_m=CUDA_NEAR,baseline_design_margin_m=BASELINE_MARGIN_M,
            source_rgb_truth_available_at_new_camera=False,near_roi_from_fixed_mesh_keypoints=True)
    return views,masks


def dark_panel(out, side, pair, polygons, rgb, pixels):
    points = np.array([point for region in polygons.values() for polygon in region['polygons'] for point in polygon])
    wh = pair['camera']['wh']
    crop = r10.native_box([*(points.min(axis=0)-28),*(points.max(axis=0)+28)],wh)
    overlay = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(overlay)
    for name, region in polygons.items():
        for polygon in region['polygons']:
            draw.polygon([tuple(point) for point in polygon],outline='yellow' if name=='eye' else '#00e4ff',width=1)
    return dict(r10.strip(out/f'dark_{pair["role"]}-{side}.png',crop,(np.asarray(overlay),*pixels)),frame=pair['frame'],
                columns=['source_with_eye_yellow_hair_cyan','black','white','alpha'])


def render(path, side, lease_fd):
    require(os.path.samestat(os.fstat(lease_fd),review.GPU_LEASE.stat()), 'Missing inherited GPU lease.')
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan,out = read(path),path.parent
    info = plan[side]
    root = Path(info['run'])
    sys.path.insert(0,str(Path(info['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    asset = Path(plan['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render as raster

    rows = read(pin(root/'data/training-inputs.json',info['training_inputs_sha256']))['records']
    saved = round1.delivered(root,asset,rows,info['protected_holdouts'])
    results,panels = {},{}
    with gpu_lock(),torch.no_grad(),np.load(pin(out/'near-rois.npz',plan['near_roi_sha256']),allow_pickle=False) as masks:
        model,_ = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,saved.mesh,saved.facts,asset)
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        for pair in plan['pairs']:
            role,frame = pair['role'],pair['frame']
            spec = plan['near_views'][role]
            slot = pair[f'{side}_slot']
            require(frame == spec['frame'] and slot == spec['slot'], 'Wrong native pose for close camera.')
            model.select_mesh_by_timestep(slot)
            cam,record = camera(spec['near_camera'],0,pose_index=slot)
            E = np.asarray(record['w2c'],dtype=np.float64)
            centers = model.get_xyz.detach().cpu().numpy()
            min_depth = float((centers@E[2,:3]+E[2,3]).min())
            require(min_depth > CUDA_NEAR,'A Gaussian center was culled by the physical dolly near plane.')
            if side != 'new':
                name = 'R15' if side=='old' else 'R16'
                require(abs(min_depth-spec['cpu_baseline_min_near_center_depth_m'][name]) < 6e-6,
                        'CUDA close depth differs from the preregistered CPU baseline.')
            pixels,error = scale_pixels(raster,render_pose,cam,model,pipe,black,white,slot,1.)
            proxy = masks[role]
            require(prior.r9.native.hash_mask(proxy) == spec['face_proxy_sha256'],'Dolly face proxy changed.')
            panels[f'physical_near_{role}'] = dict(r10.strip(out/f'physical_near_{role}-{side}.png',spec['crop_xyxy'],pixels),
                                                  frame=frame,columns=['black','white','alpha'])
            results[role] = dict(camera=record,pose_slot=slot,min_gaussian_center_depth_m=min_depth,
                projection_matrix_znear_m=float(cam.znear),actual_raster_near_cutoff_m=CUDA_NEAR,
                composition_max_abs=error,face_proxy_alpha=face_alpha(pixels[2],proxy),
                float_rgb=prior.float_summary(pixels),source_rgb_truth_available=False)
            source,source_cam = camera(spec['source_camera'],0,pose_index=slot)
            native,_ = scale_pixels(raster,render_pose,source,model,pipe,black,white,slot,1.)
            with Image.open(pin(pair['source'],pair['source_sha256'])) as image:
                rgb = np.asarray(image.convert('RGB'))
            annotation = plan['independent_annotations'][role]
            panels[f'dark_{role}'] = dark_panel(out,side,pair,annotation['regions'],rgb,native)
            results[role]['native_dark_camera'] = source_cam
    review.save(out/f'{side}-near.json',dict(physical_near=results,panels=panels))
