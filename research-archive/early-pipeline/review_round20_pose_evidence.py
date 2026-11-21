"""Verify R20 revised native poses, supplemental face crops and isolated holdouts."""

import fcntl
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

import review_smpl_delivery_round19 as baseline
import review_round18_supplement as auxiliary
from review_bound_scale_coverage import face_alpha, scale_pixels


pin,read,sha,require,descriptor = baseline.pin,baseline.read,baseline.sha,baseline.require,baseline.descriptor
review,round1,r10 = baseline.review,baseline.round1,baseline.r10
FIT = Path('fit/head-alignment')
HELDOUT = Path('gaussian-run/step-010000/heldout-replay')
REGIONS = {'right_eye':np.arange(86,92),'left_eye':np.arange(92,98),'nose':np.arange(77,86),
           'mouth':np.arange(98,118),'right_ear':np.asarray([17]),'left_ear':np.asarray([18])}


def proof_files(run,wrapper,terminal):
    proof = {}
    for key,file in (('head_alignment_holdout_plan',run/FIT/'holdout-plan.json'),
                     ('head_alignment_holdout_poses',run/FIT/'holdout-poses.npz'),
                     ('head_alignment_holdout_policy',run/FIT/'holdout-policy.json'),
                     ('heldout_replay',run/HELDOUT/'result.json')):
        proof[key] = descriptor(file)
    require(terminal['heldout_bound_replay']==HELDOUT.as_posix()+'/result.json'
            and terminal['heldout_bound_replay_sha256']==proof['heldout_replay']['sha256']==wrapper['heldout_replay']['sha256']
            and wrapper['head_fit']['holdout_plan']['plan_sha256']==proof['head_alignment_holdout_plan']['sha256']
            and wrapper['head_fit']['holdout_plan']['poses_sha256']==proof['head_alignment_holdout_poses']['sha256'],
            'The completed terminal does not pin the independent replay and pose plan.')
    report = read(run/HELDOUT/'result.json')
    require(report['schema']=='shuolee-pose-conditioned-holdout-replay/1' and report['optimizer_updates']==0
            and report['fully_unobserved_prediction'] is False and report['uses_frozen_clothing_offset'] is True
            and report['holdout_plan_sha256']==proof['head_alignment_holdout_plan']['sha256']
            and report['holdout_pose_sha256']==proof['head_alignment_holdout_poses']['sha256'],
            'Heldout replay does not pin the frozen evaluation-only pose plan.')
    mesh = run/HELDOUT/report['mesh_file']
    proof['heldout_mesh'] = descriptor(pin(mesh,report['mesh_sha256']))
    face = run/auxiliary.FOLDER
    descriptor_fields = wrapper.get('face_supplement',{})
    for key in ('admission','training','replay','poses'):
        file = face/(f'{key}.npz' if key=='poses' else f'{key}.json')
        if key=='poses' and not file.exists():
            continue
        digest = descriptor_fields.get(f'{key}_sha256')
        proof[f'supplement_{key}'] = descriptor(pin(file,digest)) if digest else descriptor(file)
    auxiliary.bundle_members(run,proof)
    return proof


def pose_metrics(observed,projected):
    errors = np.linalg.norm(projected-observed[:,:2],axis=1)
    valid = np.isfinite(observed).all(axis=1)&np.isfinite(errors)&(observed[:,2]>=.5)
    result = {}
    for name,indices in REGIONS.items():
        selected = errors[indices[valid[indices]]]
        result[name] = dict(points=len(selected),rms_px=float(np.sqrt(np.mean(selected**2))) if len(selected) else None,
                            max_px=float(selected.max()) if len(selected) else None)
    return result


def fit_plan(run,plan):
    proof = plan['proof']
    audit = read(pin(proof['head_alignment']['file'],proof['head_alignment']['sha256']))
    algorithm = audit.get('fitting',audit)
    require(algorithm['schema']=='shuolee-native-head-alignment/1' and algorithm['original_frames']==plan['matched_frames']
            and algorithm['retained_holdouts']==plan['new']['protected_holdouts']
            and algorithm['shared_shape_updates']==algorithm['per_frame_scale_parameters']==0,
            'The native alignment receipt changed the original frame, scale or holdout contract.')
    baseline_run = Path(plan['r19']['run'])
    base = pin(baseline_run/'fit/fitted.npz',plan['r19']['fitted_sha256'])
    current = pin(run/'fit/fitted.npz',plan['new']['fitted_sha256'])
    require(audit['active']['fit_sha256']==sha(current)
            and audit['active']['candidate_sha256']==proof['joint_candidates']['sha256']
            and sha(base)==proof['head_fit_preimage_fitted']['sha256'], 'The revised fitting is not bound to its saved preimage.')
    report,dolly_distance = {},{}
    with np.load(base,allow_pickle=False) as before,np.load(current,allow_pickle=False) as after:
        require(set(before.files)==set(after.files) and all(before[key].shape==after[key].shape for key in before.files)
                and before['frame_ids'].tolist()==after['frame_ids'].tolist()==plan['matched_frames'],
                'The revised head fit changed the original pose layout.')
        frozen = ('faces','K_integer_centers','w2c','source_w2c','source_frames','image_sha256','mask_sha256',
                  'observed_native_keypoints','global_orient','camera_translation','world_translation',
                  'left_hand_pose','right_hand_pose')
        require(all(np.array_equal(before[key],after[key]) for key in frozen),
                'A non-head pose, fixed camera, independent observation or source identity changed.')
        require(after['shared_betas'].shape==(1,10) and np.array_equal(after['shared_betas'],after['betas'])
                and np.array_equal(before['shared_betas'],after['shared_betas']),
                'The final head/body fit must share one betas vector.')
        body = [slot for slot in range(21) if slot not in (11,14)]
        require(np.array_equal(before['body_pose_axis_angle'].reshape(-1,21,3)[:,body],
                               after['body_pose_axis_angle'].reshape(-1,21,3)[:,body]),
                'Revised head fitting changed a non-head or non-neck body joint.')
        names = after['frame_ids'].tolist()
        for slot,frame in enumerate(names):
            observed = before['observed_native_keypoints'][slot]
            a = pose_metrics(observed,before['joints_projected_native'][slot])
            b = pose_metrics(observed,after['joints_projected_native'][slot])
            delta = np.linalg.norm(after['vertices_world'][slot]-before['vertices_world'][slot],axis=1)
            report[frame] = dict(before=a,after=b,changed_vertices=int((delta>1e-6).sum()),
                max_mesh_move_m=float(delta.max()),shared_betas=float(np.linalg.norm(after['shared_betas']-before['shared_betas'])))
        for role,view in plan.get('near_views',{}).items():
            slot = names.index(view['frame'])
            obs = before['observed_native_keypoints'][slot,67:118]
            original = before['joints_camera'][slot,67:118]
            K = np.asarray(view['source_camera']['K'])
            native = original@K.T
            eligible = (obs[:,2]>=.5)&(original[:,2]>.2)&(np.linalg.norm(native[:,:2]/native[:,2:]-.5-obs[:,:2],axis=1)<=6.)
            require(int(eligible.sum())==view['projected_inner_face_landmarks'],'The close camera used a different face anchor set.')
            revised = np.median(after['joints_camera'][slot,67:118][eligible],axis=0).astype(np.float64)
            movement = np.asarray(view['near_camera']['w2c'])[:3,3]-np.asarray(view['source_camera']['w2c'])[:3,3]
            near = revised+movement
            require(near[2]>.2 and np.linalg.norm(near)<np.linalg.norm(revised),'The fixed camera did not approach the revised face.')
            dolly_distance[role] = dict(frame=view['frame'],revised_native_face_distance_m=float(np.linalg.norm(revised)),
                                       revised_near_face_distance_m=float(np.linalg.norm(near)),
                                       same_fixed_landmark_slots=int(eligible.sum()))
    candidates_before = pin(baseline_run/'fit/joint-candidates.npz',proof['head_fit_preimage_joint-candidates']['sha256'])
    candidates_after = pin(proof['joint_candidates']['file'],proof['joint_candidates']['sha256'])
    candidates = {}
    with np.load(candidates_before,allow_pickle=False) as before,np.load(candidates_after,allow_pickle=False) as after:
        names = before['frame_ids'].tolist()
        require(names==after['frame_ids'].tolist() and not set(names).intersection(plan['new']['protected_holdouts'])
                and all(np.array_equal(before[key],after[key]) for key in ('faces','source_frames','image_sha256',
                    'mask_sha256','K_integer_centers','observed_native_keypoints')),
                'The R20 head branch altered the auxiliary camera/observation identity or mixed heldout frames.')
        for slot,frame in enumerate(names):
            observed = before['observed_native_keypoints'][slot]
            candidates[frame] = dict(before=pose_metrics(observed,before['joints_projected_native'][slot]),
                                     after=pose_metrics(observed,after['joints_projected_native'][slot]))
    summary = dict(frame_count=len(report),mesh_changed_frames=sum(row['changed_vertices']>0 for row in report.values()),
        shared_betas_l2_change=next(iter(report.values()))['shared_betas'],native_K_and_w2c_identical=True,
        frozen_non_head_pose_and_hand_parameters=True,ear_semantic_indices={key:REGIONS[key].tolist() for key in ('right_ear','left_ear')},
        detector_rms_is_independent_physical_identity=False)
    screened = {row['frame']:row for row in algorithm['frames']}
    require(len(screened)==len(algorithm['frames']) and set(screened)==set(algorithm['fitted_frames']),
            'Duplicate or missing visibility-screened native alignment row.')
    return dict(summary=summary,frames=report,candidates=candidates,screened=screened,
                changed_frames=algorithm['changed_frames'],physical_dolly_revised_face=dolly_distance,semantic_mapping=algorithm['mapping'],algorithm_audit=proof['head_alignment'],
                algorithm_state=audit.get('state'),archived_fitted=descriptor(base),revised_fitted=descriptor(current))


def supplement_plan(run,plan):
    proof = plan['proof']
    report = read(pin(proof['supplement_replay']['file'],proof['supplement_replay']['sha256']))
    admission = read(pin(proof['supplement_admission']['file'],proof['supplement_admission']['sha256']))
    training = read(pin(proof['supplement_training']['file'],proof['supplement_training']['sha256']))
    raw = {row['frame']:row for row in read(run/'data/inputs.json')['records']}
    final = read(run/'gaussian-run/step-010000/bound-replay.json')
    require(report['shared_bound_ply_sha256']==final['bound_ply_sha256']
            and len(report['records'])==len(admission['chosen_frame_ids'])<=4
            and [row['frame'] for row in report['records']]==admission['chosen_frame_ids']
            and admission['full_body_frame_ids']==plan['matched_frames'],
            'The new auxiliary poses do not belong to the delivered shared model.')
    frames = []
    for index,row in enumerate(report['records']):
        frame = row['frame']
        source = raw[frame]
        require(frame not in set(plan['matched_frames'])|set(plan['new']['protected_holdouts'])
                and row['role']=='face_only' and row['pose_index']==24+index
                and all(row[key]==source[key] for key in ('source_frame_index','source_sha256','mask_sha256'))
                and row['camera']['K']==source['K'] and row['camera']['wh']==source['wh'],
                'A supplemental pose used a different source or entered another role.')
        valid = pin(run/auxiliary.FOLDER/row['valid_mask'],row['valid_mask_sha256'])
        rgb = pin(source['image'],row['source_sha256'])
        mask = pin(source['mask'],row['mask_sha256'])
        with Image.open(valid) as valid_image,Image.open(mask) as original_mask:
            samples = np.asarray(valid_image.convert('L'))==255
            original = np.asarray(original_mask.convert('L').crop(row['box']))
            require(samples.shape==original.shape and int(samples.sum())==row['valid_pixels']
                    and (original[samples]>=253).all(),'Face-only mask extends outside reliable independent foreground.')
        decision = next(row for row in admission['candidates'] if row['frame']==frame)
        require(decision['status']=='selected_face_supplement','A replayed face failed admission.')
        frames.append(dict(row,source=str(rgb),source_mask=str(mask),valid_path=str(valid),admission=decision))
    if frames:
        item = proof['supplement_poses']
        with np.load(pin(item['file'],item['sha256']),allow_pickle=False) as archive:
            require(archive['frame_ids'].tolist()==[row['frame'] for row in frames]
                    and archive['pose_indices'].tolist()==[row['pose_index'] for row in frames]
                    and np.isfinite(archive['vertices_world']).all(), 'Auxiliary sidecar pose identity differs.')
    return dict(frames=frames,full_body_coverage=admission['full_body_coverage'],
        supplemented_coverage=admission['supplemented_coverage'],
        face_loss_updates=training['face_loss_updates'],separate_optimizer_updates=training['separate_optimizer_updates'],
        committed_updates=training.get('committed_updates',0),body_frames=len(plan['matched_frames']),
        held_out_frames=plan['new']['protected_holdouts'],review_geometry='same R20 auxiliary fitted mesh for all four endpoints')


def holdout_plan(run,plan):
    proof = plan['proof']
    replay = read(pin(proof['heldout_replay']['file'],proof['heldout_replay']['sha256']))
    raw = {row['frame']:row for row in read(run/'data/inputs.json')['records']}
    holdout_roster = plan['new']['protected_holdouts']
    manifest = read(pin(proof['head_alignment_holdout_plan']['file'],proof['head_alignment_holdout_plan']['sha256']))
    require(manifest['frames']==holdout_roster and manifest['state']=='frozen_before_gaussian'
            and manifest['gaussian_updates']==manifest['shared_identity_updates']==0
            and manifest['use_final_render_for_pose_fit'] is False and manifest['use_rgb_for_gaussian_fitting'] is False
            and manifest['policy_sha256']==proof['head_alignment_holdout_policy']['sha256']
            and manifest['native_inputs_sha256']==sha(run/'data/inputs.json')
            and manifest['observations_sha256']==sha(run/'data/observations.npz')
            and manifest['training_inputs_sha256']==plan['new']['training_inputs_sha256']
            and replay['bound_ply_sha256']==read(run/'gaussian-run/step-010000/bound-replay.json')['bound_ply_sha256'],
            'The holdout fit used a different native roster, state, or Gaussian artifact.')
    require(len(replay['records'])==4 and {row['frame'] for row in replay['records']}==set(holdout_roster)
            and not set(holdout_roster).intersection(plan['matched_frames']+ [row['frame'] for row in plan['supplement']['frames']]),
            'An independent replay was omitted or appears in a training role.')
    frames = []
    for index,(row,original_plan) in enumerate(zip(replay['records'],manifest['records'],strict=True)):
        frame = row['frame']
        source = raw[frame]
        require(all(row[key]==source[key] for key in ('source_frame_index','source_sha256','mask_sha256'))
                and row['camera']['wh']==source['wh'] and row['camera']['K']==source['K']
                and all(row['camera'][key]==original_plan[key] for key in ('frame','wh','K','w2c'))
                and row['fitted_pose_index']==original_plan['pose_index']==index
                and row['camera']['pose_index']==row['pose_index']==24+index,
                'An independent holdout was projected on a different source, mask or K.')
        original = pin(source['image'],row['source_sha256'])
        mask = pin(source['mask'],row['mask_sha256'])
        files = {key:descriptor(pin(run/HELDOUT/item['file'],item['sha256'])) for key,item in row['files'].items()}
        require(set(files)=={'black','white','alpha','mesh','feature'}, 'The holdout replay media is incomplete.')
        crop = row['crop_xyxy']
        require(list(r10.native_box(crop,source['wh']))==crop,'Heldout crop is outside its native grid.')
        for key,item in files.items():
            with Image.open(item['file']) as output:
                expected = (crop[2]-crop[0],crop[3]-crop[1]) if key in ('mesh','feature') else tuple(source['wh'])
                require(output.size==expected,'Heldout original-sized crop or image changed.')
        frames.append(dict(row,source=str(original),source_mask=str(mask),files=files,crop_xyxy=crop,mesh_index=index))
    item = proof['heldout_mesh']
    with np.load(pin(item['file'],item['sha256']),allow_pickle=False) as mesh:
        require(mesh['frame_ids'].tolist()==[row['frame'] for row in frames]
                and mesh['pose_indices'].tolist()==[row['pose_index'] for row in frames]
                and mesh['vertices_world'].shape==(len(frames),10475,3)
                and np.isfinite(mesh['vertices_world']).all(), 'Heldout cloth/mesh pose identity differs.')
    return dict(frames=frames,training_frames=24,supplemental_frames=len(plan['supplement']['frames']),
                evaluation_only_frames=4,geometry_scope='common R20 independently fitted holdout pose and cloth surface',
                measurement_scope='Pixels excluded from Gaussian training; camera/body pose inferred per holdout frame.')


def render_heldouts(path,side,fd):
    plan = read(path)
    out = path.parent
    require(os.path.samestat(os.fstat(fd),review.GPU_LEASE.stat()),'Missing inherited GPU lease.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model_info = plan[side]
    run = Path(model_info['run'])
    sys.path.insert(0,str(Path(model_info['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    asset = Path(plan['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render as raster

    rows = read(pin(run/'data/training-inputs.json',model_info['training_inputs_sha256']))['records']
    saved = round1.delivered(run,asset,rows,model_info['protected_holdouts'])
    item = plan['proof']['heldout_mesh']
    records,panels = {},{}
    with gpu_lock(),torch.no_grad(),np.load(pin(item['file'],item['sha256']),allow_pickle=False) as archive:
        model,_ = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,saved.mesh,saved.facts,asset)
        require(torch.cuda.device_count()==1 and np.array_equal(archive['faces'],model.faces.detach().cpu().numpy()),
                'R20 heldout replay changed ordered faces or CUDA device.')
        model.vertices_sequence = torch.cat((model.vertices_sequence,torch.as_tensor(archive['vertices_world'],device='cuda')))
        model.num_timesteps = len(model.vertices_sequence)
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        for held in plan['holdout']['frames']:
            frame,index = held['frame'],held['mesh_index']
            render_slot = len(rows)+index
            model.select_mesh_by_timestep(render_slot)
            cam,actual = camera(held['camera'],index,pose_index=render_slot)
            require(all(actual[key]==held['camera'][key] for key in ('frame','K','w2c','wh')),
                    'Independent holdout camera did not reproduce the frozen K and w2c.')
            pixels,error = scale_pixels(raster,render_pose,cam,model,pipe,black,white,render_slot,1.)
            source = pin(held['source'],held['source_sha256'])
            mask = pin(held['source_mask'],held['mask_sha256'])
            with Image.open(source) as raw,Image.open(mask) as image_mask:
                original = np.asarray(raw.convert('RGB'))
                matte = np.asarray(image_mask.convert('L'))
            good,background = matte>=253,matte<=2
            require(good.any() and background.any(),'Independent heldout native mask has no reliable support.')
            if side=='new':
                for label,actual_pixel in zip(('black','white','alpha'),pixels,strict=True):
                    file = held['files'][label]
                    with Image.open(pin(file['file'],file['sha256'])) as captured:
                        expected = np.asarray(captured.convert('L' if label=='alpha' else 'RGB'),dtype=np.int16)
                    observed = np.rint(np.clip(actual_pixel,0,1)*255).astype(np.int16)
                    require(expected.shape==observed.shape and int(np.abs(expected-observed).max())<=1,
                            'Independent R20 production RGB or alpha differs from bound PLY readback.')
            key = f'heldout_{frame}'
            with Image.open(held['files']['mesh']['file']) as m, Image.open(held['files']['feature']['file']) as f:
                overlays = (m.convert('RGB'),f.convert('RGB'))
            panels[key] = dict(r10.strip(out/f'{key}-{side}.png',held['crop_xyxy'],(original,*pixels,*overlays)),frame=frame,
                               columns=['source_rgb','black','white','alpha','mesh','features'])
            records[frame] = dict(camera=actual,source_sha256=held['source_sha256'],mask_sha256=held['mask_sha256'],
                actual_holdout_mesh_sha256=item['sha256'],pose_index=render_slot,
                foreground_rgb_mae=float(np.abs(pixels[0][good]-original[good].astype(np.float32)/255).mean()),
                alpha=face_alpha(pixels[2],good),exterior_half_alpha_pixels=int((pixels[2][background]>=.5).sum()),
                composition_max_abs=error,source_pose_estimated=True,training_role='evaluation_only')
    review.save(out/f'{side}-heldout.json',dict(heldout_native=records,panels=panels))


def verify_holdout(plan,arms):
    frames = plan['holdout']['frames']
    require(all(set(arm['heldout_native'])=={row['frame'] for row in frames} for arm in arms.values()),
            'An endpoint missed an independent heldout replay.')
    for row in frames:
        recs = [arm['heldout_native'][row['frame']] for arm in arms.values()]
        require(all(record['camera']==recs[0]['camera'] and record['source_sha256']==row['source_sha256']
                    and record['mask_sha256']==row['mask_sha256'] and record['training_role']=='evaluation_only'
                    and record['actual_holdout_mesh_sha256']==plan['proof']['heldout_mesh']['sha256'] for record in recs),
                'An endpoint changed the matched independent holdout source, pose or camera.')


def fallback_face_plan(run,plan,previous):
    """Keep previously named face views for diagnosis when revised admission rejects them."""
    admitted = {row['frame'] for row in plan['supplement']['frames']}
    pending = [row for row in previous if row['frame'] not in admitted]
    if not pending:
        return []
    raw = {row['frame']:row for row in read(run/'data/inputs.json')['records']}
    decisions = {row['frame']:row for row in read(run/auxiliary.FOLDER/'admission.json')['candidates']}
    candidate = plan['proof']['joint_candidates']
    held = set(plan['new']['protected_holdouts'])
    result = []
    with np.load(pin(candidate['file'],candidate['sha256']),allow_pickle=False) as poses:
        indices = {name:i for i,name in enumerate(poses['frame_ids'].tolist())}
        for old in pending:
            frame = old['frame']
            index = indices[frame]
            row = raw[frame]
            require(frame not in held and frame not in plan['matched_frames'] and frame in decisions
                    and row['source_frame_index']==int(poses['source_frames'][index])==old['source_frame_index']
                    and row['source_sha256']==poses['image_sha256'][index].item()==old['source_sha256']
                    and row['mask_sha256']==poses['mask_sha256'][index].item()==old['mask_sha256']
                    and row['K']==old['camera']['K'] and decisions[frame]['status']!='selected_face_supplement',
                    'A fitted but unadmitted face control changed its original source identity or role.')
            valid = pin(old['valid_path'],old['valid_mask_sha256'])
            source = pin(row['image'],row['source_sha256'])
            mask = pin(row['mask'],row['mask_sha256'])
            result.append(dict(frame=frame,source_frame_index=old['source_frame_index'],camera=old['camera'],
                source=str(source),source_sha256=old['source_sha256'],source_mask=str(mask),mask_sha256=old['mask_sha256'],
                valid_path=str(valid),valid_mask_sha256=old['valid_mask_sha256'],valid_pixels=old['valid_pixels'],
                box=old['box'],candidate_index=index,role='fit_only',decision=decisions[frame]['status'],
                decision_reasons=decisions[frame]['reasons']))
    return result


def render_fit_only_faces(path,side,fd):
    plan,out = read(path),path.parent
    controls = plan['supplement']['historical_fit_only_controls']
    if not controls:
        review.save(out/f'{side}-fit-only-faces.json',dict(fit_only_face_native={},panels={}))
        return
    require(os.path.samestat(os.fstat(fd),review.GPU_LEASE.stat()),'Missing inherited GPU lease.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info = plan[side]
    run = Path(info['run'])
    sys.path.insert(0,str(Path(info['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.bound_review_panels import _mesh_panel
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    asset = Path(plan['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render as raster

    rows = read(pin(run/'data/training-inputs.json',info['training_inputs_sha256']))['records']
    saved = round1.delivered(run,asset,rows,info['protected_holdouts'])
    candidate = plan['proof']['joint_candidates']
    records,panels = {},{}
    with gpu_lock(),torch.no_grad(),np.load(pin(candidate['file'],candidate['sha256']),allow_pickle=False) as arrays:
        model,_ = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,saved.mesh,saved.facts,asset)
        positions = []
        for row in controls:
            E = np.asarray(row['camera']['w2c'],dtype=np.float64)
            vertices = arrays['vertices_camera'][row['candidate_index']]
            positions.append(((vertices.astype(np.float64)-E[:3,3])@E[:3,:3]).astype(np.float32))
        require(torch.cuda.device_count()==1 and np.array_equal(arrays['faces'],model.faces.cpu().numpy()),
                'Fit-only auxiliary replay changed topology or GPU roster.')
        model.vertices_sequence = torch.cat((model.vertices_sequence,torch.as_tensor(np.stack(positions),device='cuda')))
        model.num_timesteps = len(model.vertices_sequence)
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        for offset,row in enumerate(controls):
            frame,slot = row['frame'],len(rows)+offset
            model.select_mesh_by_timestep(slot)
            cam,actual = camera(row['camera'],offset,pose_index=slot)
            pixels,error = scale_pixels(raster,render_pose,cam,model,pipe,black,white,slot,1.)
            x0,y0,x1,y1 = box = row['box']
            with Image.open(pin(row['source'],row['source_sha256'])) as source:
                original = source.convert('RGB').crop(box)
            with Image.open(pin(row['valid_path'],row['valid_mask_sha256'])) as source:
                mask = source.convert('L')
            index = row['candidate_index']
            mesh = _mesh_panel(original,box,arrays['vertices_camera'][index],arrays['faces'],arrays['K_integer_centers'][index])
            key = f'face_control_{frame}'
            panels[key] = dict(r10.strip(out/f'{key}-{side}.png',box,(original,mask.convert('RGB'),mesh,*pixels)),frame=frame,
                               columns=['source_rgb','historical_valid_face_pixels','revised_fit_only_mesh','black','white','alpha'])
            valid = np.asarray(mask)==255
            require(int(valid.sum())==row['valid_pixels'],'Fit-only control source support changed.')
            black_crop = pixels[0][y0:y1,x0:x1]
            alpha_crop = pixels[2][y0:y1,x0:x1]
            records[frame] = dict(camera=actual,source_sha256=row['source_sha256'],valid_mask_sha256=row['valid_mask_sha256'],
                revised_fit_sha256=candidate['sha256'],foreground_rgb_mae=float(np.abs(black_crop[valid]-
                    np.asarray(original,dtype=np.float32)[valid]/255.).mean()),
                foreground_alpha=face_alpha(alpha_crop,valid),composition_max_abs=error,training_role='fit_only')
    review.save(out/f'{side}-fit-only-faces.json',dict(fit_only_face_native=records,panels=panels))


def isolated_fit_panels(out,plan):
    """Show only the native fit change, before any clothing offset or Gaussian rendering."""
    sys.path.insert(0,str(Path(plan['new']['project'])/'src'))
    from shuolee_portrait.bound_review_panels import _features, _mesh_panel

    evidence = plan['fit_audit']
    old,new = evidence['archived_fitted'],evidence['revised_fitted']
    original = [(item['frame'],item['source'],item['source_sha256'],item['head_crop_xyxy'])
                for item in plan['pairs'] if item['head_crop_xyxy']]
    aux = []
    if plan['subject']=='016':
        f55 = next(item for item in plan['common_special'] if item['frame']=='frame_000055')
        aux.append((f55['frame'],f55['source'],f55['source_sha256'],f55['head_crop_xyxy']))
    candidate_before = plan['proof']['head_fit_preimage_joint-candidates']
    candidate_after = plan['proof']['joint_candidates']
    bundles = [(original,old,new),(aux,candidate_before,candidate_after)]
    panels = []
    for views,before,after in bundles:
        if not views:
            continue
        with np.load(pin(before['file'],before['sha256']),allow_pickle=False) as old_fit, \
             np.load(pin(after['file'],after['sha256']),allow_pickle=False) as new_fit:
            names = old_fit['frame_ids'].tolist()
            for frame,source,source_sha,box in views:
                slot = names.index(frame)
                with Image.open(pin(source,source_sha)) as raw:
                    image = raw.convert('RGB').crop(box)
                parts = [image]
                for array in (old_fit,new_fit):
                    require(array['frame_ids'][slot].item()==frame and array['image_sha256'][slot].item()==source_sha,
                            'Native fit overlay changed the source pose.')
                    parts.extend((_mesh_panel(image,box,array['vertices_camera'][slot],array['faces'],array['K_integer_centers'][slot]),
                                  _features(image,box,array['observed_native_keypoints'][slot],array['joints_projected_native'][slot])))
                key = f'fit_head_{frame}'
                panels.append(dict(r10.strip(out/f'{key}.png',box,parts),view=key,frame=frame,new_frame=frame,
                    columns=['source_rgb','old_mesh','old_features','revised_mesh','revised_features'],
                    rows=['native_fit_before_and_after_no_gaussian']))
    return panels
