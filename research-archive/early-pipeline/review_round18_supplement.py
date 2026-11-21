"""Read and compare admitted face-only evidence on the frozen candidate poses."""

import fcntl
from hashlib import sha256
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

import review_smpl_delivery_round17 as prior
from review_bound_scale_coverage import face_alpha, scale_pixels


pin,read,sha,require,descriptor = prior.pin,prior.read,prior.sha,prior.require,prior.descriptor
r10,review,round1 = prior.r10,prior.review,prior.round1
FOLDER = Path('gaussian-run/face-supplement')
FIXTURE = prior.DELIVERY.with_name('smpl-delivery-round18')/'input-audit/cuda-face-supplement-proof.json'


def bundle_members(run, proof):
    """Pin only the output files named by the supplemental replay manifest."""
    replay_desc = proof['supplement_replay']
    replay = read(pin(replay_desc['file'],replay_desc['sha256']))
    keys = ['supplement_admission','supplement_training','supplement_replay']
    if replay['records']:
        keys.append('supplement_poses')
    members = [pin(proof[key]['file'],proof[key]['sha256']) for key in keys]
    folder = run/FOLDER
    for row in replay['records']:
        frame = row['frame']
        require(re.fullmatch(r'frame_\d{6}',frame) is not None and row['valid_mask']==f'{frame}-valid.png',
                'Unexpected face-only mask filename.')
        members.append(pin(folder/row['valid_mask'],row['valid_mask_sha256']))
        require(set(row['files'])=={'panel','gaussian','alpha'},'Supplemental replay inventory is incomplete.')
        for key,item in row['files'].items():
            require(item['file']==f'{frame}-{key}.png','Unexpected face-only file path.')
            members.append(pin(folder/item['file'],item['sha256']))
    require(len(members)==len(set(members)) and all(file.parent==folder for file in members),
            'Supplemental replay paths must be unique and contained.')
    return members


def completed_proof(run,wrapper,integration):
    info = integration['face_supplement']
    terminal = wrapper['face_supplement']
    folder = run/FOLDER
    require(info['file']=='face-supplement/training.json','The face-only training receipt points elsewhere.')
    train = pin(folder/'training.json',info['sha256'])
    training = read(train)
    admission = pin(folder/'admission.json',training['admission_sha256'])
    replay_path = pin(folder/'replay.json',terminal['replay_sha256'])
    replay = read(replay_path)
    prereg = read(pin(run.parent/'preregistration.json',wrapper['preregistration_sha256']))
    fixture = pin(FIXTURE,prereg['cuda_fixture_sha256'])
    control = read(fixture)
    pixels,scene = control['pixel_loss'],control['bound_scene']
    require(control['state']=='passed' and all(pixels[key] is True for key in
                ('unknown_nan_loss_unchanged','unknown_gradients_zero','black_foreground_rgb_alpha_gradients_nonzero',
                 'empty_roi_rejected')) and scene['body_gradient_rows']==0 and scene['visible_head_gradient_rows']>0
            and scene['same_optimizer_update'] is True and scene['full_view_count']==24
            and scene['face_guard']['background_gain_px']==0 and scene['face_guard']['failures']
            and scene['native_replay_rgb_max_abs']<.005,'CUDA face-only integration controls did not pass.')
    require(training['schema']=='face-supplement-training-v1' and training['state']=='completed'
            and training['main_optimizer_updates']==integration['optimizer_updates']==10000
            and training['separate_optimizer_updates']==0 and training['update_interval']==4
            and 0<=training['face_loss_updates']==info['face_loss_updates']<=training['update_budget']<=1800
            and sum(row['updates'] for row in training['views'])==training['face_loss_updates']
            and info['views']==len(training['views'])==len(replay['records'])
            and terminal['training_sha256']==info['sha256'] and terminal['admission_sha256']==training['admission_sha256']
            and terminal['poses_sha256']==replay['poses_sha256']
            and terminal['frames']==[row['frame'] for row in replay['records']]
            and terminal['face_loss_updates']==training['face_loss_updates'] and terminal['separate_optimizer_updates']==0
            and replay['schema']=='face-supplement-replay-v1', 'R18 face-only work changed the main optimizer contract.')
    bound = read(run/'gaussian-run/step-010000/bound-replay.json')
    require(replay['shared_bound_ply_sha256']==bound['bound_ply_sha256'],
            'Face-only replay did not use the same terminal bound Gaussian set.')
    proof = dict(supplement_admission=descriptor(admission),supplement_training=descriptor(train),
                 supplement_replay=descriptor(replay_path),cuda_face_supplement_proof=descriptor(fixture))
    if replay['records']:
        proof['supplement_poses'] = descriptor(pin(folder/'poses.npz',replay['poses_sha256']))
    else:
        require(replay['poses_sha256'] is None and training['face_loss_updates']==0,
                'Empty face-only roster contains poses or optimizer work.')
    bundle_members(run,proof)
    return proof


def plan(run,review_plan):
    proof = review_plan['proof']
    records = read(Path(proof['supplement_replay']['file']))['records']
    training = read(Path(proof['supplement_training']['file']))
    admission = read(pin(proof['supplement_admission']['file'],proof['supplement_admission']['sha256']))
    raw = {row['frame']:row for row in read(run/'data/inputs.json')['records']}
    train = set(review_plan['matched_frames'])
    held = set(review_plan['new']['protected_holdouts'])
    require(admission['schema']=='shuolee-face-supplement/1' and admission['state']=='admission_complete'
            and admission['chosen_frame_ids']==[item['frame'] for item in records]
            and admission['full_body_frame_ids']==review_plan['matched_frames']
            and set(admission['retained_holdouts'])==held and admission['heldout_pixels_read'] is False
            and admission['physical_angle_calibration'] is False and admission['frozen_shared_shape_and_poses'] is True,
            'Face-only admission mixed the full-body or held-out rosters.')
    decisions = {item['frame']:item for item in admission['candidates']}
    require(len(decisions)==len(admission['candidates']) and not (set(decisions) & (train|held)),
            'The face-only candidate inventory includes protected or duplicate inputs.')
    fit_path = pin(proof['joint_candidates']['file'],proof['joint_candidates']['sha256'])
    frame_plan = []
    with np.load(fit_path,allow_pickle=False) as fits:
        candidates = {frame:index for index,frame in enumerate(fits['frame_ids'].tolist())}
        for index,(row,report) in enumerate(zip(records,training['views'],strict=True)):
            frame = row['frame']
            source,fit = raw[frame],candidates[frame]
            decision = decisions[frame]
            native = decision['native_camera']
            require(all(row[key]==value for key,value in report.items()) and row['role']=='face_only'
                    and row['pose_index']==len(train)+index and frame not in (train|held) and row['updates']>0
                    and decision['status']=='selected_face_supplement'
                    and row['camera']['K']==source['K'] and row['camera']['wh']==source['wh']
                    and row['camera']['w2c']==native['w2c'] and native['source_w2c']==source['w2c']
                    and row['source_frame_index']==source['source_frame_index']==int(fits['source_frames'][fit])
                    and row['source_sha256']==source['source_sha256']==fits['image_sha256'][fit].item()
                    and row['mask_sha256']==source['mask_sha256']==fits['mask_sha256'][fit].item()
                    and 0<=row['valid_rgb_max_abs']<.005,'Face-only replay or source candidate identity differs.')
            x0,y0,x1,y1 = row['box']
            require(0<=x0<x1<=source['wh'][0] and 0<=y0<y1<=source['wh'][1],'Native face-only crop is invalid.')
            folder = run/FOLDER
            valid = pin(folder/row['valid_mask'],row['valid_mask_sha256'])
            original = pin(source['image'],row['source_sha256'])
            mask = pin(source['mask'],row['mask_sha256'])
            with Image.open(valid) as data,Image.open(mask) as prior_mask:
                inside = np.asarray(data.convert('L'))
                alpha = np.asarray(prior_mask.convert('L').crop(row['box']))
            roi = decision['native_roi']
            require(inside.shape==alpha.shape==(y1-y0,x1-x0) and set(np.unique(inside))<={0,255}
                    and int((inside==255).sum())==row['valid_pixels']==roi['valid_pixels']
                    and row['box']==roi['box'] and (alpha[inside==255]>=253).all()
                    and sha256((inside==255).tobytes()).hexdigest()==roi['valid_sha256']
                    and sha256((alpha.astype(np.float32)/255).tobytes()).hexdigest()==roi['alpha_sha256'],
                    'Face-only valid pixels extend outside independent reliable foreground.')
            panel = row['files']['panel']
            with Image.open(pin(folder/panel['file'],panel['sha256'])) as card,Image.open(original) as image:
                width,height = x1-x0,y1-y0
                require(card.size==(4*width,height+26),'Face-only native panel changed shape.')
                rgb = np.asarray(image.convert('RGB').crop(row['box']))
                require(np.array_equal(np.asarray(card.convert('RGB').crop((0,26,width,height+26))),rgb),
                        'Face-only four-up native source pixels changed.')
            frame_plan.append(dict(row,source=str(original),source_mask=str(mask),valid_path=str(valid),
                                   candidate_fit_index=fit,admission=decision))
        if records:
            item = proof['supplement_poses']
            with np.load(pin(item['file'],item['sha256']),allow_pickle=False) as poses:
                require(poses['frame_ids'].tolist()==[row['frame'] for row in records]
                        and poses['pose_indices'].tolist()==[row['pose_index'] for row in records]
                        and np.array_equal(poses['faces'],fits['faces']) and poses['vertices_world'].shape==(len(records),10475,3)
                        and np.isfinite(poses['vertices_world']).all(),'Face-only posed mesh identity differs.')
                for index,row in enumerate(frame_plan):
                    E = np.asarray(row['camera']['w2c'],dtype=np.float64)
                    transformed = poses['vertices_world'][index]@E[:3,:3].T+E[:3,3]
                    np.testing.assert_allclose(transformed,fits['vertices_camera'][row['candidate_fit_index']],rtol=0,atol=3e-6)
    require(len(frame_plan)==len({row['frame'] for row in frame_plan}), 'Duplicate face-only source pose.')
    special = {}
    if review_plan['subject']=='016':
        frame = 'frame_000055'
        case = decisions[frame]
        archive = review_plan['special_016'][frame]
        require(case['source_index']==archive['source_frame_index'] and case['image_sha256']==archive['source_sha256']
                and case['mask_sha256']==archive['new_mask_sha256'] and archive['R17']=='fit_only'
                and frame not in (train|held),'f055 candidate identity or prior eligibility changed.')
        special[frame] = case
    return dict(frames=frame_plan,face_loss_updates=training['face_loss_updates'],
        separate_optimizer_updates=training['separate_optimizer_updates'],body_frames=len(train),
        admission_sha256=proof['supplement_admission']['sha256'],training_sha256=proof['supplement_training']['sha256'],
        full_body_coverage=admission['full_body_coverage'],supplemented_coverage=admission['supplemented_coverage'],
        special=special,
        held_out_frames=sorted(held),pose_scope='frozen R18 face-only candidate vertices shared across all three endpoints')


def render(plan_path,side,lease_fd):
    plan,out = read(plan_path),plan_path.parent
    face = plan['supplement']['frames']
    result,panels = {},{}
    if not face:
        review.save(out/f'{side}-supplement.json',dict(supplement_native=result,panels=panels))
        return
    require(os.path.samestat(os.fstat(lease_fd),review.GPU_LEASE.stat()), 'Missing inherited GPU lease.')
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info = plan[side]
    run = Path(info['run'])
    sys.path.insert(0,str(Path(info['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    asset = Path(plan['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render as raster

    rows = read(pin(run/'data/training-inputs.json',info['training_inputs_sha256']))['records']
    saved = round1.delivered(run,asset,rows,info['protected_holdouts'])
    desc = plan['proof']['supplement_poses']
    aux_run = Path(plan['new']['run'])/FOLDER
    with gpu_lock(),torch.no_grad(),np.load(pin(desc['file'],desc['sha256']),allow_pickle=False) as poses:
        model,_ = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,saved.mesh,saved.facts,asset)
        require(torch.cuda.device_count()==1 and len(model.vertices_sequence)==len(rows)==24
                and np.array_equal(poses['faces'],model.faces.detach().cpu().numpy()),'Supplemental replay topology differs.')
        model.vertices_sequence = torch.cat((model.vertices_sequence,torch.as_tensor(poses['vertices_world'],device='cuda')))
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        for item in face:
            frame,slot = item['frame'],item['pose_index']
            model.select_mesh_by_timestep(slot)
            cam,actual = camera(item['camera'],slot,pose_index=slot)
            require(actual==item['camera'],'Supplemental camera or pose slot differs from training replay.')
            pixels,error = scale_pixels(raster,render_pose,cam,model,pipe,black,white,slot,1.)
            x0,y0,x1,y1 = box = item['box']
            with Image.open(pin(item['source'],item['source_sha256'])) as image:
                source = image.convert('RGB').crop(box)
            with Image.open(pin(item['valid_path'],item['valid_mask_sha256'])) as image:
                verified = image.convert('L')
            valid = np.asarray(verified)==255
            mesh_info = item['files']['panel']
            with Image.open(pin(aux_run/mesh_info['file'],mesh_info['sha256'])) as image:
                width,height = source.size
                mesh = image.convert('RGB').crop((2*width,26,3*width,26+height))
            black_crop = pixels[0][y0:y1,x0:x1]
            alpha_crop = pixels[2][y0:y1,x0:x1]
            delta = None
            if side=='new':
                delta = {}
                for key,pix in (('gaussian',black_crop),('alpha',alpha_crop)):
                    entry = item['files'][key]
                    with Image.open(pin(aux_run/entry['file'],entry['sha256'])) as img:
                        got = np.rint(np.clip(pix,0,1)*255).astype(np.int16)
                        delta[key] = int(np.abs(np.asarray(img,dtype=np.int16)-got).max())
                    require(delta[key]<=1,'R18 face-only training replay pixels differ from fixed review.')
            label = f'face_only_{frame}'
            panels[label] = dict(r10.strip(out/f'{label}-{side}.png',box,(source,verified.convert('RGB'),mesh,*pixels)),
                frame=frame,columns=['source_rgb','verified_face_pixels','frozen_mesh','black','white','alpha'])
            rgb = np.asarray(source,dtype=np.float32)/255.
            result[frame] = dict(camera=actual,source_sha256=item['source_sha256'],native_valid_pixels=int(valid.sum()),
                valid_mask_sha256=item['valid_mask_sha256'],posed_mesh_file_sha256=desc['sha256'],
                foreground_rgb_mae=float(np.abs(black_crop[valid]-rgb[valid]).mean()),
                foreground_alpha=face_alpha(alpha_crop,valid),composition_max_abs=error,terminal_readback_max_abs_8bit=delta)
    review.save(out/f'{side}-supplement.json',dict(supplement_native=result,panels=panels))


def verify_render_receipts(plan,arms):
    frames = plan['supplement']['frames']
    require(all(set(arm['supplement_native'])=={item['frame'] for item in frames} for arm in arms.values()),
            'Face-only native comparison missed an admitted source.')
    for item in frames:
        frame = item['frame']
        records = [arm['supplement_native'][frame] for arm in arms.values()]
        require(all(record['camera']==item['camera'] and record['native_valid_pixels']==item['valid_pixels']
                    and record['source_sha256']==item['source_sha256'] and record['valid_mask_sha256']==item['valid_mask_sha256']
                    and record['posed_mesh_file_sha256']==plan['proof']['supplement_poses']['sha256'] for record in records),
                'Three-arm supplemental source, camera or valid-pixel support differs.')


def verify_size_guards(audit,frames):
    """Check that every real size decision kept the supplemental pixels foreground-only."""
    def check(trial):
        if trial is None:
            return
        roster = trial.get('face_supplement_frames',[])
        require([item['frame'] for item in roster]==frames
                and all(item.get('background_gain_px',0)==0 for item in roster)
                and (not trial['accepted'] or all(not item['failures'] for item in roster)),
                'A size decision skipped supplemental foreground or introduced a background vote.')
    for event in audit['events']:
        check(event.get('removal_trial'))
        check(event.get('face_supplement_trial'))
        for attempt in event.get('local_optimizations',[]):
            check(attempt['replacement_trial'])
        fallback = event.get('fallback_replacement')
        if fallback and fallback['trial']:
            for trial in fallback['trial']['trials']:
                check(trial)
