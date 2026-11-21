"""Verify and replay the frozen main and final endpoints of one R19 training run."""

import fcntl
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import review_smpl_delivery_round17 as prior
import review_round18_supplement as auxiliary
from review_bound_scale_coverage import scale_pixels


pin, read, sha, require, descriptor = prior.pin, prior.read, prior.sha, prior.require, prior.descriptor
r16, r14, r13, review, round1, dolly = prior.r16, prior.r14, prior.r13, prior.review, prior.round1, prior.dolly
rebound = prior.rebound
SNAPSHOT_FILES = ('world.ply', 'reference-render.png', 'reference-mesh.npz', 'native-ply-parity.json',
                  'bound-local.ply', 'bound-replay.json')
MAIN = Path('gaussian-run/step-010000-main')
PROTECTED = Path('gaussian-run/protected-face')


def completed_proof(run, wrapper, integration, delivery, project):
    prereg = read(pin(run.parent/'preregistration.json', wrapper['preregistration_sha256']))
    fixture = pin(delivery/'input-audit/cuda-protected-face-proof.json', prereg['cuda_fixture_sha256'])
    control = read(fixture)
    require(control['state']=='passed' and control['real_same_endpoint']['same_endpoint']
            and control['real_same_endpoint']['baseline_verified'] and control['real_same_endpoint']['final_verified'],
            'The pinned protected face CUDA controls did not pass.')
    for name,digest in control['source_sha256'].items():
        pin(project/name,digest)
    proof = dict(cuda_protected_face_proof=descriptor(fixture))
    protected = wrapper['protected_face']
    endpoint_path = pin(run/PROTECTED/'main-endpoint.json', protected['main_endpoint_sha256'])
    audit_path = pin(run/PROTECTED/'optimization.json', protected['sha256'])
    proof.update(protected_face_main_endpoint=descriptor(endpoint_path),protected_face_optimization=descriptor(audit_path))
    audit = read(audit_path)
    endpoint = read(endpoint_path)
    require(audit['schema']=='shuolee-protected-face/1' and audit['state'] in ('committed','rolled_back')
            and protected['state']==audit['state'] and integration['protected_face']['sha256']==protected['sha256']
            and audit['baseline']['all_guards_pass'] and audit['final']['all_guards_pass']
            and audit['unprotected']['same_endpoint'] and audit['rng_unchanged']
            and endpoint['main_optimizer_updates']==10000 and endpoint['supplemental_updates']==0
            and endpoint['snapshot']==MAIN.name and set(endpoint['files_sha256'])==set(SNAPSHOT_FILES),
            'The same-run facial stage or original-only endpoint has no verified terminal decision.')
    for name in SNAPSHOT_FILES:
        proof[f'protected_face_main_{Path(name).stem}'] = descriptor(pin(run/MAIN/name,endpoint['files_sha256'][name]))
    if audit.get('optimizer_state'):
        item = audit['optimizer_state']
        proof['protected_face_optimizer_state'] = descriptor(pin(run/PROTECTED/item['file'],item['sha256']))
    folder = run/auxiliary.FOLDER
    terminal = wrapper['face_supplement']
    for name in ('admission','training','replay','poses'):
        filename = f'{name}.npz' if name=='poses' else f'{name}.json'
        proof[f'supplement_{name}'] = descriptor(pin(folder/filename,terminal[f'{name}_sha256']))
    replay = read(folder/'replay.json')
    require([row['frame'] for row in replay['records']]==terminal['frames']
            and replay['shared_bound_ply_sha256']==sha(run/'gaussian-run/step-010000/bound-local.ply'),
            'The final supplemental views do not belong to the final bound model.')
    auxiliary.bundle_members(run,proof)
    return proof


def auxiliary_plan(run, plan, archived):
    proof = plan['proof']
    replay = read(pin(proof['supplement_replay']['file'],proof['supplement_replay']['sha256']))
    train = read(pin(proof['supplement_training']['file'],proof['supplement_training']['sha256']))
    admission = read(pin(proof['supplement_admission']['file'],proof['supplement_admission']['sha256']))
    previous = Path(plan['mid']['run'])/auxiliary.FOLDER
    raw = {row['frame']:row for row in read(run/'data/inputs.json')['records']}
    require(set(archived['held_out_frames']).isdisjoint(admission['chosen_frame_ids'])
            and admission['full_body_frame_ids']==plan['matched_frames']
            and archived['held_out_frames']==sorted(plan['new']['protected_holdouts'])
            and [row['frame'] for row in archived['frames']]==admission['chosen_frame_ids']
            and len(replay['records'])==len(admission['chosen_frame_ids'])==4
            and admission['heldout_pixels_read'] is False and admission['frozen_shared_shape_and_poses'] is True,
            'Main and facial rosters, exclusions or frozen poses changed.')
    with np.load(pin(previous/'poses.npz',read(previous/'replay.json')['poses_sha256']),allow_pickle=False) as old, \
         np.load(pin(proof['supplement_poses']['file'],proof['supplement_poses']['sha256']),allow_pickle=False) as new:
        require(set(old.files)==set(new.files) and all(np.array_equal(old[key],new[key]) for key in old.files),
                'A supplemental frozen pose changed from R18.')
    decisions = {row['frame']:row for row in admission['candidates']}
    frames = []
    for row,prior_row in zip(replay['records'],archived['frames'],strict=True):
        frame = row['frame']
        require(all(row[key]==prior_row[key] for key in ('frame','source_frame_index','source_sha256','mask_sha256',
                    'valid_mask_sha256','valid_pixels','box','camera','role','pose_index'))
                and frame not in plan['matched_frames'] and decisions[frame]==prior_row['admission']
                and raw[frame]['source_sha256']==row['source_sha256'] and raw[frame]['mask_sha256']==row['mask_sha256'],
                'A supplemented source camera, frozen identity, region or mask changed.')
        folder = run/auxiliary.FOLDER
        valid = pin(folder/row['valid_mask'],row['valid_mask_sha256'])
        source,mask = (pin(raw[frame][field],row[digest]) for field,digest in (('image','source_sha256'),('mask','mask_sha256')))
        x0,y0,x1,y1 = row['box']
        with Image.open(valid) as v,Image.open(mask) as m,Image.open(source) as s, \
             Image.open(pin(folder/row['files']['panel']['file'],row['files']['panel']['sha256'])) as p:
            valid_pixels = np.asarray(v.convert('L'))==255
            rgb = np.asarray(s.convert('RGB').crop(row['box']))
            require(valid_pixels.shape==(y1-y0,x1-x0) and int(valid_pixels.sum())==row['valid_pixels']
                    and (np.asarray(m.convert('L').crop(row['box']))[valid_pixels]>=253).all()
                    and p.size==(4*(x1-x0),y1-y0+26)
                    and np.array_equal(np.asarray(p.convert('RGB').crop((0,26,x1-x0,y1-y0+26))),rgb),
                    'A supplemental mask or original image panel differs.')
        frames.append(dict(row,source=str(source),source_mask=str(mask),valid_path=str(valid),
                           candidate_fit_index=prior_row['candidate_fit_index'],admission=decisions[frame]))
    return dict(archived,frames=frames,face_loss_updates=train['face_loss_updates'],
        separate_optimizer_updates=train['separate_optimizer_updates'],
        admission_sha256=proof['supplement_admission']['sha256'],training_sha256=proof['supplement_training']['sha256'],
        pose_scope='identical frozen R18/R19 candidate vertices in each replay')


def plan_main(plan):
    proof = plan['proof']
    run = Path(plan['new']['run'])
    manifest = read(pin(proof['protected_face_main_bound-replay']['file'],proof['protected_face_main_bound-replay']['sha256']))
    terminal = read(pin(run/'result.json',plan['new']['terminal_sha256']))
    _, final_replay_path, final = round1.delivery_paths(run,terminal)
    final_replay = read(final_replay_path)
    require(manifest['frames']==final['frames'] and manifest['points']==final['points']
            and manifest['reference_frame_id']==final['reference_frame_id']
            and manifest['source_files_sha256']==final['source_files_sha256']
            and all(manifest['inputs_sha256'][key]==final['inputs_sha256'][key]
                    for key in ('fitted','training_rows')),
            'Main and final bound snapshots use different fit, camera or topology inputs.')
    local = pin(run/MAIN/'bound-local.ply',manifest['bound_ply_sha256'])
    world = pin(run/MAIN/'world.ply',manifest['inputs_sha256']['world_ply'])
    mesh_path = pin(run/MAIN/'reference-mesh.npz',manifest['inputs_sha256']['reference_mesh'])
    with np.load(mesh_path,allow_pickle=False) as main_mesh, \
         np.load(final_replay_path.parents[1]/'reference-mesh.npz',allow_pickle=False) as final_mesh:
        require(set(main_mesh.files)==set(final_mesh.files)
                and all(np.array_equal(main_mesh[key],final_mesh[key]) for key in main_mesh.files),
                'Post-main refinement changed the shared mesh or clothing displacement.')
    # Replay the main and final surfaces on the same frozen body poses. Main
    # Gaussian appearance/shape always comes from its own local snapshot.
    staged = Path(plan['output'])/'main-snapshot'
    plan['main'] = dict(plan['new'],snapshot=str(staged))
    plan['main_snapshot'] = {name:proof[f'protected_face_main_{Path(name).stem}'] for name in SNAPSHOT_FILES}
    plan['main_bound_replay_sha256'] = proof['protected_face_main_bound-replay']['sha256']
    plan['round19_stage_sha256'] = sha(Path(__file__))
    native_positions = r14.PlyData.read(world)['vertex'].data
    xyz = np.stack([native_positions[name] for name in ('x','y','z')],axis=1)
    common_ref = plan['common_reference']
    with np.load(pin(common_ref['file'],common_ref['sha256']),allow_pickle=False) as fit:
        common = r13.common_centers(fit['vertices_world'],fit['faces'],local)
    arm = plan['static_arms']['new']
    plan['static_arms']['main'] = dict(arm,ply=str(world),ply_sha256=sha(world),reference_mesh_sha256=sha(mesh_path),
                                     bound_ply=str(local),bound_ply_sha256=sha(local),world_ply=str(world),world_ply_sha256=sha(world),
                                     world_to_viewer=dict(origin_world=[0.,0.,0.],rotation=np.eye(3).tolist(),scale=1.))
    plan['terminal_sh0_rgb_range']['main'] = r14.dc_range(local)
    require(r14.dc_range(world)==plan['terminal_sh0_rgb_range']['main'],'Main raw SH0 color and alpha differ.')
    for spec in plan['static_views'].values():
        camera = spec['fixed_world_camera']
        E = np.asarray(camera['w2c'])
        depth, common_depth = (float((points@E[2,:3]+E[2,3]).min()) for points in (xyz,common))
        require(min(depth,common_depth)>.2,'Main stage crosses a fixed static CUDA near plane.')
        spec['arms']['main'] = dict(camera=camera,min_center_depth_m=depth)
        spec['common_min_center_depth_m']['main'] = common_depth
    for row in plan['pairs']:
        row['main_slot'],row['main_pose_sha256'] = row['new_slot'],row['new_pose_sha256']
    plan['main_near_min_depth'] = {}
    with np.load(pin(final_replay_path.parent/final_replay['mesh_file'],final_replay['mesh_sha256']),allow_pickle=False) as posed:
        for role,spec in plan['near_views'].items():
            slot = spec['slot']
            require(posed['frame_ids'][slot].item()==spec['frame'],'Frozen final pose order differs from the dolly.')
            points = r13.common_centers(posed['vertices_world'][slot],posed['faces'],local)
            E = np.asarray(spec['near_camera']['w2c'])
            depth = float((points@E[2,:3]+E[2,3]).min())
            require(depth>.2,'Main stage crosses the CUDA dolly cutoff.')
            plan['main_near_min_depth'][role] = depth


def capture_native(path, side, fd, worker):
    """Retain native same-frame black, white, and alpha bytes for both stages."""
    plan = read(path)
    out = path.parent/f'native-stage-{side}'
    out.mkdir(exist_ok=False)
    frames = iter(plan['matched_frames'])
    entries = []

    def capture(*args,**kwargs):
        frame = next(frames)
        pixels,error = scale_pixels(*args,**kwargs)
        record = dict(frame=frame,files={})
        for name,field in zip(('black','white','alpha'),pixels,strict=True):
            data = np.rint(np.clip(field,0,1)*255).astype(np.uint8)
            target = out/f'{frame}-{name}.png'
            Image.fromarray(data).save(target,compress_level=1)
            record['files'][name] = descriptor(target)
        entries.append(record)
        return pixels,error

    rebound(worker,scale_pixels=capture)(path,side,fd)
    require([row['frame'] for row in entries]==plan['matched_frames'],'Stage native replay missed a named view.')
    review.save(out/'result.json',dict(records=entries,exact_native_no_resize=True))


def generate_main_pose_replay(path, plan, fd):
    info = plan['main']
    run = Path(info['run'])
    stage = Path(info['snapshot'])
    stage.mkdir(exist_ok=False)
    for name,item in plan['main_snapshot'].items():
        shutil.copyfile(pin(item['file'],item['sha256']),stage/name)
        pin(stage/name,item['sha256'])
    require(os.path.samestat(os.fstat(fd),review.GPU_LEASE.stat()),'Missing inherited GPU lease.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(Path(info['project'])/'src'))
    from shuolee_portrait import head_appearance, runtime as rt
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import replay_official_bound

    rt.configure(Path(plan['asset']).parents[1])
    with gpu_lock(),patch.object(head_appearance,'snapshot_name',return_value=os.path.relpath(stage,run/'gaussian-run')):
        receipt = replay_official_bound(run)
    require(receipt['bound_replay_sha256']==plan['main_bound_replay_sha256']
            and [row['frame'] for row in receipt['records']]==plan['matched_frames'],
            'The independent main-stage pose replay loaded a different bound snapshot.')


def render_main(path,fd):
    plan = read(path)
    pin(Path(__file__),plan['round19_stage_sha256'])
    generate_main_pose_replay(path,plan,fd)

    def delivered(run,asset,rows,holdouts):
        from shuolee_portrait.bound_replay import _inputs

        snapshot = Path(plan['main']['snapshot'])
        bound = read(pin(snapshot/'bound-replay.json',plan['main_bound_replay_sha256']))
        pin(snapshot/'bound-local.ply',bound['bound_ply_sha256'])
        replay = read(snapshot/'posed-replay/result.json')
        require(replay['bound_replay_sha256']==plan['main_bound_replay_sha256']
                and replay['retained_holdouts']==holdouts
                and [item['frame'] for item in replay['records']]==[row['frame'] for row in rows],
                'Main replay identity differs from the selected camera roster.')
        fit,mesh,facts = _inputs(run,snapshot,asset,bound)
        return SimpleNamespace(snapshot=snapshot,fit=fit,mesh=mesh,facts=facts,bound=bound,replay=replay)

    def dolly_plan(current):
        data = read(current)
        if current==path:
            for role,view in data['near_views'].items():
                view['cpu_baseline_min_near_center_depth_m']['R16'] = plan['main_near_min_depth'][role]
        return data

    override = SimpleNamespace(delivered=delivered)
    actual = rebound(prior.r10.render_worker,round1=override)
    # The standard actual worker adds a separate unclipped static receipt.
    rebound(r16.actual_worker,r10=SimpleNamespace(render_worker=actual))(path,'main',fd)
    capture_native(path,'main',fd,rebound(r16.native_float_worker,round1=override))
    r16.render_common_worker(path,'main',fd)
    rebound(dolly.render,read=dolly_plan,round1=override)(path,'main',fd)
    rebound(auxiliary.render,round1=override)(path,'main',fd)


def native_inventory(out,plan,main,final):
    inventories = {}
    for side in ('main','new'):
        folder = out/f'native-stage-{side}'
        path = folder/'result.json'
        receipt = read(path)
        require(receipt['exact_native_no_resize'] is True
                and [row['frame'] for row in receipt['records']]==plan['matched_frames'],
                'Main/final archived stage images missed an exact native frame.')
        score = main['float_native'] if side=='main' else final['float_native']
        for row in receipt['records']:
            require(set(row['files'])=={'black','white','alpha'},'Native RGB/alpha inventory is incomplete.')
            for name,item in row['files'].items():
                path_png = pin(item['file'],item['sha256'])
                require(path_png.parent==folder and path_png.name==f'{row["frame"]}-{name}.png',
                        'Stage native image path changed.')
                with Image.open(path_png) as image:
                    require(list(image.size)==score[row['frame']]['camera']['wh']
                            and image.mode==('L' if name=='alpha' else 'RGB'),
                            'Native stage image dimensions or channels changed.')
        inventories[side] = dict(manifest=descriptor(path),records=receipt['records'])
    pair = {}
    for name in plan['matched_frames']:
        before,after = main['float_native'][name],final['float_native'][name]
        require(all(before[key]==after[key] for key in ('camera','source_sha256','mask_sha256','reliable_foreground_pixels')),
                'Main and final native scores are not from the same camera and pixels.')
        pair[name] = dict(rgb_changed=before['black_float_sha256']!=after['black_float_sha256'],
                          alpha_changed=before['alpha_float_sha256']!=after['alpha_float_sha256'],
                          foreground_rgb_mae_delta=after['foreground_rgb_mae_raw']-before['foreground_rgb_mae_raw'],
                          foreground_hole_fraction_delta=after['foreground_hole_fraction_at_half']-before['foreground_hole_fraction_at_half'],
                          exterior_spill_pixels_delta=after['exterior_spill_pixels_at_half']-before['exterior_spill_pixels_at_half'])
    return inventories,pair


def stage_panels(out,main,final):
    panels = []
    require(main['panels'].keys()==final['panels'].keys(),'Main and final panel inventories differ.')
    for role,before in main['panels'].items():
        if not (role.endswith('_head') or role.startswith(('face_only_','physical_near_','dark_','common_'))):
            continue
        after = final['panels'][role]
        require(all(before[key]==after[key] for key in ('frame','columns','crop_xyxy')),
                'A protected comparison panel changed its pixels, crop or pose.')
        with Image.open(pin(out/before['file'],before['sha256'])) as image_a, \
             Image.open(pin(out/after['file'],after['sha256'])) as image_b:
            width,height = image_a.size
            require(image_b.size==(width,height),'Stage panel sizes differ.')
            image = Image.new('RGB',(width,2*height))
            image.paste(image_a.convert('RGB'),(0,0))
            image.paste(image_b.convert('RGB'),(0,height))
        file = out/f'stage_{role}-{before["frame"]}.png'
        image.save(file)
        panels.append(dict(view='stage_'+role,file=file.name,sha256=sha(file),frame=before['frame'],
                           new_frame=after['frame'],columns=before['columns'],crop_xyxy=before['crop_xyxy'],
                           rows=['round19_main','round19_protected_final']))
    return panels


def assemble_stage(out,plan,final):
    main = read(out/'main-result.json')
    for suffix in ('common','near','supplement'):
        receipt = read(out/f'main-{suffix}.json')
        main['panels'].update(receipt.pop('panels'))
        main.update(receipt)
    main['float_native'] = read(out/'main-float-native.json')
    main['float_actual'] = read(out/'main-float-actual.json')
    auxiliary.verify_render_receipts(plan,dict(main=main,final=final))
    for role,spec in plan['near_views'].items():
        before,after = (arm['physical_near'][role] for arm in (main,final))
        require(all(before[key]==after[key] for key in ('camera','pose_slot','projection_matrix_znear_m','actual_raster_near_cutoff_m'))
                and before['camera']['w2c']==spec['near_camera']['w2c'],
                'Main and final close cameras differ.')
    for pair,before,after in zip(plan['pairs'],main['native'],final['native'],strict=True):
        require(all(before[key]==after[key] for key in ('frame','native_camera','pose_sha256','source_sha256',
                    'score_mask_sha256','own_mask_sha256','fresh_score_mask_sha256','posed_mesh_silhouette'))
                and before['frame']==pair['frame'] and before['native_archive_max_abs_8bit']<=1,
                'The main/final native regional evidence changed a pose or source identity.')
    for name,spec in plan['static_views'].items():
        a,b = (arm['common'][name] for arm in (main,final))
        require(all(a[key]==b[key] for key in ('camera','common_mesh_sha256','fixed_world_camera'))
                and a['fixed_world_camera']==spec['fixed_world_camera'],
                'The main/final common static geometry or camera changed.')
    r13.check_source_pixels(out,plan,(main,final))
    archive,pairs = native_inventory(out,plan,main,final)
    audit_item = plan['proof']['protected_face_optimization']
    audit = read(pin(audit_item['file'],audit_item['sha256']))
    if audit['state']=='rolled_back':
        require(all(not row['rgb_changed'] and not row['alpha_changed'] for row in pairs.values())
                and all(main['supplement_native'][frame]['foreground_alpha']==row['foreground_alpha']
                        and main['supplement_native'][frame]['foreground_rgb_mae']==row['foreground_rgb_mae']
                        for frame,row in final['supplement_native'].items())
                and all(main['physical_near'][key]['float_rgb']==row['float_rgb']
                        for key,row in final['physical_near'].items()),
                'Independent main/final native or dolly replay changed after a rollback.')
    panels = stage_panels(out,main,final)
    return dict(main=main,all_native_images=archive,native_pair_deltas=pairs,
                main_pose_replay=descriptor(Path(plan['main']['snapshot'])/'posed-replay/result.json'),
                optimizer_audit=audit_item,decision=audit['state'],
                rollback_native_pixels_bitwise_equal=audit['state']=='rolled_back',
                no_refit=True,exact_full_body_views=len(pairs),exact_face_views=len(plan['supplement']['frames']),
                same_camera_masks_and_pose=True),panels


def check_holdouts(run,holdout_frames):
    raw_path = run/'data/inputs.json'
    raw = {row['frame']:row for row in read(raw_path)['records']}
    fit_path = run/'fit/joint-candidates.npz'
    selected = run/'fit/fitted.npz'
    with np.load(fit_path,allow_pickle=False) as candidates,np.load(selected,allow_pickle=False) as training:
        seen = set(candidates['frame_ids'].tolist()) | set(training['frame_ids'].tolist())
    require(len(set(holdout_frames))==4 and not seen.intersection(holdout_frames),
            'The four locked holdouts now contain fit records. Update the independent-review plan before rendering.')
    frames = []
    for frame in holdout_frames:
        row = raw[frame]
        original = pin(row['image'],row['source_sha256'])
        mask = pin(row['mask'],row['mask_sha256'])
        require(row['camera_kind']=='camera_relative_fixedK_body_motion'
                and row['camera_registration']['static_trajectory_claim'] is False,
                'The frozen heldout camera domain changed.')
        frames.append(dict(frame=frame,source_frame_index=row['source_frame_index'],source_sha256=sha(original),
             mask_sha256=sha(mask),camera_kind=row['camera_kind'],conditional_intrinsics=row['K'],
             image_wh=row['wh'],frozen_fitted_pose=False,gaussian_replay_available=False))
    return dict(frames=frames,raw_native_index=descriptor(raw_path),joint_fit_inventory=descriptor(fit_path),
                named_training_fit=descriptor(selected),independent_bound_replay_available=False,
                scope='Source RGB, mask, conditional K and 2D observations exist; fitted holdout poses are absent.')


def garment_audit(out,plan,arms):
    """Score original native shoulder and torso band with fixed reliable masks."""
    from scipy.ndimage import label

    cfg = plan['brightness_contract']
    luma = np.asarray(cfg['encoded_luma_rgb_weights'],dtype=np.float32)
    results = {}
    for pair in plan['pairs']:
        role = pair['role']
        x0,y0,x1,y1 = pair['crop_xyxy']
        width,height = x1-x0,y1-y0
        with Image.open(pin(pair['fresh_score_mask'],pair['fresh_score_mask_sha256'])) as matte, \
             Image.open(pin(pair['source'],pair['source_sha256'])) as source:
            mask = np.asarray(matte.convert('L').crop(pair['crop_xyxy']))
            native_rgb = np.asarray(source.convert('RGB').crop(pair['crop_xyxy']))
        ys = np.flatnonzero((mask>=253).any(axis=1))
        require(len(ys)>0,'A fixed native clothing band has no source foreground.')
        lo,hi = np.rint(ys[0]+(ys[-1]-ys[0]+1)*np.asarray(cfg['person_height_fraction'])).astype(int)
        band = (np.arange(height)[:,None]>=lo) & (np.arange(height)[:,None]<hi)
        fg,bg = band & (mask>=253),band & (mask<=2)
        require(fg.any() and bg.any(),'A fixed clothing band lacks reliable interior or exterior.')
        item = dict(frame=pair['frame'],role=role,crop_xyxy=pair['crop_xyxy'],absolute_y_band=[y0+int(lo),y0+int(hi)],
                    fresh_source_mask_sha256=pair['fresh_score_mask_sha256'],foreground_pixels=int(fg.sum()),
                    exterior_pixels=int(bg.sum()),scope='native torso-height foreground, no garment semantic label',arms={})
        for side,arm in arms.items():
            panel = arm['panels'][role]
            require(panel['crop_xyxy']==pair['crop_xyxy'] and panel['columns']==['source_rgb','black','white','alpha'],
                    'Fixed clothing band image layout changed.')
            with Image.open(pin(out/panel['file'],panel['sha256'])) as image:
                data = np.asarray(image.convert('RGB'))
            require(data.shape==(height,width*4,3) and np.array_equal(data[:,:width],native_rgb),
                    'Fixed clothing band differs from the original pixels.')
            black,alpha = data[:,width:2*width],data[:,3*width:,0]
            holes = fg & (alpha<128)
            excess = fg & (alpha>=cfg['opaque_alpha_u8']) & (black@luma>=cfg['bright_luma_u8']) \
                         & (native_rgb@luma<=cfg['native_source_luma_ceiling_u8'])
            labels,count = label(excess,np.ones((3,3),dtype=bool))
            item['arms'][side] = dict(rgb_mae=float(np.abs(black[fg].astype(np.float32)-native_rgb[fg]).mean()/255),
                                      alpha_half_hole_pixels=int(holes.sum()),alpha_half_hole_fraction=float(holes.sum()/fg.sum()),
                                      alpha_half_spill_pixels=int((bg&(alpha>=128)).sum()),
                                      excess_bright_pixels=int(excess.sum()),
                                      largest_excess_bright_component=int(np.bincount(labels.ravel())[1:].max()) if count else 0)
        results[role] = item
    return results
