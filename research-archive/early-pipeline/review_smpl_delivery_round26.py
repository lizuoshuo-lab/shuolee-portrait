"""Compare native face pixels and fixed cameras against the verified R22 model."""

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round22 as qa


pin,checked,read,sha,require,descriptor,rebound = qa.pin,qa.checked,qa.read,qa.sha,qa.require,qa.descriptor,qa.rebound
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r26')
BASELINE_SHA = 'c0f1b8c35c5c645deb64db1700486aedc20e7aa56d20de2abc079294242ae924'
DELIVERY = qa.launch.DELIVERY.with_name('smpl-delivery-round26')
NEW = {'r26_10':10000,'r26':18000}
ARMS = ('r22',*NEW)
LABELS = ('R22 18,000','R26 10,000','R26 18,000')
FACE_LANDMARKS = {'right_eye':range(36,42),'left_eye':range(42,48),
                  'nose':range(27,36),'lips':range(48,68)}
PRODUCTION_ROLES = {'selected_fit':'training','holdout':'pose_conditioned_holdout','auxiliary_face':'face_only'}


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round26-{subject}'


def native_crop(points,weights,size,padding,minimum,within=None):
    supported = np.isfinite(points).all(axis=1)&np.isfinite(weights)&(weights>=.5)
    selected = points[supported]
    if len(selected)<minimum:
        return dict(box=None,reliable_detector_points=len(selected))
    low,high = selected.min(axis=0),selected.max(axis=0)
    extent = high-low
    if (extent<2).any():
        return dict(box=None,reliable_detector_points=len(selected))
    pad = np.maximum(extent*padding,5)
    x0,y0,x1,y1 = within if within is not None else [0,0,*size]
    start = np.maximum(np.floor(low-pad).astype(int),[x0,y0])
    end = np.minimum(np.ceil(high+pad).astype(int),[x1,y1])
    box = [int(v) for v in (*start,*end)] if (end>start).all() else None
    return dict(box=box,reliable_detector_points=len(selected))


def source_rois(observations,rows):
    with np.load(observations,allow_pickle=False) as archive:
        names,frames = archive['joint_names'].tolist(),archive['frame_ids'].tolist()
        points,weights = archive['keypoints'].copy(),archive['scores'].copy()
    indices = np.asarray([names.index(f'face-{index}') for index in range(68)])
    result = {}
    for row in rows:
        frame = row['frame']
        slot = frames.index(frame)
        xy,score = points[slot,indices],weights[slot,indices]
        # Original detector positions establish fixed pixel windows, with no
        # displacement fitting or model-driven crop adjustment.
        overall = native_crop(xy,score,row['wh'],.35,8,within=row.get('verified_box'))
        parts = {name:native_crop(xy[group],score[group],row['wh'],.35,3,within=overall['box'])
                 for name,group in FACE_LANDMARKS.items()}
        result[frame] = dict(face=overall,parts=parts,
            scope='Native detector crop. Landmark visibility and physical identity require source review.')
    return result


def history(subject):
    index = pin(qa.launch.DELIVERY/'review-receipt.json',BASELINE_SHA)
    signed = read(index)['subjects'][subject]
    old_review_path = checked(signed['review'])
    old = read(old_review_path)
    old_plan = read(checked(old['plan']))
    require(old['state']=='paired_native_and_static_views_rendered' and old['commit']==old_plan['commit']==qa.COMMIT
            and old['subject']==old_plan['subject']==subject and old['static_camera_count']==50,
            'R26 requires the exact published R22 camera controls.')
    pose = old_plan['poses']['r22']
    control = old_plan['static_grid']['arms']['r22']
    pin(control['ply'],control['ply_sha256'])
    pin(control['camera_path'],control['camera_sha256'])
    grid = dict(old_plan['static_grid'],arms={'r22':control})
    for entry in grid['proof'].values():
        checked(entry)
    static = old['static']['r22']['receipt']
    qa.grid_records(old_plan,'r22',checked(static))

    run = Path(pose['new']['run'])
    source = pin(run/'data/training-inputs.json',pose['new']['training_inputs_sha256'])
    fit = pin(run/'fit/fitted.npz',pose['new']['fitted_sha256'])
    observed = read(checked(pose['proof']['head_alignment_holdout_plan']))
    observation_path = pin(run/'data/observations.npz',observed['observations_sha256'])
    selected = read(source)['records']
    terminal_path = pin(run/'result.json',pose['new']['terminal_sha256'])
    terminal = read(terminal_path)
    replay = pin(run/terminal['bound_replay'],terminal['bound_replay_sha256'])
    replay_meta = read(replay)
    selected_mesh = descriptor(pin(replay.parent/replay_meta['mesh_file'],replay_meta['mesh_sha256']))

    def native(row,role,slot):
        cam = row.get('camera',row)
        if role=='selected_fit':
            source,mask = row['image'],row['mask']
        else:
            source,mask = row['source'],row['source_mask']
        source_path = pin(source,row['source_sha256'])
        mask_path = pin(mask,row['mask_sha256'])
        return dict({key:cam[key] for key in ('frame','K','wh','w2c')},role=role,pose_index=slot,
            image=descriptor(source_path),mask=descriptor(mask_path),
            verified_box=row.get('box') if role=='auxiliary_face' else None,
            verified_face_mask=descriptor(pin(row['valid_path'],row['valid_mask_sha256'])) if role=='auxiliary_face' else None)

    holdout = pose['holdout']['frames']
    auxiliary = pose['supplement']['frames']
    require((len(selected),len(holdout),len(auxiliary))==(24,4,4),'R22 native roster differs from 24 + 4 + 4.')
    rows = [native(row,role,slot) for role,group in
            (('selected_fit',selected),('holdout',holdout),('auxiliary_face',auxiliary))
            for slot,row in enumerate(group)]
    require(len({row['frame'] for row in rows})==32,'Native selected, held-out and auxiliary source roles overlap.')
    meshes = dict(selected_fit=selected_mesh,holdout=pose['proof']['heldout_mesh'],auxiliary_face=pose['proof']['supplement_poses'])
    topology = None
    for role,item in meshes.items():
        with np.load(checked(item),allow_pickle=False) as archive:
            require(archive['frame_ids'].tolist()==[row['frame'] for row in rows if row['role']==role],
                    'The frozen native pose geometry roster differs from its camera roster.')
            if topology is None:
                topology = archive['faces'].copy()
            else:
                require(np.array_equal(topology,archive['faces']),'The frozen native pose topologies differ.')
    return dict(subject=subject,baseline_review=descriptor(old_review_path),baseline_index=descriptor(index),
        baseline_archive=signed['archive'],asset=old_plan['asset'],static_grid=grid,static_archive={'r22':static},
        baseline_pose=pose,frozen=dict(rows=rows,meshes=meshes,fit=descriptor(fit),observations=descriptor(observation_path),
                                      selected=descriptor(source),reference_frame=pose['new']['reference_frame_id']),
        rois=source_rois(observation_path,rows))


def archive_members(path,expected):
    with ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names)==len(set(names)) and set(expected)<=set(names) and archive.testzip() is None,
                'A delivered R26 package is incomplete or corrupt.')
        for name,digest in expected.items():
            require(sha256(archive.read(name)).hexdigest()==digest,f'Changed R26 package entry {name}.')


def scan_snapshot(run,endpoint,frozen,grid):
    step = endpoint['step']
    snapshot = run/'gaussian-run'/endpoint['path']
    require(snapshot.name==f'raw-step-{step:06d}' and read(snapshot/'result.json')==endpoint,
            'R26 endpoint does not match its own step receipt.')
    state = pin(snapshot/'main-state.pt',endpoint['state_sha256'])
    drive = snapshot/'drive'
    spec_path = drive/'drive.json'
    spec = read(spec_path)
    rows = frozen['rows']
    require(spec['schema']=='shuolee-face-bound-v1' and spec['sh_degree']==0
            and [row['frame'] for row in spec['cameras']]==[row['frame'] for row in rows]
            and all(all(camera[key]==row[key] for key in ('frame','wh','K','w2c'))
                    and camera['pose_index']==slot and camera['pixel_convention']=='pixel_corners'
                    for slot,(camera,row) in enumerate(zip(spec['cameras'],rows,strict=True)))
            and spec['roles']==[PRODUCTION_ROLES[row['role']] for row in rows]
            and rows[spec['reference_pose_index']]['frame']==frozen['reference_frame'],
            'The R26 native drive camera, pose or frame order differs from R22.')
    local = pin(snapshot/'bound-local.ply',spec['files']['bound-local.ply'])
    pin(drive/local.name,sha(local))
    motion = pin(drive/'motion.npz',spec['files']['motion.npz'])
    with np.load(motion,allow_pickle=False) as data:
        require(data['frame_ids'].tolist()==[row['frame'] for row in rows]
                and len(data['stable_ids'])==len(data['labels'])==len(data['spill'])==endpoint['points'],
                'R26 native drive frame IDs or per-point lineage differs.')
        cursor = 0
        for role,proof in frozen['meshes'].items():
            with np.load(checked(proof),allow_pickle=False) as mesh:
                length = len(mesh['frame_ids'])
                require(np.array_equal(mesh['faces'],data['faces'])
                        and np.array_equal(mesh['vertices_world'],data['vertices_world'][cursor:cursor+length]),
                        f'R26 changed the registered {role} motion mesh.')
                cursor += length
        require(cursor==len(data['vertices_world'])==32,'A R26 native pose lacks exact source geometry.')
    portable_path = snapshot/'portable-result.json'
    portable = read(portable_path)
    ply = pin(snapshot/portable['ply'],portable['sha256'])
    camera = pin(snapshot/portable['camera'],portable['camera_sha256'])
    scene = pin(snapshot/portable['scene'],portable['scene_sha256'])
    world = pin(snapshot/'world.ply',portable['source_world_sha256'])
    provenance = read(camera.with_name('camera_provenance.json'))
    parity = read(snapshot/'native-ply-parity.json')
    require(portable['state']=='completed' and portable['step']==step and portable['format_passed']
            and parity['passed'] and parity['sha256']==sha(world) and parity['reference_frame_id']==frozen['reference_frame']
            and parity['reference_pose_index']==spec['reference_pose_index']
            and provenance['files_sha256']['cameras.json']==sha(camera) and provenance['cameras_share_model_world'],
            'R26 portable PLY, reference pose and camera metadata disagree.')
    pin(snapshot/'reference-mesh.npz',parity['mesh_sha256'])
    package = pin(snapshot/endpoint['drive_package'],endpoint['drive_package_sha256'])
    archive_members(package,{'drive/drive.json':sha(spec_path),'reference/scene.zip':sha(scene),
                            **{'drive/'+name:digest for name,digest in spec['files'].items()}})
    with ZipFile(scene) as archive:
        names = {Path(name).name:name for name in archive.namelist()}
    archive_members(scene,{names['cameras.json']:sha(camera),names[ply.name]:sha(ply)})
    transform = provenance['world_to_viewer']
    native_data = PlyData.read(world)['vertex'].data
    viewer_data = PlyData.read(ply)['vertex'].data
    xyz_native = np.column_stack([native_data[key] for key in ('x','y','z')])
    xyz_viewer = np.column_stack([viewer_data[key] for key in ('x','y','z')])
    require(transform['scale']==1. and len(xyz_native)==len(xyz_viewer)==endpoint['points']==portable['points'],
            'R26 static preview changed world scale or Gaussian row count.')
    np.testing.assert_allclose(xyz_viewer@np.asarray(transform['rotation'])+np.asarray(transform['origin_world']),
                               xyz_native,rtol=0,atol=2e-6)

    replay,receipts = {},{}
    for folder,expected in (('drive-replay',rows[:28]),('face-supplement',rows[28:])):
        path = snapshot/folder/'result.json'
        meta = read(path)
        require(meta['optimizer_updates']==0 and not meta['quality_accepted']
                and [row['frame'] for row in meta['records']]==[row['frame'] for row in expected],
                'R26 native delivery replay used the wrong training, holdout or auxiliary role.')
        receipts[folder] = descriptor(path)
        for row,record in zip(expected,meta['records'],strict=True):
            require(record['role']==PRODUCTION_ROLES[row['role']] and record['exact_geometry_roundtrip']
                    and record['camera']==spec['cameras'][record['pose_index']]
                    and record['box']==record['box_xyxy'] and set(record['files'])==set(record['files_sha256']),
                    'R26 native delivery replay camera or crop differs from its package.')
            media = {name:descriptor(pin(path.parent/file,record['files_sha256'][name]))
                     for name,file in record['files'].items()}
            replay[row['frame']] = dict(record,files=media)
    arm = dict(ply=str(ply),ply_sha256=sha(ply),camera_path=str(camera),camera_sha256=sha(camera),
        world=descriptor(world),local=descriptor(local),world_to_viewer=transform,ref=frozen['reference_frame'],
        project=str(PROJECT),run=str(run),model_class='actual_delivered_static_portable_ply')
    return qa.project_static(arm,grid),dict(run=str(run),snapshot=str(snapshot),project=str(PROJECT),
        drive=descriptor(spec_path),package=descriptor(package),portable=descriptor(portable_path),scene=descriptor(scene),
        checkpoint=descriptor(state),replay=replay,replay_receipts=receipts,step=step,points=endpoint['points'])


def verify_trajectory(snapshots,config):
    from audit_r23_numerical import read_state

    draws,events = None,None
    for name,step in NEW.items():
        state = read_state(checked(snapshots[name]['checkpoint']))
        require(state['step']==step==len(state['draws']) and state['pinned_inputs']==config['pinned_inputs']
                and all(isinstance(slot,int) and 0<=slot<24 for slot in state['draws']),
                'R26 optimizer state has different training inputs or draw count.')
        if draws is not None:
            require(state['draws'][:len(draws)]==draws and state['split_events'][:len(events)]==events,
                    'R26 10,000 and 18,000 steps did not use one uninterrupted training trajectory.')
        draws,events = state['draws'],state['split_events']


def plan(subject,result_hash,commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Use the full R26 source commit.')
    pointer = (PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['/usr/bin/wslpath','-u',pointer],text=True).strip()
    git = ['/usr/bin/git','--git-dir',git_dir,'--work-tree',str(PROJECT)]
    current = subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()
    require(current==commit,'Use the exact R26 worktree.')
    implementation = {}
    for part in ('bundle','geometry','split','training'):
        filename = f'face_surface_{part}.py'
        relative = f'src/shuolee_portrait/{filename}'
        content = subprocess.check_output([*git,'show',f'{commit}:{relative}'])
        implementation[filename] = descriptor(pin(PROJECT/relative,sha256(content).hexdigest()))
    run = root_for(subject)/'run'
    target = pin(run/'gaussian-run/result.json',result_hash)
    terminal = read(target)
    progress = run/'progress.json'
    require(terminal['state']=='checkpoint_verified' and not terminal['quality_accepted']
            and terminal['steps']==18000 and read(progress)['state']=='completed'
            and [row['step'] for row in terminal['endpoints']]==list(NEW.values()),
            'Independent R26 review requires both completed same-run endpoints.')
    baseline = history(subject)
    frozen,grid = baseline['frozen'],baseline['static_grid']
    old = baseline['baseline_pose']['new']
    cfg_path = run/'gaussian-run/config.json'
    config = read(cfg_path)
    pin(run/'data/training-inputs.json',frozen['selected']['sha256'])
    pin(run/'fit/fitted.npz',frozen['fit']['sha256'])
    for role,key in (('selected_fit','train_frames'),('holdout','holdout_frames'),('auxiliary_face','auxiliary_frames')):
        require(config[key]==[row['frame'] for row in frozen['rows'] if row['role']==role],
                'The R26 training config changed the original native roster.')
    require(config['execution']=='main' and config['auxiliary_frequency']==8
            and sum(terminal['auxiliary_updates'].values())==18000//8
            and set(terminal['auxiliary_updates'])==set(config['auxiliary_frames']),
            'The R26 run does not have the registered main and auxiliary draw schedule.')
    snapshots = {'r22':dict(run=old['run'],snapshot=str(Path(old['run'])/'gaussian-run/step-018000'),
                           project=old['project'])}
    for name,item in zip(NEW,terminal['endpoints'],strict=True):
        grid['arms'][name],snapshots[name] = scan_snapshot(run,item,frozen,grid)
    verify_trajectory(snapshots,config)
    return dict(baseline,schema='shuolee-r26-native-face-review/1',commit=commit,result=descriptor(target),
        snapshots=snapshots,training_config=descriptor(cfg_path),training_progress=descriptor(progress),
        split_events=descriptor(run/'gaussian-run/split-events.json'),implementation=implementation,
        script=descriptor(Path(__file__)),quality_accepted=False)


def native_scores(source,valid,pixels,roi):
    report = dict(roi)
    if roi['box'] is None:
        return report
    x0,y0,x1,y1 = roi['box']
    sub = np.s_[y0:y1,x0:x1]
    mask = valid[sub]
    black,_,alpha = pixels
    values = alpha[sub][mask]
    report.update(foreground_pixels=int(mask.sum()),low_alpha_095_pixels=int((values<.95).sum()),
                  foreground_rgb_mae=float(np.abs(black[sub][mask]-source[sub][mask]/255.).mean())
                  if len(values) else None,
                  foreground_mean_transmission=float((1.-values).mean(dtype=np.float64)) if len(values) else None)
    return report


def native_media(directory,frame,crop,pixels):
    x0,y0,x1,y1 = crop
    saved = {}
    for key,value in zip(('black','white','alpha'),pixels,strict=True):
        path = directory/f'{frame}-{key}.png'
        array = np.rint(np.clip(value[y0:y1,x0:x1],0,1)*255.).astype(np.uint8)
        with path.open('xb') as stream:
            Image.fromarray(array).save(stream,format='PNG',compress_level=1)
        saved[key] = descriptor(path)
    return saved


def check_native_pixels(production,row,camera,pixels,source,mask,valid):
    if production is None:
        return None
    require(production['source_sha256']==row['image']['sha256']
            and production['mask_sha256']==row['mask']['sha256']
            and all(production['camera'][key]==camera[key] for key in ('K','wh','w2c','frame')),
            'Independent R26 native replay changed the exact input raster, mask or camera.')
    x0,y0,x1,y1 = production['box']
    sub = np.s_[y0:y1,x0:x1]
    inputs = [('source','RGB',source),('mask','L',mask)]
    if row['role']=='auxiliary_face':
        inputs.append(('valid','L',valid.astype(np.uint8)*255))
    for field,mode,expected in inputs:
        with Image.open(checked(production['files'][field])) as image:
            require(np.array_equal(np.asarray(image.convert(mode)),expected[sub]),
                    'The delivered R26 replay changed the original source, mask or verified auxiliary pixels.')
    errors = {}
    for field,value in zip(('black','white','alpha'),pixels,strict=True):
        with Image.open(checked(production['files'][field])) as png:
            recorded = np.asarray(png.convert('L' if field=='alpha' else 'RGB'),dtype=np.int16)
        crop = np.rint(np.clip(value[sub],0,1)*255.).astype(np.int16)
        require(crop.shape==recorded.shape,'R26 delivered native crop differs from its declared box.')
        errors[field] = int(np.max(np.abs(crop-recorded)))
        require(errors[field]<=1,'Independent native R26 replay differs from the delivered raster.')
    return errors


def replay_native(path,current,name,fd):
    require(os.path.samestat(os.fstat(fd),qa.review.GPU_LEASE.stat()),'Missing inherited GPU lease.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    arm = current['snapshots'][name]
    project = Path(arm['project'])
    sys.path.insert(0,str(project/'src'))
    import shuolee_portrait
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera,render_pose

    require(Path(shuolee_portrait.__file__).resolve().is_relative_to((project/'src').resolve()),
            'The native review worker loaded another renderer checkout.')
    asset = Path(current['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render

    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = current['frozen']
    saved = read(checked(current['baseline_archive']))['selected_panels']
    output = path.parent/'native'/name
    output.mkdir(parents=True,exist_ok=False)
    pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    results = {}
    with gpu_lock(),torch.no_grad():
        if name=='r22':
            model,manifest = load_bound_snapshot(Path(arm['run']),Path(arm['snapshot']),asset_path=asset)
            require(len(model.vertices_sequence)==24,'Expected all original selected R22 mesh poses.')
            for role in ('holdout','auxiliary_face'):
                with np.load(checked(frozen['meshes'][role]),allow_pickle=False) as mesh:
                    require(np.array_equal(model.faces.cpu().numpy(),mesh['faces']),
                            'The R22 baseline replay changes native pose topology.')
                    model.vertices_sequence = torch.cat((model.vertices_sequence,torch.as_tensor(
                        mesh['vertices_world'],dtype=model.vertices_sequence.dtype,device='cuda')))
            model.num_timesteps = len(model.vertices_sequence)
        else:
            from shuolee_portrait.face_surface_bundle import load_face_drive

            model,manifest = load_face_drive(checked(arm['drive']).parent)
        require(torch.cuda.device_count()==1 and model.num_timesteps==len(frozen['rows']) and manifest['sh_degree']==0,
                'Expected the complete native pose roster and one local SH0 renderer.')
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        for slot,row in enumerate(frozen['rows']):
            frame,role = row['frame'],row['role']
            model.select_mesh_by_timestep(slot)
            cam,actual = camera({key:row[key] for key in ('frame','wh','K','w2c')},slot,pose_index=slot)
            require(all(actual[key]==row[key] for key in ('frame','wh','K','w2c')),'R26 changed the native camera.')
            pixels,error = qa.static.scale_pixels(render,render_pose,cam,model,pipe,black,white,slot,1.)
            with Image.open(checked(row['image'])) as image, Image.open(checked(row['mask'])) as matte:
                source = np.asarray(image.convert('RGB'))
                matte_pixels = np.asarray(matte.convert('L'))
            valid = matte_pixels>=128
            require((source.shape[1],source.shape[0])==tuple(row['wh']) and source.shape[:2]==valid.shape,
                    'R26 decoded a different native input raster.')
            if row['verified_face_mask']:
                x0,y0,x1,y1 = row['verified_box']
                with Image.open(checked(row['verified_face_mask'])) as mask:
                    region = np.asarray(mask.convert('L'))==255
                require(region.shape==(y1-y0,x1-x0),'The native auxiliary face mask changed crop size.')
                valid = np.zeros_like(valid)
                valid[y0:y1,x0:x1] = region
            crops = current['rois'][frame]
            boxes = dict(face=crops['face'],**crops['parts'])
            metrics = {}
            for part,box in boxes.items():
                score = native_scores(source,valid,pixels,box)
                if box['box'] is not None:
                    panel = qa.static.r10.strip(output/f'{frame}-{part}.png',box['box'],(source,*pixels))
                    score['panel'] = descriptor(pin(output/panel['file'],panel['sha256']))
                metrics[part] = score
            body = None
            if role=='selected_fit':
                original = saved[frame]
                require(original['source_sha256']==row['image']['sha256'],'The selected record changed its native source.')
                crop = original['crop_xyxy']
                body = dict(crop_xyxy=crop,files=native_media(output,frame,crop,pixels))
            differences = check_native_pixels(arm.get('replay',{}).get(frame),row,actual,pixels,source,matte_pixels,valid)
            results[frame] = dict(role=role,camera=actual,source_sha256=row['image']['sha256'],
                mask_sha256=row['mask']['sha256'],regions=metrics,body=body,
                native_render_error_8bit=differences,black_white_alpha_composition_max_abs=error)
    qa.review.save(output/'result.json',dict(records=results,native_poses=28,auxiliary_face_poses=4))


def score_summary(native):
    summary = {}
    for role in ('selected_fit','holdout','auxiliary_face'):
        rows = [row for row in native.values() if row['role']==role]
        by_part = {}
        for part in ('face',*FACE_LANDMARKS):
            values = [row['regions'][part] for row in rows
                      if row['regions'][part].get('foreground_rgb_mae') is not None]
            area = sum(v['foreground_pixels'] for v in values)
            by_part[part] = dict(crops=len(values),foreground_pixels=area,
                mean_crop_rgb_mae=float(np.mean([v['foreground_rgb_mae'] for v in values])) if area else None,
                low_alpha_095_fraction=sum(v['low_alpha_095_pixels'] for v in values)/area if area else None)
        summary[role] = dict(frames=len(rows),regions=by_part)
    return summary


def assemble(path,current):
    output = path.parent
    panels_path = output/'paired'
    panels_path.mkdir(exist_ok=True)
    static,native,panels = {},{},{}
    for name in ARMS:
        receipt = (checked(current['static_archive'][name]) if name=='r22' else
                   output/'static-grid'/name/'result.json')
        static[name] = dict(receipt=descriptor(receipt),records=qa.grid_records(current,name,receipt))
        independent = read(output/'native'/name/'result.json')
        require((independent['native_poses'],independent['auxiliary_face_poses'])==(28,4),
                'An R26 native comparison has an incomplete pose roster.')
        native[name] = independent['records']
    for key in current['static_grid']['camera_roster']:
        items = [static[name]['records'][key]['panel'] for name in ARMS]
        require(all((item['crop_xyxy'],item['columns'])==(items[0]['crop_xyxy'],items[0]['columns'])
                    for item in items),'An R26 fixed face camera changed its crop or rendering columns.')
        panels[key] = qa.stack(panels_path/f'{key}.png',[checked(item) for item in items],LABELS)
    for row in current['frozen']['rows']:
        frame = row['frame']
        entries = [native[name][frame] for name in ARMS]
        require(all(all(item['camera'][key]==row[key] for key in ('K','wh','w2c','frame'))
                    and item['role']==row['role'] and item['source_sha256']==row['image']['sha256']
                    and item['mask_sha256']==row['mask']['sha256'] for item in entries),
                'R26 comparison uses a different original view, camera, mask or role.')
        for part in ('face',*FACE_LANDMARKS):
            items = [item['regions'][part] for item in entries]
            require(all((item['box'],item.get('foreground_pixels'))==(items[0]['box'],items[0].get('foreground_pixels'))
                        for item in items),'An R26 paired face region changed its original crop or mask.')
            if items[0]['box'] is not None:
                key = f'native_{frame}_{part}'
                panels[key] = dict(qa.stack(panels_path/f'{key}.png',[checked(item['panel']) for item in items],LABELS),
                                   crop_xyxy=items[0]['box'],frame=frame,role=row['role'],region=part)
    result = dict(state='paired_native_face_and_static_views_rendered',subject=current['subject'],commit=current['commit'],
        static=static,static_camera_count=50,native=native,native_poses=28,auxiliary_face_poses=4,panels=panels,
        native_scores={name:score_summary(native[name]) for name in ARMS},plan=descriptor(path),quality_accepted=False)
    target = output/'result.json'
    if target.exists():
        require(read(target)==result,'Preserve the completed R26 independent review.')
    else:
        qa.review.save(target,result)
    print(json.dumps(descriptor(target)),flush=True)


def worker(path,name,fd):
    current = read(path)
    checked(current['script'])
    for item in current['implementation'].values():
        checked(item)
    if name in NEW and not (path.parent/'static-grid'/name/'result.json').is_file():
        rebound(qa.static.render_grid,PORTABLE=ARMS)(path,name,fd)
    replay_native(path,current,name,fd)


def render(path):
    with qa.review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ARMS:
            if (path.parent/'native'/name/'result.json').is_file():
                continue
            with (path.parent/f'{name}.log').open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),name,str(lease.fileno())],
                               pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-history','check','render','assemble'))
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    parser.add_argument('--result-sha')
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action=='check-history':
        baseline = history(args.subject)
        rows = baseline['frozen']['rows']
        print(json.dumps(dict(subject=args.subject,cameras=len(baseline['static_grid']['camera_roster']),
            face_crops=sum(bool(item['face']['box']) for item in baseline['rois'].values()),
            native={role:[row['frame'] for row in rows if row['role']==role] for role in PRODUCTION_ROLES})),flush=True)
        return
    if not args.result_sha or not args.commit:
        parser.error('Use the actual complete R26 result and source commit.')
    current = plan(args.subject,args.result_sha,args.commit)
    if args.action=='check':
        print(json.dumps(dict(subject=args.subject,cameras=50,native_poses=28,auxiliary_face_poses=4,
                              points={name:current['snapshots'][name]['points'] for name in NEW})),flush=True)
        return
    out = root_for(args.subject)/'view-comparison'
    out.mkdir(exist_ok=True)
    path = out/'plan.json'
    if not path.exists():
        qa.review.save(path,current)
    require(read(path)==current,'The saved R26 independent review plan changed.')
    if args.action=='render':
        render(path)
        print(json.dumps(dict(subject=args.subject,state='cuda_review_released_gpu',plan=descriptor(path))),flush=True)
    assemble(path,current)


if __name__=='__main__':
    main()
