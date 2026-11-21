"""Verify R23 delivered static and driven exports on the existing fixed R22 evaluation."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round22 as qa


pin, checked, read, sha, require, descriptor, rebound = qa.pin, qa.checked, qa.read, qa.sha, qa.require, qa.descriptor, qa.rebound
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-drivable-clothing-20261111')
COMMIT = '7c98ece3831ee7c411e20d2ea7b4c432d15e75f0'
STATIC_BASE = ('r15','r21','r22_10','r22')
NEW = {'r23_10':10000,'r23':18000}
STATIC_ARMS = (*STATIC_BASE,*NEW)
NATIVE_ARMS = ('r22',*NEW)
LABELS = ('R22 18,000','R23 10,000','R23 18,000')
BODY_JOINTS = ('left_shoulder','right_shoulder','left_hip','right_hip')
HAND_JOINTS = ('left_wrist','right_wrist')
IMPLEMENTATION = {
    'drivable_clothing_training.py':'src/shuolee_portrait/drivable_clothing_training.py',
    'drivable_clothing_bundle.py':'src/shuolee_portrait/drivable_clothing_bundle.py',
    'clothing_gaussian_model.py':'src/avatar_core/scene/clothing_gaussian_model.py',
}


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round23-{subject}'


def scan_drive(snapshot, endpoint, frozen, grid):
    step = endpoint['step']
    package = pin(snapshot/endpoint['drive_package'],endpoint['drive_package_sha256'])
    drive = snapshot/'drive'
    manifest_path = drive/'drive.json'
    manifest = read(manifest_path)
    motion = pin(drive/'motion.npz',manifest['files_sha256']['motion.npz'])
    portable_path = snapshot/'portable-result.json'
    portable = read(portable_path)
    ply = pin(snapshot/portable['ply'],portable['sha256'])
    camera = pin(snapshot/portable['camera'],portable['camera_sha256'])
    scene = pin(snapshot/portable['scene'],portable['scene_sha256'])
    provenance = read(camera.with_name('camera_provenance.json'))
    native_world = pin(snapshot/'world.ply',portable['source_world_sha256'])
    parity = read(snapshot/'native-ply-parity.json')
    require(portable['state']=='completed' and portable['step']==step and portable['format_passed']
            and parity['passed'] and parity['sha256']==portable['source_world_sha256']
            and provenance['files_sha256']['cameras.json']==portable['camera_sha256']
            and provenance['cameras_share_model_world']
            and manifest['reference_frame']==frozen['ref']==parity['reference_frame_id']
            and manifest['sh_degree']==0 and manifest['pixel_convention']=='pixel_corners'
            and manifest['clothing_points']+manifest['fixed_points']==endpoint['points']==portable['points'],
            'The R23 portable scene, drive motion, or metric reference changed.')
    pin(snapshot/'reference-mesh.npz',frozen['mesh_sha256'])
    packed = {'drive/drive.json':descriptor(manifest_path),'reference/scene.zip':descriptor(scene)}
    for name,digest in manifest['files_sha256'].items():
        packed['drive/'+name] = descriptor(pin(drive/name,digest))
    with ZipFile(package) as archive:
        require(len(archive.namelist())==len(packed) and set(archive.namelist())==set(packed)
                and archive.testzip() is None,'The complete drive ZIP changed.')
        for name,item in packed.items():
            require(hashlib.sha256(archive.read(name)).hexdigest()==item['sha256'],f'Changed drive package entry: {name}')
    with ZipFile(scene) as archive:
        for name,path in (('cameras.json',camera),(ply.name,ply)):
            member, = [member for member in archive.namelist() if Path(member).name==name]
            require(hashlib.sha256(archive.read(member)).hexdigest()==sha(path),'Static scene ZIP bytes differ.')
    replay_file = snapshot/'drive-replay/result.json'
    replay = read(replay_file)
    names = [row['frame'] for row in frozen['rows']]
    require(replay['optimizer_updates']==0 and not replay['independent_full_clothing_surface_ground_truth']
            and manifest['frame_ids']==names==[row['frame'] for row in replay['records']],
            'The R23 drive replay does not cover the original 24 + 4 frames.')
    with np.load(motion,allow_pickle=False) as arrays, \
         np.load(checked(frozen['selected_mesh']),allow_pickle=False) as selected, \
         np.load(checked(frozen['heldout_mesh']),allow_pickle=False) as heldout:
        require(arrays['frame_ids'].tolist()==names and arrays['reference_index'].item()==names.index(frozen['ref'])
                and np.array_equal(arrays['camera_K'],[row['K'] for row in frozen['rows']])
                and np.array_equal(arrays['camera_w2c'],[row['w2c'] for row in frozen['rows']])
                and np.array_equal(arrays['camera_wh'],[row['wh'] for row in frozen['rows']])
                and arrays['frame_ids'].tolist()==selected['frame_ids'].tolist()+heldout['frame_ids'].tolist()
                and np.array_equal(arrays['faces'],selected['faces'])
                and np.array_equal(arrays['faces'],heldout['faces'])
                and np.array_equal(arrays['vertices_world'][:24],selected['vertices_world'])
                and np.array_equal(arrays['vertices_world'][24:],heldout['vertices_world']),
                'R23 native drive camera or geometry changed from R22.')
    for slot,(record,row) in enumerate(zip(replay['records'],frozen['rows'],strict=True)):
        require(record['pose_index']==slot and record['exact_geometry_roundtrip']
                and record['source_sha256']==row['source_sha256'] and record['mask_sha256']==row['mask_sha256']
                and record['role']==('training' if slot<24 else 'pose_conditioned_holdout')
                and set(record['files'])=={'black','white','alpha'},'A named R23 native pose differs.')
        require(all(np.isfinite(record[key]) and record[key]>=0 for key in
                    ('cloth_center_motion_median_m','root_removed_center_motion_median_m',
                     'root_removed_center_motion_p95_m')),
                'A named pose lacks finite canonical clothing motion evidence.')
    xyz = lambda data: np.column_stack([data[key] for key in ('x','y','z')])
    T = provenance['world_to_viewer']
    world_xyz = xyz(PlyData.read(native_world)['vertex'].data)
    viewer_xyz = xyz(PlyData.read(ply)['vertex'].data)
    require(T['scale']==1. and len(world_xyz)==len(viewer_xyz)==endpoint['points'],
            'R23 reference export changed its geometry scale or row count.')
    np.testing.assert_allclose(viewer_xyz@np.asarray(T['rotation'])+np.asarray(T['origin_world']),
                               world_xyz,rtol=0,atol=2e-6)
    arm = dict(ply=str(ply),ply_sha256=sha(ply),camera_path=str(camera),camera_sha256=sha(camera),
               world=descriptor(native_world),world_to_viewer=provenance['world_to_viewer'],ref=frozen['ref'],
               project=str(PROJECT),run=str(snapshot.parent.parent),model_class='actual_delivered_static_portable_ply')
    return qa.project_static(arm,grid),dict(drive=descriptor(manifest_path),motion=descriptor(motion),
               package=descriptor(package),portable=descriptor(portable_path),scene=descriptor(scene),
               replay=descriptor(replay_file),records=replay['records'],reference_fixed_rows=packed['drive/fixed-bound.ply'],
               points=endpoint['points'],clothing_points=endpoint['free_garment_points'],step=step)


def source_rois(observations, rows):
    with np.load(observations,allow_pickle=False) as archive:
        joint_names, frames = archive['joint_names'].tolist(),archive['frame_ids'].tolist()
        coords, scores = archive['keypoints'].copy(),archive['scores'].copy()
    j = [joint_names.index(name) for name in (*BODY_JOINTS,*HAND_JOINTS)]
    rois = {}
    for row in rows:
        frame = row['frame']
        slot = frames.index(frame)
        coords_row, scores_row = coords[slot,j],scores[slot,j]
        valid = np.isfinite(coords_row).all(axis=1)&np.isfinite(scores_row)&(scores_row>=.5)
        if not valid[:4].all():
            rois[frame] = dict(state='torso_joints_unreliable',box=None)
            continue
        shoulder = coords_row[:2,1].min()
        hip = coords_row[2:4,1].max()
        height = hip-shoulder
        if height<24:
            rois[frame] = dict(state='projected_torso_not_resolved',box=None)
            continue
        w,h = row['wh']
        x0 = max(0,int(np.floor(coords_row[:4,0].min()-.20*height)))
        x1 = min(w,int(np.ceil(coords_row[:4,0].max()+.20*height)))
        y0 = max(0,int(np.floor(shoulder-.06*height)))
        y1 = min(h,int(np.ceil(hip+.10*height)))
        rois[frame] = dict(state='source_torso_roi',box=[x0,y0,x1,y1],
            hands=coords_row[4:][valid[4:]].tolist(),hand_radius_px=max(7.,.13*height),
            face_box=row.get('face_box_xyxy'),
            scope='Visible source torso-band foreground. Clothing and exposed skin may both occur.')
    return rois


def plan(subject, expected):
    old_run = qa.launch.root_for(subject)/'run'
    prior_path = qa.launch.root_for(subject)/'view-comparison/result.json'
    old_review = read(prior_path)
    previous = read(checked(old_review['plan']))
    require(old_review['state']=='paired_native_and_static_views_rendered' and old_review['commit']==qa.COMMIT
            and old_review['frozen_fit_reused'] and old_review['static_camera_count']==50,
            'Complete the paired R22 baseline first.')
    pose = previous['poses']['r22']
    fit_path = pin(old_run/'fit/fitted.npz',pose['new']['fitted_sha256'])
    selected = read(pin(old_run/'data/training-inputs.json',pose['new']['training_inputs_sha256']))['records']
    held = [dict(row,image=row['source'],mask=row['source_mask'])
            for row in pose['holdout']['frames']]
    rows = []
    for row in (*selected,*held):
        native = row.get('camera',row)
        rows.append(dict({key:native[key] for key in ('frame','wh','K','w2c')},
            **{key:row[key] for key in ('image','mask','source_sha256','mask_sha256')},
            face_box_xyxy=row.get('face_box_xyxy')))
    require(len(selected)==24 and len(held)==4 and len({row['frame'] for row in rows})==28,
            'The original held-out camera roster overlaps training.')
    old_result = read(pin(old_run/'result.json',pose['new']['terminal_sha256']))
    replay_path = pin(old_run/old_result['bound_replay'],old_result['bound_replay_sha256'])
    replay = read(replay_path)
    selected_mesh = descriptor(pin(replay_path.parent/replay['mesh_file'],replay['mesh_sha256']))
    frozen = dict(rows=rows,ref=pose['new']['reference_frame_id'],
        mesh_sha256=previous['raw_equivalence']['r22']['frozen_reference_mesh']['sha256'],
        fit=descriptor(fit_path),selected_mesh=selected_mesh,
        heldout_mesh=pose['proof']['heldout_mesh'],source_pose=pose['new'])
    obs = old_run/'data/observations.npz'
    obs_plan = read(checked(pose['proof']['head_alignment_holdout_plan']))
    rois = source_rois(pin(obs,obs_plan['observations_sha256']),rows)
    run = root_for(subject)/'run'
    result_path = pin(run/'result.json',expected)
    terminal = read(result_path)
    require(terminal['state']=='completed' and terminal['total_optimizer_updates']==18000
            and terminal['input_pose_updates']==0 and not terminal['quality_accepted']
            and [item['step'] for item in terminal['endpoints']]==list(NEW.values()),
            'R23 requires both exports from the same completed garment fit.')
    config_path = run/'gaussian-run/config.json'
    config = read(config_path)
    require(set(config['implementation_sha256'])==set(IMPLEMENTATION)
            and Path(config['baseline_run'])==old_run,
            'The R23 training implementation or exact baseline identity differs.')
    code = {name:descriptor(pin(PROJECT/path,config['implementation_sha256'][name]))
            for name,path in IMPLEMENTATION.items()}
    old_grid = previous['static_grid']
    grid = dict(old_grid,arms={name:old_grid['arms'][name] for name in STATIC_BASE})
    historic = {name:old_review['static'][name]['receipt'] for name in STATIC_BASE}
    for name,desc in historic.items():
        arm = grid['arms'][name]
        pin(arm['ply'],arm['ply_sha256'])
        pin(arm['camera_path'],arm['camera_sha256'])
        qa.grid_records(previous,name,checked(desc))
    drives = {}
    for (name,step),endpoint in zip(NEW.items(),terminal['endpoints'],strict=True):
        require(endpoint['path']==f'raw-step-{step:06d}','The exported R23 endpoint is not its registered optimizer snapshot.')
        snapshot = run/'gaussian-run'/endpoint['path']
        grid['arms'][name],drives[name] = scan_drive(snapshot,endpoint,frozen,grid)
    require(drives['r23_10']['reference_fixed_rows']['sha256']==drives['r23']['reference_fixed_rows']['sha256'],
            'The two R23 durations changed the frozen non-clothing PLY parameters.')
    loss = run/'gaussian-run/loss.jsonl'
    entries = [json.loads(line) for line in loss.read_text(encoding='utf-8').splitlines()]
    require([row['step'] for row in entries]==[1,*range(100,18001,100)]
            and all(row['state']=='training' for row in entries),
            'The training curve does not cover the uninterrupted 18,000 updates.')
    return dict(schema='shuolee-r23-independent-review/1',subject=subject,commit=COMMIT,asset=previous['asset'],
        static_grid=grid,static_archive=historic,frozen=frozen,rois=rois,drives=drives,
        r22_review=descriptor(prior_path),result=descriptor(result_path),script=descriptor(Path(__file__)),
        training_log=descriptor(loss),training_config=descriptor(config_path),implementation=code,quality_accepted=False)


def score(source, mask, pixels, roi):
    if roi['box'] is None:
        return dict(roi,state=roi['state'])
    x0,y0,x1,y1 = roi['box']
    rgb,alpha = pixels[0][y0:y1,x0:x1],pixels[2][y0:y1,x0:x1]
    foreground = mask[y0:y1,x0:x1]>=253
    exterior = mask[y0:y1,x0:x1]<=2
    allowed = np.ones_like(foreground)
    if roi['face_box']:
        xx0,yy0 = np.floor(roi['face_box'][:2]).astype(int)
        xx1,yy1 = np.ceil(roi['face_box'][2:]).astype(int)
        allowed[max(0,yy0-y0):max(0,yy1-y0),max(0,xx0-x0):max(0,xx1-x0)] = False
    y,x = np.ogrid[y0:y1,x0:x1]
    for hx,hy in roi['hands']:
        allowed &= ((x-hx)**2+(y-hy)**2)>roi['hand_radius_px']**2
    fg,bg = foreground&allowed,exterior&allowed
    target = source[y0:y1,x0:x1].astype(np.float32)/255
    return dict(roi,foreground_pixels=int(fg.sum()),exterior_pixels=int(bg.sum()),
                foreground_rgb_mae=float(np.abs(rgb[fg]-target[fg]).mean()) if fg.any() else None,
                foreground_half_alpha_holes=int((alpha[fg]<.5).sum()),
                exterior_half_alpha_spill=int((alpha[bg]>=.5).sum()))


def native_alpha(mask, pixels, face_box):
    alpha = pixels[2]
    known = mask>=253
    result = dict(foreground_samples=int(known.sum()),
                  foreground_below_095_fraction=float((alpha[known]<.95).mean()))
    if face_box:
        x0,y0 = np.maximum(np.floor(face_box[:2]).astype(int),[0,0])
        x1,y1 = np.minimum(np.ceil(face_box[2:]).astype(int),mask.shape[::-1])
        inside = known[y0:y1,x0:x1]
        if inside.any():
            result.update(face_samples=int(inside.sum()),
                          face_below_095_fraction=float((alpha[y0:y1,x0:x1][inside]<.95).mean()))
    return result


def worker(plan_path,name,fd):
    current = read(plan_path)
    checked(current['script'])
    for item in current['implementation'].values():
        checked(item)
    require(os.path.samestat(os.fstat(fd),qa.review.GPU_LEASE.stat()),'Missing inherited GPU lease.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if name in NEW and not (plan_path.parent/'static-grid'/name/'result.json').exists():
        rebound(qa.static.render_grid,PORTABLE=STATIC_ARMS)(plan_path,name,fd)
    project = PROJECT if name in NEW else qa.launch.PROJECT
    sys.path.insert(0,str(project/'src'))
    import shuolee_portrait
    require(Path(shuolee_portrait.__file__).resolve().is_relative_to((project/'src').resolve()),
            'The review worker imported another renderer checkout.')
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    asset = Path(current['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render

    frozen = current['frozen']
    out = plan_path.parent/'native'/name
    out.mkdir(parents=True,exist_ok=False)
    pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    records = {}
    with gpu_lock(),torch.no_grad():
        require(torch.cuda.device_count()==1,'Use one CUDA device for native clothing evaluation.')
        if name in NEW:
            from shuolee_portrait.drivable_clothing_bundle import load_drive_bundle
            model, _ = load_drive_bundle(checked(current['drives'][name]['drive']).parent)
            replay = checked(current['drives'][name]['replay']).parent
        else:
            from shuolee_portrait.bound_replay import load_bound_snapshot
            run = Path(frozen['source_pose']['run'])
            model, _ = load_bound_snapshot(run,run/'gaussian-run/step-018000',asset_path=asset)
            with np.load(checked(frozen['heldout_mesh']),allow_pickle=False) as archive:
                model.vertices_sequence = torch.cat((model.vertices_sequence,torch.as_tensor(archive['vertices_world'],device='cuda')))
            model.num_timesteps = len(model.vertices_sequence)
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        for slot,row in enumerate(frozen['rows']):
            frame = row['frame']
            model.select_mesh_by_timestep(slot)
            cam,actual = camera(row,slot,pose_index=slot)
            require(all(actual[key]==row[key] for key in ('frame','wh','K','w2c')),'The comparison altered the source camera.')
            pixels,error = qa.static.scale_pixels(render,render_pose,cam,model,pipe,black,white,slot,1.)
            with Image.open(pin(row['image'],row['source_sha256'])) as original, Image.open(pin(row['mask'],row['mask_sha256'])) as matte:
                source,mask = np.asarray(original.convert('RGB')),np.asarray(matte.convert('L'))
            media = {}
            if name in NEW:
                production = current['drives'][name]['records'][slot]
                box = production['box_xyxy']
                for field,value in zip(('black','white','alpha'),pixels,strict=True):
                    x0,y0,x1,y1 = box
                    crop = np.rint(np.clip(value[y0:y1,x0:x1],0,1)*255).astype(np.int16)
                    with Image.open(replay/production['files'][field]) as image:
                        reference = np.asarray(image.convert('L' if field=='alpha' else 'RGB'),dtype=np.int16)
                    require(crop.shape==reference.shape and np.abs(crop-reference).max()<=1,
                            'Independent driven rendering differs from the delivered native replay.')
                    media[field] = descriptor(replay/production['files'][field])
                for suffix,mode,expected in (('source','RGB',source),('mask','L',mask)):
                    with Image.open(replay/f'{frame}-{suffix}.png') as image:
                        require(np.array_equal(np.asarray(image.convert(mode)),expected[y0:y1,x0:x1]),
                                'The drive package replay cropped different source RGB or mask pixels.')
            roi = current['rois'][frame]
            metrics = score(source,mask,pixels,roi)
            if roi['box'] is not None:
                panel = qa.static.r10.strip(out/f'{frame}.png',roi['box'],(source,*pixels))
                metrics['panel'] = descriptor(pin(out/panel['file'],panel['sha256']))
            records[frame] = dict(camera=actual,role='selected_fit' if slot<24 else 'holdout',
                source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],roi=metrics,
                black_white_alpha_composition_max_abs=error,delivered_media=media,
                alpha_regression=native_alpha(mask,pixels,row.get('face_box_xyxy')))
    qa.review.save(out/'result.json',dict(records=records,rows=len(records)))


def run(path,current):
    out = path.parent
    with qa.review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in NATIVE_ARMS:
            receipt = out/'native'/name/'result.json'
            if receipt.exists():
                continue
            with (out/f'{name}.log').open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),name,str(lease.fileno())],
                    pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    paired = out/'paired'
    paired.mkdir(exist_ok=True)
    grid,native,panels = {},{},{}
    for name in STATIC_ARMS:
        result = checked(current['static_archive'][name]) if name in STATIC_BASE else out/'static-grid'/name/'result.json'
        grid[name] = dict(receipt=descriptor(result),records=qa.grid_records(current,name,result))
    for view in current['static_grid']['camera_roster']:
        parts = [grid[name]['records'][view]['panel'] for name in STATIC_ARMS]
        require(all(item['crop_xyxy']==parts[0]['crop_xyxy'] and item['columns']==parts[0]['columns'] for item in parts),
                'Static R23 diagnostic crop changed.')
        panels[view] = qa.stack(paired/f'{view}.png',[checked(item) for item in parts],
                               ('R15','R21 18,000','R22 10,000','R22 18,000','R23 10,000','R23 18,000'))
    for name in NATIVE_ARMS:
        native[name] = read(out/'native'/name/'result.json')['records']
    for row in current['frozen']['rows']:
        frame = row['frame']
        items = [native[name][frame] for name in NATIVE_ARMS]
        require(all(item['camera']==items[0]['camera'] and item['source_sha256']==row['source_sha256']
                    and item['mask_sha256']==row['mask_sha256'] and item['role']==items[0]['role'] for item in items),
                'A native ROI used a different source or projection.')
        if current['rois'][frame]['box'] is not None:
            panels['native_torso_'+frame] = qa.stack(paired/f'native_torso_{frame}.png',
                [checked(item['roi']['panel']) for item in items],LABELS)
    summary = {}
    for name in NATIVE_ARMS:
        summary[name] = {}
        for role in ('selected_fit','holdout'):
            group = [item for item in native[name].values() if item['role']==role]
            values = [item['roi'] for item in group if item['roi'].get('foreground_rgb_mae') is not None]
            summary[name][role] = dict(frames=len(group),visible_torso_frames=len(values),
                mean_torso_rgb_mae=float(np.mean([item['foreground_rgb_mae'] for item in values])),
                torso_fg_half_alpha_holes=sum(item['foreground_half_alpha_holes'] for item in values),
                torso_bg_half_alpha_spill=sum(item['exterior_half_alpha_spill'] for item in values),
                mean_native_foreground_below_095=float(np.mean(
                    [item['alpha_regression']['foreground_below_095_fraction'] for item in group])))
    camera_roster = current['static_grid']['camera_roster']
    close_pitches = [key for key,row in camera_roster.items() if row['distance_class']=='close' and row['pitch_deg']]
    require(len(close_pitches)==20,'The close pitch review lost a registered camera.')
    for name in NATIVE_ARMS:
        rows = [grid[name]['records'][key] for key in close_pitches]
        interior = [row['opacity']['visible_face_proxy']['low_alpha_095_fraction'] for row in rows]
        ring = [row['anatomical_exterior_ring']['alpha_mass']/row['anatomical_exterior_ring']['pixels'] for row in rows]
        summary[name]['close_pitch'] = dict(views=len(rows),
            mean_low_alpha_on_visible_face_proxy=float(np.mean([value for value in interior if value is not None])),
            mean_anatomical_exterior_ring_alpha=float(np.mean(ring)))
    result = dict(state='paired_driven_and_static_views_rendered',commit=COMMIT,subject=current['subject'],
                  plan=descriptor(path),static=grid,static_camera_count=50,native=native,native_poses=28,
                  panels=panels,paired_native_metrics=summary,quality_accepted=False)
    target = out/'result.json'
    if not target.exists():
        qa.review.save(target,result)
    else:
        require(read(target)==result,'Preserve the completed R23 review.')
    print(json.dumps(descriptor(target)),flush=True)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    parser.add_argument('--result-sha',required=True)
    args = parser.parse_args()
    current = plan(args.subject,args.result_sha)
    if args.action=='check':
        print(json.dumps(dict(subject=args.subject,static_cameras=50,native_poses=28,
            visible_torso_rois=sum(item['box'] is not None for item in current['rois'].values()))),flush=True)
        return
    out = root_for(args.subject)/'view-comparison'
    out.mkdir(exist_ok=True)
    path = out/'plan.json'
    if not path.exists():
        qa.review.save(path,current)
    require(read(path)==current,'The saved R23 review plan differs.')
    run(path,current)


if __name__=='__main__':
    main()
