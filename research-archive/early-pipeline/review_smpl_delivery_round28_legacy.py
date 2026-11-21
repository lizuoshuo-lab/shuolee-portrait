"""Compare available original poses and 50 fixed camera views for transferred R28 subjects."""

import argparse
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0,str(Path(__file__).resolve().parent))
import archive_smpl_delivery_round23 as panels
import review_smpl_delivery_round28 as review


base,qa = review.base,review.qa
pin,checked,read,sha,require,descriptor,rebound = review.pin,review.checked,review.read,review.sha,review.require,review.descriptor,review.rebound
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r28-transfer')
SUBJECTS = {'001':(24,4),'005':(23,4),'008':(19,3)}
NEW = review.NEW
ARMS = ('source',*NEW)
CORE_MODULES = (*review.CORE_MODULES,'face_surface_legacy.py')
LABELS = ('Starting model','R28 10,000','R28 18,000')
RECORD_TITLES = ('Original native RGB','Starting model','R28 10,000','R28 18,000','Frozen SMPL-X','Native 2D features')


def source(subject):
    path = review.DELIVERY/'input-audit'/f'{subject}-legacy-source.json'
    spec = read(path)
    main,hold = spec['selected_records'],spec['heldout_records']
    require(spec['schema']=='shuolee-r28-legacy-bound-source/1' and spec['subject']==subject
            and (len(main),len(hold))==SUBJECTS[subject] and not spec['auxiliary_records'],
            'R28 legacy source has another subject or original pose roster.')
    frozen = dict(ref=spec['reference_frame'],fit=spec['fit'],selected=spec['training_inputs'],
        meshes={'selected_fit':spec['pose_mesh'],'holdout':spec['heldout_pose_mesh']},rows=[])
    for slot,row in enumerate((*main,*hold)):
        role = 'selected_fit' if slot<len(main) else 'holdout'
        frozen['rows'].append(dict({key:row[key] for key in ('frame','wh','K','w2c')},pose_index=slot,role=role,
            image=descriptor(pin(row['image'],row['source_sha256'])),mask=descriptor(pin(row['mask'],row['mask_sha256']))))
    require(len({row['frame'] for row in frozen['rows']})==len(frozen['rows']) and
            frozen['rows'][spec['reference_pose_index']]['frame']==frozen['ref'],'Original legacy pose names overlap.')
    for role,item in frozen['meshes'].items():
        with np.load(checked(item),allow_pickle=False) as mesh:
            require(mesh['frame_ids'].tolist()==[row['frame'] for row in frozen['rows'] if row['role']==role],
                    'Legacy named original geometry differs from its images.')
    return descriptor(path),spec,frozen


def static_arm(ply,camera,world,local,frame,run):
    transform = read(camera.with_name('camera_provenance.json'))
    require(transform['files_sha256']['cameras.json']==sha(camera) and transform['cameras_share_model_world'],
            'Camera and portable Gaussian geometry do not share a coordinate system.')
    return dict(ply=str(ply),ply_sha256=sha(ply),camera_path=str(camera),camera_sha256=sha(camera),
        world=descriptor(world),local=descriptor(local),world_to_viewer=transform['world_to_viewer'],
        ref=frame,project=str(PROJECT),run=str(run),model_class='actual_delivered_static_portable_ply')


def fixed_cameras(spec,frozen):
    import review_round21_camera_grid as cameras

    root = Path(spec['source_root'])
    terminal = read(checked(spec['terminal']))
    ply = pin(root/terminal['ply'],terminal['sha256'])
    camera = root/terminal['camera']
    scene = pin(root/terminal['scene'],terminal['scene_sha256'])
    with ZipFile(scene) as package:
        names = {Path(name).name:name for name in package.namelist()}
    base.archive_members(scene,{names['cameras.json']:sha(camera),names[ply.name]:sha(ply)})
    control = static_arm(ply,camera,checked(spec['world_ply']),checked(spec['local_ply']),frozen['ref'],root)
    values = PlyData.read(checked(spec['world_ply']))['vertex'].data
    centers = np.column_stack([values[key] for key in ('x','y','z')])
    rows = {row['frame']:row for row in spec['selected_records']}
    model = SimpleNamespace(provenance=dict(reference_frame_id=frozen['ref'],run=str(root),
        fitted_sha256=spec['fit_sha256']),rows=rows,slots={key:i for i,key in enumerate(rows)})
    old = dict(model=model,snapshot=Path(spec['snapshot']),bound=read(checked(spec['bound_manifest'])),
               static=control,centers_world=centers)
    ref = cameras.fixed_reference({'R20':old})
    views = cameras.grid({'R20':old},ref)
    manifest = dict(subject=spec['subject'],reference_frame=frozen['ref'],views=views,
        native_wh=ref['row']['wh'],head_proxy_lbs_joint_ids=list(cameras.HEAD_JOINTS),head_proxy_lbs_threshold=.5,
        proof=dict(reference_fitted=spec['fit'],reference_mesh=spec['reference_mesh'],smplx_asset=spec['asset']))
    return manifest,control


def save_exact(path,value):
    if path.exists():
        require(read(path)==value,f'Preserve existing R28 delivery: {path}.')
    else:
        qa.review.save(path,value)
    return descriptor(path)


def projection_plan(out,manifest,control):
    sys.path.insert(0,str(PROJECT/'src'))
    import review_round21_head_proxy as proxy

    proof = save_exact(out/'fixed-cameras.json',manifest)
    regions,arrays = proxy.build(manifest)
    buffer = io.BytesIO()
    np.savez_compressed(buffer,**arrays)
    path = out/'fixed-crop-support.npz'
    data = buffer.getvalue()
    if path.exists():
        require(path.read_bytes()==data,'Preserve the original static crop supports.')
    else:
        with path.open('xb') as stream:
            stream.write(data)
    grid = dict(camera_roster=manifest['views'],proxies=regions,arms={},
        proof=dict(camera_manifest=proof,proxy_masks=descriptor(path)))
    grid['arms']['source'] = qa.project_static(control,grid)
    return grid


def snapshot(run,item,spec,frozen,grid):
    path = run/'gaussian-run'/item['path']
    require(path.name==f'raw-step-{item["step"]:06d}' and read(path/'result.json')==item,
            'R28 legacy snapshot path differs from the delivered endpoint.')
    portable_path = path/'portable-result.json'
    portable = read(portable_path)
    ply = pin(path/portable['ply'],portable['sha256'])
    camera = pin(path/portable['camera'],portable['camera_sha256'])
    scene = pin(path/portable['scene'],portable['scene_sha256'])
    world = pin(path/'world.ply',portable['source_world_sha256'])
    package = pin(path/item['drive_package'],item['drive_package_sha256'])
    drive_path = path/'drive/drive.json'
    drive = read(drive_path)
    local = pin(path/'bound-local.ply',drive['files']['bound-local.ply'])
    motion = pin(path/'drive/motion.npz',drive['files']['motion.npz'])
    pin(path/'reference-mesh.npz',spec['reference_mesh']['sha256'])
    require(portable['format_passed'] and portable['step']==item['step'] and portable['points']==item['points']
            and drive['schema']=='shuolee-face-bound-v1' and drive['reference_pose_index']==spec['reference_pose_index']
            and drive['roles']==[base.PRODUCTION_ROLES[row['role']] for row in frozen['rows']]
            and all(all(record[key]==row[key] for key in ('frame','wh','K','w2c')) and record['pose_index']==slot
                    for slot,(record,row) in enumerate(zip(drive['cameras'],frozen['rows'],strict=True))),
            'R28 legacy drive camera, pose or role changed.')
    with np.load(motion,allow_pickle=False) as combined:
        count = 0
        require(combined['frame_ids'].tolist()==[row['frame'] for row in frozen['rows']],
                'R28 legacy drive changed named pose IDs.')
        for evidence in frozen['meshes'].values():
            with np.load(checked(evidence),allow_pickle=False) as mesh:
                end = count+len(mesh['frame_ids'])
                require(np.array_equal(combined['faces'],mesh['faces']) and
                        np.array_equal(combined['vertices_world'][count:end],mesh['vertices_world']),
                        'R28 legacy drive changed the original fitted vertices.')
                count = end
        require(count==len(combined['vertices_world']),'Unexpected extra pose in the legacy drive.')
    base.archive_members(package,{'drive/drive.json':sha(drive_path),'reference/scene.zip':sha(scene),
        **{'drive/'+name:digest for name,digest in drive['files'].items()}})
    with ZipFile(scene) as archive:
        names = {Path(name).name:name for name in archive.namelist()}
    base.archive_members(scene,{names['cameras.json']:sha(camera),names[ply.name]:sha(ply)})
    replay_path = path/'drive-replay/result.json'
    replay = read(replay_path)
    require(replay['optimizer_updates']==0 and [row['frame'] for row in replay['records']]==
            [row['frame'] for row in frozen['rows']],'R28 legacy replay lost an original camera pose.')
    media = {}
    for record,row in zip(replay['records'],frozen['rows'],strict=True):
        require(record['source_sha256']==row['image']['sha256'] and record['mask_sha256']==row['mask']['sha256']
                and record['camera']==drive['cameras'][record['pose_index']]
                and record['role']==base.PRODUCTION_ROLES[row['role']] and record['exact_geometry_roundtrip']
                and set(record['files'])=={'source','mask','black','white','alpha'},
                'R28 legacy native replay does not match the archived inputs.')
        files = {key:descriptor(pin(replay_path.parent/leaf,record['files_sha256'][key]))
                 for key,leaf in record['files'].items()}
        media[row['frame']] = dict(crop=record['box_xyxy'],files=files)
    arm = qa.project_static(static_arm(ply,camera,world,local,frozen['ref'],run),grid)
    return arm,dict(portable=descriptor(portable_path),scene=descriptor(scene),package=descriptor(package),
        replay_receipts={'drive-replay':descriptor(replay_path)},media=media,step=item['step'],points=item['points'])


def plan(subject,commit,digest,out):
    implementation = rebound(review.prior.checked_implementation,PROJECT=PROJECT,CORE_MODULES=CORE_MODULES)(commit)
    receipt,spec,frozen = source(subject)
    run = review.root_for(subject)/'run'
    path = pin(run/'gaussian-run/result.json',digest)
    result = read(path)
    require(result['state']=='checkpoint_verified' and read(run/'progress.json')['state']=='completed'
            and [row['step'] for row in result['endpoints']]==list(NEW.values()),'R28 legacy endpoints are incomplete.')
    config = read(run/'gaussian-run/config.json')
    require(config['pinned_inputs']['legacy_source']==receipt['sha256'] and
            config['train_frames']==[row['frame'] for row in frozen['rows'] if row['role']=='selected_fit'] and
            config['holdout_frames']==[row['frame'] for row in frozen['rows'] if row['role']=='holdout'] and
            not config['auxiliary_frames'],'R28 legacy training changed its named source roster.')
    reserved=spec.get('reserved_holdouts',[row['frame'] for row in spec['heldout_records']])
    unavailable=spec.get('unavailable_holdouts',[])
    require(config['reserved_holdouts']==reserved and
            set(reserved)==set(config['holdout_frames'])|set(unavailable),
            'The unavailable formal pose was lost or replaced in the legacy roster.')
    manifest,control = fixed_cameras(spec,frozen)
    grid = projection_plan(out,manifest,control)
    snapshots = {}
    for name,item in zip(NEW,result['endpoints'],strict=True):
        grid['arms'][name],snapshots[name] = snapshot(run,item,spec,frozen,grid)
    return dict(subject=subject,commit=commit,schema='shuolee-r28-legacy-image-review/1',frozen=frozen,
        legacy_source=receipt,static_grid=grid,asset=str(checked(spec['asset'])),snapshots=snapshots,
        implementation=implementation,script=descriptor(Path(__file__)),quality_accepted=False,
        unavailable_holdout_frames=unavailable,
        unavailable_holdout_reasons={frame:'outside_capture_arc_without_archived_pose' for frame in unavailable})


def formal_crop(path,source):
    """Locate the signed unresized source column in the original native image."""
    with Image.open(path) as panel, Image.open(source) as image:
        require(panel.width%3==0,'The historical formal panel is not a three-column RGB strip.')
        width,height=panel.width//3,panel.height
        rgb=np.asarray(image.convert('RGB'))
        sample=np.asarray(panel.crop((0,0,width,height)).convert('RGB'))
    x,y=width//2,height//2
    anchor=sample[y-12:y+13,x-12:x+13]
    values=cv2.matchTemplate(rgb,anchor,cv2.TM_SQDIFF)
    _,_,(left,top),_=cv2.minMaxLoc(values)
    x0,y0=left-x+12,top-y+12
    require(0<=x0 and 0<=y0 and np.array_equal(rgb[y0:y0+height,x0:x0+width],sample),
            'The original RGB column of the formal holdout is not an exact native crop.')
    return [x0,y0,x0+width,y0+height]


def native_panels(out,current):
    spec = read(checked(current['legacy_source']))
    terminal = read(checked(spec['terminal']))
    path = pin(Path(spec['source_root'])/terminal['bound_panels'],terminal['bound_panels_sha256'])
    old = {row['frame']:row for row in read(path)['panels']}
    heldout = {row['frame']:row['baseline_panel'] for row in spec['heldout_records']}
    selected,result = {},{}
    folder = out/'native-paired'
    folder.mkdir(exist_ok=True)
    paste = rebound(panels.paste_panel,TITLE=RECORD_TITLES)
    for row in current['frozen']['rows']:
        frame,role = row['frame'],row['role']
        renders = [current['snapshots'][name]['media'][frame] for name in NEW]
        boxes = [item['crop'] for item in renders]
        full = boxes[0]
        require(all(box==full for box in boxes),'R28 legacy durations crop different native pixels.')
        source = panels.crop_file(checked(row['image']),full)
        matte = panels.crop_file(checked(row['mask']),full).convert('L')
        for item in renders:
            with Image.open(checked(item['files']['source'])) as original, Image.open(checked(item['files']['mask'])) as mask:
                require(np.array_equal(np.asarray(original.convert('RGB')),np.asarray(source)) and
                        np.array_equal(np.asarray(mask.convert('L')),np.asarray(matte)),
                        'R28 legacy replay changed native RGB or mask pixels.')
        if role=='selected_fit':
            previous = old[frame]
            box = previous['crop_xyxy']
            archived = pin(path.parent/previous['panel'],previous['panel_sha256'])
            require(previous['source_sha256']==row['image']['sha256'],'Legacy selected panel has a different source.')
            crop = panels.crop_overlap(full,box)
            with Image.open(archived) as image:
                header = image.height-(box[3]-box[1])
                require(image.width==4*(box[2]-box[0]) and header>=0,'Invalid legacy selected panel layout.')
            original,before,mesh,features = (panels.crop_file(archived,crop,box,col=i,header=header) for i in range(4))
            require(np.array_equal(np.asarray(original),np.asarray(panels.crop_file(checked(row['image']),crop))),
                    'Legacy original panel pixels do not match the signed source.')
            early,late = (panels.crop_file(checked(item['files']['black']),crop,item['crop']) for item in renders)
            result[frame] = paste(folder/f'{frame}.png',(original,before,early,late,mesh,features))
            selected[frame] = dict(result[frame],frame=frame,source_sha256=row['image']['sha256'],crop_xyxy=crop)
        else:
            archived=checked(heldout[frame])
            box=formal_crop(archived,checked(row['image']))
            crop=panels.crop_overlap(full,box)
            original,before=(panels.crop_file(archived,crop,box,col=i) for i in (0,1))
            early,late=(panels.crop_file(checked(item['files']['black']),crop,item['crop']) for item in renders)
            proof=qa.static.r10.strip(folder/f'{frame}.png',crop,(original,before,early,late))
            result[frame]=descriptor(pin(folder/proof['file'],proof['sha256']))
    return selected,result


def assemble(path,current):
    out = path.parent
    folder = out/'paired'
    folder.mkdir(exist_ok=True)
    static,images = {},{}
    for name in ARMS:
        receipt = out/'static-grid'/name/'result.json'
        static[name] = dict(receipt=descriptor(receipt),records=qa.grid_records(current,name,receipt))
    for key in current['static_grid']['camera_roster']:
        sources = [checked(static[name]['records'][key]['panel']) for name in ARMS]
        images[key] = qa.stack(folder/f'{key}.png',sources,LABELS)
    selected,native = native_panels(out,current)
    images.update({f'native_{frame}_face':item for frame,item in native.items()})
    result = dict(state='paired_native_face_and_static_views_rendered',subject=current['subject'],commit=current['commit'],
        static=static,static_camera_count=50,native_poses=len(native),auxiliary_face_poses=0,panels=images,selected=selected,
        plan=descriptor(path),quality_accepted=False,unavailable_holdout_frames=current['unavailable_holdout_frames'],
        unavailable_holdout_reasons=current['unavailable_holdout_reasons'])
    print(json.dumps(save_exact(out/'result.json',result)),flush=True)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        path,name,fd = Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
        current = read(path)
        checked(current['script'])
        for evidence in current['implementation'].values():
            checked(evidence)
        if not (path.parent/'static-grid'/name/'result.json').is_file():
            rebound(qa.static.render_grid,PORTABLE=ARMS)(path,name,fd)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-source','check','render','assemble'))
    parser.add_argument('subject',choices=SUBJECTS)
    parser.add_argument('--result-sha')
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action=='check-source':
        receipt,spec,frozen = source(args.subject)
        manifest,_ = fixed_cameras(spec,frozen)
        print(json.dumps(dict(source=receipt,cameras=len(manifest['views']),native=len(frozen['rows']))),flush=True)
        return
    if not args.result_sha or not args.commit:
        parser.error('Use the exact R28 legacy terminal hash and commit.')
    out = review.root_for(args.subject)/'view-comparison'
    out.mkdir(exist_ok=True)
    current = plan(args.subject,args.commit,args.result_sha,out)
    path = out/'plan.json'
    save_exact(path,current)
    if args.action=='check':
        print(json.dumps(dict(plan=descriptor(path),cameras=len(current['static_grid']['camera_roster']),
                              native=len(current['frozen']['rows']))),flush=True)
        return
    if args.action=='render':
        rebound(base.render,ARMS=ARMS,__file__=__file__)(path)
        print(json.dumps(dict(subject=args.subject,state='cuda_review_released_gpu',plan=descriptor(path))),flush=True)
    assemble(path,current)


if __name__=='__main__':
    main()
