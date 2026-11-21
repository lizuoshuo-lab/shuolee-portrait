"""Package the preserved same-run main snapshot and native paired R19 evidence."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

import review_round19_stage as stage


pin, read, require, sha = stage.pin, stage.read, stage.require, stage.sha


def comparison_bundle(run,output,review_path,views,plan):
    comparison = views['protected_stage']
    require(comparison['exact_full_body_views']==24 and comparison['exact_face_views']==4
            and comparison['same_camera_masks_and_pose'] is True,'An unpaired stage cannot be archived.')
    proof = plan['proof']
    files = {}

    def add(path,digest,name=None):
        path = pin(path,digest)
        key = name if name else path.relative_to(run).as_posix()
        require(not Path(key).is_absolute() and '..' not in Path(key).parts and key not in files,
                'Invalid or duplicate protected stage member.')
        files[key] = (path,digest)

    for item in plan['main_snapshot'].values():
        add(item['file'],item['sha256'])
    for key in ('protected_face_main_endpoint','protected_face_optimization','protected_face_optimizer_state'):
        if key in proof:
            item = proof[key]
            add(item['file'],item['sha256'])
    for name in ('fit/fitted.npz','fit/selection.json','data/training-inputs.json','gaussian-run/training-cameras.json'):
        path = run/name
        add(path,sha(path))
    for path in stage.auxiliary.bundle_members(run,proof):
        add(path,sha(path))
    final = run/'gaussian-run/step-010000'
    for name in stage.SNAPSHOT_FILES:
        path = final/name
        add(path,sha(path))
    for key in ('main','new'):
        prefix = 'native-before' if key=='main' else 'native-after'
        arm = comparison['all_native_images'][key]
        add(arm['manifest']['file'],arm['manifest']['sha256'],f'{prefix}/result.json')
        for row in arm['records']:
            require(set(row['files'])=={'black','white','alpha'},'A native comparison lost a channel.')
            for item in row['files'].values():
                add(item['file'],item['sha256'],f'{prefix}/{Path(item["file"]).name}')
    main_pose = comparison['main_pose_replay']
    path = pin(main_pose['file'],main_pose['sha256'])
    posed = read(path)
    add(path,main_pose['sha256'],'main-poses/result.json')
    add(path.parent/posed['mesh_file'],posed['mesh_sha256'],'main-poses/'+posed['mesh_file'])
    terminal = read(run/'result.json')
    final_path = pin(run/terminal['bound_replay'],terminal['bound_replay_sha256'])
    final_pose = read(final_path)
    require([item['frame'] for item in final_pose['records']]==plan['matched_frames'],
            'Final posed roster changed from the same-run main endpoint.')
    add(final_path,terminal['bound_replay_sha256'],'final-poses/result.json')
    final_mesh = final_path.parent/final_pose['mesh_file']
    main_mesh = path.parent/posed['mesh_file']
    add(final_mesh,final_pose['mesh_sha256'],'final-poses/'+final_pose['mesh_file'])
    with np.load(main_mesh,allow_pickle=False) as a,np.load(final_mesh,allow_pickle=False) as b:
        require(set(a.files)==set(b.files) and all(np.array_equal(a[key],b[key]) for key in a.files),
                'Final posed mesh arrays changed from the same-run main endpoint.')
    for prefix,folder,manifest in (('main-poses',path.parent,posed),('final-poses',final_path.parent,final_pose)):
        for record in manifest['records']:
            add(folder/record['image'],record['image_sha256'],f'{prefix}/{record["image"]}')
    pose_manifest = output/'named-pose-roster.json'
    slots = [dict(frame=row['frame'],pose_index=row['pose_index'],role='full_body',
                  source_frame_index=row['source_frame_index'],source_sha256=row['source_sha256'],
                  mask_sha256=row['mask_sha256'],camera=row['camera'],mesh='final-poses/'+final_pose['mesh_file'])
             for row in final_pose['records']]
    slots += [dict(frame=row['frame'],pose_index=row['pose_index'],role='face_only',
                   source_frame_index=row['source_frame_index'],source_sha256=row['source_sha256'],
                   mask_sha256=row['mask_sha256'],camera=row['camera'],
                   valid_mask='gaussian-run/face-supplement/'+row['valid_mask'],
                   valid_mask_sha256=row['valid_mask_sha256'],mesh='gaussian-run/face-supplement/poses.npz')
              for row in plan['supplement']['frames']]
    require(len(slots)==len({row['frame'] for row in slots})==28
            and [row['pose_index'] for row in slots]==list(range(28)),
            'The 28-slot sidecar contains a duplicate frame or wrong pose slot.')
    roster = dict(schema='shuolee-r19-28-pose-evidence-v1',reference_frame_id=final_pose['reference_frame_id'],
                  full_body=24,face_only=4,retained_holdouts=final_pose['retained_holdouts'],slots=slots,
                  full_body_mesh_sha256=final_pose['mesh_sha256'],face_pose_sha256=proof['supplement_poses']['sha256'])
    if pose_manifest.exists():
        require(read(pose_manifest)==roster,'Preserve the archived 28-slot pose inventory.')
    else:
        with pose_manifest.open('x',encoding='utf-8') as file:
            json.dump(roster,file,indent=2,ensure_ascii=False,allow_nan=False)
            file.write('\n')
    add(pose_manifest,sha(pose_manifest),pose_manifest.name)
    for panel in views['panels']:
        if panel['view'].startswith('stage_'):
            add(review_path.parent/panel['file'],panel['sha256'],'paired-panels/'+panel['file'])
    add(review_path,sha(review_path),'comparison.json')
    binding = output/'protected-face-comparison.zip'
    if not binding.exists():
        with TemporaryDirectory(prefix='.round19-zip-',dir=output) as directory:
            temporary = Path(directory)/binding.name
            with ZipFile(temporary,'x') as package:
                for name,(path,_) in files.items():
                    package.write(path,name)
            require(not binding.exists(),'A protected stage comparison package already exists.')
            temporary.rename(binding)
    with ZipFile(binding) as package:
        require(len(package.namelist())==len(files) and set(package.namelist())==set(files) and package.testzip() is None,
                'Protected stage ZIP inventory or CRC differs.')
        for name,(_,digest) in files.items():
            with package.open(name) as member:
                actual = hashlib.sha256()
                for block in iter(lambda: member.read(1024*1024),b''):
                    actual.update(block)
                require(actual.hexdigest()==digest,f'Protected stage member differs: {name}.')
    return dict(protected_face_comparison=str(binding),protected_face_comparison_sha256=sha(binding),
                protected_face_comparison_members=len(files),protected_face_comparison_native_frames=24,
                protected_face_comparison_aux_poses=4,named_pose_roster=str(pose_manifest),
                named_pose_roster_sha256=sha(pose_manifest))
