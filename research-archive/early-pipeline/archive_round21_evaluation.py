"""Archive R21 frozen evaluation poses, static cameras and head-support evidence."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from review_round20_pose_evidence import pin,read,require,sha


def bundle_evaluation(run,output,plan,views,matched):
    proof = plan['proof']
    files = {}

    def add(path,digest,name=None):
        path = pin(path,digest)
        label = name or path.relative_to(run).as_posix()
        require(not Path(label).is_absolute() and '..' not in Path(label).parts and label not in files,
                'Unexpected or repeated evaluation package member.')
        files[label] = (path,digest)
        return label

    for key in ('head_alignment','head_alignment_holdout_plan','head_alignment_holdout_poses',
                'head_alignment_holdout_policy','heldout_replay','heldout_mesh',
                'head_fit_preimage_fitted','head_fit_preimage_joint-candidates','head_fit_preimage_official-fit-result'):
        item = proof[key]
        add(item['file'],item['sha256'])
    plan_item = proof['head_alignment_holdout_plan']
    frozen = read(pin(plan_item['file'],plan_item['sha256']))
    add(run/'data/inputs.json',frozen['native_inputs_sha256'])
    add(run/'data/observations.npz',frozen['observations_sha256'])
    for row in frozen['records']:
        official = row['official_fit']
        path = (run/official['path']).resolve(strict=True)
        require(path.is_relative_to(run/'fit/head-alignment/holdout-official'/row['frame']),
                'A holdout fit receipt points beyond its own fixed run.')
        add(path,official['sha256'])
    for key,item in proof.items():
        if key.startswith(('surface_attachment_','duration_raw_')) or key in (
                'duration_continuity','duration_secondary_terminal','main_training_loss'):
            add(item['file'],item['sha256'])
    for key,name in (('user_view_priority','frozen-user-view-priority.json'),
                     ('current_user_view_priority','registered-surface-duration-policy.json')):
        item = proof[key]
        add(item['file'],item['sha256'],'static-camera-evidence/'+name)
    for key in ('csv','image'):
        item = views['training_curve'][key]
        add(item['file'],item['sha256'],'training-curve/'+Path(item['file']).name)
    add(matched['file'],matched['sha256'],'surface-attachment/matched-stable-geometry.json')
    for raw in views['duration']['raw_snapshots'].values():
        item = raw['portable_result']
        add(item['file'],item['sha256'])
        for item in raw['files'].values():
            add(item['file'],item['sha256'])
    ten = plan['ten_pose_evidence']
    for key in ('heldout_replay','heldout_mesh'):
        item = ten['proof'][key]
        add(item['file'],item['sha256'])
    for row in ten['holdout']['frames']:
        for item in row['files'].values():
            add(item['file'],item['sha256'])
    for key,item in plan['static_grid']['proof'].items():
        name = None if key=='staged_camera_manifest' else 'static-camera-evidence/'+Path(item['file']).name
        add(item['file'],item['sha256'],name)
    grid = views['static_grid']
    grid_path = pin(grid['file'],grid['sha256'])
    grid_result = read(grid_path)
    require(grid_result['per_arm_views']==50 and grid_result['actual_portable_arms']==['r15','r20','r21_10','r21']
            and grid_result['same_run_controls']==['before10','before18'],
            'A registered static arm was omitted from the archive.')
    add(grid_path,grid['sha256'],'static-grid/result.json')
    for side,arm in grid_result['arms'].items():
        item = arm['receipt']
        add(item['file'],item['sha256'],f'static-grid/{side}/result.json')
        for key,row in arm['records'].items():
            image = row['panel']
            add(image['file'],image['sha256'],f'static-grid/{side}/{key}.png')
    terminal = read(run/'result.json')
    replay_path = pin(run/terminal['bound_replay'],terminal['bound_replay_sha256'])
    replay = read(replay_path)
    body_mesh = add(replay_path.parent/replay['mesh_file'],replay['mesh_sha256'])
    for name in ('fit/fitted.npz','fit/joint-candidates.npz','data/training-inputs.json','gaussian-run/training-cameras.json'):
        path = run/name
        add(path,sha(path))
    aux = plan['supplement']['frames']
    aux_mesh = None
    if aux:
        item = proof['supplement_poses']
        aux_mesh = add(item['file'],item['sha256'])
    slots = [dict(frame=row['frame'],package_slot=index,fit_role='full_body',runtime_pose_index=row['pose_index'],
                  source_frame_index=row['source_frame_index'],source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
                  camera=row['camera'],mesh=body_mesh) for index,row in enumerate(replay['records'])]
    for row in aux:
        valid = add(row['valid_path'],row['valid_mask_sha256'])
        slots.append(dict(frame=row['frame'],package_slot=len(slots),fit_role='face_only',runtime_pose_index=row['pose_index'],
            source_frame_index=row['source_frame_index'],source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
            camera=row['camera'],mesh=aux_mesh,valid_mask=valid,valid_mask_sha256=row['valid_mask_sha256']))
    held = plan['holdout']['frames']
    for row in held:
        media = {key:add(item['file'],item['sha256']) for key,item in row['files'].items()}
        slots.append(dict(frame=row['frame'],package_slot=len(slots),fit_role='pose_conditioned_holdout',
            runtime_pose_index=row['pose_index'],fitted_pose_index=row['fitted_pose_index'],
            source_frame_index=row['source_frame_index'],source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
            camera=row['camera'],mesh=Path(proof['heldout_mesh']['file']).relative_to(run).as_posix(),files=media))
    require(len(slots)==len({row['frame'] for row in slots})==24+len(aux)+len(held) and len(held)==4,
            'A package role reused an incompatible pose identity.')
    roster = dict(schema='shuolee-r21-separated-pose-roster-v1', reference_frame=replay['reference_frame_id'],
                  full_body=24,face_only=len(aux),pose_conditioned_holdout=len(held),slots=slots,
                  layout='Package slots are unique. Runtime full-body/face and holdout replays are separate tracks.',
                  holdout_gaussian_updates=0,holdout_shared_identity_updates=0)
    roster_file = output/'named-pose-roster.json'
    if roster_file.exists():
        require(read(roster_file)==roster,'Preserve the existing distinct-role pose index.')
    else:
        with roster_file.open('x',encoding='utf-8') as stream:
            json.dump(roster,stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write('\n')
    add(roster_file,sha(roster_file),roster_file.name)
    view_dir = Path(plan['output'])
    for panel in views['panels']:
        if panel['view'].startswith(('heldout_','face_control_','fit_head_','static_grid_')):
            add(view_dir/panel['file'],panel['sha256'],'review/'+panel['file'])
    comparison = output/'pose-conditioned-evaluation.zip'
    if not comparison.exists():
        with TemporaryDirectory(prefix='.r21-pose-',dir=output) as temp:
            stage = Path(temp)/comparison.name
            with ZipFile(stage,'x') as package:
                for name,(path,_) in files.items():
                    package.write(path,name)
            require(not comparison.exists(),'An evaluation package already exists.')
            stage.rename(comparison)
    with ZipFile(comparison) as package:
        require(len(package.namelist())==len(files) and set(package.namelist())==set(files) and package.testzip() is None,
                'The pose-conditioned evaluation ZIP inventory or CRC differs.')
        for name,(_,digest) in files.items():
            value = hashlib.sha256()
            with package.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    value.update(block)
            require(value.hexdigest()==digest,f'Changed evaluation ZIP member: {name}')
    return dict(pose_conditioned_evaluation=str(comparison),pose_conditioned_evaluation_sha256=sha(comparison),
                evaluation_member_count=len(files),static_camera_count=50,static_portable_arms=4,
                same_run_static_control_arms=2,named_pose_roster=str(roster_file),named_pose_roster_sha256=sha(roster_file),
                primary_full_body=24,face_only=len(aux),pose_conditioned_holdout=len(held))
