"""Verify and archive R11 native and common-reference comparison outputs."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round11 as reviewer
from archive_smpl_delivery_round1 import bundle, common, verify_scene
from archive_smpl_delivery_round2 import copy_proof


OUTPUT = Path(__file__).resolve().parents[1] / 'outputs/smpl-delivery-round11'


def collect(subject, commit, delivery):
    terminal,wrapper,proof = reviewer.completed_training(subject,commit)
    root = reviewer.root_for(subject)
    run = root/'run'
    review_path = root/'view-comparison/result.json'
    views = reviewer.read(review_path)
    plan_path = reviewer.pin(review_path.with_name('plan.json'),views['plan_sha256'])
    plan = reviewer.read(plan_path)
    require,sha,pin = reviewer.require,reviewer.sha,reviewer.pin
    require(views['state'] == 'round11_delivery_views_complete' and views['subject'] == subject
            and views['commit'] == plan['commit'] == commit and plan['proof'] == views['proof']
            and all(plan['proof'][key] == value for key,value in proof.items())
            and plan['new']['terminal_sha256'] == wrapper['result_sha256'],
            'Archive requires the completed review of the exact protected fit and training run.')
    snapshot, replay_path, _ = reviewer.round1.delivery_paths(run,terminal)
    replay = reviewer.read(replay_path)
    bound_path = pin(run/terminal['bound_panels'],terminal['bound_panels_sha256'])
    bound = reviewer.read(bound_path)
    selection_path = run/'fit/selection.json'
    selection = reviewer.read(selection_path)
    rows = reviewer.read(pin(run/'data/training-inputs.json',plan['new']['training_inputs_sha256']))['records']
    order = [row['frame'] for row in rows]
    selected = selection['selected_frames']
    require(order == [row['frame'] for row in bound['panels']] == [row['frame'] for row in replay['records']]
            and 0 < len(order) == len(set(order)) == len(selected) <= 24 and set(order) == set(selected)
            and selection['retained_holdouts'] == replay['retained_holdouts']
            and set(order).isdisjoint(selection['retained_holdouts'])
            and selection['reference_frame'] == order[0] == terminal['reference_frame_id']
            and bound['same_frame_bound_gaussians'] is bound['native_crops'] is True,
            'Selected source, mesh and Gaussian pose identities differ.')
    mesh = verify_scene(run,terminal)
    subject_dir = delivery/subject
    packages = bundle(run,snapshot,terminal,subject_dir)
    ply = run/terminal['ply']
    exported = copy_proof(subject_dir, {'portable_ply':(ply,terminal['sha256'])})
    proof_files = {key:(Path(value['file']),value['sha256']) for key,value in plan['proof'].items()}
    evidence = {
        'preregistration':root/'preregistration.json',
        'protected_inputs':pin(root/'protected-inputs.json',wrapper['protected_input_inventory_sha256']),
        'selection':selection_path,
        'pre_joint_selection':Path(plan['proof']['pre_joint_selection']['file']),
        'official_fit':run/'fit/official-fit-result.json',
        'world_scale':run/'gaussian-run/world-scale.json',
        'effective_config':run/'gaussian-run/config.json',
        'training_launch':root/'launch.json',
        'initialization':run/'gaussian-run/initialization.json',
        'review':review_path,
        'review_plan':plan_path,
        'fixed_native_rois':reviewer.pin(review_path.with_name('rois.npz'),plan['roi_sha256']),
        'selected_panel_receipt':bound_path,
    }
    require(sha(evidence['preregistration']) == wrapper['preregistration_sha256']
            and sha(evidence['pre_joint_selection']) == plan['proof']['pre_joint_selection']['sha256'],
            'Fit or launch evidence changed after the terminal.')
    for key,path in evidence.items():
        proof_files[key] = (path,sha(path))
    proof_dir = subject_dir/'evidence'
    proof_dir.mkdir(exist_ok=True)
    # The review and panel receipts share a basename. Keep each under its named key.
    copied = {}
    for key, item in proof_files.items():
        target = proof_dir/key
        target.mkdir(exist_ok=True)
        copied.update(copy_proof(target,{key:item}))
    records = common.RECORDS/subject
    require(records.resolve(strict=True) == records and records.is_dir(), 'Use the existing person record folder.')
    tag = f'交付11_{commit[:7]}'
    images,selected_panels = [],{}
    with TemporaryDirectory(prefix='shuolee-round11-record-') as scratch:
        files,pins = [],{}

        def add(source, name, expected, crop=None):
            pin(source,expected)
            pins[source] = expected
            target,stage = records/name,Path(scratch)/name
            if crop is None:
                with Image.open(source) as image:
                    image.convert('RGB').save(stage,quality=95,subsampling=0)
            else:
                common.record_jpeg(source,stage,crop)
            digest = sha(stage)
            if target.exists():
                pin(target,digest)
            else:
                files.append((stage,target))
            return dict(path=str(target),sha256=digest)

        for panel in views['panels']:
            filename = Path(panel['file'])
            require(filename.name == panel['file'] and filename.suffix == '.png', 'Unexpected review filename.')
            image = add(review_path.parent/filename,f'分析_{subject}_{tag}_{filename.stem}.jpg',panel['sha256'])
            images.append(dict(panel,archive=image))
        for row, panel in zip(rows,bound['panels'],strict=True):
            frame = panel['frame']
            require(frame == row['frame'] and panel['panel'] == frame+'.png'
                    and panel['source_sha256'] == row['source_sha256']
                    and panel['source_frame_index'] == row['source_frame_index'], 'Native selected panel identity differs.')
            path = bound_path.parent/panel['panel']
            image = add(path,f'3DGS_{subject}_{tag}_原图-高斯-Mesh-特征点_{frame}.jpg',
                        panel['panel_sha256'],panel['crop_xyxy'])
            selected_panels[frame] = dict(frame=frame,path=str(path),sha256=panel['panel_sha256'],
                source_frame_index=row['source_frame_index'],crop_xyxy=panel['crop_xyxy'])
            images.append(dict(frame=frame,view='driven_four_panel',archive=image))
        common.publish_images(files,pins)

    result_path = run/'result.json'
    receipt = dict(subject=subject,commit=commit,baseline_commit=reviewer.BASE_COMMIT,
        points=terminal['points'],frames=len(rows),reference=terminal['reference_frame_id'],
        selected_frame_ids=selected,selected_frame_panels=[selected_panels[frame] for frame in selected],
        retained_holdouts=selection['retained_holdouts'],coverage=selection['coverage'],
        baseline_coverage=plan['selections']['old']['coverage'],pre_joint_coverage=plan['selections']['pre_joint'].get('coverage'),
        result=str(result_path),result_sha256=wrapper['result_sha256'],mesh=str(mesh),
        review=str(review_path),review_sha256=sha(review_path),images=images,packages=packages,
        portable_ply=exported['portable_ply'],posed_replay=str(replay_path),
        bound_terminal=str(snapshot/'bound-replay.json'),evidence=copied,
        effective_config=copied['effective_config']['path'],effective_config_sha256=copied['effective_config']['sha256'],
        training_proof=copied,refinement=reviewer.read(Path(proof['integration']['file'])),
        special_016=views['special_016'],common_reference=views['common_reference'],
        fit_method=wrapper.get('fit_method'),r10_native_inputs_reused=True,quality_accepted=False)
    path = subject_dir/'receipt.json'
    if path.exists():
        require(reviewer.read(path) == receipt, 'Preserve the existing delivery receipt.')
    else:
        with path.open('x',encoding='utf-8') as file:
            json.dump(receipt,file,indent=2,ensure_ascii=False,allow_nan=False)
            file.write('\n')
    print(json.dumps(dict(subject=subject,receipt=str(path),sha256=sha(path),images=len(images),
                          selected_frames=len(selected)),ensure_ascii=False),flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=reviewer.SUBJECTS)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--delivery',type=Path,default=OUTPUT)
    args = parser.parse_args()
    collect(args.subject,args.commit,args.delivery)
