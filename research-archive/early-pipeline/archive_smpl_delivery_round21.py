"""Verify and archive the R21 fixed static PLY grid, native and holdout comparisons."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round21 as reviewer
from archive_smpl_delivery_round1 import common, verify_scene
from archive_smpl_delivery_round18 import bundle
from archive_round21_evaluation import bundle_evaluation
from archive_smpl_delivery_round2 import copy_proof


OUTPUT = Path(__file__).resolve().parents[1] / 'outputs/smpl-delivery-round21'
SELECTED_COLUMNS = ['原图','10,000 步高斯','18,000 步高斯','共享衣物网格','2D 特征与投影']


def five_panel(path,ten,new,crop):
    """Merge exact native crops from both standard same-frame bound panels."""
    require = reviewer.require
    x0,y0,x1,y1 = crop
    width,height = x1-x0,y1-y0
    titles = ('Original native RGB','Bound Gaussian · 10,000','Bound Gaussian · 18,000',
              'Shared clothing mesh','2D input (green) / fit (magenta)')
    with Image.open(ten) as first, Image.open(new) as second:
        require(first.size==second.size==(4*width,height+26),'Source bound-panel dimensions differ.')
        early,late = first.convert('RGB'),second.convert('RGB')
    def tile(canvas,index):
        return canvas.crop((index*width,26,(index+1)*width,26+height))
    require(all(ImageChops.difference(tile(early,i),tile(late,i)).getbbox() is None for i in (0,2,3)),
            'The same-frame original, mesh or projected feature overlay differs across durations.')
    result = Image.new('RGB',(5*width,height+26),'#181818')
    draw = ImageDraw.Draw(result)
    parts = (tile(late,0),tile(early,1),tile(late,1),tile(late,2),tile(late,3))
    for index,(label,image) in enumerate(zip(titles,parts,strict=True)):
        result.paste(image,(index*width,26))
        draw.text((index*width+5,5),label,fill='white')
    if path.exists():
        with Image.open(path) as prior:
            require(prior.size==result.size and ImageChops.difference(prior.convert('RGB'),result).getbbox() is None,
                    'Preserve the existing selected five-panel native pixels.')
    else:
        with path.open('xb') as stream:
            result.save(stream,format='PNG')


def collect(subject, commit, delivery):
    terminal,wrapper,proof,_,duration,surface = reviewer.completed_training(subject,commit)
    root = reviewer.root_for(subject)
    run = root/'run'
    review_path = root/'view-comparison/result.json'
    views = reviewer.read(review_path)
    plan_path = reviewer.pin(review_path.with_name('plan.json'),views['plan_sha256'])
    plan = reviewer.read(plan_path)
    require,sha,pin = reviewer.require,reviewer.sha,reviewer.pin
    require(views['state'] == 'round21_delivery_views_complete' and views['subject'] == subject
            and views['commit'] == plan['commit'] == commit and plan['proof'] == views['proof']
            and all(plan['proof'][key] == value for key,value in proof.items())
            and duration==views['duration']==plan['duration'] and surface==views['surface_attachment']==plan['surface_attachment']
            and views['candidate_panels'] == plan['candidate_panels'] and views['supplement'] == plan['supplement']
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
    matched_path = delivery/'input-audit'/f'matched-stable-geometry-{subject}.json'
    matched = reviewer.read(matched_path)
    require(matched['schema']=='shuolee-r21-matched-stable-geometry/1' and matched['run']==str(run)
            and matched['raw_steps']==[10000,18000] and matched['reference_frame']==terminal['reference_frame_id'],
            'The same-ID surface audit does not belong to this uninterrupted run.')
    for row in matched['raw']:
        raw = duration['raw_snapshots'][str(row['step'])]
        require(row['files']['state']==raw['optimizer_state'] and row['files']['manifest']==raw['manifest'],
                'The same-ID audit changed a raw checkpoint or its recorded hash.')
    matched_item = reviewer.descriptor(matched_path)
    packages = bundle(run,snapshot,terminal,subject_dir,plan['proof'])
    packages.update(bundle_evaluation(run,subject_dir,plan,views,matched_item))

    def portable_files(source,result,target):
        ply, camera = (source/result[name] for name in ('ply','camera'))
        camera_provenance = camera.with_name('camera_provenance.json')
        provenance = reviewer.read(camera_provenance)
        return copy_proof(target, {'portable_ply':(ply,result['sha256']),
            'portable_camera':(camera,provenance['files_sha256']['cameras.json']),
            'camera_provenance':(camera_provenance,sha(camera_provenance))})

    exported = portable_files(run,terminal,subject_dir)
    ten_run = run/'duration-10000'
    ten = reviewer.read(pin(ten_run/'result.json',plan['ten']['terminal_sha256']))
    ten_snapshot,ten_replay,_ = reviewer.round1.delivery_paths(ten_run,ten)
    ten_proof = plan['ten_pose_evidence']['proof']
    ten_panels_path = pin(ten_run/ten['bound_panels'],ten['bound_panels_sha256'])
    ten_panels = reviewer.read(ten_panels_path)
    ten_named = reviewer.read(ten_replay)
    require(ten_panels['same_frame_bound_gaussians'] is ten_panels['native_crops'] is True
            and ten_panels['mesh_sha256']==bound['mesh_sha256']
            and [row['frame'] for row in ten_panels['panels']]==[row['frame'] for row in ten_named['records']]==order,
            'The 10k source panel track changed the shared posed mesh or frame roster.')
    ten_output = subject_dir/'duration-10000'
    ten_mesh = verify_scene(ten_run,ten)
    ten_packages = bundle(ten_run,ten_snapshot,ten,ten_output,ten_proof)
    secondary = dict(step=10000,points=ten['points'],reference_frame=ten['reference_frame_id'],packages=ten_packages,
        mesh=str(ten_mesh),posed_replay=str(ten_replay),result=plan['proof']['duration_secondary_terminal'],
        **portable_files(ten_run,ten,ten_output))
    proof_files = {key:(Path(value['file']),value['sha256']) for key,value in plan['proof'].items()
                   if key not in ('delivered_r15_ply','delivered_r15_camera','delivered_r20_ply','delivered_r20_camera')}
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
        'scored_native_rois':reviewer.pin(review_path.with_name('rois.npz'),plan['roi_sha256']),
        'scored_near_rois':reviewer.pin(review_path.with_name('near-rois.npz'),plan['near_roi_sha256']),
        'selected_panel_receipt':bound_path,
        'selected_10k_panel_receipt':ten_panels_path,
        'training_curve_csv':pin(views['training_curve']['csv']['file'],views['training_curve']['csv']['sha256']),
        'training_curve_image':pin(views['training_curve']['image']['file'],views['training_curve']['image']['sha256']),
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
    tag = f'交付21_{commit[:7]}'
    images,selected_panels,candidate_images = [],{},[]
    selected_dir = subject_dir/'selected-views'
    selected_dir.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='shuolee-round21-record-') as scratch:
        files,pins = [],{}

        def add(source, name, expected):
            pin(source,expected)
            pins[source] = expected
            target,stage = records/name,Path(scratch)/name
            with Image.open(source) as image:
                image.convert('RGB').save(stage,quality=95,subsampling=0)
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
        curve = views['training_curve']['image']
        images.append(dict(view='training_curve',archive=add(Path(curve['file']),
                      f'分析_{subject}_{tag}_10000-18000_主训练曲线.jpg',curve['sha256'])))
        for panel in views['candidate_panels']:
            source = Path(panel['file'])
            name = f'分析_{subject}_{tag}_step{panel["step"]:06d}_parent{panel["parent_id"]:07d}_{source.stem}_{panel["frame"]}.jpg'
            candidate_images.append(dict(panel, archive=add(source,name,panel['sha256'])))
        for row,panel,earlier,new_render,ten_render in zip(rows,bound['panels'],ten_panels['panels'],
                                                          replay['records'],ten_named['records'],strict=True):
            frame = panel['frame']
            require(frame == row['frame'] and panel['panel'] == frame+'.png'
                    and panel['source_sha256'] == row['source_sha256']
                    and panel['source_frame_index'] == row['source_frame_index']
                    and all(panel[key]==earlier[key] for key in ('frame','source_sha256','source_frame_index','crop_xyxy','pose_index','panel'))
                    and new_render['image_sha256']==panel['gaussian_sha256']
                    and ten_render['image_sha256']==earlier['gaussian_sha256']
                    and new_render['camera']==ten_render['camera'], 'Native selected panel identity differs.')
            new_path = pin(bound_path.parent/panel['panel'],panel['panel_sha256'])
            ten_path = pin(ten_panels_path.parent/earlier['panel'],earlier['panel_sha256'])
            combined = selected_dir/f'{frame}.png'
            five_panel(combined,ten_path,new_path,panel['crop_xyxy'])
            digest = sha(combined)
            image = add(combined,f'3DGS_{subject}_{tag}_原图-高斯10k-高斯18k-Mesh-特征点_{frame}.jpg',digest)
            selected_panels[frame] = dict(frame=frame,path=str(combined),sha256=digest,columns=SELECTED_COLUMNS,
                source_frame_index=row['source_frame_index'],crop_xyxy=panel['crop_xyxy'],
                input_panels={'10000':reviewer.descriptor(ten_path),'18000':reviewer.descriptor(new_path)})
            images.append(dict(frame=frame,view='paired_duration_five_panel',archive=image))
        common.publish_images(files,pins)

    result_path = run/'result.json'
    receipt = dict(subject=subject,commit=commit,baseline_commit=reviewer.R20_COMMIT,round15_baseline_commit=reviewer.R15_COMMIT,
        points=terminal['points'],frames=len(rows),reference=terminal['reference_frame_id'],
        selected_frame_ids=selected,selected_frame_panels=[selected_panels[frame] for frame in selected],
        retained_holdouts=selection['retained_holdouts'],coverage=selection['coverage'],
        baseline_coverage=reviewer.read(Path(plan['proof']['round20_archive']['file']))['coverage'],pose_relative_azimuth=views['pose_relative_azimuth'],
        result=str(result_path),result_sha256=wrapper['result_sha256'],mesh=str(mesh),
        review=str(review_path),review_sha256=sha(review_path),images=images,packages=packages,
        candidate_images=candidate_images,
        portable_ply=exported['portable_ply'],portable_camera=exported['portable_camera'],
        camera_provenance=exported['camera_provenance'],posed_replay=str(replay_path),
        bound_terminal=str(snapshot/'bound-replay.json'),evidence=copied,
        effective_config=copied['effective_config']['path'],effective_config_sha256=copied['effective_config']['sha256'],
        training_proof=copied,refinement=reviewer.read(Path(proof['integration']['file'])),
        special_016=views['special_016'],common_reference=views['common_reference'],
        terminal_sh0_rgb_range=views['terminal_sh0_rgb_range'],
        frozen_original_camera_and_selection=True,exact_round20_fit_reused=True,common_historical_geometry_fixed=True,
        near_views=views['near_views'],geometry_summary=views['geometry_summary'],supplement=views['supplement'],
        protected_face=wrapper.get('protected_face'),fit_audit=views['fit_audit'],
        heldout=views['heldout'],native_torso_band=views['native_torso_band'],
        surface_attachment=views['surface_attachment'],static_grid=views['static_grid'],static_grid_proof=views['static_grid_proof'],
        duration=views['duration'],training_curve=views['training_curve'],matched_geometry=matched_item,
        secondary_endpoint=secondary,
        baseline_delivered_r15={key:plan['proof']['delivered_r15_'+key] for key in ('ply','camera')},
        baseline_delivered_r20={key:plan['proof']['delivered_r20_'+key] for key in ('ply','camera')},
        fit_method=wrapper.get('fit_method'),r10_native_inputs_reused=True,quality_accepted=False)
    path = subject_dir/'receipt.json'
    if path.exists():
        require(reviewer.read(path) == receipt, 'Preserve the existing delivery receipt.')
    else:
        with path.open('x',encoding='utf-8') as file:
            json.dump(receipt,file,indent=2,ensure_ascii=False,allow_nan=False)
            file.write('\n')
    print(json.dumps(dict(subject=subject,receipt=str(path),sha256=sha(path),images=len(images),
                          candidate_images=len(candidate_images),
                          selected_frames=len(selected)),ensure_ascii=False),flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=reviewer.SUBJECTS)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--delivery',type=Path,default=OUTPUT)
    args = parser.parse_args()
    collect(args.subject,args.commit,args.delivery)
