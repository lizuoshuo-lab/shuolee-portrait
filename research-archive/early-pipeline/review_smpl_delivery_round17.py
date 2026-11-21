"""Compare R15, R16 and R17 on identical native, common and physical close cameras."""

import argparse
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round16 as r16
import review_round17_audit as audit17
import review_round17_dolly as dolly
import review_round17_regions as annotations


r14,r13,r10,r9,review,round1 = r16.r14,r16.r13,r16.r10,r16.r9,r16.review,r16.round1
pin,read,sha,require,descriptor = r16.pin,r16.read,r16.sha,r16.require,r16.descriptor
rebound = r9.transfer.rebound
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-multiview-quality-20261111')
DELIVERY = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round17'
BASE_COMMIT = r16.BASE_COMMIT
R16_COMMIT = 'af19f3a36880115472398c22396194143c0970c9'
R16_RECEIPT_SHA = '99593659dc9ab660b983aa71eeff280d9c5760e7a02167c02ac91e1bfc32c9bb'
DIRECTOR = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs/round16_director_review_20261111.json')
DIRECTOR_SHA = 'f7068409e5832d39f956dee47da189f1e9fa60ee6a5bb1b3f21071e35675e1c8'
SUBJECTS = r16.SUBJECTS
SIDES = ('old','mid','new')


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round17-{subject}'


def baselines(subject):
    old,_,_,old_plan,proof = r16.verify_base(subject)
    receipt_path = pin(r16.DELIVERY/'receipt.json',R16_RECEIPT_SHA)
    receipt = read(receipt_path)
    arm = receipt['subjects'][subject]
    require(receipt['commit'] == arm['commit'] == R16_COMMIT and arm['exact_round15_fit_reused'],
            'R16 endpoint identity differs from the pinned aggregate receipt.')
    path = pin(arm['review'],arm['review_sha256'])
    result = read(path)
    plan_path = pin(path.with_name('plan.json'),result['plan_sha256'])
    plan = read(plan_path)
    mid = review.planned_model(Path(arm['result']).parent,r16.PROJECT,arm['result_sha256'])
    require(result['state']=='round16_delivery_views_complete' and plan['new']==mid.provenance
            and plan['old']==old.provenance and plan['common_reference']==old_plan['common_reference']
            and plan['common_special']==old_plan['common_special'] and plan['roi_sha256']==old_plan['roi_sha256']
            and result['round15_float_native_bitwise_equal'] and result['all_paired_native_same_source_mask_camera_pose']
            and list(old.rows)==list(mid.rows) and all(old.rows[name]==mid.rows[name]
                and review.pose_digest(old,name)==review.pose_digest(mid,name) for name in old.rows),
            'R16 is not a same-fit, same-view archived endpoint.')
    director_path = pin(DIRECTOR,DIRECTOR_SHA)
    require(read(director_path)['receipt_sha256']==R16_RECEIPT_SHA,'R16 director proof points elsewhere.')
    proof.update(round16_receipt=descriptor(receipt_path),round16_review=descriptor(path),
                 round16_review_plan=descriptor(plan_path),round16_director=descriptor(director_path))
    return old,mid,plan,proof


def annotation_dir(subject):
    return DELIVERY/'input-audit'/f'review-{subject}'


def write_or_pin(path,data):
    if path.exists():
        require(path.read_bytes()==data,f'Preregistered review input differs: {path}.')
    else:
        with path.open('xb') as output:
            output.write(data)


def npz(path,values):
    buffer = BytesIO()
    np.savez_compressed(buffer,**values)
    write_or_pin(path,buffer.getvalue())
    return descriptor(path)


def prepare(subject):
    """Freeze camera and manual source-only annotations before a new endpoint exists."""
    require(not (root_for(subject)/'run/result.json').exists(),'The close views and annotations must precede the R17 endpoint.')
    old,mid,lock,proof = baselines(subject)
    out = annotation_dir(subject)
    out.mkdir(parents=True,exist_ok=True)
    path = out/'registered-review-inputs.json'
    with np.load(pin(proof['fixed_native_rois']['file'],proof['fixed_native_rois']['sha256']),allow_pickle=False) as previous:
        masks = dict(previous)
    native = {role:dict(regions) for role,regions in lock['native_regions'].items()}
    records = {}
    for pair in lock['pairs']:
        role = pair['role']
        regions,record = annotations.regions(subject,pair,pin,sha,r9.native.hash_mask)
        masks.update({f'{role}_{name}':mask for name,mask in regions.items()})
        native[role].update({name:r9.native.hash_mask(mask) for name,mask in regions.items()})
        records[role] = record
    cameras,near = dolly.plan_native_dollies(subject,lock,old,mid)
    roi = npz(out/'rois.npz',masks)
    near_roi = npz(out/'near-rois.npz',near)
    receipt = dict(schema='shuolee-round17-preregistered-source-and-dolly-v1',subject=subject,
        r15_commit=BASE_COMMIT,r16_commit=R16_COMMIT,fitted_sha256=old.provenance['fitted_sha256'],
        source_roi_method=annotations.identity(),dolly_method=descriptor(Path(dolly.__file__)),
        source_roi=roi,near_roi=near_roi,independent_annotations=records,near_views=cameras,native_regions=native)
    write_or_pin(path,(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(subject=subject,registered=str(path),sha256=sha(path),roles=list(records)),ensure_ascii=False),flush=True)
    return receipt


def registered(subject,old):
    path = annotation_dir(subject)/'registered-review-inputs.json'
    r = read(path)
    require(r['subject']==subject and r['r15_commit']==BASE_COMMIT and r['r16_commit']==R16_COMMIT
            and r['fitted_sha256']==old.provenance['fitted_sha256'] and r['source_roi_method']==annotations.identity(),
            'Preregistered camera/ROI plan identity differs.')
    for key in ('source_roi','near_roi','dolly_method'):
        pin(r[key]['file'],r[key]['sha256'])
    return r,descriptor(path)


def completed_training(subject,commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Use the completed R17 candidate commit.')
    rebound(r9.transfer.base.common.git_state,PROJECT=PROJECT)(commit)
    root = root_for(subject)
    run = root/'run'
    wrapper_path = root/'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    terminal = read(pin(run/'result.json',wrapper['result_sha256']))
    require(wrapper['state']==terminal['state']=='completed' and wrapper['candidate_commit']==commit
            and wrapper['baseline_commit']==R16_COMMIT and wrapper['quality_baseline_commit']==BASE_COMMIT
            and wrapper['fit_method']=='reused_exact_round15_fit'
            and wrapper['same_round15_fit_and_training_selection'] is wrapper['same_round16_fit_and_training_selection'] is True
            and terminal['gaussian_from_scratch'] is terminal['format_passed'] is terminal['fit_reused'] is True
            and terminal['fit_calls']==0 and terminal['steps']==10000 and terminal['final_snapshot']=='step-010000',
            'Missing verified R17 fixed-fit 10,000-step terminal.')
    integration_path = run/'gaussian-run/integration.json'
    integration = read(integration_path)
    require(integration['optimizer_updates']==10000 and integration['initial_points']==20908
            and integration['optimizer_initial_state_entries']==0 and integration['no_prior_gaussian_loaded'] is True,
            'R17 optimizer or one-Gaussian-per-triangle initialization differs.')
    color = pin(run/'gaussian-run/color-constraint.json',integration['sh0_color_constraint']['sha256'])
    densification = pin(run/'gaussian-run/densification-audit.json',integration['densification_audit']['sha256'])
    frames = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
    _,trials = audit17.verify(read(densification),integration,frames)
    preregistration = read(pin(root/'preregistration.json',wrapper['preregistration_sha256']))
    fixture = pin(DELIVERY/'input-audit/cuda-multiview-quality-proof.json',preregistration['cuda_fixture_sha256'])
    control = read(fixture)
    require(control['state']=='passed' and control['optimized']['accepted'] is True
            and control['rejected']['accepted'] is False and control['optimized']['views']==control['rejected']['views']==24
            and control['fallback']['trial']['accepted'] is True and control['fallback']['trial']['views']==24
            and control['post_prune_rollback']['restored_model_and_optimizer'] is True,
            'R17 CUDA controls for optimized commit, rejection, fallback and rollback did not pass.')
    for name,digest in control['source_sha256'].items():
        pin(PROJECT/name,digest)
    proof = dict(wrapper=descriptor(wrapper_path),integration=descriptor(integration_path),
        color_constraint=descriptor(color),densification_audit=descriptor(densification),
        cuda_multiview_quality_proof=descriptor(fixture),joint=descriptor(run/'fit/joint-body-refinement.json'),
        joint_candidates=descriptor(run/'fit/joint-candidates.npz'),shared_body=descriptor(run/'fit/shared-body.json'))
    proof.update(audit17.event_proofs(run,trials))
    if subject=='006':
        proof['round15_unchanged_retry'] = r16.r15.verified_retry(r16.r15.root_for(subject),BASE_COMMIT)
    for index,panel in enumerate(audit17.panels(run,trials)):
        proof[f'candidate_panel_{index:02d}'] = {key:panel[key] for key in ('file','sha256')}
    return terminal,wrapper,proof


def preflight(subject,commit):
    old,mid,lock,proof = baselines(subject)
    fixed,registered_proof = registered(subject,old)
    if not commit:
        return dict(subject=subject,registered=registered_proof,frames=len(old.rows),roles=list(fixed['near_views']))
    plan = rebound(r16.preflight,PROJECT=PROJECT,root_for=root_for,completed_training=completed_training,
        check_local_optimization=audit17.verify,candidate_panels=audit17.panels)(subject,commit)
    plan.update(schema='shuolee-round17-three-arm-review-v1',mid=mid.provenance,
                native_regions=fixed['native_regions'],independent_annotations=fixed['independent_annotations'],
                near_views=fixed['near_views'],near_roi_sha256=fixed['near_roi']['sha256'],
                historical_roi_sha256=plan['roi_sha256'],roi_sha256=fixed['source_roi']['sha256'])
    plan['proof'].update(proof,preregistered_review_inputs=registered_proof,
                         native_roi_with_annotations=fixed['source_roi'],close_face_proxy=fixed['near_roi'])
    run,new = root_for(subject)/'run',review.planned_model(root_for(subject)/'run',PROJECT,plan['new']['terminal_sha256'])
    mid_run = Path(mid.provenance['run'])
    for name in r16.FIXED_FILES:
        pin(run/name,sha(mid_run/name))
    parts = dict(old=round1.delivery_paths(mid_run,read(mid_run/'result.json')),
                 new=round1.delivery_paths(run,read(run/'result.json')))
    views,arms,_,colors = r14.common_plans(mid,new,lock,parts)
    plan['static_arms']['mid'] = arms['old']
    plan['terminal_sh0_rgb_range']['mid'] = colors['old']
    require(colors['old']['points_above_one']==colors['old']['points_below_zero']==0,
            'R16 archived point color range differs.')
    for name,view in views.items():
        destination = plan['static_views'][name]
        require(destination['fixed_world_camera']==view['fixed_world_camera'] and destination['crop_xyxy']==view['crop_xyxy']
                and destination['arms']['new']==view['arms']['new'],'R16/R17 static comparison camera differs.')
        destination['arms']['mid'] = view['arms']['old']
        destination['common_min_center_depth_m']['mid'] = view['common_min_center_depth_m']['old']
    for pair in plan['pairs']:
        frame = pair['frame']
        pair.update(mid_slot=mid.slots[frame],mid_pose_sha256=review.pose_digest(mid,frame))
        require(pair['old_slot']==pair['mid_slot']==pair['new_slot']
                and pair['old_pose_sha256']==pair['mid_pose_sha256']==pair['new_pose_sha256'],
                'Three-arm native pose identity changed.')
    for frame,status in plan['special_016'].items():
        status['R17'] = status['R15']
        require(status['R16']==lock['special_016'][frame]['R16']==status['R17'],'Special 016 frame class changed.')
    plan['round17_reviewer_sha256'] = sha(Path(__file__))
    return plan


def worker(path,side,fd):
    pin(Path(__file__),read(path)['round17_reviewer_sha256'])
    r16.actual_worker(path,side,fd)
    r16.native_float_worker(path,side,fd)
    r16.render_common_worker(path,side,fd)
    dolly.render(path,side,fd)


def geometry_summary(arm):
    from review_bound_scale_coverage import geometry
    import torch

    run = Path(arm['run'])
    terminal = read(pin(run/'result.json',arm['terminal_sha256']))
    snapshot,_,bound = round1.delivery_paths(run,terminal)
    _,mesh,faces = r13.mesh_parts(snapshot,bound)
    local = r14.PlyData.read(pin(snapshot/'bound-local.ply',bound['bound_ply_sha256']))['vertex'].data
    world = r14.PlyData.read(pin(snapshot/'world.ply',bound['inputs_sha256']['world_ply']))['vertex'].data
    sigma = np.exp(np.stack([world[f'scale_{i}'] for i in range(3)],axis=1))
    model = SimpleNamespace(get_xyz=torch.from_numpy(np.stack([world[key] for key in ('x','y','z')],axis=1)),
        get_scaling=torch.from_numpy(sigma),binding=torch.from_numpy(local['binding_0'].astype(np.int64)),
        vertices_sequence=torch.from_numpy(mesh[None]),faces=torch.from_numpy(faces.astype(np.int64)),timestep=0)
    ratio = np.quantile(sigma.max(axis=1)/sigma.min(axis=1),[.5,.95]).tolist()
    return dict(geometry(model),axis_ratio_p50_p95=ratio,posed_mesh_sha256=bound['inputs_sha256']['reference_mesh'],
                reference_frame=arm['reference_frame_id'])


def merge_three(out,plan,arms):
    require(all(arm['panels'].keys()==arms['old']['panels'].keys() for arm in arms.values()),'Three-arm view inventory differs.')
    panels = []
    for name in arms['old']['panels']:
        items = [arms[side]['panels'][name] for side in SIDES]
        first = items[0]
        require(all(item['crop_xyxy']==first['crop_xyxy'] and item['columns']==first['columns'] for item in items),
                'Three-arm panel crops or columns differ.')
        images = []
        for item in items:
            with Image.open(pin(out/item['file'],item['sha256'])) as image:
                images.append(image.convert('RGB'))
        width,height = images[0].size
        require(all(image.size==(width,height) for image in images),'Three-arm panel pixel grids differ.')
        sheet = Image.new('RGB',(width,height*3))
        for index,image in enumerate(images):
            sheet.paste(image,(0,index*height))
        path = out/f'{name}-{first["frame"]}.png'
        sheet.save(path)
        panels.append(dict(view=name,file=path.name,sha256=sha(path),frame=first['frame'],new_frame=items[-1]['frame'],
            crop_xyxy=first['crop_xyxy'],columns=first['columns'],rows=[f'round15_{BASE_COMMIT[:7]}',
                f'round16_{R16_COMMIT[:7]}',f'round17_{plan["commit"][:7]}']))
    historical = r16.fit_only_panel(out,plan)
    for panel in historical:
        panel['rows'] = ['round10_historical','round15_round16_round17_identical_fit']
    return panels+historical


def assemble(path,plan):
    require(read(path)==json.loads(json.dumps(plan,allow_nan=False)),'Saved three-arm plan differs.')
    out = path.parent
    pin(out/'rois.npz',plan['roi_sha256'])
    pin(out/'near-rois.npz',plan['near_roi_sha256'])
    arms = {}
    for side in SIDES:
        result = read(out/f'{side}-result.json')
        for source in ('common','near'):
            block = read(out/f'{side}-{source}.json')
            result['panels'].update(block.pop('panels'))
            result.update(block)
        result['float_native'] = read(out/f'{side}-float-native.json')
        result['float_actual'] = read(out/f'{side}-float-actual.json')
        require(set(result['float_native'])==set(plan['matched_frames'])
                and set(result['float_actual'])==set(plan['static_views']),'A float replay missed fixed views.')
        arms[side] = result
    for side,version in (('old','round15'),('mid','round16')):
        proof = plan['proof'][f'{version}_review']
        baseline = read(pin(proof['file'],proof['sha256']))['new']['float_native']
        for frame,record in arms[side]['float_native'].items():
            for key in ('camera','source_sha256','mask_sha256','black_float_sha256','alpha_float_sha256'):
                require(record[key]==baseline[frame][key],'Baseline native float replay differs from its frozen archive.')
            require(all(record[key]==arms['new']['float_native'][frame][key] for key in ('camera','source_sha256','mask_sha256')),
                    'R17 native RGB/mask/camera pairing differs.')
    r13.check_render_receipts(plan,arms['old'],arms['new'])
    mid_as_old = dict(plan,static_arms=dict(plan['static_arms'],old=plan['static_arms']['mid']),
        static_views={key:dict(view,arms=dict(view['arms'],old=view['arms']['mid'])) for key,view in plan['static_views'].items()})
    # The old-arm archive check in R13 is R15-specific. Independently bind the R16
    # controlled crops to its own archived bytes before applying the same checks.
    proof = plan['proof']['round16_review']
    history = Path(proof['file'])
    old_panels = read(pin(history,proof['sha256']))['new']['panels']
    for item in plan['common_special']:
        key = item['role']
        saved,current = old_panels[key],arms['mid']['panels'][key]
        with Image.open(pin(history.parent/saved['file'],saved['sha256'])) as a, Image.open(pin(out/current['file'],current['sha256'])) as b:
            delta = int(np.abs(np.asarray(a,dtype=np.int16)-np.asarray(b,dtype=np.int16)).max())
        require(delta<=1,'R16 controlled face crop differs from the archived replay.')
        arms['mid']['common_special'][item['frame']]['archive_max_abs_8bit'] = delta
    r13.check_render_receipts(mid_as_old,arms['mid'],arms['new'])
    r13.check_source_pixels(out,plan,arms.values())
    for role,spec in plan['near_views'].items():
        records = [arms[side]['physical_near'][role] for side in SIDES]
        require(all(record['camera']==records[0]['camera'] and record['pose_slot']==spec['slot']
                    and record['face_proxy_alpha']['pixels']==records[0]['face_proxy_alpha']['pixels'] for record in records)
                and records[0]['camera']['w2c']==spec['near_camera']['w2c'], 'Physical near camera or projected face region differs.')
    brightness = r13.brightness_diagnostic(out,plan,arms['old'],arms['new'])
    previous = r13.brightness_diagnostic(out,plan,arms['old'],arms['mid'])
    for role,row in brightness.items():
        require(row['old']==previous[role]['old'] and row['crop_relative_y_band']==previous[role]['crop_relative_y_band'],
                'Brightness baseline pixel band differs.')
        row['mid'] = previous[role]['new']
    panels = merge_three(out,plan,arms)
    output = out/'result.json'
    require(not output.exists(),'Preserve the completed R17 comparison receipt.')
    review.save(output,dict(state='round17_delivery_views_complete',subject=plan['subject'],commit=plan['commit'],
        plan_sha256=sha(path),assembler_sha256=sha(Path(__file__)),panels=panels,**arms,
        proof=plan['proof'],special_016=plan['special_016'],common_reference=plan['common_reference'],
        pose_relative_azimuth=plan['pose_relative_azimuth'],terminal_sh0_rgb_range=plan['terminal_sh0_rgb_range'],
        historical_R9_native=plan['historical_R9_native'],historical_R10_native=plan['historical_R10_native'],
        historical_R10_brightness=plan['historical_R10_brightness'],brightness_diagnostic=brightness,
        geometry_summary={side:geometry_summary(plan[side]) for side in SIDES},
        candidate_panels=plan['candidate_panels'],independent_annotations=plan['independent_annotations'],near_views=plan['near_views'],
        all_paired_native_same_source_mask_camera_pose=True,round15_and_round16_float_native_bitwise_equal=True,quality_accepted=False))
    print(f'{output} {sha(output)}',flush=True)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','check','render','assemble'))
    parser.add_argument('subject',choices=SUBJECTS)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action!='render':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.action=='prepare':
        prepare(args.subject)
        return
    if args.action!='check' and not args.commit:
        parser.error('Use the exact completed R17 candidate commit.')
    plan = preflight(args.subject,args.commit)
    if args.action=='check':
        print(json.dumps({key:plan[key] for key in ('subject','registered','frames','roles','matched_frames','near_views') if key in plan}),flush=True)
        return
    out = Path(plan['output'])
    path = out/'plan.json'
    if args.action=='assemble':
        assemble(path,plan)
        return
    with review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        for key,name in (('native_roi_with_annotations','rois.npz'),('close_face_proxy','near-rois.npz')):
            record = plan['proof'][key]
            shutil.copyfile(pin(record['file'],record['sha256']),out/name)
        review.save(path,plan)
        for side in SIDES:
            with (out/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                    pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assemble(path,plan)


if __name__=='__main__':
    main()
