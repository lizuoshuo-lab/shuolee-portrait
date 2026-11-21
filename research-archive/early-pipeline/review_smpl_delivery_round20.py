"""Compare R15/R18/R19/R20 on fixed source cameras and clearly scoped meshes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round19 as r19
import review_round20_pose_evidence as pose
import review_round18_supplement as auxiliary


r16,r14,r13,r10,r9 = r19.r16,r19.r14,r19.r13,r19.r10,r19.r9
review,round1,audit17,dolly = r19.review,r19.round1,r19.audit17,r19.dolly
pin,read,sha,require,descriptor,rebound = r19.pin,r19.read,r19.sha,r19.require,r19.descriptor,r19.rebound
SUBJECTS = r19.SUBJECTS
SIDES = ('old','mid','r19','new')
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-align-20261111')
DELIVERY = r19.DELIVERY.with_name('smpl-delivery-round20')
R19_COMMIT = '3415c5cff4a54d84effbdc82611a7bb96eeb4705'
R19_RECEIPT_SHA = '6b936ff2abc550701b719c3c7fc8c88ea665a5b655a88e619587a0059d564628'
DIRECTOR = r19.DIRECTOR.with_name('round19_director_review_20261111.json')
DIRECTOR_SHA = '3456109c68deb55b9ce3029d634e8514d1ab3981e56c6a81814f09e4d18e0f6d'
HISTORICAL_LABELS = ('round15_'+r19.BASE_COMMIT[:7],'round18_'+r19.R18_COMMIT[:7],'round19_'+R19_COMMIT[:7])
FIT_DIR = Path('fit/head-alignment')
DELIVERED_R15 = Path('/mnt/d/00/点云/首批_20261111_R15')


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round20-{subject}'


def history(subject):
    aggregate = pin(r19.DELIVERY/'receipt.json',R19_RECEIPT_SHA)
    ledger = read(aggregate)
    director = pin(DIRECTOR,DIRECTOR_SHA)
    require(ledger['commit']==read(director)['commit']==R19_COMMIT
            and read(director)['receipt_sha256']==R19_RECEIPT_SHA,'The R19 decision or baseline changed.')
    arm = ledger['subjects'][subject]
    delivered = pin(r19.DELIVERY/subject/'receipt.json',arm['delivery']['sha256'])
    endpoint = read(delivered)
    comparison = pin(endpoint['review'],arm['review_sha256'])
    receipts = read(comparison)
    previous_plan = pin(comparison.with_name('plan.json'),receipts['plan_sha256'])
    plan = read(previous_plan)
    require(plan['commit']==endpoint['commit']==R19_COMMIT
            and receipts['all_paired_native_same_source_mask_camera_pose']
            and receipts['round15_and_round18_float_native_bitwise_equal']
            and len(plan['matched_frames'])==24 and len(plan['supplement']['frames'])==4
            and not plan['heldout_pose_availability']['independent_bound_replay_available'],
            'The R19 reference images, camera roster or qualified comparisons differ.')
    models = {}
    for side,historical in (('old','old'),('mid','mid'),('r19','new')):
        info = plan[historical]
        model = review.planned_model(Path(info['run']),Path(info['project']),info['terminal_sha256'])
        require(model.provenance==info,'A pinned historical endpoint has changed.')
        models[side] = model
    require(all(list(model.rows)==plan['matched_frames'] for model in models.values()),'Historical roster order changed.')
    published = DELIVERED_R15/subject
    stable = plan['static_arms']['old']
    published_ply = pin(published/f'{subject}_R15_10000步.ply',stable['ply_sha256'])
    published_cameras = pin(published/'cameras.json',stable['camera_sha256'])
    return models,plan,receipts,dict(delivered_r15_ply=descriptor(published_ply),
        delivered_r15_camera=descriptor(published_cameras),round19_receipt=descriptor(aggregate),round19_director=descriptor(director),
        round19_review=descriptor(comparison),round19_review_plan=descriptor(previous_plan),round19_archive=descriptor(delivered))


def completed_training(subject,commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Use the full completed R20 commit.')
    rebound(r9.transfer.base.common.git_state,PROJECT=PROJECT)(commit)
    root = root_for(subject)
    run = root/'run'
    wrapper_path = root/'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    terminal = read(pin(run/'result.json',wrapper['result_sha256']))
    require(wrapper['state']==terminal['state']=='completed' and wrapper['candidate_commit']==commit
            and wrapper['baseline_commit']==R19_COMMIT and terminal['gaussian_from_scratch'] is True
            and terminal['format_passed'] and terminal['steps']==10000 and terminal['final_snapshot']=='step-010000',
            'The R20 wrapper and final SH0 endpoint are incomplete.')
    integration_path = run/'gaussian-run/integration.json'
    integration = read(integration_path)
    require(integration['optimizer_updates']==10000 and integration['initial_points']==20908
            and integration['optimizer_initial_state_entries']==0 and integration['no_prior_gaussian_loaded'] is True,
            'R20 must use the ordinary full training schedule and fresh parameters.')
    audit = pin(run/'gaussian-run/densification-audit.json',integration['densification_audit']['sha256'])
    color = pin(run/'gaussian-run/color-constraint.json',integration['sh0_color_constraint']['sha256'])
    frames = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
    _,trials = audit17.verify(read(audit),integration,frames)
    fit = pin(run/FIT_DIR/'result.json',wrapper['head_fit']['correction']['result_sha256'])
    proof = dict(wrapper=descriptor(wrapper_path),integration=descriptor(integration_path),head_alignment=descriptor(fit),
        joint=descriptor(run/'fit/joint-body-refinement.json'),joint_candidates=descriptor(run/'fit/joint-candidates.npz'),
        shared_body=descriptor(run/'fit/shared-body.json'),color_constraint=descriptor(color),densification_audit=descriptor(audit))
    proof.update(audit17.event_proofs(run,trials))
    candidate_panels = audit17.panels(run,trials)
    for index,panel in enumerate(candidate_panels):
        proof[f'candidate_panel_{index:02d}'] = {key:panel[key] for key in ('file','sha256')}
    proof.update(pose.proof_files(run,wrapper,terminal))
    for name,digest in read(fit)['source_sha256'].items():
        proof['head_fit_preimage_'+Path(name).stem] = descriptor(pin(run/FIT_DIR/'source'/name,digest))
    return terminal,wrapper,proof,candidate_panels


def preflight(subject,commit=None):
    old,locked,archive,proof = history(subject)
    if commit is None:
        return dict(subject=subject,history=proof['round19_review']['sha256'],native=len(locked['matched_frames']),
                    supplementary=[row['frame'] for row in locked['supplement']['frames']],
                    raw_holdouts=locked['new']['protected_holdouts'],native_roles=[row['role'] for row in locked['pairs']])
    terminal,wrapper,actual,candidate_panels = completed_training(subject,commit)
    run = root_for(subject)/'run'
    new = review.planned_model(run,PROJECT,wrapper['result_sha256'])
    models = dict(old,new=new)
    require(list(new.rows)==locked['matched_frames'] and new.provenance['reference_frame_id']==locked['new']['reference_frame_id']
            and new.provenance['protected_holdouts']==locked['new']['protected_holdouts'],
            'The same-camera QA requires the exact original 24 roster and heldouts.')
    old_rows = old['r19'].rows
    for frame,row in new.rows.items():
        require(all(row[key]==old_rows[frame][key] for key in (*review.NATIVE_KEYS,'mask_sha256','K_integer_centers')),
                f'The revised fitting changed source, mask or native projection for {frame}.')
    require(np.array_equal(new.fit['faces'],old['r19'].fit['faces']), 'The ordered SMPL-X topology changed.')
    proof.update(actual)
    for key in ('round15_review','common_special_posed_meshes','common_reference_mesh','pre_joint_selection',
                'native_roi_with_annotations','close_face_proxy','preregistered_review_inputs'):
        if key in locked['proof']:
            desc = locked['proof'][key]
            proof[key] = descriptor(pin(desc['file'],desc['sha256']))
    parts = {side:round1.delivery_paths(Path(arm.provenance['run']),
                 read(Path(arm.provenance['run'])/'result.json')) for side,arm in models.items()}
    r16.check_source_equivalence(Path(old['r19'].provenance['project']),PROJECT,parts['r19'][2]['source_files_sha256'],
                                 parts['new'][2]['source_files_sha256'])
    views,static,common,colors = r14.common_plans(old['old'],new,locked,parts)
    for side,archived in (('old','old'),('mid','mid'),('r19','new')):
        static[side] = locked['static_arms'][archived]
        colors[side] = locked['terminal_sh0_rgb_range'][archived]
        for key,view in views.items():
            historical = locked['static_views'][key]
            require(all(view[field]==historical[field] for field in ('fixed_world_camera','crop_xyxy','common_mesh_sha256')),
                    'The controlled static world camera or crop changed.')
            view['arms'][side] = historical['arms'][archived]
            view['common_min_center_depth_m'][side] = historical['common_min_center_depth_m'][archived]
    require(all(color['points_above_one']==color['points_below_zero']==0 for color in colors.values()),
            'An endpoint has SH0 point colors outside LDR.')
    pairs = []
    for saved in locked['pairs']:
        frame = saved['frame']
        row = dict(saved)
        for side,model in models.items():
            row[f'{side}_slot'] = model.slots[frame]
            row[f'{side}_pose_sha256'] = review.pose_digest(model,frame)
        require(all(row[f'{side}_slot']==row['old_slot'] for side in SIDES),'A named original pose moved slots.')
        pairs.append(row)
    plan = dict(schema='shuolee-round20-head-alignment-review-v1',subject=subject,commit=commit,
        **{side:model.provenance for side,model in models.items()},
        output=str(root_for(subject)/'view-comparison'),asset=locked['asset'],proof=proof,
        reviewer_sha256=locked['reviewer_sha256'],round16_reviewer_sha256=sha(Path(r16.__file__)),
        round20_reviewer_sha256=sha(Path(__file__)),pose_reviewer_sha256=sha(Path(pose.__file__)),
        pairs=pairs,matched_frames=locked['matched_frames'],static_views=views,static_arms=static,common_reference=common,
        common_special=locked['common_special'],special_016=locked['special_016'],native_regions=locked['native_regions'],
        independent_annotations=locked['independent_annotations'],near_views=locked['near_views'],roi_sha256=locked['roi_sha256'],
        near_roi_sha256=locked['near_roi_sha256'],brightness_contract=locked['brightness_contract'],
        pose_relative_azimuth=locked['pose_relative_azimuth'],terminal_sh0_rgb_range=colors,
        candidate_panels=candidate_panels,quality_accepted=False)
    plan['fit_audit'] = pose.fit_plan(run,plan)
    plan['supplement'] = pose.supplement_plan(run,plan)
    plan['supplement']['historical_fit_only_controls'] = pose.fallback_face_plan(run,plan,locked['supplement']['frames'])
    plan['holdout'] = pose.holdout_plan(run,plan)
    if subject=='016':
        f51,f55 = plan['special_016']['frame_000051'],plan['special_016']['frame_000055']
        f51['R20'] = 'full_body'
        f55['R20'] = 'face_only' if 'frame_000055' in {row['frame'] for row in plan['supplement']['frames']} else 'fit_only'
    return plan


def worker(path,side,fd):
    plan = read(path)
    pin(Path(__file__),plan['round20_reviewer_sha256'])
    pin(Path(pose.__file__),plan['pose_reviewer_sha256'])
    if side=='new':
        r16.actual_worker(path,side,fd)
        r16.native_float_worker(path,side,fd)
        r16.render_common_worker(path,side,fd)
        dolly.render(path,side,fd)
    auxiliary.render(path,side,fd)
    pose.render_fit_only_faces(path,side,fd)
    pose.render_heldouts(path,side,fd)


def collected(out,plan,history_receipt):
    hist_path = Path(plan['proof']['round19_review']['file']).parent
    arms = {}
    for side,archived in (('old','old'),('mid','mid'),('r19','new')):
        previous = history_receipt[archived]
        arm = dict(previous,panels={})
        for label,item in previous['panels'].items():
            if label.startswith('face_only_'):
                continue
            filename = item['file'] if side!='r19' else item['file'].removesuffix('-new.png')+'-r19.png'
            source = pin(hist_path/item['file'],item['sha256'])
            target = out/filename
            if not target.exists():
                shutil.copyfile(source,target)
            pin(target,item['sha256'])
            arm['panels'][label] = dict(item,file=filename)
        arms[side] = arm
    result = read(out/'new-result.json')
    for source in ('common','near'):
        extra = read(out/f'new-{source}.json')
        result['panels'].update(extra.pop('panels'))
        result.update(extra)
    result['float_native'] = read(out/'new-float-native.json')
    result['float_actual'] = read(out/'new-float-actual.json')
    arms['new'] = result
    for side,arm in arms.items():
        for suffix in ('supplement','fit-only-faces','heldout'):
            extra = read(out/f'{side}-{suffix}.json')
            arm['panels'].update(extra.pop('panels'))
            arm.update(extra)
    return arms


def merge(out,plan,arms):
    labels = (*HISTORICAL_LABELS,'round20_'+plan['commit'][:7])
    panels = []
    require(all(set(arm['panels'])==set(arms['new']['panels']) for arm in arms.values()),'Paired panel inventory differs.')
    for name,first in arms['old']['panels'].items():
        items = [arms[side]['panels'][name] for side in SIDES]
        require(all(item['crop_xyxy']==first['crop_xyxy'] and item['columns']==first['columns'] for item in items),
                'The fixed native display grids or columns changed.')
        images = []
        for item in items:
            with Image.open(pin(out/item['file'],item['sha256'])) as source:
                images.append(source.convert('RGB'))
        width,height = images[0].size
        require(all(image.size==(width,height) for image in images),'The same-camera panels differ in pixel grid.')
        canvas = Image.new('RGB',(width,height*len(SIDES)))
        for index,image in enumerate(images):
            canvas.paste(image,(0,index*height))
        path = out/f'{name}-{first["frame"]}.png'
        canvas.save(path)
        panels.append(dict(view=name,file=path.name,sha256=sha(path),frame=first['frame'],new_frame=items[-1]['frame'],
                           crop_xyxy=first['crop_xyxy'],columns=first['columns'],rows=list(labels)))
    return panels


def assemble(path,plan):
    out = path.parent
    require(read(path)==json.loads(json.dumps(plan,allow_nan=False)),'The saved R20 review plan differs.')
    pin(out/'rois.npz',plan['roi_sha256'])
    pin(out/'near-rois.npz',plan['near_roi_sha256'])
    historical = plan['proof']['round19_review']
    archived = read(pin(historical['file'],historical['sha256']))
    arms = collected(out,plan,archived)
    for frame in plan['matched_frames']:
        expected = arms['new']['float_native'][frame]
        for side in SIDES:
            observed = arms[side]['float_native'][frame]
            require(all(observed[key]==expected[key] for key in ('camera','source_sha256','mask_sha256')),
                    'An actual native model was compared under a different camera or source.')
    r13.check_render_receipts(plan,arms['old'],arms['new'])
    r13.check_source_pixels(out,plan,arms.values())
    for name,view in plan['near_views'].items():
        records = [arms[side]['physical_near'][name] for side in SIDES]
        require(all(row['camera']==records[0]['camera'] and row['pose_slot']==view['slot']
                    and row['camera']['w2c']==view['near_camera']['w2c'] for row in records),
                'Physical close-camera transform or source pose slot changed.')
    auxiliary.verify_render_receipts(plan,arms)
    pose.verify_holdout(plan,arms)
    for row in plan['supplement']['historical_fit_only_controls']:
        records = [arm['fit_only_face_native'][row['frame']] for arm in arms.values()]
        require(all(item['camera']==records[0]['camera'] and item['valid_mask_sha256']==row['valid_mask_sha256']
                    and item['source_sha256']==row['source_sha256'] and item['training_role']=='fit_only' for item in records),
                'An unadmitted face control changed its frozen pixels or role.')
    brightness = {}
    for side in SIDES[1:]:
        arm = r13.brightness_diagnostic(out,plan,arms['old'],arms[side])
        for role,item in arm.items():
            slot = brightness.setdefault(role,{key:value for key,value in item.items() if key!='new'})
            require(slot['old']==item['old'] and slot['crop_relative_y_band']==item['crop_relative_y_band'],
                    'The locked brightness band moved.')
            slot[side] = item['new']
    panels = merge(out,plan,arms) + pose.isolated_fit_panels(out,plan)
    output = out/'result.json'
    require(not output.exists(),'Preserve the completed R20 comparison receipt.')
    review.save(output,dict(state='round20_delivery_views_complete',subject=plan['subject'],commit=plan['commit'],
        plan_sha256=sha(path),assembler_sha256=sha(Path(__file__)),panels=panels,**arms,proof=plan['proof'],
        fit_audit=plan['fit_audit'],supplement=plan['supplement'],heldout=plan['holdout'],
        special_016=plan['special_016'],common_reference=plan['common_reference'],near_views=plan['near_views'],
        independent_annotations=plan['independent_annotations'],brightness_diagnostic=brightness,
        terminal_sh0_rgb_range=plan['terminal_sh0_rgb_range'],pose_relative_azimuth=plan['pose_relative_azimuth'],
        native_torso_band=r19.stage.garment_audit(out,plan,arms),
        geometry_summary={side:r19.r17.geometry_summary(plan[side]) for side in SIDES},candidate_panels=plan['candidate_panels'],
        all_paired_native_same_source_mask_camera=True,original_native_uses_each_endpoints_fitted_pose=True,
        common_historical_geometry_fixed=True,quality_accepted=False))
    print(f'{output} {sha(output)}',flush=True)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render','assemble'))
    parser.add_argument('subject',choices=SUBJECTS)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action!='render':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.action!='check' and not args.commit:
        parser.error('Use the full completed R20 commit.')
    plan = preflight(args.subject,args.commit)
    if args.action=='check':
        print(json.dumps(plan if not args.commit else dict(subject=args.subject,fit_audit=plan['fit_audit']['summary'],
               native=len(plan['matched_frames']),auxiliary=len(plan['supplement']['frames']),heldout=len(plan['holdout']['frames'])),
               ensure_ascii=False),flush=True)
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
