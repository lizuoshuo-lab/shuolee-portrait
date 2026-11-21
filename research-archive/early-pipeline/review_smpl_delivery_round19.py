"""Review R15, R18 and R19 at the exact original and physical close cameras."""

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
import review_smpl_delivery_round17 as r17
import review_smpl_delivery_round18 as r18
import review_round18_supplement as auxiliary
import review_round19_stage as stage


r16,r14,r13,r10,r9 = r17.r16,r17.r14,r17.r13,r17.r10,r17.r9
review,round1,audit17,dolly = r17.review,r17.round1,r17.audit17,r17.dolly
pin,read,sha,require,descriptor = r17.pin,r17.read,r17.sha,r17.require,r17.descriptor
rebound = r17.rebound
SIDES,SUBJECTS = r17.SIDES,r17.SUBJECTS
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-protected-face-20261111')
DELIVERY = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round19'
BASE_COMMIT = r17.BASE_COMMIT
R18_COMMIT = '9f1bd057fe50fb6c9d93ee69a9ddcb31d941fd9d'
R18_RECEIPT_SHA = '2bc141e4d9e017542393a09c68815f97ecfdcc4c7b9ec148c96cfbf5d3e0d1f4'
DIRECTOR = r17.DIRECTOR.with_name('round18_director_review_20261111.json')
DIRECTOR_SHA = '453fb58304f84ee99c4a6366d33df047d0c9a05122b1755cde3e1d69550cbfdd'


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round19-{subject}'


def baselines(subject):
    old,_,_,_,proof = r16.verify_base(subject)
    receipt_path = pin(r18.DELIVERY/'receipt.json',R18_RECEIPT_SHA)
    receipt = read(receipt_path)
    arm = receipt['subjects'][subject]
    require(receipt['commit']==arm['commit']==R18_COMMIT and arm['exact_round15_fit_reused'] and arm['exact_round17_fit_reused'],
            'R18 baseline is not the pinned exact whole-body endpoint.')
    compare_path = pin(arm['review'],arm['review_sha256'])
    result = read(compare_path)
    plan_path = pin(compare_path.with_name('plan.json'),result['plan_sha256'])
    lock = read(plan_path)
    mid = review.planned_model(Path(arm['result']).parent,r18.PROJECT,arm['result_sha256'])
    fixed,reg = r17.registered(subject,old)
    require(result['state']=='round18_delivery_views_complete' and lock['new']==mid.provenance
            and lock['old']==old.provenance and lock['near_views']==fixed['near_views']
            and lock['roi_sha256']==fixed['source_roi']['sha256'] and lock['near_roi_sha256']==fixed['near_roi']['sha256']
            and lock['native_regions']==fixed['native_regions'] and lock['independent_annotations']==fixed['independent_annotations']
            and result['round15_and_round17_float_native_bitwise_equal']
            and list(old.rows)==list(mid.rows) and all(old.rows[key]==mid.rows[key]
                and review.pose_digest(old,key)==review.pose_digest(mid,key) for key in old.rows),
            'Frozen R18 whole-body, near-camera or foreground identity differs.')
    director = pin(DIRECTOR,DIRECTOR_SHA)
    require(read(director)['receipt_sha256']==R18_RECEIPT_SHA,'R18 director evidence points elsewhere.')
    proof.update(round18_receipt=descriptor(receipt_path),round18_review=descriptor(compare_path),
                 round18_review_plan=descriptor(plan_path),round18_director=descriptor(director),
                 round17_registered_review_inputs=reg)
    return old,mid,lock,proof


def completed_training(subject,commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Use the full completed R19 commit.')
    rebound(r9.transfer.base.common.git_state,PROJECT=PROJECT)(commit)
    root = root_for(subject)
    run = root/'run'
    wrapper_path = root/'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    terminal = read(pin(run/'result.json',wrapper['result_sha256']))
    require(wrapper['state']==terminal['state']=='completed' and wrapper['candidate_commit']==commit
            and wrapper['baseline_commit']==wrapper['quality_baseline_commit']==R18_COMMIT
            and wrapper['fit_method']=='reused_exact_round18_fit' and wrapper['same_round18_fit_and_training_selection'] is True
            and terminal['gaussian_from_scratch'] is terminal['format_passed'] is terminal['fit_reused'] is True
            and terminal['fit_calls']==0 and terminal['steps']==10000 and terminal['final_snapshot']=='step-010000',
            'Missing verified R19 10,000-step fixed-body terminal.')
    integration_path = run/'gaussian-run/integration.json'
    integration = read(integration_path)
    require(integration['optimizer_updates']==10000 and integration['initial_points']==20908
            and integration['optimizer_initial_state_entries']==0 and integration['no_prior_gaussian_loaded'] is True,
            'The fixed main 10,000-step profile differs.')
    color = pin(run/'gaussian-run/color-constraint.json',integration['sh0_color_constraint']['sha256'])
    densification = pin(run/'gaussian-run/densification-audit.json',integration['densification_audit']['sha256'])
    frames = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
    densification_data = read(densification)
    _,trials = audit17.verify(densification_data,integration,frames)
    proof = dict(wrapper=descriptor(wrapper_path),integration=descriptor(integration_path),
        color_constraint=descriptor(color),densification_audit=descriptor(densification),
        joint=descriptor(run/'fit/joint-body-refinement.json'),joint_candidates=descriptor(run/'fit/joint-candidates.npz'),
        shared_body=descriptor(run/'fit/shared-body.json'))
    proof.update(audit17.event_proofs(run,trials))
    if subject=='006':
        proof['round15_unchanged_retry'] = r16.r15.verified_retry(r16.r15.root_for(subject),BASE_COMMIT)
    for index,panel in enumerate(audit17.panels(run,trials)):
        proof[f'candidate_panel_{index:02d}'] = {key:panel[key] for key in ('file','sha256')}
    proof.update(stage.completed_proof(run,wrapper,integration,DELIVERY,PROJECT))
    return terminal,wrapper,proof


def preflight(subject,commit):
    if not commit:
        old,mid,plan,proof = baselines(subject)
        return dict(subject=subject,whole_body_frames=len(old.rows),near_roles=list(plan['near_views']),
                    prior_review_sha256=proof['round18_review']['sha256'],
                    heldout_pose_availability=stage.check_holdouts(Path(mid.provenance['run']),mid.provenance['protected_holdouts']))
    plan = rebound(r17.preflight,PROJECT=PROJECT,root_for=root_for,baselines=baselines,
                   completed_training=completed_training)(subject,commit)
    plan['schema'] = 'shuolee-round19-main-protected-face-review-v1'
    prior = plan['proof']['round18_review']
    history = read(pin(prior['file'],prior['sha256']))
    plan['mid_near_min_depth'] = {role:item['min_gaussian_center_depth_m']
                                  for role,item in history['new']['physical_near'].items()}
    plan['round19_reviewer_sha256'] = sha(Path(__file__))
    plan['supplement'] = stage.auxiliary_plan(root_for(subject)/'run',plan,history['supplement'])
    stage.plan_main(plan)
    plan['heldout_pose_availability'] = stage.check_holdouts(root_for(subject)/'run',plan['new']['protected_holdouts'])
    supplements = {row['frame'] for row in plan['supplement']['frames']}
    for frame,status in plan['special_016'].items():
        status['R18'] = history['special_016'][frame]['R18']
        status['R19'] = 'face_only_admitted' if frame in supplements else status['R18']
    return plan


def worker(path,side,fd):
    plan = read(path)
    pin(Path(__file__),plan['round19_reviewer_sha256'])
    if side=='main':
        stage.render_main(path,fd)
        return
    r16.actual_worker(path,side,fd)
    if side=='new':
        stage.capture_native(path,side,fd,r16.native_float_worker)
    else:
        r16.native_float_worker(path,side,fd)
    r16.render_common_worker(path,side,fd)
    if side=='mid':
        # Keep the original R17 dolly renderer and camera bytes. Replace only
        # its expected mid-arm depth with the archived R18 CUDA measurement.
        def mid_read(current):
            value = read(current)
            if current==path:
                for role,view in value['near_views'].items():
                    view['cpu_baseline_min_near_center_depth_m']['R16'] = plan['mid_near_min_depth'][role]
            return value
        rebound(dolly.render,read=mid_read)(path,side,fd)
    else:
        dolly.render(path,side,fd)
    auxiliary.render(path,side,fd)


def assemble(path,plan):
    require(read(path)==json.loads(json.dumps(plan,allow_nan=False)),'Saved R19 review plan differs.')
    out = path.parent
    pin(out/'rois.npz',plan['roi_sha256'])
    pin(out/'near-rois.npz',plan['near_roi_sha256'])
    arms = {}
    for side in SIDES:
        result = read(out/f'{side}-result.json')
        for source in ('common','near','supplement'):
            block = read(out/f'{side}-{source}.json')
            result['panels'].update(block.pop('panels'))
            result.update(block)
        result['float_native'] = read(out/f'{side}-float-native.json')
        result['float_actual'] = read(out/f'{side}-float-actual.json')
        require(set(result['float_native'])==set(plan['matched_frames'])
                and set(result['float_actual'])==set(plan['static_views']),'A float replay missed the fixed camera roster.')
        arms[side] = result
    for side,version in (('old','round15'),('mid','round18')):
        proof = plan['proof'][f'{version}_review']
        baseline = read(pin(proof['file'],proof['sha256']))['new']['float_native']
        for frame,record in arms[side]['float_native'].items():
            for key in ('camera','source_sha256','mask_sha256','black_float_sha256','alpha_float_sha256'):
                require(record[key]==baseline[frame][key],'Archived baseline native RGB or alpha drifted.')
            require(all(record[key]==arms['new']['float_native'][frame][key] for key in ('camera','source_sha256','mask_sha256')),
                    'R19 body replay used a different source image, mask or camera.')
    r13.check_render_receipts(plan,arms['old'],arms['new'])
    mid_as_old = dict(plan,static_arms=dict(plan['static_arms'],old=plan['static_arms']['mid']),
        static_views={key:dict(view,arms=dict(view['arms'],old=view['arms']['mid'])) for key,view in plan['static_views'].items()})
    mid_proof = plan['proof']['round18_review']
    history = Path(mid_proof['file'])
    old_panels = read(pin(history,mid_proof['sha256']))['new']['panels']
    for item in plan['common_special']:
        key = item['role']
        archived,current = old_panels[key],arms['mid']['panels'][key]
        with Image.open(pin(history.parent/archived['file'],archived['sha256'])) as a, Image.open(pin(out/current['file'],current['sha256'])) as b:
            delta = int(np.abs(np.asarray(a,dtype=np.int16)-np.asarray(b,dtype=np.int16)).max())
        require(delta<=1,'R18 controlled crop does not reproduce its archived pixels.')
        arms['mid']['common_special'][item['frame']]['archive_max_abs_8bit'] = delta
    r13.check_render_receipts(mid_as_old,arms['mid'],arms['new'])
    r13.check_source_pixels(out,plan,arms.values())
    for role,spec in plan['near_views'].items():
        records = [arms[side]['physical_near'][role] for side in SIDES]
        require(all(record['camera']==records[0]['camera'] and record['pose_slot']==spec['slot']
                    and record['face_proxy_alpha']['pixels']==records[0]['face_proxy_alpha']['pixels'] for record in records)
                and records[0]['camera']['w2c']==spec['near_camera']['w2c'], 'Physical close camera or face proxy changed.')
    auxiliary.verify_render_receipts(plan,arms)
    brightness = r13.brightness_diagnostic(out,plan,arms['old'],arms['new'])
    previous = r13.brightness_diagnostic(out,plan,arms['old'],arms['mid'])
    for role,row in brightness.items():
        require(row['old']==previous[role]['old'] and row['crop_relative_y_band']==previous[role]['crop_relative_y_band'],
                'Brightness reference band changed.')
        row['mid'] = previous[role]['new']
    panels = r17.merge_three(out,plan,arms)
    for panel in panels:
        panel['rows'] = (['round10_historical','round15_fit_only_historical'] if panel['view']=='available_record'
                         else [f'round15_{BASE_COMMIT[:7]}',f'round18_{R18_COMMIT[:7]}',f'round19_{plan["commit"][:7]}'])
    stage_data,stage_panels = stage.assemble_stage(out,plan,arms['new'])
    panels.extend(stage_panels)
    main_brightness = r13.brightness_diagnostic(out,plan,arms['old'],stage_data['main'])
    for role,row in brightness.items():
        require(row['old']==main_brightness[role]['old'] and row['crop_relative_y_band']==main_brightness[role]['crop_relative_y_band'],
                'The protected-stage bright-pixel support differs from the historical band.')
        row['main'] = main_brightness[role]['new']
    garment = stage.garment_audit(out,plan,dict(arms,main=stage_data['main']))
    output = out/'result.json'
    require(not output.exists(),'Preserve the completed R19 comparison receipt.')
    review.save(output,dict(state='round19_delivery_views_complete',subject=plan['subject'],commit=plan['commit'],
        plan_sha256=sha(path),assembler_sha256=sha(Path(__file__)),panels=panels,**arms,
        proof=plan['proof'],special_016=plan['special_016'],common_reference=plan['common_reference'],
        pose_relative_azimuth=plan['pose_relative_azimuth'],terminal_sh0_rgb_range=plan['terminal_sh0_rgb_range'],
        historical_R9_native=plan['historical_R9_native'],historical_R10_native=plan['historical_R10_native'],
        historical_R10_brightness=plan['historical_R10_brightness'],brightness_diagnostic=brightness,
        geometry_summary={side:r17.geometry_summary(plan[side]) for side in SIDES},
        candidate_panels=plan['candidate_panels'],independent_annotations=plan['independent_annotations'],
        near_views=plan['near_views'],supplement=plan['supplement'],protected_stage=stage_data,
        heldout_pose_availability=plan['heldout_pose_availability'],native_torso_band=garment,
        all_paired_native_same_source_mask_camera_pose=True,round15_and_round18_float_native_bitwise_equal=True,quality_accepted=False))
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
        parser.error('Use the full completed R19 candidate commit.')
    plan = preflight(args.subject,args.commit)
    if args.action=='check':
        print(json.dumps(dict(subject=args.subject,whole_body_frames=len(plan['matched_frames']) if args.commit else plan['whole_body_frames'],
            near_roles=list(plan.get('near_views',plan.get('near_roles'))),
            auxiliary_frames=[row['frame'] for row in plan.get('supplement',{}).get('frames',[])],
            heldout_frozen_pose_available=plan['heldout_pose_availability']['independent_bound_replay_available']),
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
        for side in (*SIDES,'main'):
            with (out/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                    pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assemble(path,plan)


if __name__=='__main__':
    main()
