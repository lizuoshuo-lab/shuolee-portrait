"""Review fixed static PLY cameras and frozen native pose tracks for R21."""

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
from plyfile import PlyData

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round20 as r20
import review_round21_static_replay as static_replay
import review_round21_io as stage_io
import review_round21_audit as audit21
import review_round21_training_curve as training_curve
import review_round18_supplement as auxiliary


r16,r14,r13,r10,r9 = r20.r16,r20.r14,r20.r13,r20.r10,r20.r9
review,round1,audit17,dolly,pose = r20.review,r20.round1,r20.audit17,r20.dolly,r20.pose
pin,read,sha,require,descriptor,rebound = r20.pin,r20.read,r20.sha,r20.require,r20.descriptor,r20.rebound
SUBJECTS = r20.SUBJECTS
SIDES = ('old','mid','ten','new')
WORKERS = (*SIDES,'before10','before18')
STATIC_NAMES = dict(old='r15',mid='r20',ten='r21_10',new='r21',before10='before10',before18='before18')
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-support-20261111')
DELIVERY = r20.DELIVERY.with_name('smpl-delivery-round21')
R20_COMMIT = 'ad52fbee2d78527e022ef145b20a497c4c4e2fa1'
R20_RECEIPT_SHA = 'be40feef7d37dc34de765746ac8f44ab4a1a02e83fd553a3cd228525def52b9d'
R20_REVIEW_SHA = '7bc24d053359b96143c07a0aabbfc0d53ab6cbfaee6e6b1c685e05bf923306d1'
R15_COMMIT = r20.r19.BASE_COMMIT
FIT_DIR = r20.FIT_DIR
FROZEN_FILES = ('fit/fitted.npz','fit/joint-candidates.npz','fit/official-fit-result.json','fit/selection.json',
                'fit/selection-before-joint.json','fit/joint-body-refinement.json','fit/shared-body.json','data/inputs.json',
                'data/training-inputs.json','data/observations.npz','fit/head-alignment/result.json',
                'fit/head-alignment/holdout-poses.npz','fit/head-alignment/holdout-plan.json','fit/head-alignment/holdout-policy.json')


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round21-{subject}'


def history(subject):
    import review_round21_camera_grid as cameras

    historical_policy = DELIVERY/'input-audit/frozen-user-view-priority.json'
    baseline,proof = rebound(cameras.read_baselines,POLICY=historical_policy)(subject)
    prior = pin(r20.DELIVERY/'review-receipt.json',R20_REVIEW_SHA)
    review_index = read(prior)
    require(review_index['commit']==R20_COMMIT and review_index['subjects'][subject]['heldout_views']==4,
            'The pinned R20 independent review or heldout scope changed.')
    row = review_index['subjects'][subject]['review']
    result_file = pin(row['file'],row['sha256'])
    result = read(result_file)
    plan_file = pin(result_file.with_name('plan.json'),result['plan_sha256'])
    plan = read(plan_file)
    old,mid = (baseline[name]['model'] for name in ('R15','R20'))
    require(result['commit']==R20_COMMIT and plan['old']==old.provenance and plan['new']==mid.provenance
            and result['all_paired_native_same_source_mask_camera'] and len(plan['matched_frames'])==24,
            'The pinned R20 shared cameras or source identities differ.')
    proof.update(round20_review_receipt=descriptor(prior),round20_review=descriptor(result_file),
                 round20_review_plan=descriptor(plan_file))
    return old,mid,plan,result,proof


def duration_evidence(run,wrapper,terminal):
    """Pin one uninterrupted run, two untouched raw states and the 10k export branch."""
    verified = wrapper['duration_comparison']
    continuity = verified['continuity']
    require(continuity['single_main_training'] and continuity['endpoint_postprocess_started_after_18000']
            and continuity['raw_10000_is_prefix'], 'The wrapper did not prove uninterrupted main training.')
    duration_path = pin(run/continuity['file'],continuity['sha256'])
    duration = read(duration_path)
    normal = terminal['duration_comparison']
    require(all(normal[key]==value for key,value in duration.items()) and normal['sha256']==continuity['sha256'],
            'Duration receipt differs between the wrapper and ordinary run.')
    ten_run = run/'duration-10000'
    ten_path = pin(run/normal['secondary']['result'],verified['intermediate']['result_sha256'])
    require(ten_path==ten_run/'result.json' and verified['terminal']['result_sha256']==wrapper['result_sha256'],
            'The secondary or primary export terminal does not match the paired run.')
    ten = read(ten_path)
    require(ten['state']=='completed' and ten['format_passed'] and ten['steps']==10000
            and ten['final_snapshot']=='step-010000' and terminal['steps']==18000,
            'The same-run 10k branch or ordinary 18k export is incomplete.')
    proof = dict(duration_secondary_terminal=descriptor(ten_path),duration_continuity=descriptor(duration_path),
                 main_training_loss=descriptor(pin(run/continuity['loss_log'],continuity['loss_log_sha256'])))
    comparison = dict(schema='shuolee-r21-10k-18k-review/1',primary_steps=18000,secondary_steps=10000,
                      primary_path=str(run),secondary_path=str(ten_run),continuity=continuity,raw_snapshots={},
                      intermediate_protected_face=verified['intermediate']['protected_face'])
    for step,registered,verified_raw in zip((10000,18000),duration['endpoints'],continuity['raw_endpoints'],strict=True):
        folder = run/registered['raw']
        require(folder==run/'gaussian-run'/f'raw-step-{step:06d}' and registered['step']==verified_raw['step']==step,
                'A raw checkpoint has an unexpected stage or location.')
        path = pin(folder/'state.json',registered['raw_receipt_sha256'])
        manifest = read(path)
        require(manifest['state_sha256']==registered['raw_state_sha256']==verified_raw['state_sha256']
                and manifest['files_sha256']==verified_raw['files_sha256'],
                'The raw optimizer state or its native files changed after wrapper verification.')
        raw_state = descriptor(pin(folder/'state.pt',manifest['state_sha256']))
        files = {name:descriptor(pin(folder/name,digest)) for name,digest in manifest['files_sha256'].items()}
        parity = read(folder/'native-ply-parity.json')
        require(parity['passed'] and parity['sha256']==files['world.ply']['sha256']
                and parity['mesh_sha256']==files['reference-mesh.npz']['sha256'],
                'A raw checkpoint has different saved native PLY bytes or mesh.')
        portable_path = pin(run/registered['raw_portable_result'],registered['raw_portable_result_sha256'])
        portable = read(portable_path)
        require(portable['source_world_sha256']==files['world.ply']['sha256'] and portable['step']==step
                and portable['format_passed'], 'The raw portable endpoint differs from its verified world snapshot.')
        portable_files = {name:descriptor(pin(folder/portable[name],portable[digest])) for name,digest in
                          (('ply','sha256'),('camera','camera_sha256'),('scene','scene_sha256'))}
        proof[f'duration_raw_{step}_manifest'] = descriptor(path)
        comparison['raw_snapshots'][str(step)] = dict(manifest=descriptor(path),optimizer_state=raw_state,files=files,
            portable_result=descriptor(portable_path),portable_files=portable_files)
    return dict(proof=proof,comparison=comparison)


def attachment_evidence(run,wrapper,terminal,duration):
    verify = wrapper['duration_comparison']
    branches = {10000:(run/'duration-10000',verify['intermediate']),18000:(run,verify['terminal'])}
    proof, receipts = {},{}
    for step,(branch,verified) in branches.items():
        branch_terminal = terminal if step==18000 else read(pin(branch/'result.json',verified['result_sha256']))
        attachment = branch_terminal['unified_main_training']['surface_attachment']
        path = pin(branch/'gaussian-run'/attachment['file'],attachment['sha256'])
        result = read(path)
        raw = duration['raw_snapshots'][str(step)]
        raw_item = raw['files']['attachment.json']
        raw_report = read(pin(raw_item['file'],raw_item['sha256']))
        require(result['schema']=='shuolee-surface-attachment-v1' and result['step']==step
                and result['training']==raw_report and result['postprocessing_changes_geometry'] is False,
                'The branch surface receipt does not reproduce its untouched native observations.')
        guide = result['clothing_guide']
        guide_path = pin(path.parent/guide['file'],guide['sha256'])
        support = read(guide_path)
        mesh_dir = branch/'gaussian-run'/branch_terminal['final_snapshot']
        mesh = pin(mesh_dir/'reference-mesh.npz',result['reference_mesh_sha256'])
        require(sha(mesh)==raw['files']['reference-mesh.npz']['sha256']==support['active_mesh_sha256'],
                'Raw and final evaluation do not use the same fixed clothing reference mesh.')
        support_file = pin(path.parent/support['support_file'],support['support_file_sha256'])
        before = PlyData.read(pin(raw['files']['world.ply']['file'],raw['files']['world.ply']['sha256']))['vertex'].data
        parity = read(mesh_dir/'native-ply-parity.json')
        after = PlyData.read(pin(mesh_dir/'world.ply',parity['sha256']))['vertex'].data
        fields = ('x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3')
        require(len(before)==len(after) and all(np.array_equal(before[name],after[name]) for name in fields),
                'The same-run protected appearance stage altered centers, sizes, or orientations.')
        proof.update({f'surface_attachment_{step}_result':descriptor(path),
                      f'surface_attachment_{step}_guide':descriptor(guide_path),
                      f'surface_attachment_{step}_support':descriptor(support_file)})
        receipts[str(step)] = dict(file=descriptor(path),guide=descriptor(guide_path),
            supported_file=descriptor(support_file),raw_attachment=raw_item,raw_final_same_world_geometry=True,
            wrapper=verified['surface_attachment'])
    a,b = (read(pin(receipts[str(step)]['file']['file'],receipts[str(step)]['file']['sha256'])) for step in (10000,18000))
    require(a['reference_mesh_sha256']==b['reference_mesh_sha256'] and a['clothing_guide']==b['clothing_guide']
            and a['training']['initial']==b['training']['initial'],
            'The shared clothing surface or initialization changed between durations.')
    return dict(proof=proof,branches=receipts)


def completed_training(subject,commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Use the full completed R21 commit.')
    rebound(r9.transfer.base.common.git_state,PROJECT=PROJECT)(commit)
    root = root_for(subject)
    run = root/'run'
    original_path = root/'wrapper-terminal.json'
    original = read(original_path)
    wrapper_path,wrapper = original_path,original
    if original['state']!='completed':
        wrapper_path = root/'wrapper-terminal-frozen-policy-verified.json'
        wrapper = read(wrapper_path)
        amend = wrapper['verification_amendment']
        require(original['state']=='incomplete' and original['returncode']==0
                and amend['original_terminal_sha256']==sha(original_path)
                and amend['original_validation_error']==original['validation_error']
                and amend['training_repeated'] is False, 'The read-only policy verification does not bind the original run.')
    prereg = read(pin(root/'preregistration.json',wrapper['preregistration_sha256']))
    registered = pin(DELIVERY/'input-audit/registered-surface-duration-policy.json',prereg['quality_policy_sha256'])
    if wrapper_path!=original_path:
        require(wrapper['verification_amendment']['registered_policy_sha256']==sha(registered),
                'The read-only policy verification used different policy bytes.')
    terminal = read(pin(run/'result.json',wrapper['result_sha256']))
    require(wrapper['state']==terminal['state']=='completed' and wrapper['candidate_commit']==commit
            and wrapper['baseline_commit']==R20_COMMIT and wrapper['fit_method']=='reused_exact_round20_corrected_fit'
            and wrapper['same_round20_fit_and_training_selection'] is True and terminal['gaussian_from_scratch'] is True
            and terminal['format_passed'] and terminal['steps']==18000 and terminal['final_snapshot']=='step-018000',
            'The R21 wrapper, pinned fit or final SH0 endpoint is incomplete.')
    integration_path = run/'gaussian-run/integration.json'
    integration = read(integration_path)
    require(terminal['unified_main_training']==integration and integration['optimizer_updates']==18000
            and integration['initial_points']==20908 and integration['optimizer_initial_state_entries']==0
            and integration['no_prior_gaussian_loaded'] is True,'R21 changed the normal source fit or main training schedule.')
    audit = pin(run/'gaussian-run/densification-audit.json',integration['densification_audit']['sha256'])
    color = pin(run/'gaussian-run/color-constraint.json',integration['sh0_color_constraint']['sha256'])
    frames = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
    _,trials = audit21.verify(read(audit),integration,frames)
    fit = pin(run/FIT_DIR/'result.json',wrapper['head_fit']['correction']['result_sha256'])
    proof = dict(wrapper=descriptor(wrapper_path),integration=descriptor(integration_path),head_alignment=descriptor(fit),
        current_user_view_priority=descriptor(registered),joint=descriptor(run/'fit/joint-body-refinement.json'),
        joint_candidates=descriptor(run/'fit/joint-candidates.npz'),
        shared_body=descriptor(run/'fit/shared-body.json'),color_constraint=descriptor(color),densification_audit=descriptor(audit))
    if wrapper_path!=original_path:
        proof['original_wrapper_terminal'] = descriptor(original_path)
    proof.update(audit17.event_proofs(run,trials))
    candidate_panels = audit17.panels(run,trials)
    for index,panel in enumerate(candidate_panels):
        proof[f'candidate_panel_{index:02d}'] = {key:panel[key] for key in ('file','sha256')}
    proof.update(stage_io.active_pose_proofs(run,wrapper,terminal))
    for name,digest in read(fit)['source_sha256'].items():
        proof['head_fit_preimage_'+Path(name).stem] = descriptor(pin(run/FIT_DIR/'source'/name,digest))
    duration = duration_evidence(run,wrapper,terminal)
    proof.update(duration['proof'])
    surface = attachment_evidence(run,wrapper,terminal,duration['comparison'])
    proof.update(surface['proof'])
    return terminal,wrapper,proof,candidate_panels,duration['comparison'],surface['branches']


def preflight(subject,commit=None):
    old,mid,locked,archive,proof = history(subject)
    if commit is None:
        from review_round21_head_proxy import camera_manifest

        path = camera_manifest(subject)
        return dict(subject=subject,history=proof['round20_review']['sha256'],native=len(locked['matched_frames']),
                    static_cameras=len(read(path)['views']),camera_manifest_sha256=sha(path),
                    prior_auxiliary=[row['frame'] for row in locked['supplement']['frames']],
                    raw_holdouts=locked['new']['protected_holdouts'])
    terminal,wrapper,actual,candidate_panels,duration,surface = completed_training(subject,commit)
    run = root_for(subject)/'run'
    new = stage_io.planned_model(run,PROJECT,wrapper['result_sha256'],18000)
    ten_run = run/'duration-10000'
    ten_desc = actual['duration_secondary_terminal']
    ten_terminal = read(pin(ten_desc['file'],ten_desc['sha256']))
    ten = stage_io.planned_model(ten_run,PROJECT,ten_desc['sha256'],10000)
    old_run = Path(mid.provenance['run'])
    for name in FROZEN_FILES:
        pin(run/name,sha(old_run/name))
        pin(ten_run/name,sha(old_run/name))
    require(new.rows==ten.rows==mid.rows and list(new.rows)==locked['matched_frames']
            and new.provenance['reference_frame_id']==mid.provenance['reference_frame_id']
            and new.provenance['protected_holdouts']==mid.provenance['protected_holdouts']
            and all(review.pose_digest(new,frame)==review.pose_digest(ten,frame)==review.pose_digest(mid,frame) for frame in new.rows),
            'R21 changed a source frame, mask, calibrated camera, reference or fitted pose.')
    proof.update(actual)
    for key in ('round15_review','common_special_posed_meshes','common_reference_mesh','pre_joint_selection',
                'native_roi_with_annotations','close_face_proxy','preregistered_review_inputs'):
        if key in locked['proof']:
            desc = locked['proof'][key]
            proof[key] = descriptor(pin(desc['file'],desc['sha256']))
    models = dict(old=old,mid=mid,ten=ten,new=new)
    parts = {side:round1.delivery_paths(Path(arm.provenance['run']),
                 read(Path(arm.provenance['run'])/'result.json')) for side,arm in models.items()}
    r16.check_source_equivalence(r20.PROJECT,PROJECT,parts['mid'][2]['source_files_sha256'],parts['new'][2]['source_files_sha256'])
    views,static,common,colors = stage_io.common_plans(old,new,locked,parts)
    ten_views,ten_static,_,ten_colors = stage_io.common_plans(old,ten,locked,dict(old=parts['old'],new=parts['ten']))
    static['ten'],colors['ten'] = ten_static['new'],ten_colors['new']
    static['mid'] = locked['static_arms']['new']
    colors['mid'] = locked['terminal_sh0_rgb_range']['new']
    for key,view in views.items():
        historical = locked['static_views'][key]
        require(all(view[field]==historical[field] for field in ('fixed_world_camera','crop_xyxy','common_mesh_sha256')),
                'Legacy static evidence changed the fixed world camera.')
        view['arms']['mid'] = historical['arms']['new']
        view['arms']['ten'] = ten_views[key]['arms']['new']
        view['common_min_center_depth_m']['ten'] = ten_views[key]['common_min_center_depth_m']['new']
        view['common_min_center_depth_m']['mid'] = historical['common_min_center_depth_m']['new']
    pairs = []
    for saved in locked['pairs']:
        frame = saved['frame']
        row = dict(saved)
        for side,model in models.items():
            row[f'{side}_slot'] = model.slots[frame]
            row[f'{side}_pose_sha256'] = review.pose_digest(model,frame)
        require(all(row[f'{side}_slot']==row['old_slot'] for side in SIDES),'A named original pose moved slots.')
        pairs.append(row)
    plan = dict(schema='shuolee-round21-static-ply-head-support-review-v1',subject=subject,commit=commit,
        **{side:model.provenance for side,model in models.items()},output=str(root_for(subject)/'view-comparison'),
        asset=locked['asset'],proof=proof,reviewer_sha256=locked['reviewer_sha256'],round16_reviewer_sha256=sha(Path(r16.__file__)),
        round21_reviewer_sha256=sha(Path(__file__)),static_reviewer_sha256=sha(Path(static_replay.__file__)),
        pairs=pairs,matched_frames=locked['matched_frames'],static_views=views,static_arms=static,common_reference=common,
        common_special=locked['common_special'],special_016=locked['special_016'],native_regions=locked['native_regions'],
        independent_annotations=locked['independent_annotations'],near_views=locked['near_views'],roi_sha256=locked['roi_sha256'],
        near_roi_sha256=locked['near_roi_sha256'],brightness_contract=locked['brightness_contract'],
        pose_relative_azimuth=locked['pose_relative_azimuth'],terminal_sh0_rgb_range=colors,
        candidate_panels=candidate_panels,fit_audit=locked['fit_audit'],duration=duration,surface_attachment=surface,quality_accepted=False)
    plan['supplement'],plan['holdout'] = stage_io.pose_plans(run,plan,terminal)
    ten_wrapper = dict(wrapper['duration_comparison']['intermediate'],head_fit=wrapper['head_fit'])
    ten_proof = stage_io.active_pose_proofs(ten_run,ten_wrapper,ten_terminal)
    ten_supplement,ten_holdout = stage_io.pose_plans(ten_run,dict(plan,proof=dict(proof,**ten_proof),new=ten.provenance),ten_terminal)
    for key in ('supplement_poses','heldout_mesh'):
        with np.load(proof[key]['file'],allow_pickle=False) as first, np.load(ten_proof[key]['file'],allow_pickle=False) as second:
            require(first.files==second.files and all(np.array_equal(first[field],second[field]) for field in first.files),
                    'The 10k and 18k render branches do not share one fixed auxiliary/evaluation geometry.')
    plan['ten_pose_evidence'] = dict(supplement=ten_supplement,holdout=ten_holdout,proof=ten_proof)
    require([row['frame'] for row in plan['supplement']['frames']]==[row['frame'] for row in locked['supplement']['frames']]
            and all(all(a[key]==b[key] for key in ('source_sha256','mask_sha256','camera','valid_mask_sha256','box'))
                    for a,b in zip(plan['supplement']['frames'],locked['supplement']['frames'],strict=True)),
            'R21 changed the frozen face-only observation or geometry inputs.')
    require([row['frame'] for row in plan['holdout']['frames']]==[row['frame'] for row in locked['holdout']['frames']]
            and all(all(a[key]==b[key] for key in ('source_sha256','mask_sha256','camera'))
                    for a,b in zip(plan['holdout']['frames'],locked['holdout']['frames'],strict=True)),
            'R21 changed an evaluation-only camera or pose observation.')
    plan['static_grid'] = static_replay.prepare(plan,old.provenance,mid.provenance)
    return plan


def worker(path,side,fd):
    plan = read(path)
    pin(Path(__file__),plan['round21_reviewer_sha256'])
    pin(Path(static_replay.__file__),plan['static_reviewer_sha256'])
    static_replay.render_grid(path,STATIC_NAMES[side],fd)
    if side.startswith('before'):
        return
    if side in ('ten','new'):
        r16.actual_worker(path,side,fd)
        r16.native_float_worker(path,side,fd)
        r16.render_common_worker(path,side,fd)
        if side=='new':
            dolly.render(path,side,fd)
        else:
            def without_older_depth_gate(current):
                doc = read(current)
                if current==path:
                    for pair in doc['pairs']:
                        spec = doc['near_views'][pair['role']]
                        endpoint = dolly.endpoint(doc['subject'],doc['ten'],[pair['frame']])
                        centers = endpoint['centers'][pair['frame']]
                        E = np.asarray(spec['near_camera']['w2c'])
                        spec['cpu_baseline_min_near_center_depth_m']['R16'] = float((centers@E[2,:3]+E[2,3]).min())
                return doc
            rebound(dolly.render,read=without_older_depth_gate)(path,side,fd)
    auxiliary.render(path,side,fd)
    pose.render_heldouts(path,side,fd)


def collected(out,plan,history_receipt):
    hist_path = Path(plan['proof']['round20_review']['file']).parent
    arms = {}
    for side,archived in (('old','old'),('mid','new')):
        previous = history_receipt[archived]
        arm = dict(previous,panels={})
        for label,item in previous['panels'].items():
            if label.startswith(('face_only_','face_control_','heldout_')):
                continue
            filename = item['file'] if side=='old' else item['file'].removesuffix('-new.png')+'-mid.png'
            source = pin(hist_path/item['file'],item['sha256'])
            target = out/filename
            if not target.exists():
                shutil.copyfile(source,target)
            pin(target,item['sha256'])
            arm['panels'][label] = dict(item,file=filename)
        arms[side] = arm
    for side in ('ten','new'):
        result = read(out/f'{side}-result.json')
        for suffix in ('common','near'):
            extra = read(out/f'{side}-{suffix}.json')
            result['panels'].update(extra.pop('panels'))
            result.update(extra)
        result['float_native'] = read(out/f'{side}-float-native.json')
        result['float_actual'] = read(out/f'{side}-float-actual.json')
        arms[side] = result
    for side,arm in arms.items():
        for suffix in ('supplement','heldout'):
            extra = read(out/f'{side}-{suffix}.json')
            arm['panels'].update(extra.pop('panels'))
            arm.update(extra)
    return arms


def verify_ten_readback(out,plan,arm):
    """Compare actual 10k media with fixed native 10k render panels after quantization."""
    evidence = plan['ten_pose_evidence']
    runs = Path(plan['ten']['run'])
    checks = {}
    for label,group,box_key,fields in (
            ('face_only',evidence['supplement']['frames'],'box',(('gaussian',3),('alpha',5))),
            ('heldout',evidence['holdout']['frames'],'crop_xyxy',(('black',1),('white',2),('alpha',3)))):
        for row in group:
            frame = row['frame']
            entry = arm['panels'][f'{label}_{frame}']
            require(list(row[box_key])==entry['crop_xyxy'],'The 10k replay review box moved.')
            x0,y0,x1,y1 = row[box_key]
            width,height = x1-x0,y1-y0
            errors = {}
            with Image.open(pin(out/entry['file'],entry['sha256'])) as panel:
                for key,column in fields:
                    info = row['files'][key]
                    file = runs/auxiliary.FOLDER/info['file'] if label=='face_only' else Path(info['file'])
                    mode = 'L' if key=='alpha' else 'RGB'
                    with Image.open(pin(file,info['sha256'])) as source:
                        region = source if label=='face_only' else source.crop(row[box_key])
                        expected = np.asarray(region.convert(mode),dtype=np.int16)
                    observed = np.asarray(panel.crop((column*width,0,(column+1)*width,height)).convert(mode),dtype=np.int16)
                    require(expected.shape==observed.shape,'The 10k terminal native readback dimensions differ.')
                    errors[key] = int(np.abs(expected-observed).max())
                    require(errors[key]<=1,'The 10k terminal native readback differs by more than one 8-bit level.')
            checks[f'{label}:{frame}'] = errors
    return checks


def assemble(path,plan):
    out = path.parent
    require(read(path)==json.loads(json.dumps(plan,allow_nan=False)),'The saved R21 review plan differs.')
    pin(out/'rois.npz',plan['roi_sha256'])
    pin(out/'near-rois.npz',plan['near_roi_sha256'])
    desc = plan['proof']['round20_review']
    old = read(pin(desc['file'],desc['sha256']))
    arms = collected(out,plan,old)
    for frame in plan['matched_frames']:
        expected = arms['new']['float_native'][frame]
        for side in SIDES:
            observed = arms[side]['float_native'][frame]
            require(all(observed[key]==expected[key] for key in ('camera','source_sha256','mask_sha256')),
                    'A native comparator changed the registered pixel source or camera.')
    r13.check_render_receipts(plan,arms['old'],arms['new'])
    for side in ('mid','ten'):
        comparator = dict(plan,static_arms=dict(plan['static_arms'],new=plan['static_arms'][side]),
            static_views={key:dict(view,arms=dict(view['arms'],new=view['arms'][side]))
                          for key,view in plan['static_views'].items()})
        r13.check_render_receipts(comparator,arms['old'],arms[side])
    r13.check_source_pixels(out,plan,arms.values())
    auxiliary.verify_render_receipts(plan,arms)
    pose.verify_holdout(plan,arms)
    ten_readback = verify_ten_readback(out,plan,arms['ten'])
    for name,view in plan['near_views'].items():
        rows = [arms[side]['physical_near'][name] for side in SIDES]
        require(all(row['camera']==rows[0]['camera'] and row['pose_slot']==view['slot'] for row in rows),
                'The actual-pose close camera differs across endpoints.')
    brightness = r13.brightness_diagnostic(out,plan,arms['old'],arms['new'])
    for side in ('mid','ten'):
        prior = r13.brightness_diagnostic(out,plan,arms['old'],arms[side])
        for role,item in brightness.items():
            require(item['old']==prior[role]['old'] and item['crop_relative_y_band']==prior[role]['crop_relative_y_band'],
                    'The fixed native torso band moved.')
            item[side] = prior[role]['new']
    labels = ('round15_'+R15_COMMIT[:7],'round20_'+R20_COMMIT[:7],'round21_10k_'+plan['commit'][:7])
    panels = rebound(r20.merge,SIDES=SIDES,HISTORICAL_LABELS=labels)(out,plan,arms)
    for panel in panels:
        panel['rows'][-1] = 'round21_18k_'+plan['commit'][:7]
    static_evidence,static_panels = static_replay.combine(out,plan)
    curve = training_curve.build(out,plan['proof']['main_training_loss'])
    output = out/'result.json'
    require(not output.exists(),'Preserve the completed R21 comparison receipt.')
    review.save(output,dict(state='round21_delivery_views_complete',subject=plan['subject'],commit=plan['commit'],
        plan_sha256=sha(path),assembler_sha256=sha(Path(__file__)),panels=panels+static_panels,**arms,proof=plan['proof'],
        fit_audit=plan['fit_audit'],supplement=plan['supplement'],heldout=plan['holdout'],surface_attachment=plan['surface_attachment'],
        static_grid=static_evidence,static_grid_proof=plan['static_grid']['proof'],
        duration=plan['duration'],training_curve=curve,
        same_run_10k_pose_readback=ten_readback,
        special_016=plan['special_016'],common_reference=plan['common_reference'],near_views=plan['near_views'],
        independent_annotations=plan['independent_annotations'],brightness_diagnostic=brightness,
        terminal_sh0_rgb_range=plan['terminal_sh0_rgb_range'],pose_relative_azimuth=plan['pose_relative_azimuth'],
        native_torso_band=r20.r19.stage.garment_audit(out,plan,arms),
        geometry_summary={side:r20.r19.r17.geometry_summary(plan[side]) for side in SIDES},candidate_panels=plan['candidate_panels'],
        all_paired_native_same_source_mask_camera=True,exact_round20_fit_preserved=True,
        actual_portable_static_cameras=50,common_historical_geometry_fixed=True,quality_accepted=False))
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
        parser.error('Use the full completed R21 commit.')
    plan = preflight(args.subject,args.commit)
    if args.action=='check':
        print(json.dumps(plan if not args.commit else dict(subject=args.subject,native=len(plan['matched_frames']),
            auxiliary=len(plan['supplement']['frames']),heldout=len(plan['holdout']['frames']),
            static=len(plan['static_grid']['camera_roster'])),ensure_ascii=False),flush=True)
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
        for side in WORKERS:
            with (out/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                               pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assemble(path,plan)


if __name__=='__main__':
    main()
