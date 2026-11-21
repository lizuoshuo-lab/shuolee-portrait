"""Compare R15 densification with the exact R14 fitted endpoint."""

import argparse
import ast
import fcntl
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_smpl_delivery_round14 as r14

r13 = r14.r13

r10, r9, review, round1 = r13.r10, r13.r9, r13.review, r13.round1
read, require, sha, pin = r13.read, r13.require, r13.sha, r13.pin
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-densification-20261111')
DELIVERY = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round15'
BASE_COMMIT = '7f325f2ed89e91e0d028256329a57bc021060fa2'
BASE_SHA = '04d3cf14ea2aaf65cf9e5612f0b95d8f6388029d90af6ba9f6344e27d7255705'
SUBJECTS = r13.SUBJECTS
FIXED_FILES = (
    'fit/fitted.npz', 'fit/selection.json', 'fit/selection-before-joint.json',
    'fit/joint-candidates.npz', 'fit/joint-body-refinement.json', 'fit/official-fit-result.json',
    'fit/shared-body.json', 'data/training-inputs.json', 'data/inputs.json', 'data/observations.npz',
)


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round15-{subject}'


def descriptor(path):
    return dict(file=str(path), sha256=sha(path))


def verify_base(subject):
    receipt_path = pin(DELIVERY.with_name('smpl-delivery-round14')/'receipt.json', BASE_SHA)
    receipt = read(receipt_path)
    arm = receipt['subjects'][subject]
    require(receipt['commit'] == arm['commit'] == BASE_COMMIT, 'Wrong R14 training baseline.')
    compare_path = pin(arm['review'], arm['review_sha256'])
    comparison = read(compare_path)
    plan_path = pin(compare_path.with_name('plan.json'), comparison['plan_sha256'])
    lock = read(plan_path)
    require(comparison['state'] == 'round14_delivery_views_complete'
            and comparison['commit'] == lock['commit'] == BASE_COMMIT
            and lock['subject'] == subject, 'Invalid R14 comparison archive.')
    old = review.planned_model(Path(arm['result']).parent, r14.PROJECT, arm['result_sha256'])
    require(old.provenance == lock['new'] and comparison['all_paired_native_same_source_mask_camera_pose'],
            'R14 comparison does not bind the training endpoint.')
    require(arm['exact_round13_fit_reused'], 'R14 baseline lacks the verified input lock.')
    proofs = dict(round14_receipt=descriptor(receipt_path), round14_review=descriptor(compare_path),
                  round14_review_plan=descriptor(plan_path),
                  fixed_native_rois=descriptor(pin(plan_path.with_name('rois.npz'), lock['roi_sha256'])))
    return old, arm, comparison, lock, proofs


def check_densification(audit, integration):
    events, totals = audit['events'], audit['totals']
    require(audit['state'] == 'completed' and audit['radius_units'].startswith('native pixels')
            and audit['stock_events'] == len(events) == integration['stock_densifications']
            and all(a['step'] < b['step'] for a,b in zip(events,events[1:]))
            and audit['terminal'] == integration['densification_audit']['terminal']
            and audit['terminal']['live_rows_over_local_cap'] == 0,
            'Densification audit does not match the terminal.')
    for key, value in totals.items():
        require(value == sum(event.get(key,0) for event in events), f'Invalid densification total: {key}.')
    for event in events:
        before, after = (event[f'screen_candidates_{position}_stock'] for position in ('before','after'))
        require(0 <= after <= before and before-after == event['screen_parents_replaced_by_gradient_split'],
                'Screen candidate topology mapping differs.')
        if event['screen_limit_px'] is None:
            require(before == after == 0, 'Unexpected early screen-size decision.')
            continue
        candidates = event['size_candidates']
        require(after == event['screen_candidates'] <= candidates
                and 0 <= event['foreground_supported'] <= event['assessed'] <= candidates
                and candidates == event['protected']+event['size_pruned']+event['size_replaced']
                and event['screen_pruned'] <= event['size_pruned']
                and event['replacement_children'] == 2*event['size_replaced'],
                'Native size decisions fail candidate accounting.')
        for key, count in (('removal_trial',event['size_pruned']),('replacement_trial',event['size_replaced'])):
            if count:
                trial = event[key]
                check = trial['final_trial'] if key == 'replacement_trial' else trial
                require(trial['accepted'] and trial['proposed'] == count and check['accepted']
                        and len(check['frames']) == 24 and all(not row['failures'] for row in check['frames']),
                        'A size-driven removal or replacement lacks complete foreground protection.')
                if key == 'replacement_trial':
                    radius = check['native_radius_check']
                    require(radius['all_decreased'] and len(radius['before_px']) == len(radius['child_max_px']) == count
                            and all(a>b for a,b in zip(radius['before_px'],radius['child_max_px'],strict=True)),
                            'A size-driven replacement failed to reduce its raster radius.')


def verified_retry(root, commit):
    path = DELIVERY/'input-audit/006-retry-parity.json'
    record = read(path)
    require(record['state'] == 'retry_passed_original_step1000_gate' and record['subject'] == '006'
            and record['candidate_commit'] == commit and record['fit_reused']
            and not record['root_cause_established'] and record['max_abs_gate'] == .005
            and record['mean_abs_gate'] == 2e-6 and Path(record['retry']['root']) == root,
            'Unexpected unchanged-retry record.')
    first, retry = record['first'], record['retry']
    for attempt, items in ((first,('parity','terminal','launch','entry_log')),(retry,('parity','launch'))):
        for key in items:
            pin(attempt[key]['path'],attempt[key]['sha256'])
    failed = read(Path(first['terminal']['path']))
    require(failed['state'] == 'failed' and failed['candidate_commit'] == commit
            and first['parity']['passed'] is False and retry['parity']['passed'] is True,
            '006 retry did not preserve the original rejected snapshot.')
    for attempt in (first,retry):
        parity = read(Path(attempt['parity']['path']))
        require(parity['checks'] == attempt['parity']['checks'] and parity['passed'] == attempt['parity']['passed'],
                '006 retry and snapshot parity evidence differ.')
    for name, digest in record['same_input_and_profile_sha256'].items():
        pin(Path(first['root'])/name,digest)
        pin(root/name,digest)
    return descriptor(path)


def completed_training(subject, commit):
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'Pin the exact completed R15 candidate.')
    r13.r9.transfer.rebound(r13.r9.transfer.base.common.git_state, PROJECT=PROJECT)(commit)
    root = root_for(subject)
    run = root/'run'
    wrapper_path = root/'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    terminal = read(pin(run/'result.json', wrapper['result_sha256']))
    require(wrapper['state'] == terminal['state'] == 'completed' and wrapper['candidate_commit'] == commit
            and terminal['gaussian_from_scratch'] is terminal['format_passed'] is terminal['fit_reused'] is True
            and terminal['fit_calls'] == 0 and terminal['steps'] == 10000
            and terminal['final_snapshot'] == 'step-010000', 'Missing verified R15 fixed-fit terminal.')
    integration_path = run/'gaussian-run/integration.json'
    integration = read(integration_path)
    require(integration['optimizer_updates'] == 10000 and integration['initial_points'] == 20908
            and integration['optimizer_initial_state_entries'] == 0 and integration['no_prior_gaussian_loaded'] is True,
            'R15 Gaussian training did not start from the protected empty optimizer.')
    color = pin(run/'gaussian-run/color-constraint.json', integration['sh0_color_constraint']['sha256'])
    densification = pin(run/'gaussian-run/densification-audit.json', integration['densification_audit']['sha256'])
    check_densification(read(densification), integration)
    preregistration = read(pin(root/'preregistration.json',wrapper['preregistration_sha256']))
    fixtures = pin(DELIVERY/'input-audit/cuda-densification-proof.json',preregistration['cuda_fixture_sha256'])
    require(read(fixtures)['state'] == 'passed', 'The registered CUDA fixture evidence is incomplete.')
    require(wrapper['fit_method'] == 'reused_exact_round14_fit'
            and wrapper['same_round14_fit_and_training_selection'] is True,
            'R15 wrapper does not bind the exact R14 fitted training selection.')
    proof = dict(wrapper=descriptor(wrapper_path), integration=descriptor(integration_path),
        color_constraint=descriptor(color), densification_audit=descriptor(densification),
        cuda_densification_proof=descriptor(fixtures), joint=descriptor(run/'fit/joint-body-refinement.json'),
        joint_candidates=descriptor(run/'fit/joint-candidates.npz'))
    if subject == '006':
        proof['unchanged_retry'] = verified_retry(root,commit)
    return terminal, wrapper, proof


def check_source_equivalence(left, right, old, new):
    method_contracts = {
        'src/avatar_core/scene/gaussian_model.py': (
            'GaussianModel', {'setup_functions', 'get_xyz', 'get_scaling', 'get_rotation',
                              'get_features', 'get_opacity', 'get_covariance', 'select_mesh_by_timestep',
                              'load_ply', 'save_ply', 'construct_list_of_attributes'}),
        'src/avatar_core/scene/smplx_gaussian_model.py': (
            'SMPLXGaussianModel', {'select_mesh_by_timestep', 'get_covariance', 'export_world_ply'}),
    }
    require(set(old) == set(new) and method_contracts.keys() <= old.keys(),
            'The renderer source inventory changed.')
    round1.check_source_equivalence(left, right, {k:v for k,v in old.items() if k not in method_contracts},
                                   {k:v for k,v in new.items() if k not in method_contracts})

    def render_methods(path, class_name, names):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        methods = {node.name:ast.dump(node) for node in cls.body if isinstance(node,ast.FunctionDef)
                   and node.name in names}
        require(set(methods) == names, f'A {class_name} replay method is absent.')
        return methods

    for source, (class_name, names) in method_contracts.items():
        before, after = pin(left/source,old[source]), pin(right/source,new[source])
        require(render_methods(before,class_name,names) == render_methods(after,class_name,names),
                f'A {class_name} replay method changed.')


def preflight(subject, commit=None):
    old, arm, historical, lock, proofs = verify_base(subject)
    if commit is None:
        return dict(subject=subject, baseline=arm['commit'], reference=lock['common_reference'],
                    native_roles=[row['role'] for row in lock['pairs']], selected=len(old.rows))
    terminal, wrapper, proof = completed_training(subject, commit)
    proof.update(proofs)
    run, old_run = root_for(subject)/'run', Path(old.provenance['run'])
    new = review.planned_model(run, PROJECT, wrapper['result_sha256'])
    for name in FIXED_FILES:
        pin(run/name, sha(old_run/name))
    require(all(old.provenance[key] == new.provenance[key] for key in
                ('fitted_sha256', 'training_inputs_sha256', 'protected_holdouts', 'reference_frame_id'))
            and list(old.rows) == list(new.rows) and len(old.rows) == 24
            and all(old.rows[name] == new.rows[name] and review.pose_digest(old,name) == review.pose_digest(new,name)
                    for name in old.rows), 'R15 changed a native camera, mask, fitted pose or selected frame.')
    parts = dict(old=round1.delivery_paths(old_run,read(old_run/'result.json')),new=round1.delivery_paths(run,terminal))
    check_source_equivalence(r14.PROJECT, PROJECT, parts['old'][2]['source_files_sha256'],
                                   parts['new'][2]['source_files_sha256'])
    for field in ('fitted', 'training_rows'):
        lhs = parts['old'][2]['inputs_sha256'].get(field)
        require(lhs is not None and lhs == parts['new'][2]['inputs_sha256'].get(field),
                f'R15 changed the saved binding input {field}.')
    views, arms, common, colors = r14.common_plans(old, new, lock, parts)
    require(all(color['points_above_one'] == color['points_below_zero'] == 0 for color in colors.values()),
            'Delivered trained SH0 colors lie outside LDR.')
    pairs = []
    for pair in lock['pairs']:
        frame = pair['frame']
        pose = review.pose_digest(old, frame)
        require(old.slots[frame] == new.slots[frame] == pair['new_slot'] and pose == pair['new_pose_sha256'],
                'The locked R14 native pose differs from the actual pair.')
        pairs.append(dict(pair, old_slot=old.slots[frame], new_slot=new.slots[frame],
                          old_pose_sha256=pose, new_pose_sha256=pose))
    special = {}
    for frame, row in lock['special_016'].items():
        special[frame] = dict(row, R15=row['R14'])
        require((frame in new.rows) == (row['R14'] == 'trained'), 'A special-frame identity changed.')
    if special:
        proof['common_special_posed_meshes'] = lock['proof']['common_special_posed_meshes']
    proof['common_reference_mesh'] = descriptor(Path(common['file']))
    proof['pre_joint_selection'] = descriptor(run/'fit/selection-before-joint.json')
    selection = read(run/'fit/selection.json')
    return dict(schema='shuolee-round15-densification-review-v1',subject=subject,commit=commit,old=old.provenance,
        new=new.provenance,proof=proof,reviewer_sha256=r13.R10_REVIEWER_SHA,
        round15_reviewer_sha256=sha(Path(__file__)), asset=str(review.ASSET), output=str(root_for(subject)/'view-comparison'),
        pairs=pairs,matched_frames=list(old.rows),static_views=views,static_arms=arms,common_reference=common,
        common_special=lock['common_special'],special_016=special,pose_relative_azimuth=lock['pose_relative_azimuth'],
        selections=dict(old=selection,new=selection,round14=selection,pre_joint=read(run/'fit/selection-before-joint.json')),
        native_regions=lock['native_regions'],roi_sha256=lock['roi_sha256'], brightness_contract=lock['brightness_contract'],
        historical_R9_native=historical['historical_R9_native'],
        historical_R10_native=historical['historical_R10_native'],
        historical_R10_brightness=historical['historical_R10_brightness'],
        same_native_score_mask=True, same_native_fitted_pose=True, terminal_sh0_rgb_range=colors,
        common_static_source_rgb_truth=False,quality_accepted=False)


float_summary = r14.float_summary
native_float_worker = r14.native_float_worker
actual_worker = r14.actual_worker


def render_common_worker(plan_path,side,lease_fd):
    """Replay only the trained local Gaussian parameters on the exact common mesh."""
    require(os.path.samestat(os.fstat(lease_fd),review.GPU_LEASE.stat()), 'Missing inherited GPU lease.')
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan,out = read(plan_path),plan_path.parent
    pin(Path(__file__),plan['round15_reviewer_sha256'])
    sys.path.insert(0,str(Path(plan[side]['project'])/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import inspect_bound_ply
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    from review_bound_scale_coverage import face_alpha, scale_pixels

    rt.configure(Path(plan['asset']).parents[1])
    from gaussian_renderer import render
    from scene.smplx_gaussian_model import SMPLXGaussianModel

    ref,arm = plan['common_reference'],plan['static_arms'][side]
    with np.load(pin(ref['file'],ref['sha256']),allow_pickle=False) as archive:
        vertices,faces = archive['vertices_world'],archive['faces']
        require(str(archive['reference_frame_id'].item()) == ref['frame'], 'Common mesh frame differs.')
    special = plan['common_special']
    posed = [vertices]
    if special:
        mesh = special[0]
        with np.load(pin(mesh['common_mesh_file'],mesh['common_mesh_file_sha256']),allow_pickle=False) as archive:
            require(np.array_equal(faces,archive['faces']),'Special R10 mesh topology differs.')
            indices = {frame:index for index,frame in enumerate(archive['frame_ids'].tolist())}
            positions = archive['vertices_world']
            for item in special:
                index = indices[item['frame']]
                require(archive['pose_indices'][index] == item['old_slot']
                        and archive['image_sha256'][index].item() == item['source_sha256']
                        and archive['mask_sha256'][index].item() == item['score_mask_sha256'],
                        'Special R10 mesh archive identity differs.')
                posed.append(positions[index])
        require(all(array.shape == vertices.shape and array.dtype == vertices.dtype and np.isfinite(array).all()
                    for array in posed), 'Invalid special R10 mesh sequence.')
    path = pin(arm['bound_ply'],arm['bound_ply_sha256'])
    ids,counts = inspect_bound_ply(path,len(faces))
    records,panels = {},{}
    with gpu_lock(),torch.no_grad():
        require(torch.cuda.device_count() == 1, 'Expose one CUDA device.')
        model = SMPLXGaussianModel(0)
        model.vertices_sequence = torch.as_tensor(np.stack(posed),device='cuda')
        model.faces = torch.as_tensor(faces,dtype=torch.long,device='cuda')
        model.num_timesteps = len(posed)
        model.load_ply(str(path))
        model.binding_counter = torch.as_tensor(counts,dtype=torch.int32,device='cuda')
        model.select_mesh_by_timestep(0)
        require(model.optimizer is None and model.active_sh_degree == 0 and len(model.get_xyz) == arm['points']
                and torch.equal(model.binding.long(),torch.as_tensor(ids,device='cuda')),
                'Loaded local parameters or ordered triangle IDs differ.')
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        positions = model.get_xyz.detach().cpu().numpy()
        for name,view in plan['static_views'].items():
            row,crop = view['fixed_world_camera'],view['crop_xyxy']
            cam,record = camera(row,0,pose_index=0)
            E = np.asarray(row['w2c'])
            depth = float((positions@E[2,:3]+E[2,3]).min())
            require(depth > .2 and abs(depth-view['common_min_center_depth_m'][side]) < 5e-6,
                    'CUDA center depth differs from the matched camera plan.')
            pixels,error = scale_pixels(render,render_pose,cam,model,pipe,black,white,0,1.)
            x0,y0,x1,y1 = crop
            alpha = pixels[2][y0:y1,x0:x1]
            occupied = pixels[2] >= .5
            panel_name = f'common_{name}'
            panel = dict(r10.strip(out/f'{panel_name}-{side}.png',crop,pixels),frame=ref['frame'],
                         columns=['black','white','alpha'])
            panels[panel_name] = panel
            records[name] = dict(camera=record,fixed_world_camera=row,common_mesh_sha256=ref['sha256'],
                panel=panel,actual_reference_frame=arm['reference_frame'],local_bound_ply_sha256=arm['bound_ply_sha256'],
                source_rgb_compared=False,composition_max_abs=error,min_center_depth_m=depth,
                alpha_ge_half_in_crop=int(occupied[y0:y1,x0:x1].sum()),
                alpha_ge_half_outside_crop=int(occupied.sum()-occupied[y0:y1,x0:x1].sum()),
                alpha_mass_in_crop=float(alpha.sum(dtype=np.float64)),float_rgb=float_summary(pixels))
        common_special = {}
        with np.load(pin(out/'rois.npz',plan['roi_sha256']),allow_pickle=False) as rois:
            for slot,item in enumerate(special,start=1):
                frame,role = item['frame'],item['role']
                model.select_mesh_by_timestep(slot)
                row = dict(item['camera'],frame=frame)
                cam,record = camera(row,0,pose_index=slot)
                pixels,error = scale_pixels(render,render_pose,cam,model,pipe,black,white,slot,1.)
                delta = None
                if side == 'old':
                    proof = plan['proof']['round14_review']
                    baseline_path = pin(proof['file'], proof['sha256'])
                    expected = read(baseline_path)['new']['panels'][role]
                    require(expected['crop_xyxy'] == item['crop_xyxy']
                            and expected['columns'] == ['source_rgb','black','white','alpha'],
                            'The historical R14 controlled crop differs.')
                    x0,y0,x1,y1 = expected['crop_xyxy']
                    width = x1-x0
                    with Image.open(pin(baseline_path.parent/expected['file'],expected['sha256'])) as saved:
                        original = np.asarray(saved.convert('RGB'),dtype=np.int16)[:,width:2*width]
                    observed = np.rint(np.clip(pixels[0][y0:y1,x0:x1],0,1)*255).astype(np.int16)
                    delta = int(np.abs(observed-original).max())
                    require(delta <= 1,'Common special replay differs from the actual R14 archive.')
                with Image.open(pin(item['source'],item['source_sha256'])) as source:
                    rgb = np.asarray(source.convert('RGB'))
                with Image.open(pin(item['score_mask'],item['score_mask_sha256'])) as mask:
                    matte = np.asarray(mask.convert('L'),dtype=np.float32)/255.
                stats = r10.mask_metrics(pixels[2],matte,np.zeros(matte.shape,dtype=bool))
                regions = {}
                for name,digest in plan['native_regions'][role].items():
                    roi = rois[f'{role}_{name}']
                    require(r9.native.hash_mask(roi) == digest,'Special native ROI changed.')
                    alpha = np.rint(np.clip(pixels[2][roi],0,1)*255)
                    regions[name] = dict(alpha=face_alpha(pixels[2],roi),alpha_u8_lt_243=int((alpha<243).sum()),
                        black_rgb_mae=float(np.abs(np.clip(pixels[0][roi],0,1)-rgb[roi].astype(np.float32)/255).mean()))
                for label,crop in ((role,item['crop_xyxy']),(f'{role}_head',item['head_crop_xyxy'])):
                    if crop:
                        panels[label] = dict(r10.strip(out/f'{label}-{side}.png',crop,(rgb,*pixels)),frame=frame,
                                             columns=['source_rgb','black','white','alpha'])
                common_special[frame] = dict(camera=record,crop_xyxy=item['crop_xyxy'],head_crop_xyxy=item['head_crop_xyxy'],
                    common_mesh_file_sha256=item['common_mesh_file_sha256'],common_mesh_pose_index=item['old_slot'],
                    source_sha256=item['source_sha256'],score_mask_sha256=item['score_mask_sha256'],
                    local_bound_ply_sha256=arm['bound_ply_sha256'],composition_max_abs=error,archive_max_abs_8bit=delta,
                    R13_fitted_pose_used=False,R14_fitted_pose_used=False,R15_fitted_pose_used=False,mask_metrics=stats,
                    region_metrics=regions,float_rgb=float_summary(pixels))
    review.save(out/f'{side}-common.json',dict(common=records,common_special=common_special,panels=panels,optimizer_updates=0))


def fit_only_panel(out, plan):
    if not plan['special_016']:
        return []
    proof = plan['proof']['round14_review']
    historical = pin(proof['file'],proof['sha256'])
    panels = []
    for item in read(historical)['panels']:
        if item['view'] != 'available_record' or plan['special_016'][item['frame']]['R15'] != 'fit_only':
            continue
        source = pin(historical.parent/item['file'],item['sha256'])
        shutil.copyfile(source,out/item['file'])
        panels.append(dict(item,rows=['round10_historical', 'round14_round15_identical_fit']))
    return panels


def assemble(path, expected):
    plan = read(path)
    require(plan == json.loads(json.dumps(expected,allow_nan=False)), 'The saved comparison plan changed.')
    out = path.parent
    pin(out/'rois.npz',plan['roi_sha256'])
    prior = plan['proof']['round14_review']
    baseline_native = read(pin(prior['file'],prior['sha256']))['new']['float_native']
    results = []
    for side in ('old','new'):
        native = read(out/f'{side}-result.json')
        common = read(out/f'{side}-common.json')
        native['panels'].update(common.pop('panels'))
        native.update(common)
        native['float_native'] = read(out/f'{side}-float-native.json')
        native['float_actual'] = read(out/f'{side}-float-actual.json')
        require(set(native['float_native']) == set(plan['matched_frames'])
                and set(native['float_actual']) == set(plan['static_views']), 'Float audit missed a paired render.')
        results.append(native)
    old,new = results
    for frame, record in old['float_native'].items():
        other, baseline = new['float_native'][frame],baseline_native[frame]
        require(record['source_sha256'] == other['source_sha256'] == baseline['source_sha256']
                and record['mask_sha256'] == other['mask_sha256'] == baseline['mask_sha256']
                and record['camera'] == other['camera'] == baseline['camera']
                and record['archive_max_abs_8bit'] <= 1 and other['archive_max_abs_8bit'] <= 1,
                'The paired float native camera, image, mask or archive differs.')
        require(record['black_float_sha256'] == baseline['black_float_sha256']
                and record['alpha_float_sha256'] == baseline['alpha_float_sha256'],
                'The reproduced R14 native float RGB or alpha differs from its archive.')
    r13.check_render_receipts(plan,old,new)
    r13.check_source_pixels(out,plan,results)
    brightness = r13.brightness_diagnostic(out,plan,old,new)
    panels = r10.merge(out,plan,old,new)
    for panel in panels:
        panel['rows'] = [f'round14_{BASE_COMMIT[:7]}',f'round15_{plan["commit"][:7]}']
    panels += fit_only_panel(out,plan)
    path_out = out/'result.json'
    require(not path_out.exists(),'Preserve the completed densification review receipt.')
    review.save(path_out,dict(state='round15_delivery_views_complete',subject=plan['subject'],commit=plan['commit'],
        plan_sha256=sha(path),assembler_sha256=sha(Path(__file__)),panels=panels,old=old,new=new,proof=plan['proof'],
        special_016=plan['special_016'],common_reference=plan['common_reference'],
        pose_relative_azimuth=plan['pose_relative_azimuth'],terminal_sh0_rgb_range=plan['terminal_sh0_rgb_range'],
        historical_R9_native=plan['historical_R9_native'],historical_R10_native=plan['historical_R10_native'],
        historical_R10_brightness=plan['historical_R10_brightness'],brightness_diagnostic=brightness,
        all_paired_native_same_source_mask_camera_pose=True,round14_float_native_bitwise_equal=True,quality_accepted=False))
    print(f'{path_out} {sha(path_out)}',flush=True)


def main():
    if len(sys.argv) == 5 and sys.argv[1] in ('_worker','_common'):
        path,side,fd = Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
        if sys.argv[1] == '_worker':
            actual_worker(path,side,fd)
            native_float_worker(path,side,fd)
        render_common_worker(path,side,fd)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render','assemble'))
    parser.add_argument('subject',choices=SUBJECTS)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action in ('check','assemble'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.action != 'check' and not args.commit:
        parser.error('Review requires a completed R15 candidate commit.')
    plan = preflight(args.subject,args.commit)
    if args.action == 'check':
        print(json.dumps({key:plan[key] for key in ('subject','baseline','reference','native_roles','selected',
            'matched_frames','terminal_sh0_rgb_range') if key in plan}),flush=True)
        return
    out = Path(plan['output'])
    path = out/'plan.json'
    if args.action == 'assemble':
        assemble(path,plan)
        return
    with review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        roi = plan['proof']['fixed_native_rois']
        shutil.copyfile(pin(roi['file'],roi['sha256']),out/'rois.npz')
        review.save(path,plan)
        for side in ('old','new'):
            with (out/f'{side}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),side,str(lease.fileno())],
                    pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assemble(path,plan)


if __name__ == '__main__':
    main()
