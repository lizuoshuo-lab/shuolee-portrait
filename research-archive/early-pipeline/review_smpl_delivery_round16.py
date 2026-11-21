"""Compare R16 densification with the exact R15 fitted endpoint."""

import argparse
import fcntl
import json
import math
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
import review_smpl_delivery_round15 as r15

r14, r13 = r15.r14, r15.r13

r10, r9, review, round1 = r13.r10, r13.r9, r13.review, r13.round1
read, require, sha, pin = r13.read, r13.require, r13.sha, r13.pin
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-size-optimization-20261111')
DELIVERY = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round16'
BASE_COMMIT = '6785a212b304d38a31f699933b23096d38ceb4a4'
BASE_SHA = '617dc00c2cc6fa1e913732ec3713c4ac8624e342ef18bf715608647d985c0eb1'
SUBJECTS = r13.SUBJECTS
FIXED_FILES = (
    'fit/fitted.npz', 'fit/selection.json', 'fit/selection-before-joint.json',
    'fit/joint-candidates.npz', 'fit/joint-body-refinement.json', 'fit/official-fit-result.json',
    'fit/shared-body.json', 'data/training-inputs.json', 'data/inputs.json', 'data/observations.npz',
)


def root_for(subject):
    return review.LAB/f'shuolee-smpl-delivery-round16-{subject}'


def descriptor(path):
    return dict(file=str(path), sha256=sha(path))


def verify_base(subject):
    receipt_path = pin(DELIVERY.with_name('smpl-delivery-round15')/'receipt.json', BASE_SHA)
    receipt = read(receipt_path)
    arm = receipt['subjects'][subject]
    require(receipt['commit'] == arm['commit'] == BASE_COMMIT, 'Wrong R15 training baseline.')
    compare_path = pin(arm['review'], arm['review_sha256'])
    comparison = read(compare_path)
    plan_path = pin(compare_path.with_name('plan.json'), comparison['plan_sha256'])
    lock = read(plan_path)
    require(comparison['state'] == 'round15_delivery_views_complete'
            and comparison['commit'] == lock['commit'] == BASE_COMMIT
            and lock['subject'] == subject, 'Invalid R15 comparison archive.')
    old = review.planned_model(Path(arm['result']).parent, r15.PROJECT, arm['result_sha256'])
    require(old.provenance == lock['new'] and comparison['all_paired_native_same_source_mask_camera_pose'],
            'R15 comparison does not bind the training endpoint.')
    require(arm['exact_round14_fit_reused'], 'R15 baseline lacks the verified input lock.')
    proofs = dict(round15_receipt=descriptor(receipt_path), round15_review=descriptor(compare_path),
                  round15_review_plan=descriptor(plan_path),
                  fixed_native_rois=descriptor(pin(plan_path.with_name('rois.npz'), lock['roi_sha256'])))
    return old, arm, comparison, lock, proofs


def check_densification(audit, integration, ordered_frames=None):
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
                and (event['replacement_children'] >= 3*event['size_replaced'] if event['size_replaced']
                     else event['replacement_children'] == 0),
                'Native size decisions fail candidate accounting.')
        for key, count in (('removal_trial',event['size_pruned']),('replacement_trial',event['size_replaced'])):
            if count:
                trial = event[key]
                check = trial['final_trial'] if key == 'replacement_trial' else trial
                require(trial['accepted'] and trial['proposed'] == count and check['accepted']
                        and len(check['frames']) == 24 and all(not row['failures'] for row in check['frames']),
                        'A size-driven removal or replacement lacks complete foreground protection.')
                if ordered_frames is not None:
                    require([row['frame'] for row in check['frames']] == ordered_frames,
                            'A size decision was verified against a different native frame inventory.')
                if key == 'replacement_trial':
                    radius = check['native_radius_check']
                    require(radius['all_decreased'] and len(radius['before_px']) == len(radius['child_max_px']) == count
                            and all(a>b for a,b in zip(radius['before_px'],radius['child_max_px'],strict=True)),
                            'A size-driven replacement failed to reduce its raster radius.')



def check_local_optimization(audit, integration, ordered_frames):
    summary = audit['candidate_optimization']
    require(summary == integration['densification_audit']['candidate_optimization']
            and summary['main_optimizer_updates'] == integration['optimizer_updates'] == 10000,
            'Candidate work changed the main optimizer budget or terminal identity.')
    trials = []
    born, retired = set(), set()
    for event in audit['events']:
        trial = event.get('local_optimization')
        if trial is None:
            require(event.get('size_replaced', 0) == 0, 'A committed replacement has no local optimizer evidence.')
            continue
        updates, seconds = trial['optimizer_updates'], trial['seconds']
        require(type(updates) is int and 0 <= updates <= trial['optimizer_update_budget'] <= 24
                and math.isfinite(seconds) and seconds >= 0
                and math.isfinite(trial['verification_seconds']) and trial['verification_seconds'] >= 0,
                'The local candidate budget record is invalid.')
        ids, rows = trial['stable_parent_ids'], trial['parent_rows']
        accepted, rejected = trial['committed_parent_ids'], trial['rejected_parent_ids']
        require(len(ids) == len(set(ids)) and all(type(identifier) is int and identifier >= 0 for identifier in ids)
                and 0 < len(ids) == len(rows) == len(set(rows)) <= 8
                and not retired.intersection(ids) and len(accepted) == event['size_replaced']
                and len(set(accepted) | set(rejected)) == len(accepted)+len(rejected) == len(ids)
                and set(accepted) | set(rejected) == set(ids), 'Stable candidate lineage IDs are invalid.')
        k, views = trial['children_per_parent'], trial['views']
        require(k >= 3 and len(views) <= trial['optimizer_view_budget'] <= 6
                and len(views) == len({view['view_index'] for view in views}), 'Invalid multi-child trial shape.')
        for view in views:
            index = view['view_index']
            require(0 <= index < len(ordered_frames) and view['frame'] == ordered_frames[index]
                    and len(view['parent']['radii_px']) == len(ids), 'Local diagnostic camera identity differs.')
            for stage in ('parent', 'proposal', 'optimized'):
                metric = view[stage]
                require(all(math.isfinite(metric[name]) and metric[name] >= 0 for name in ('rgb_mae','alpha_error')),
                        'Local native RGB or alpha metric is invalid.')
                radii = np.asarray(metric['radii_px'])
                require(radii.shape == ((len(ids),) if stage == 'parent' else (len(ids), k))
                        and np.isfinite(radii).all() and (radii >= 0).all(), 'Local native raster radii are invalid.')
        if accepted:
            check, commit = event['replacement_trial'], trial['commit']
            mapping = dict(zip(rows, ids, strict=True))
            children = commit['child_ids']
            require(trial['status'] == 'fitted' and updates > 0 and views
                    and [mapping[row] for row in check['selected_candidate_ids']] == accepted == commit['parent_ids']
                    and len(set(children)) == len(children) == commit['children'] == event['replacement_children']
                    == k*len(accepted) and not (set(children) & (born | retired | set(ids)))
                    and set(commit['moment_groups']) == {'xyz', 'scaling', 'opacity', 'f_dc'},
                    'Candidate commit or transferred Adam moment inventory differs.')
            born.update(children)
            retired.update(accepted)
        else:
            require('commit' not in trial, 'A rejected candidate event includes committed state.')
        trials.append((event, trial))
    optimized_ids = {identifier for _, trial in trials for identifier in trial['stable_parent_ids']}
    optimized_events = sum(len(trial['stable_parent_ids']) for _, trial in trials)
    require(summary['events'] == len(trials)
            and summary['size_candidate_events'] == audit['totals']['size_candidates']
            and len(optimized_ids) <= summary['unique_size_candidate_parents'] <= summary['size_candidate_events']
            and summary['repeated_size_candidate_events'] == summary['size_candidate_events']-summary['unique_size_candidate_parents']
            and summary['optimized_parent_events'] == optimized_events
            and summary['unique_optimized_parents'] == len(optimized_ids)
            and summary['repeated_optimized_parent_events'] == optimized_events-len(optimized_ids)
            and summary['optimizer_updates'] == sum(trial['optimizer_updates'] for _, trial in trials)
            and math.isclose(summary['seconds'], sum(trial['seconds'] for _, trial in trials), abs_tol=1e-9),
            'Candidate event, unique-parent or timing totals differ.')
    return summary, trials


def candidate_panels(run, trial_events):
    """Pin the first successful and first rejected real trial crops with source identity."""
    with_images = [(event, trial) for event, trial in trial_events
                   if trial['representatives'] and trial['decision_representatives']]
    require(all(trial['decision_representatives'] for _, trial in trial_events if trial['committed_parent_ids']),
            'A successful local candidate event lacks live decision diagnostics.')
    accepted = next(((event, trial) for event, trial in with_images if trial['committed_parent_ids']), None)
    rejected = next(((event, trial) for event, trial in with_images if not trial['committed_parent_ids']), None)
    if rejected is None:
        rejected = next(((event, trial) for event, trial in with_images if trial['rejected_parent_ids']), None)
    chosen = {}
    for pair in (accepted, rejected):
        if pair:
            chosen[pair[0]['step']] = pair
    rows = {row['frame']: row for row in read(run/'data/training-inputs.json')['records']}
    panels = []
    for step, (event, trial) in sorted(chosen.items()):
        views = {view['frame']: view for view in trial['views']}
        folder = f'step-{step:06d}'
        require(trial['diagnostic_subdir'] == folder, 'Candidate diagnostic event folder differs.')
        decisions = {item['frame']: item for item in trial['decision_representatives']}
        require(set(decisions) == {item['frame'] for item in trial['representatives']},
                'Local optimization and live decision crop inventory differs.')
        for representative in trial['representatives']:
            frame = representative['frame']
            view = views[frame]
            row = rows[frame]
            x0, y0, x1, y1 = box = representative['box']
            width, height = row['wh']
            require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, 'Candidate diagnostic crop is invalid.')
            with Image.open(pin(row['image'], row['source_sha256'])) as original:
                source_rgb = np.asarray(original.convert('RGB').crop(box))
            width, height = x1-x0, y1-y0
            stride = max(width, 105)+3
            descriptors, stage_images = [], []
            for kind, item, stages in (
                    ('representative', representative, ['source','parent','proposal','optimized']),
                    ('decision', decisions[frame], ['source','live_final_scene'])):
                filename = f'{kind}-view-{view["view_index"]:02d}.png'
                require(item['path'] == filename and item['view_index'] == view['view_index'] and item['box'] == box,
                        'Candidate before/after diagnostic identities differ.')
                path = run/'gaussian-run/size-child-diagnostics'/folder/filename
                with Image.open(pin(path, item['sha256'])) as image:
                    require(image.format == 'PNG' and image.mode == 'RGB'
                            and image.size == (len(stages)*stride+3, 2*height+39),
                            'Candidate panel has a different native crop layout.')
                    rgb = np.asarray(image)
                require(np.array_equal(rgb[20:20+height,3:3+width],source_rgb),
                        'Candidate panel source is not the exact native RGB crop.')
                stage_images.append(rgb[20:23+2*height,3+stride:3+stride+width])
                descriptors.append(dict(descriptor(path), stages=stages, view=f'size_child_{kind}'))
            delta = int(np.abs(stage_images[0].astype(np.int16)-stage_images[1]).max())
            facts = dict(frame=frame, step=step, native_crop=box, parent_to_live_max_abs_8bit=delta,
                committed_parent_ids=trial['committed_parent_ids'], rejected_parent_ids=trial['rejected_parent_ids'],
                raw_view_metrics=view, optimized_children_per_parent=trial['children_per_parent'])
            panels.extend(dict(item, **facts) for item in descriptors)
    return panels


def completed_training(subject, commit):
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'Pin the exact completed R16 candidate.')
    r13.r9.transfer.rebound(r13.r9.transfer.base.common.git_state, PROJECT=PROJECT)(commit)
    root = root_for(subject)
    run = root/'run'
    wrapper_path = root/'wrapper-terminal.json'
    wrapper = read(wrapper_path)
    terminal = read(pin(run/'result.json', wrapper['result_sha256']))
    require(wrapper['state'] == terminal['state'] == 'completed' and wrapper['candidate_commit'] == commit
            and terminal['gaussian_from_scratch'] is terminal['format_passed'] is terminal['fit_reused'] is True
            and terminal['fit_calls'] == 0 and terminal['steps'] == 10000
            and terminal['final_snapshot'] == 'step-010000', 'Missing verified R16 fixed-fit terminal.')
    integration_path = run/'gaussian-run/integration.json'
    integration = read(integration_path)
    require(integration['optimizer_updates'] == 10000 and integration['initial_points'] == 20908
            and integration['optimizer_initial_state_entries'] == 0 and integration['no_prior_gaussian_loaded'] is True,
            'R16 Gaussian training did not start from the protected empty optimizer.')
    color = pin(run/'gaussian-run/color-constraint.json', integration['sh0_color_constraint']['sha256'])
    densification = pin(run/'gaussian-run/densification-audit.json', integration['densification_audit']['sha256'])
    audit = read(densification)
    order = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
    check_densification(audit, integration, order)
    _, trials = check_local_optimization(audit, integration, order)
    preregistration = read(pin(root/'preregistration.json',wrapper['preregistration_sha256']))
    fixtures = pin(DELIVERY/'input-audit/cuda-size-optimization-proof.json',preregistration['cuda_fixture_sha256'])
    fixture = read(fixtures)
    require(fixture['state'] == 'passed' and fixture['optimized']['accepted'] is True
            and fixture['rejected']['accepted'] is False and fixture['optimized']['views'] == fixture['rejected']['views'] == 24
            and fixture['post_prune_rollback']['restored_model_and_optimizer'] is True,
            'The registered CUDA fixture evidence is incomplete.')
    for name, digest in fixture['source_sha256'].items():
        pin(PROJECT/name, digest)
    require(wrapper['fit_method'] == 'reused_exact_round15_fit'
            and wrapper['same_round15_fit_and_training_selection'] is True,
            'R16 wrapper does not bind the exact R15 fitted training selection.')
    proof = dict(wrapper=descriptor(wrapper_path), integration=descriptor(integration_path),
        color_constraint=descriptor(color), densification_audit=descriptor(densification),
        cuda_size_optimization_proof=descriptor(fixtures), joint=descriptor(run/'fit/joint-body-refinement.json'),
        joint_candidates=descriptor(run/'fit/joint-candidates.npz'))
    if subject == '006':
        proof['round15_unchanged_retry'] = r15.verified_retry(r15.root_for(subject), BASE_COMMIT)
    for index, panel in enumerate(candidate_panels(run, trials)):
        proof[f'candidate_panel_{index:02d}'] = dict(file=panel['file'], sha256=panel['sha256'])
    return terminal, wrapper, proof


check_source_equivalence = r15.check_source_equivalence


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
                    for name in old.rows), 'R16 changed a native camera, mask, fitted pose or selected frame.')
    parts = dict(old=round1.delivery_paths(old_run,read(old_run/'result.json')),new=round1.delivery_paths(run,terminal))
    check_source_equivalence(r15.PROJECT, PROJECT, parts['old'][2]['source_files_sha256'],
                                   parts['new'][2]['source_files_sha256'])
    for field in ('fitted', 'training_rows'):
        lhs = parts['old'][2]['inputs_sha256'].get(field)
        require(lhs is not None and lhs == parts['new'][2]['inputs_sha256'].get(field),
                f'R16 changed the saved binding input {field}.')
    views, arms, common, colors = r14.common_plans(old, new, lock, parts)
    require(all(color['points_above_one'] == color['points_below_zero'] == 0 for color in colors.values()),
            'Delivered trained SH0 colors lie outside LDR.')
    pairs = []
    for pair in lock['pairs']:
        frame = pair['frame']
        pose = review.pose_digest(old, frame)
        require(old.slots[frame] == new.slots[frame] == pair['new_slot'] and pose == pair['new_pose_sha256'],
                'The locked R15 native pose differs from the actual pair.')
        pairs.append(dict(pair, old_slot=old.slots[frame], new_slot=new.slots[frame],
                          old_pose_sha256=pose, new_pose_sha256=pose))
    special = {}
    for frame, row in lock['special_016'].items():
        special[frame] = dict(row, R16=row['R15'])
        require((frame in new.rows) == (row['R15'] == 'trained'), 'A special-frame identity changed.')
    if special:
        proof['common_special_posed_meshes'] = lock['proof']['common_special_posed_meshes']
    proof['common_reference_mesh'] = descriptor(Path(common['file']))
    proof['pre_joint_selection'] = descriptor(run/'fit/selection-before-joint.json')
    selection = read(run/'fit/selection.json')
    audit = read(Path(proof['densification_audit']['file']))
    _, trials = check_local_optimization(audit, read(Path(proof['integration']['file'])), list(new.rows))
    return dict(schema='shuolee-round16-size-optimization-review-v1',subject=subject,commit=commit,old=old.provenance,
        new=new.provenance,proof=proof,reviewer_sha256=r13.R10_REVIEWER_SHA,
        round16_reviewer_sha256=sha(Path(__file__)), asset=str(review.ASSET), output=str(root_for(subject)/'view-comparison'),
        pairs=pairs,matched_frames=list(old.rows),static_views=views,static_arms=arms,common_reference=common,
        common_special=lock['common_special'],special_016=special,pose_relative_azimuth=lock['pose_relative_azimuth'],
        selections=dict(old=selection,new=selection,round15=selection,pre_joint=read(run/'fit/selection-before-joint.json')),
        native_regions=lock['native_regions'],roi_sha256=lock['roi_sha256'], brightness_contract=lock['brightness_contract'],
        historical_R9_native=historical['historical_R9_native'],
        historical_R10_native=historical['historical_R10_native'],
        historical_R10_brightness=historical['historical_R10_brightness'],
        candidate_panels=candidate_panels(run, trials),
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
    pin(Path(__file__),plan['round16_reviewer_sha256'])
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
                    proof = plan['proof']['round15_review']
                    baseline_path = pin(proof['file'], proof['sha256'])
                    expected = read(baseline_path)['new']['panels'][role]
                    require(expected['crop_xyxy'] == item['crop_xyxy']
                            and expected['columns'] == ['source_rgb','black','white','alpha'],
                            'The historical R15 controlled crop differs.')
                    x0,y0,x1,y1 = expected['crop_xyxy']
                    width = x1-x0
                    with Image.open(pin(baseline_path.parent/expected['file'],expected['sha256'])) as saved:
                        original = np.asarray(saved.convert('RGB'),dtype=np.int16)[:,width:2*width]
                    observed = np.rint(np.clip(pixels[0][y0:y1,x0:x1],0,1)*255).astype(np.int16)
                    delta = int(np.abs(observed-original).max())
                    require(delta <= 1,'Common special replay differs from the actual R15 archive.')
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
                    R13_fitted_pose_used=False,R15_fitted_pose_used=False,R16_fitted_pose_used=False,mask_metrics=stats,
                    region_metrics=regions,float_rgb=float_summary(pixels))
    review.save(out/f'{side}-common.json',dict(common=records,common_special=common_special,panels=panels,optimizer_updates=0))


def fit_only_panel(out, plan):
    if not plan['special_016']:
        return []
    proof = plan['proof']['round15_review']
    historical = pin(proof['file'],proof['sha256'])
    panels = []
    for item in read(historical)['panels']:
        if item['view'] != 'available_record' or plan['special_016'][item['frame']]['R16'] != 'fit_only':
            continue
        source = pin(historical.parent/item['file'],item['sha256'])
        shutil.copyfile(source,out/item['file'])
        panels.append(dict(item,rows=['round10_historical', 'round15_round16_identical_fit']))
    return panels


def assemble(path, expected):
    plan = read(path)
    require(plan == json.loads(json.dumps(expected,allow_nan=False)), 'The saved comparison plan changed.')
    out = path.parent
    pin(out/'rois.npz',plan['roi_sha256'])
    prior = plan['proof']['round15_review']
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
                'The reproduced R15 native float RGB or alpha differs from its archive.')
    r13.check_render_receipts(plan,old,new)
    r13.check_source_pixels(out,plan,results)
    brightness = r13.brightness_diagnostic(out,plan,old,new)
    panels = r10.merge(out,plan,old,new)
    for panel in panels:
        panel['rows'] = [f'round15_{BASE_COMMIT[:7]}',f'round16_{plan["commit"][:7]}']
    panels += fit_only_panel(out,plan)
    path_out = out/'result.json'
    require(not path_out.exists(),'Preserve the completed densification review receipt.')
    review.save(path_out,dict(state='round16_delivery_views_complete',subject=plan['subject'],commit=plan['commit'],
        plan_sha256=sha(path),assembler_sha256=sha(Path(__file__)),panels=panels,old=old,new=new,proof=plan['proof'],
        special_016=plan['special_016'],common_reference=plan['common_reference'],
        pose_relative_azimuth=plan['pose_relative_azimuth'],terminal_sh0_rgb_range=plan['terminal_sh0_rgb_range'],
        historical_R9_native=plan['historical_R9_native'],historical_R10_native=plan['historical_R10_native'],
        historical_R10_brightness=plan['historical_R10_brightness'],brightness_diagnostic=brightness,
        candidate_panels=plan['candidate_panels'],
        all_paired_native_same_source_mask_camera_pose=True,round15_float_native_bitwise_equal=True,quality_accepted=False))
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
        parser.error('Review requires a completed R16 candidate commit.')
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
