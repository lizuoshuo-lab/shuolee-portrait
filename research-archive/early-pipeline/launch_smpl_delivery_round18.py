"""Run the R18 facial supplement with the exact frozen R17 full-body inputs."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
from tempfile import TemporaryDirectory
import zipfile

import launch_unified_cold_016_006 as cold
import launch_smpl_delivery_round17 as prior


need, read, pin, sha, write = cold.need, cold.read, cold.pin, cold.sha, cold.write
check_snapshot, color_stats = prior.check_snapshot, prior.color_stats
stage, check_binding = prior.stage, prior.check_binding
snapshot_failure, verify_parity = prior.snapshot_failure, prior.verify_parity
PYTHON = cold.ENV/'bin/python'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-supplement-20261111')
PARENT_PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-multiview-quality-20261111')
PARENT = '6dc21ded72ec2695cdda2e51f428f69a5284c2c8'
QUALITY_BASELINE = PARENT
STABLE_TAG_COMMIT = 'a6166a5ffd439362ab2c17cda30f9ec408c688c4'
OUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs')
DIRECTOR_ROOT = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs')
DIRECTOR = DIRECTOR_ROOT/'round17_director_review_20261111.json'
DIRECTOR_SHA = '13ca1c4ecb6fa67636cb28838a553204f2380d97a843e186d505fbaa52e9ee92'
DELIVERY = OUT/'smpl-delivery-round17/receipt.json'
DELIVERY_SHA = '86ad5bbde53487a7314043769e88d5502da1ee8fa4f73a2d03bbf6e4af6a94c1'
CUDA_PROOF = OUT/'smpl-delivery-round18/input-audit/cuda-face-supplement-proof.json'
CUDA_FIXTURE_SOURCE = Path(__file__).with_name('probe_cuda_face_supplement_round18.py')
POLICY = 'projected_ldr_dc'
POLICY_KEY = 'sh0_color_constraint'
COLOR_C0 = .28209479177387814
MAX_WALL_SECONDS = 7200
TRAINING_STEPS = 10000
SCHEMA = 'shuolee-round18-fixed-fit-face-supplement-v1'
REQUIRED = {'src/shuolee_portrait/training.py','src/shuolee_portrait/face_supplement.py',
            'src/shuolee_portrait/face_supplement_training.py'}
ALLOWED = REQUIRED | {'src/shuolee_portrait/render_support.py','src/shuolee_portrait/world_scale.py',
                      'src/shuolee_portrait/bound_replay.py','src/shuolee_portrait/bound_review_panels.py',
                      'src/shuolee_portrait/export.py','tests/test_render_support.py','tests/test_world_scale.py'}
EVIDENCE_FILES = prior.EVIDENCE_FILES


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round18-{subject}'


def check_code(commit):
    pointer = (PROJECT/'.git').read_text().strip()
    need(pointer.startswith('gitdir: '), 'The candidate must be a linked worktree.')
    git_dir = subprocess.check_output(['/usr/bin/wslpath', '-u', pointer.removeprefix('gitdir: ')],text=True).strip()
    git = ['/usr/bin/git','--no-optional-locks','-c','core.fileMode=false','--git-dir',git_dir,'--work-tree',str(PROJECT)]
    head = subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()
    need(head == commit, f'Candidate HEAD {head} differs from requested commit.')
    stable_tag = subprocess.check_output(
        [*git,'rev-parse','refs/tags/smpl-stable-20261111^{commit}'],text=True).strip()
    need(stable_tag == STABLE_TAG_COMMIT,'The fixed local Git release tag differs from the archived release.')
    status = subprocess.check_output([*git,'status','--porcelain','--untracked-files=all'],
                                     cwd=PROJECT,text=True)
    need(not status.strip(), f'Candidate worktree is dirty: {status}')
    subprocess.run([*git,'merge-base','--is-ancestor',PARENT,head],check=True)
    changed = set(subprocess.check_output([*git,'diff','--name-only',PARENT,head],text=True).splitlines())
    focused_tests = {name for name in changed if Path(name).parent == Path('tests')
                     and Path(name).name.startswith(('test_face_supplement','test_bound_'))
                     and Path(name).suffix == '.py'}
    need(REQUIRED <= changed <= ALLOWED | focused_tests,
         f'Unexpected facial-supplement code paths: {sorted(changed)}')
    prior_gitdir = subprocess.check_output(
        ['/usr/bin/wslpath','-u',(PARENT_PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')],
        text=True).strip()
    prior_git = ['/usr/bin/git','--no-optional-locks','-c','core.fileMode=false','--git-dir',prior_gitdir,'--work-tree',str(PARENT_PROJECT)]
    prior_head = subprocess.check_output([*prior_git,'rev-parse','HEAD'],text=True).strip()
    prior_status = subprocess.check_output([*prior_git,'status','--porcelain','--untracked-files=all'],
                                            cwd=PARENT_PROJECT,text=True)
    need(prior_head == PARENT and not prior_status.strip(), 'The R17 worktree changed.')
    tracked = set(cold.SOURCE_FILES) | changed | prior.OPT_PATHS | {
        'src/shuolee_portrait/sh0_main.py','src/shuolee_portrait/world_scale.py',
        'src/shuolee_portrait/coverage_refinement.py','src/shuolee_portrait/render_support.py'}
    for name in tracked-changed:
        pin(PROJECT/name,sha(PARENT_PROJECT/name))
    return {name:sha(PROJECT/name) for name in sorted(tracked)}


def source_inventory(subject):
    """Pin R17 outputs plus every frozen non-holdout candidate and source artifact."""
    inventory = prior.source_inventory(subject)
    files = inventory['files']

    def record(path,digest=None):
        path = Path(path)
        if digest is None:
            digest = sha(path)
        else:
            pin(path,digest)
        need(files.setdefault(str(path),digest) == digest,f'Conflicting source identity for {path}.')
        return dict(path=str(path),sha256=digest)

    record(DIRECTOR,DIRECTOR_SHA)
    record(DELIVERY,DELIVERY_SHA)
    review, delivery = read(DIRECTOR), read(DELIVERY)
    need(review['round'] == 17 and review['commit'] == delivery['commit'] == PARENT
         and delivery['baseline_commit'] == prior.QUALITY_BASELINE
         and delivery['round16_baseline_commit'] == prior.PARENT
         and review['receipt_sha256'] == DELIVERY_SHA
         and review['quality_accepted'] is delivery['quality_accepted'] is False
         and review['next_round']['round'] == 18,
         'R17 report or director authorization differs.')
    need(review['findings_sha256'] == delivery['quality_findings']['sha256'],
         'The R17 director reviewed different quality findings.')
    record(DELIVERY.parent/'review-findings.json',review['findings_sha256'])
    baseline = prior.root_for(subject)
    run = baseline/'run'
    delivery_subject = delivery['subjects'][subject]
    evidence = delivery_subject['evidence']
    for label,relative in EVIDENCE_FILES.items():
        descriptor = evidence[label]
        record(descriptor['path'],descriptor['sha256'])
        record(run/relative,descriptor['sha256'])
    for label,filename in (('wrapper','wrapper-terminal.json'),('preregistration','preregistration.json'),
                           ('protected_inputs','protected-inputs.json'),('training_launch','launch.json')):
        descriptor = evidence[label]
        record(descriptor['path'],descriptor['sha256'])
        record(baseline/filename,descriptor['sha256'])
    record(evidence['cuda_multiview_quality_proof']['path'],delivery['cuda_multiview_controls']['sha256'])
    for step in (3000,6600):
        descriptor = evidence[f'candidate_event_{step:06d}']
        record(descriptor['path'],descriptor['sha256'])
        record(run/'gaussian-run/size-child-diagnostics'/f'step-{step:06d}'/'event.json',descriptor['sha256'])
    terminal = read(baseline/'wrapper-terminal.json')
    r17_plan = read(baseline/'preregistration.json')
    r17_inventory = read(baseline/'protected-inputs.json')
    need(terminal['state'] == 'completed' and terminal['candidate_commit'] == r17_plan['candidate_commit'] == PARENT
         and terminal['result_sha256'] == delivery_subject['result_sha256']
         and terminal['preregistration_sha256'] == evidence['preregistration']['sha256']
         and terminal['protected_input_inventory_sha256'] == evidence['protected_inputs']['sha256']
         and terminal['same_round15_fit_and_training_selection'] is True
         and terminal['same_round16_fit_and_training_selection'] is True
         and terminal['fit_sha256'] == inventory['fit_sha256']
         and terminal['historical_r15_006_parity'] == inventory['r15_006_parity'],
         'The archived R17 formal terminal identity differs.')
    record(run/'result.json',terminal['result_sha256'])
    r15_plan = read(Path(inventory['baseline_run']).parent/'preregistration.json')
    need(r17_plan['runtime'] == r15_plan['runtime'] and r17_plan['asset_sha256'] == r15_plan['asset_sha256']
         and r17_plan['selected_frames'] == inventory['selected_frames'] == r17_inventory['selected_frames']
         and r17_plan['retained_holdouts'] == inventory['retained_holdouts'] == r17_inventory['retained_holdouts'],
         'The archived R17 runtime or frame roster differs.')
    staged = {}
    for name,entry in inventory['staged_files'].items():
        need(r17_inventory['staged_files'][name]['sha256'] == entry['sha256'],
             f'Frozen R17 source payload differs: {name}.')
        staged[name] = record(run/name,entry['sha256'])
    frozen_profiles = {'config.json','gaussian-run/config.json','gaussian-run/training-cameras.json',
                       'gaussian-run/initialization.json'}
    profiles = {name:record(run/name,entry['sha256'] if name in frozen_profiles else None)
                for name,entry in inventory['profiles'].items()}
    for step,parity in zip((1000,TRAINING_STEPS),terminal['native_ply_parity'],strict=True):
        checked = verify_parity(run,step,inventory['reference_frame'])
        need(checked == parity,f'The archived R17 step {step} parity differs.')
        path = run/parity['file']
        record(path,parity['sha256'])
        record(path.parent/'world.ply',read(path)['sha256'])
        record(path.parent/'reference-mesh.npz',read(path)['mesh_sha256'])
    candidates = candidate_inventory(run,inventory,record)
    return dict(inventory,schema=SCHEMA,baseline_commit=PARENT,baseline_run=str(run),staged_files=staged,
                profiles=profiles,archived_round15_run=inventory['baseline_run'],archived_round17_terminal=evidence['wrapper'],
                candidate_frames=candidates,candidate_count=len(candidates))


def candidate_inventory(run,inventory,record):
    """Verify native source identities and frozen poses without using any held observation."""
    import numpy as np
    from PIL import Image

    rows = read(run/'data/inputs.json')['records']
    raw_by_frame = {row['frame']:(i,row) for i,row in enumerate(rows)}
    selected = set(inventory['selected_frames'])
    held = set(inventory['retained_holdouts'])
    selection = read(run/'fit/selection.json')
    joint = read(run/'fit/joint-body-refinement.json')
    official = read(run/'fit/official-fit-result.json')
    record(run/'fit/joint-candidates.npz',joint['candidate_sha256'])
    need(official['joint_body_refinement']['candidate_sha256'] == joint['candidate_sha256'],
         'The official fit and joint-candidate archive disagree.')
    with np.load(run/'fit/joint-candidates.npz',allow_pickle=False) as f:
        candidate = dict(f)
    with np.load(run/'fit/fitted.npz',allow_pickle=False) as f:
        fitted = dict(f)
    with np.load(run/'data/observations.npz',allow_pickle=False) as f:
        obs_ids = f['frame_ids'].tolist()
        obs_shape = f['keypoints'].shape
    ids = candidate['frame_ids'].tolist()
    expected = set(raw_by_frame)-held
    need(len(rows) == len(raw_by_frame) == len(obs_ids) == len(set(obs_ids)) == 56
         and obs_shape == (56,133,2) and set(raw_by_frame) == set(obs_ids)
         and len(ids) == len(set(ids)) == 52 and set(ids) == expected == set(joint['fit_frame_ids'])
         and selected <= expected and set(ids).isdisjoint(held)
         and np.array_equal(candidate['betas'],fitted['shared_betas'])
         and np.array_equal(candidate['faces'],fitted['faces']),
         'The complete frozen nonheld roster, observations, topology or shared fit differs.')
    obs_index = {frame:i for i,frame in enumerate(obs_ids)}
    fit_index = {frame:i for i,frame in enumerate(fitted['frame_ids'].tolist())}
    scale = float(fitted['world_scale'])
    contract = read(run/'data/training-inputs.json')['source_camera_length_contract']
    need(scale == float(contract['meters_per_camera_unit']) and math.isfinite(scale) and scale > 0,
         'The shared camera scale is missing or changed.')
    pose_fields = ('vertices_camera','joints_camera','joints_projected_native','body_pose_axis_angle','global_orient',
                   'camera_translation','pose_embedding','expression','jaw_pose','left_hand_pose','right_hand_pose',
                   'leye_pose','reye_pose')
    need(all(candidate[name].shape[0] == len(ids) and np.isfinite(candidate[name]).all() for name in pose_fields),
         'The frozen candidate pose archive contains invalid rows.')
    results = {}
    for i,frame in enumerate(ids):
        source_index,row = raw_by_frame[frame]
        need(type(row['source_frame_index']) is int and row['source_frame_index'] == candidate['source_frames'][i]
             and row['source_sha256'] == candidate['image_sha256'][i]
             and row['mask_sha256'] == candidate['mask_sha256'][i]
             and np.array_equal(row['K_integer_centers'],candidate['K_integer_centers'][i]),
             f'Candidate {frame} disagrees with its source pixels, frame ID or camera K.')
        image,mask = (record(row[k],row[d]) for k,d in (('image','source_sha256'),('mask','mask_sha256')))
        wh = tuple(row['wh'])
        with Image.open(image['path']) as raw, Image.open(mask['path']) as alpha:
            need(raw.size == alpha.size == wh and raw.format == alpha.format == 'PNG',
                 f'Candidate {frame} is not a native image and independent mask pair.')
        source_E = np.asarray(row['w2c'],dtype=np.float64)
        R = source_E[:3,:3]
        need(source_E.shape == (4,4) and np.isfinite(source_E).all()
             and np.allclose(source_E[3],[0,0,0,1],atol=1e-12,rtol=0)
             and np.allclose(R@R.T,np.eye(3),atol=1e-6,rtol=0)
             and np.isclose(np.linalg.det(R),1.,atol=1e-6,rtol=0),
             f'Candidate {frame} does not have a rigid frozen camera.')
        meters = source_E.copy()
        meters[:3,3] *= scale
        if frame in selected:
            k = fit_index[frame]
            need(np.array_equal(meters,fitted['w2c'][k]) and np.array_equal(source_E,fitted['source_w2c'][k])
                 and all(np.array_equal(candidate[key][i],fitted[key][k]) for key in pose_fields),
                 f'Candidate {frame} does not replay its matching frozen full-body pose.')
        reason = selection['reasons'][frame]
        results[frame] = dict(frame=frame,candidate_pose_index=i,source_input_index=source_index,
            observation_index=obs_index[frame],source_frame_index=row['source_frame_index'],native_wh=list(wh),
            image=image,mask=mask,K_integer_centers=row['K_integer_centers'],source_w2c=source_E.tolist(),
            metric_w2c=meters.tolist(),time_seconds=row['time_seconds'],body_selection_status=reason['status'],
            full_body_selected=frame in selected)
    if inventory['subject'] == '016':
        need('frame_000055' in results and not results['frame_000055']['full_body_selected'],
             '016 f055 identity changed from fit-only.')
    return results


def plan_for(subject, commit, cuda_proof_sha):
    code = check_code(commit)
    inventory = source_inventory(subject)
    need(re.fullmatch('[0-9a-f]{64}',cuda_proof_sha) is not None,'The approved CUDA fixture SHA-256 is required.')
    pin(CUDA_PROOF,cuda_proof_sha)
    proof = read(CUDA_PROOF)
    fixture_sources = proof['source_sha256']
    need(proof['state'] == 'passed' and REQUIRED <= fixture_sources.keys() <= code.keys()
         and all(code[name] == digest for name,digest in fixture_sources.items()),
         'The facial-supplement CUDA fixture did not pass for this exact source tree.')
    old_run = Path(inventory['baseline_run'])
    old_plan = read(old_run.parent/'preregistration.json')
    runtime = cold.cpu_runtime(cold.clean_env())
    need(runtime == old_plan['runtime'],'Training runtime differs from R17.')
    _verify_cuda_controls(proof,runtime)
    assets = old_plan['asset_sha256']
    for name,digest in assets.items():
        pin(Path(name),digest)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient space for the two full renders and their replay records.')
    with TemporaryDirectory(prefix='r18-cpu-fit-',dir=cold.LAB) as temporary:
        staging = Path(temporary)
        stage(staging,inventory)
        fit_preflight = cpu_fit_gate(staging/'run')
    old_train = read(old_run/'gaussian-run/result.json')
    need(fit_preflight['fit_sha256'] == inventory['fit_sha256']
         and fit_preflight['training_inputs_sha256'] == inventory['staged_files']['data/training-inputs.json']['sha256']
         and fit_preflight['frames'] == len(inventory['selected_frames'])
         and fit_preflight['expected_per_frame_updates'] == old_train['per_frame_updates']
         and fit_preflight['expected_source_bin_draws'] == old_train['source_bin_draws']
         and fit_preflight['face_candidate_count'] == inventory['candidate_count'] == 52
         and fit_preflight['face_admission_considered'] == inventory['candidate_count']-len(inventory['selected_frames'])
         and len(set(fit_preflight['chosen_face_frames'])) == len(fit_preflight['chosen_face_frames']) <= 4
         and set(fit_preflight['chosen_face_frames']) <= inventory['candidate_frames'].keys()
         and set(fit_preflight['chosen_face_frames']).isdisjoint(inventory['selected_frames']),
         'The normal CPU entry rejected the staged R17 fit inputs.')
    run = root_for(subject)/'run'
    command = [str(PYTHON),'-B','-u','-m','shuolee_portrait.cli','train-fitted','--run',str(run)]
    return dict(schema=SCHEMA,subject=subject,candidate_commit=commit,baseline_commit=PARENT,
                quality_baseline_commit=QUALITY_BASELINE,archived_round15_run=inventory['archived_round15_run'],
                stable_git_release_commit=STABLE_TAG_COMMIT,
                baseline_run=inventory['baseline_run'],baseline_result_sha256=inventory['profiles']['result.json']['sha256'],
                output=str(run),director_review_sha256=DIRECTOR_SHA,baseline_delivery_sha256=DELIVERY_SHA,
                archived_r15_006_parity_sha256=inventory['r15_006_parity']['proof_sha256'],
                cuda_fixture_sha256=cuda_proof_sha,
                wrapper_sha256=sha(Path(__file__)),
                verifier_source_sha256={Path(module.__file__).name:sha(Path(module.__file__))
                                        for module in (prior,prior.prior,prior.prior.prior,cold)},
                runtime=runtime,code_sha256=code,asset_sha256=assets,cpu_fit_preflight=fit_preflight,
                retained_holdouts=inventory['retained_holdouts'],
                selected_frames=inventory['selected_frames'],commands=[command],gaussian_steps=TRAINING_STEPS,
                candidate_count=inventory['candidate_count'],full_body_views=24,
                color_policy=POLICY,auto_resolved_profile_fields=['initial_points','image_cache_frames'],
                max_wall_seconds=MAX_WALL_SECONDS),inventory


def _verify_cuda_controls(proof,runtime):
    binary_digest, = [digest for name,digest in runtime['file_sha256'].items()
                       if 'diff_gaussian_rasterization' in name and Path(name).name.startswith('_C')]
    loss,scene = proof['pixel_loss'],proof['bound_scene']
    need(proof['fixture_sha256'] == sha(CUDA_FIXTURE_SOURCE)
         and proof['runtime']['raster_extension_sha256'] == binary_digest
         and all(loss[key] is True for key in ('unknown_nan_loss_unchanged','unknown_gradients_zero',
                   'black_foreground_rgb_alpha_gradients_nonzero','empty_roi_rejected'))
         and scene['body_gradient_rows'] == 0 and scene['visible_head_gradient_rows'] > 0
         and scene['same_optimizer_update'] is True and scene['full_view_count'] == 24
         and scene['face_guard']['background_gain_px'] == 0 and scene['face_guard']['failures']
         and scene['pose_index'] == 24 and scene['exact_pose_geometry']['center_max_abs_m'] == 0
         and 0 <= scene['native_replay_rgb_max_abs'] < .005,
         'The tested facial-supplement CUDA controls or native renderer differ.')


def register(subject,commit,cuda_proof_sha):
    plan, inventory = plan_for(subject,commit,cuda_proof_sha)
    root = root_for(subject)
    root.mkdir(exist_ok=True)
    need(not any(root.iterdir()), 'Registration would overwrite an existing R18 execution.')
    protected = root/'protected-inputs.json'
    write(protected,inventory)
    write(root/'preregistration.json',dict(plan,registered_at=cold.now(),protected_input_inventory_sha256=sha(protected)))
    print(dict(state='registered',subject=subject,files=len(inventory['files']),inventory_sha256=sha(protected)),flush=True)


def registered(subject,commit,cuda_proof_sha):
    root = root_for(subject)
    plan = read(root/'preregistration.json')
    protected = root/'protected-inputs.json'
    pin(protected,plan['protected_input_inventory_sha256'])
    inventory = read(protected)
    current, current_inventory = plan_for(subject,commit,cuda_proof_sha)
    need(inventory == current_inventory and all(plan[key] == value for key,value in current.items()),
         'Registered R18 candidate or protected inputs differ.')
    return plan,inventory


def cpu_fit_gate(run):
    """Exercise the normal named-fit verification and SH0 profile on CPU."""
    code = """
from collections import Counter
from hashlib import sha256
import json, sys
from pathlib import Path
import torch
from shuolee_portrait import runtime as rt
from shuolee_portrait.training import verify_official_fit, smpl_sh0_effective_config
from shuolee_portrait.stable_selection import source_bin_training_order
from shuolee_portrait.face_supplement import prepare_face_supplements

run = Path(sys.argv[1])
rt.configure(Path(rt.read(run/'run.json')['assets']),run)
fit = verify_official_fit(run)
cfg = rt.read(run/'config.json')
effective = smpl_sh0_effective_config()
if any(cfg.get(key,value) != value for key,value in effective.items()):
    raise ValueError('Staged config conflicts with the stable SH0 entry.')
rows = rt.read(run/'data/training-inputs.json')['records']
supplement, admission = prepare_face_supplements(run, rows)
admission_digest = sha256(json.dumps(admission,ensure_ascii=False,sort_keys=True,
                                    separators=(',',':'),allow_nan=False).encode()).hexdigest()
selection = rt.read(run/'fit/official-fit-result.json')['selection']
bins, order = source_bin_training_order(rows, selection, seed=effective['seed'])
draws = bytes(next(order) for _ in range(effective['iterations']))
per_frame = Counter(draws)
per_bin = Counter(bins[index] for index in draws)
if torch.cuda.is_initialized():
    raise RuntimeError('CPU fit gate unexpectedly initialized CUDA.')
print(json.dumps(dict(fit_sha256=fit['fit_sha256'],
    training_inputs_sha256=fit['training_inputs_sha256'],
    frames=len(fit['official_frame_results']),cuda_initialized=False,
    expected_sampling_order_sha256=sha256(draws).hexdigest(),
    expected_per_frame_updates=[per_frame[index] for index in range(len(rows))],
    expected_source_bin_draws=per_bin,
    face_admission_payload_sha256=admission_digest,face_candidate_count=len(admission['fit_candidate_frame_ids']),
    face_admission_considered=len(admission['candidates']),
    chosen_face_frames=[item.row['frame'] for item in supplement])))
"""
    env = dict(cold.clean_env(),CUDA_VISIBLE_DEVICES='')
    return json.loads(subprocess.check_output([str(PYTHON),'-B','-c',code,str(run)],
                                              cwd=PROJECT/'src',env=env,text=True))


def finish(root,plan,inventory,returncode,elapsed):
    run = root/'run'
    receipt = dict(state='failed' if returncode else 'incomplete',subject=plan['subject'],
        returncode=returncode,stage='train_fitted',candidate_commit=plan['candidate_commit'],
        baseline_commit=PARENT,quality_baseline_commit=QUALITY_BASELINE,command_to_end_seconds=elapsed,
        preregistration_sha256=sha(root/'preregistration.json'),
        protected_input_inventory_sha256=plan['protected_input_inventory_sha256'],quality_accepted=False,
        launch_sha256=sha(root/'launch.json'),entry_log_sha256=sha(root/'entry.log'),
        historical_r15_006_parity=inventory['r15_006_parity'])
    if returncode == 0:
        try:
            receipt.update(verify_outputs(run,plan,inventory))
        except Exception as error:
            receipt['validation_error'] = f'{type(error).__name__}: {error}'
    if receipt['state'] != 'completed':
        receipt['preserved_snapshots'] = [snapshot_failure(path,run) for path in sorted(
            (run/'gaussian-run').glob('step-*/native-ply-parity.json'))]
        receipt['preserved_size_events'] = [dict(file=str(path.relative_to(run)),sha256=sha(path))
            for path in sorted((run/'gaussian-run/size-child-diagnostics').glob('step-*/event.json'))]
        receipt['preserved_face_evidence'] = [dict(file=str(path.relative_to(run)),sha256=sha(path))
            for path in sorted((run/'gaussian-run/face-supplement').glob('*')) if path.is_file()]
        receipt['retry_performed'] = False
    write(root/'wrapper-terminal.json',receipt)
    print(dict(state=receipt['state'],subject=plan['subject'],validation_error=receipt.get('validation_error')),flush=True)
    return 0 if receipt['state']=='completed' else returncode or 1


def verify_outputs(run,plan,inventory):
    for path,digest in inventory['files'].items():
        pin(Path(path),digest)
    for name,entry in inventory['staged_files'].items():
        pin(run/name,entry['sha256'])
    for path,digest in plan['code_sha256'].items():
        pin(PROJECT/path,digest)

    def baseline(relative):
        return read(Path(inventory['profiles'][relative]['path']))

    cfg = read(run/'config.json')
    effective = read(run/'gaussian-run/config.json')
    need(cfg == baseline('config.json') and effective == baseline('gaussian-run/config.json'),
         'R18 changed the effective run profile.')
    for name in ('config.json','gaussian-run/config.json','gaussian-run/training-cameras.json',
                 'gaussian-run/initialization.json'):
        pin(run/name,inventory['profiles'][name]['sha256'])
    result = read(run/'result.json')
    trained = read(run/'gaussian-run/result.json')
    integration = read(run/'gaussian-run/integration.json')
    prior_train = baseline('gaussian-run/result.json')
    prior_integration = baseline('gaussian-run/integration.json')
    keys = ('policy','initial_points','local_parameter_sha256','optimizer_initial_state_entries',
            'optimizer_updates','no_prior_gaussian_loaded','schedule','stock_densifications',
            'frozen_surface_sha256','raw_native_rgb_target','prior_render_used_as_target')
    need(all(integration[key] == prior_integration[key] for key in keys)
         and all(trained[key] == prior_train[key] for key in ('per_frame_updates','source_bin_draws','pose_layout'))
         and result['unified_main_training'] == integration,
         'R18 changed initialization, optimization schedule, or the consumed pose/sample sequence.')

    color_path = run/'gaussian-run/color-constraint.json'
    proof = read(color_path)
    descriptor = integration[POLICY_KEY]
    need(descriptor['file'] == color_path.name and descriptor['sha256'] == sha(color_path)
         and descriptor['terminal'] == proof['terminal'] and proof['policy'] == POLICY
         and proof['optimizer_steps'] == TRAINING_STEPS and proof['topology_projections'] > 0
         and proof['projected_channels'] >= 0 and proof['cleared_outward_first_moments'] >= 0
         and abs(proof['color_basis_c0']-COLOR_C0) < 1e-12,
         'Missing in-training SH0 color constraint or audit.')
    for phase,count in (('initial',20908),('terminal',result['points'])):
        stats = proof[phase]
        need(stats['points'] == count and stats['finite'] and stats['channels_outside_bounds'] == 0
             and -.0000001 <= stats['rgb_min'] <= stats['rgb_max'] <= 1.0000001,
             f'The {phase} SH0 DC state exceeds LDR.')

    status = read(run/'status.json')
    need(status['state'] == 'completed' and status['static_export_passed']
         and result['state'] == 'completed' and result['format_passed']
         and result['gaussian_from_scratch'] and result['fit_reused'] and result['fit_calls'] == 0
         and result['sh_degree'] == 0 and result['steps'] == TRAINING_STEPS
         and result['reference_frame_id'] == inventory['reference_frame']
         and result['final_snapshot'] == 'step-010000' and result['frames'] == len(inventory['selected_frames']),
         'Incomplete fresh official Gaussian export.')
    for key,digest in (('ply','sha256'),('scene','scene_sha256'),('bound_replay','bound_replay_sha256'),
                       ('bound_panels','bound_panels_sha256')):
        target = (run/result[key]).resolve()
        need(target.is_relative_to(run),'Exported artifact escapes its run.')
        pin(target,result[digest])
    with zipfile.ZipFile(run/result['scene']) as bundle:
        with bundle.open('cameras.json') as cameras, bundle.open('point_cloud/iteration_10000/point_cloud.ply') as ply:
            need(cold.stream_sha(cameras) == sha(run/result['camera']) and cold.stream_sha(ply) == result['sha256'],
                 'Bundled camera or static PLY differs.')
    replay, panels = read(run/result['bound_replay']), read(run/result['bound_panels'])
    need([row['frame'] for row in replay['records']] == [row['frame'] for row in panels['panels']]
         == inventory['selected_frames'] and replay['retained_holdouts'] == inventory['retained_holdouts'],
         'Bound replay changed the exact R17 selected pose sequence.')
    bound = run/'gaussian-run/step-010000/bound-local.ply'
    stats = dict(bound_local=color_stats(bound),standard_world=color_stats(run/result['ply']))
    need(all(row['points'] == result['points'] for row in stats.values()),'Terminal point count differs across PLYs.')
    binding = check_binding(bound,run/'fit/fitted.npz',result['points'])
    world_scale = integration['world_scale']
    pin(run/'gaussian-run'/world_scale['file'],world_scale['sha256'])
    world_terminal = world_scale['terminal']
    need(world_terminal['step'] == TRAINING_STEPS
         and world_terminal['all_poses_max']['all']['points'] == result['points']
         and world_terminal['all_poses_max']['all']['over_limit'] == 0,
         'Terminal all-pose world scale limits do not hold.')
    early = check_snapshot(run,1000)
    late = check_snapshot(run,TRAINING_STEPS)
    parity = [verify_parity(run,step,inventory['reference_frame']) for step in effective['snapshot_steps']]
    need(late['stats'] == proof['terminal'], 'Terminal snapshot and optimizer color receipt disagree.')
    color_proof = dict(file=str(color_path.relative_to(run)),sha256=sha(color_path),policy=POLICY,
                       ply_stats=stats,snapshots=[early,late])
    face = verify_face_supplement(run,integration,effective,inventory,bound,plan['cpu_fit_preflight'])
    densification = verify_densification(run,integration,effective,inventory['selected_frames'],result['points'],
                                         face['frames'])
    selection = read(run/'fit/selection.json')
    joint = read(run/'fit/official-fit-result.json')['joint_body_refinement']
    return dict(state='completed',result_sha256=sha(run/'result.json'),points=result['points'],
        reference_frame=result['reference_frame_id'],selected_frames=selection['selected_frames'],
        coverage=selection['coverage'],joint_refinement_sha256=joint['sha256'],joint_candidate_sha256=joint['candidate_sha256'],
        world_scale_sha256=integration['world_scale']['sha256'],fit_method='reused_exact_round17_fit',
        fresh_gaussian=True,original_video_to_export=False,native_inputs_reused=True,
        reference_mesh_frozen_from_prior=False,fit_sha256=inventory['fit_sha256'],
        training_inputs_sha256=inventory['staged_files']['data/training-inputs.json']['sha256'],
        same_round15_fit_and_training_selection=True,same_round16_fit_and_training_selection=True,
        same_round17_fit_and_training_selection=True,face_supplement=face,
        face_guard=read(run/joint['file'])['face_guard'],
        color_constraint=color_proof,binding=binding,densification_audit=densification,native_ply_parity=parity)


def verify_admission(admission,inventory,frames,preflight):
    config = admission['config']
    candidates = inventory['candidate_frames']
    full = inventory['selected_frames']
    held = inventory['retained_holdouts']
    eligible_names = set(candidates)-set(full)
    archive_ids = sorted(candidates,key=lambda name: candidates[name]['candidate_pose_index'])
    audit = {row['frame']:row for row in admission['candidates']}
    digest = hashlib.sha256(json.dumps(admission,ensure_ascii=False,sort_keys=True,
                                     separators=(',',':'),allow_nan=False).encode()).hexdigest()
    staged = inventory['staged_files']
    protected = dict(native_inputs='data/inputs.json',observations='data/observations.npz',
        selection='fit/selection.json',base_fit='fit/fitted.npz',base_rows='data/training-inputs.json',
        joint_candidates='fit/joint-candidates.npz',joint_fit='fit/joint-body-refinement.json',
        official_fit='fit/official-fit-result.json')
    need(admission['schema'] == 'shuolee-face-supplement/1' and admission['state'] == 'admission_complete'
         and admission['input_sha256'] == {name:staged[path]['sha256'] for name,path in protected.items()}
         and digest == preflight['face_admission_payload_sha256']
         and admission['full_body_frame_ids'] == full and admission['retained_holdouts'] == held
         and admission['fit_candidate_frame_ids'] == archive_ids
         and admission['chosen_frame_ids'] == preflight['chosen_face_frames'] == frames
         and len(audit) == len(admission['candidates']) == len(eligible_names) == 28
         and set(audit) == eligible_names and set(audit).isdisjoint(held)
         and admission['physical_angle_calibration'] is False and admission['heldout_pixels_read'] is False
         and admission['frozen_shared_shape_and_poses'] is True and len(frames) <= config['max_views'] == 4,
         'The supplemental admission is different from the protected CPU evidence or used a holdout.')
    chosen = set(frames)
    for frame,row in audit.items():
        source = candidates[frame]
        status = row['status']
        need(row['source_index'] == source['source_frame_index'] and row['time_seconds'] == source['time_seconds']
             and row['image_sha256'] == source['image']['sha256'] and row['mask_sha256'] == source['mask']['sha256']
             and row['full_body_status'] == source['body_selection_status']
             and status in ('rejected','face_region_admitted','selected_face_supplement')
             and (status == 'selected_face_supplement') == (frame in chosen)
             and bool(row['reasons']) == (status == 'rejected'),
             f'The supplemental admission for {frame} lacks its frozen source or rejection rationale.')
        if status == 'rejected':
            continue
        span = row['face_span_px']
        near = row['near_side_from_fit']
        regions = row['regions']
        required = {near+'_eye','nose','mouth'}
        roi = row['native_roi']
        need(config['min_side_yaw_deg'] <= abs(row['fitted_head_yaw_proxy_deg']) <= config['max_side_yaw_deg']
             and abs(row['fitted_head_pitch_proxy_deg']) <= config['max_pitch_deg']
             and span >= config['min_face_span_px'] and near in ('left','right')
             and all(regions[name]['status'] == 'admitted' for name in required)
             and roi['valid_pixels'] > 0 and roi['foreground_fraction'] >= config['min_roi_foreground_fraction']
             and roi['luma_gradient_rms'] >= config['min_luma_gradient']
             and roi['saturated_fraction'] <= config['max_saturated_fraction'],
             f'Supplemental frame {frame} lacks valid near-side facial/foreground evidence.')
        for name,region in regions.items():
            if region['status'] != 'admitted':
                continue
            feature = 'eye' if name.endswith('_eye') else name
            ear = name.endswith('_ear')
            count = 1 if ear else config[f'min_{feature}_points']
            need(len(set(region['supported_points'])) == len(region['supported_points']) >= count
                 and 0 <= region['rms_px'] <= min(config['max_region_rms_px'],config['max_region_rms_over_span']*span)
                 and 0 <= region['peak_px'] <= min(config['max_region_peak_px'],config['max_region_peak_over_span']*span)
                 and (not ear or (name == near+'_ear'
                                  and row['ear_depth_gap_m'] >= config['ear_visibility_min_depth_difference_m'])),
                 f'Supplemental frame {frame} has unsupported native evidence in {name}.')
    times = sorted(audit[frame]['time_seconds'] for frame in frames)
    need(all(b-a >= config['min_supplement_spacing_s'] for a,b in zip(times,times[1:])),
         'The selected auxiliary views violate the registered source-time separation.')
    return audit


def verify_face_supplement(run,integration,effective,inventory,bound,preflight):
    """Independently check frozen same-frame replay, valid RGB/alpha pixels and update budget."""
    import numpy as np
    from PIL import Image

    directory = run/'gaussian-run/face-supplement'
    admission_path, report_path, replay_path = (directory/name for name in ('admission.json','training.json','replay.json'))
    admission, report, replay = (read(path) for path in (admission_path,report_path,replay_path))
    descriptor = integration['face_supplement']
    views = report['views']
    frames = [row['frame'] for row in views]
    candidates = inventory['candidate_frames']
    first = effective['unified_main']['warmup_steps']+1
    interval, budget = 4, 1800
    scheduled = list(range(first,TRAINING_STEPS+1,interval))[:budget] if frames else []
    need(descriptor['file'] == 'face-supplement/training.json'
         and descriptor['sha256'] == sha(report_path) and descriptor['views'] == len(frames)
         and descriptor['face_loss_updates'] == report['face_loss_updates'] == len(scheduled)
         and report['schema'] == 'face-supplement-training-v1' and report['state'] == 'completed'
         and report['admission_sha256'] == sha(admission_path)
         and report['main_optimizer_updates'] == TRAINING_STEPS and report['separate_optimizer_updates'] == 0
         and report['first_eligible_step'] == first == 2401
         and report['update_interval'] == interval and report['update_budget'] == budget
         and report['rgb_weight'] == .30 and report['alpha_weight'] == .15 and report['head_binding_faces'] > 0
         and len(set(frames)) == len(frames) and set(frames) <= candidates.keys()
         and set(frames).isdisjoint(inventory['selected_frames'])
         and set(frames).isdisjoint(inventory['retained_holdouts']),
         'The separate facial supervision changed the protected training roster or optimizer budget.')
    admission_rows = verify_admission(admission,inventory,frames,preflight)
    need(replay['schema'] == 'face-supplement-replay-v1' and replay['shared_bound_ply_sha256'] == sha(bound)
         and [row['frame'] for row in replay['records']] == frames,
         'The auxiliary replay is not the same Gaussian model and selected face-only roster.')
    pose_path = directory/'poses.npz'
    if frames:
        pin(pose_path,replay['poses_sha256'])
        with np.load(pose_path,allow_pickle=False) as f:
            poses = dict(f)
        with np.load(run/'fit/joint-candidates.npz',allow_pickle=False) as f:
            source_vertices, faces = f['vertices_camera'], f['faces']
        need(poses['frame_ids'].tolist() == frames and np.array_equal(poses['faces'],faces)
             and poses['pose_indices'].tolist() == list(range(24,24+len(frames)))
             and poses['vertices_world'].shape == (len(frames),10475,3),
             'The auxiliary pose archive does not have the protected original topology and named frame slots.')
    else:
        need(replay['poses_sha256'] is None and not pose_path.exists(),
             'A zero-candidate run published auxiliary poses.')
    file_records = []
    for i,(row,rendered) in enumerate(zip(views,replay['records'],strict=True)):
        frame = row['frame']
        frozen = candidates[frame]
        steps = scheduled[i::len(frames)]
        density_steps = sum(step <= effective['unified_main']['stock_densify_through'] for step in steps)
        cam = row['camera']
        K = np.asarray(frozen['K_integer_centers']).copy()
        K[:2,2] += .5
        E = np.asarray(frozen['metric_w2c'])
        expected_world = (source_vertices[frozen['candidate_pose_index']]-E[:3,3])@E[:3,:3]
        actual_world = poses['vertices_world'][i]
        need(row['role'] == 'face_only' and row['pose_index'] == cam['pose_index'] == 24+i
             and row['source_frame_index'] == frozen['source_frame_index']
             and row['source_sha256'] == frozen['image']['sha256'] and row['mask_sha256'] == frozen['mask']['sha256']
             and cam['frame'] == frame and cam['wh'] == frozen['native_wh']
             and cam['pixel_convention'] == 'pixel_corners' and 0 <= cam['projection_error_px'] < .001
             and np.array_equal(cam['K'],K) and np.array_equal(cam['w2c'],E)
             and np.isfinite(actual_world).all() and np.max(np.abs(actual_world-expected_world)) < 2e-6
             and row['updates'] == len(steps) > 0 and row['first_main_step'] == steps[0]
             and row['last_main_step'] == steps[-1] and row['density_observations'] == density_steps
             and 0 <= row['max_density_visible_rows']
             and all(math.isfinite(row[key]) and row[key] >= 0 for key in ('last_rgb_loss','last_alpha_loss')),
             f'Face-only frame {frame} did not use its own frozen source/pose/camera and declared update schedule.')
        x0,y0,x1,y1 = row['box']
        w,h = frozen['native_wh']
        valid_path = directory/f'{frame}-valid.png'
        need(row['valid_mask'] == valid_path.name and 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h,
             f'Invalid native validity crop for {frame}.')
        pin(valid_path,row['valid_mask_sha256'])
        with Image.open(valid_path) as valid_image, Image.open(frozen['mask']['path']) as source_mask:
            valid = np.asarray(valid_image)
            source = np.asarray(source_mask.convert('L').crop(row['box']))
            roi = admission_rows[frame]['native_roi']
            need(valid.shape == source.shape == (y1-y0,x1-x0) and np.isin(valid,[0,255]).all()
                 and row['box'] == roi['box'] and np.count_nonzero(valid) == row['valid_pixels'] == roi['valid_pixels'] > 0
                 and hashlib.sha256((valid != 0).tobytes()).hexdigest() == roi['valid_sha256']
                 and hashlib.sha256((source.astype(np.float32)/255).tobytes()).hexdigest() == roi['alpha_sha256']
                 and (source[valid != 0].astype(np.float32)/255 >= .99).all(),
                 f'Face-only validity for {frame} includes unknown or original-background pixels.')
        errors = rendered['geometry_parity']
        need(all(rendered[key] == value for key,value in row.items())
             and 0 <= rendered['valid_rgb_max_abs'] < .005
             and 0 <= errors['center_max_abs_m'] < 2e-6 and 0 <= errors['rotation_max_abs'] < 2e-5
             and 0 <= errors['scale_max_relative'] < 2e-5,
             f'Face-only frame {frame} did not replay from the standard shared bound PLY.')
        files = rendered['files']
        need(set(files) == {'panel','alpha','gaussian'},f'Incomplete face-only replay panels for {frame}.')
        for kind,desc in files.items():
            need(desc['file'] == f'{frame}-{kind}.png',f'Invalid face-only replay output path for {frame}.')
            pin(directory/desc['file'],desc['sha256'])
        file_records.append(dict(frame=frame,valid_pixels=row['valid_pixels'],pose_index=row['pose_index'],
                                 valid_mask_sha256=row['valid_mask_sha256'],files=files))
    return dict(frames=frames,training_sha256=sha(report_path),admission_sha256=sha(admission_path),
                replay_sha256=sha(replay_path),poses_sha256=replay['poses_sha256'],
                face_loss_updates=report['face_loss_updates'],separate_optimizer_updates=0,records=file_records)


_ids = prior._ids
_local_evidence = prior._local_evidence


def _native_trial(trial,ordered_frames,face_frames,screen_limit=None):
    source_accepted = (not any(row['failures'] for row in trial['frames'])
                       and (screen_limit is None or trial['native_radius_check']['all_decreased']))
    prior._native_trial(dict(trial,accepted=source_accepted),ordered_frames,screen_limit)
    supplements = trial.get('face_supplement_frames',[])
    need([row['frame'] for row in supplements] == face_frames
         and all(row.get('background_gain_px',0) == 0 for row in supplements),
         'The exact native guard changed the verified face-only roster or assigned exterior supervision.')
    accepted = source_accepted and not any(row['failures'] for row in supplements)
    need(trial['accepted'] is accepted,'Final native decision differs from the full-body and face-only guards.')
    return accepted


def _fallback_evidence(event,ordered_frames,limit,options,optimized,face_frames):
    fallback = event.get('fallback_replacement')
    if fallback is None:
        need(event['replacement_trial'] is None,'A fallback trial lacks original-path provenance.')
        return [],[]
    shortlist, eligible, committed, children = (fallback[key] for key in
        ('shortlist_parent_ids','eligible_parent_ids','committed_parent_ids','child_ids'))
    trial = fallback['trial']
    need(fallback['path'] == 'r15_fallback' and trial == event['replacement_trial']
         and all(_ids(rows) for rows in (shortlist,eligible,committed,children))
         and 1 <= len(shortlist) <= options['replacement_budget']
         and set(committed) <= set(eligible) <= set(shortlist)
         and set(shortlist).isdisjoint(optimized)
         and set(children).isdisjoint(shortlist) and len(children) == 2*len(committed),
         'The original R15 fallback lost shortlist provenance or reused a published parent.')
    if not eligible:
        need(trial is None,'The R15 fallback rendered a trial without an eligible parent.')
    else:
        trials = trial['trials']
        need(trial['initial_proposed'] == len(eligible) and 1 <= len(trials) == trial['trial_count'] <= trial['max_trials']
             and trial['max_trials'] == options['coverage_max_trials'],
             'The R15 fallback did not retain the original bounded trial policy.')
        for probe in trials:
            _native_trial(probe,ordered_frames,face_frames,limit)
        selected = trial['selected_candidate_ids']
        need(trial['accepted'] is bool(committed) and len(selected) == len(committed),
             'The original fallback commit count differs from its exact trial.')
        if committed:
            final = trial['final_trial']
            need(final in trials and final['candidate_ids'] == selected
                 and final['proposed'] == len(committed) and _native_trial(final,ordered_frames,face_frames,limit),
                 'The R15 fallback published a set without full native coverage proof.')
    return committed,children


def verify_densification(run,integration,effective,ordered_frames,points,face_frames):
    """Check the original stock schedule and independently prove both replacement paths."""
    desc = integration['densification_audit']
    need(desc['file'] == 'densification-audit.json','Unexpected densification audit path.')
    path = run/'gaussian-run'/desc['file']
    pin(path,desc['sha256'])
    audit = read(path)
    options, schedule = effective['optimization'], effective['unified_main']
    interval = options['densification_interval']
    expected = [step for step in range(interval,schedule['stock_densify_through']+1,interval)
                if step > options['densify_from_iter'] and step not in schedule['coverage_steps']]
    events = audit['events']
    need(audit['state'] == 'completed' and 'native pixels' in audit['radius_units']
         and 'zero' in audit['radius_window'] and 'face-local' in audit['local_cap_units']
         and audit['stock_events'] == integration['stock_densifications'] == len(expected)
         and [event['step'] for event in events] == expected,
         'Densification audit does not cover the expected stock schedule and units.')
    keys = ('screen_candidates_before_stock','screen_candidates_after_stock',
            'screen_parents_replaced_by_gradient_split','size_candidates','world_candidates','assessed',
            'foreground_supported','protected','size_pruned','screen_pruned','size_replaced','replacement_children',
            'optimized_size_replaced','optimized_replacement_children','fallback_size_replaced',
            'fallback_replacement_children','opacity_only_pruned')
    totals = audit['totals']
    need(totals == {key:sum(event.get(key,0) for event in events) for key in keys},
         'Densification audit totals differ from the native cycle records.')
    all_attempts = []
    event_records = []
    attempted, committed_ids, child_ids = set(), set(), set()
    guard = schedule['world_scale']['render_support']
    for event in events:
        step = event['step']
        need(all(type(event.get(key,0)) is int and event.get(key,0) >= 0 for key in keys),
             f'Invalid densification counters at step {step}.')
        limit = 20 if step > options['opacity_reset_interval'] else None
        before, after, split = (event[key] for key in keys[:3])
        need(event['screen_limit_px'] == limit and before == after+split,
             f'The original screen-radius lifecycle differs at step {step}.')
        if limit is None:
            need(not any(event.get(key,0) for key in keys[:-1])
                 and not any(key in event for key in ('local_optimizations','fallback_replacement')),
                 f'Size-triggered work ran before its schedule at step {step}.')
            continue
        candidates, world = event['size_candidates'], event['world_candidates']
        removed, replaced = event['size_pruned'], event['size_replaced']
        assessed, supported = event['assessed'], event['foreground_supported']
        need(event['screen_candidates'] == after and max(after,world) <= candidates <= after+world
             and supported <= assessed <= candidates and removed <= assessed-supported and replaced <= supported
             and removed+replaced+event['protected'] == candidates and event['screen_pruned'] <= min(after,removed),
             f'Size-triggered actions differ from the measured candidates at step {step}.')
        removal = event['removal_trial']
        if removal is not None:
            accepted = _native_trial(removal,ordered_frames,face_frames)
            need(removed == (removal['proposed'] if accepted else 0),'Removal count differs from the approved trial.')
        else:
            need(not removed,'Size-triggered removal lacks a native trial.')

        attempts = event.get('local_optimizations',[])
        fallback = event.get('fallback_replacement')
        need((not attempts and fallback is None) or step in schedule['world_scale']['split_steps'],
             f'Size replacement ran outside the fixed schedule at step {step}.')
        need(len(attempts) <= min(8,supported,guard['replacement_budget']),
             f'Local parent shortlist exceeds its registered bound at step {step}.')
        if attempts:
            saved = run/'gaussian-run/size-child-diagnostics'/f'step-{step:06d}'/'event.json'
            need(read(saved) == event,f'Step {step} persisted event differs from the final topology audit.')
            event_records.append(dict(step=step,file=str(saved.relative_to(run)),sha256=sha(saved)))
        optimized_parents, optimized_children = [], []
        scene_rows = attempts[0]['scene_rows_before'] if attempts else 0
        for attempt in attempts:
            parent, = attempt['stable_parent_ids']
            count = attempt['children_per_parent']
            need(_ids([parent]) and parent not in attempted and parent not in committed_ids
                 and attempt['scene_scope'] == 'singleton_replacement_in_current_live_scene'
                 and attempt['scene_rows_before'] == scene_rows > 0
                 and attempt['prior_optimized_commits'] == optimized_parents and count == 4,
                 f'The optimizer lost singleton scene or stable parent identity at step {step}.')
            attempted.add(parent)
            all_attempts.append(attempt)
            _local_evidence(run,step,attempt,parent,ordered_frames,effective['training_source_bins'])
            trial = attempt['replacement_trial']
            fitted = attempt['status'] in ('fitted','no_feasible_checkpoint')
            need((trial is not None) is fitted and (trial is None or trial['proposed'] == 1),
                 'Only a complete singleton optimizer may submit a native trial.')
            accepted = trial is not None and _native_trial(trial,ordered_frames,face_frames,limit)
            need(attempt['committed_parent_ids'] == ([parent] if accepted else [])
                 and attempt['rejected_parent_ids'] == ([] if accepted else [parent])
                 and ('commit' in attempt) is accepted
                 and attempt['decision_reason'] == ('optimized' if accepted else
                     'native_guard_rejected' if fitted else 'local_optimization_did_not_finish'),
                 f'The optimized replacement decision disagrees with the exact native trial at step {step}.')
            if accepted:
                commit = attempt['commit']
                children = commit['child_ids']
                need(commit['parent_ids'] == [parent] and commit['children'] == len(children) == count
                     and _ids(children) and parent not in children and set(children).isdisjoint(child_ids)
                     and commit['unchanged_adam_rows'] == scene_rows-1
                     and set(commit['moment_groups']) == {'xyz','scaling','opacity','f_dc'}
                     and 0 < attempt['checkpoint_updates'] <= attempt['optimizer_updates'],
                     f'The optimized replacement lacks transaction or Adam migration evidence at step {step}.')
                scene_rows += count-1
                optimized_parents.append(parent)
                optimized_children.extend(children)
                committed_ids.add(parent)
                child_ids.update(children)
        fallback_parents, fallback_children = _fallback_evidence(event,ordered_frames,limit,guard,optimized_parents,face_frames)
        need(set(fallback_parents).isdisjoint(committed_ids) and set(fallback_children).isdisjoint(child_ids)
             and event['optimized_size_replaced'] == len(optimized_parents)
             and event['optimized_replacement_children'] == len(optimized_children)
             and event['fallback_size_replaced'] == len(fallback_parents)
             and event['fallback_replacement_children'] == len(fallback_children)
             and event['size_replaced'] == len(optimized_parents)+len(fallback_parents)
             and event['replacement_children'] == len(optimized_children)+len(fallback_children),
             f'The two commit paths have inconsistent or overlapping counters at step {step}.')
        committed_ids.update(fallback_parents)
        child_ids.update(fallback_children)
    candidate = audit['candidate_optimization']
    size_events = totals['size_candidates']
    unique_size = candidate['unique_size_candidate_parents']
    need(desc['candidate_optimization'] == candidate and candidate['main_optimizer_updates'] == TRAINING_STEPS
         and candidate['events'] == sum(bool(event.get('local_optimizations')) for event in events)
         and candidate['attempts'] == candidate['optimized_parent_events'] == candidate['unique_optimized_parents']
             == len(attempted) == len(all_attempts)
         and candidate['size_candidate_events'] == size_events
         and len(attempted) <= unique_size <= size_events
         and candidate['repeated_size_candidate_events'] == size_events-unique_size
         and candidate['repeated_optimized_parent_events'] == 0
         and candidate['optimizer_updates'] == sum(row['optimizer_updates'] for row in all_attempts)
         and candidate['optimized_parent_commits'] == totals['optimized_size_replaced']
         and candidate['fallback_parent_commits'] == totals['fallback_size_replaced']
         and len(committed_ids) == totals['size_replaced'] and len(child_ids) == totals['replacement_children']
         and all(math.isclose(candidate[key],sum(row[key] for row in all_attempts),rel_tol=1e-9)
                 for key in ('seconds','verification_seconds')),
         'The local optimizer aggregate disagrees with the real event, topology or independent-parent evidence.')
    terminal = audit['terminal']
    local_keys = ('inherited_cap_children','child_axes_projected_at_birth','children_projected_at_birth',
                  'live_finite_cap_rows','live_rows_over_local_cap')
    need(desc['terminal'] == terminal and all(type(terminal[key]) is int and terminal[key] >= 0 for key in local_keys)
         and terminal['live_rows_over_local_cap'] == 0 and terminal['live_finite_cap_rows'] <= points
         and terminal['children_projected_at_birth'] <= terminal['inherited_cap_children']
         and terminal['children_projected_at_birth'] <= terminal['child_axes_projected_at_birth']
         <= 3*terminal['children_projected_at_birth'],
         'Terminal local covariance audit has inconsistent lineage or an inherited cap violation.')
    return dict(file=str(path.relative_to(run)),sha256=desc['sha256'],terminal=terminal,
                radius_units=audit['radius_units'],stock_events=len(events),totals=totals,
                candidate_optimization=candidate,event_records=event_records)


def launch(subject,commit,cuda_proof_sha):
    root = root_for(subject)
    plan, inventory = registered(subject,commit,cuda_proof_sha)
    with cold.LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subject == '006':
            first = root_for('016')
            terminal = read(first/'wrapper-terminal.json')
            need(terminal['state'] == 'completed' and terminal['candidate_commit'] == commit,
                 'Complete the verified 016 export before starting 006.')
            pin(first/'run/result.json',terminal['result_sha256'])
        need(not any((root/name).exists() for name in ('run','launch.json','entry.log','wrapper-terminal.json')),
             'Fresh round-eighteen output is required.')
        stage(root,inventory)
        write(root/'launch.json',dict(subject=subject,candidate_commit=commit,commands=plan['commands'],
                                      cwd=str(PROJECT/'src'),started=cold.now(),cuda_fixture_sha256=cuda_proof_sha,
                                      preregistration_sha256=sha(root/'preregistration.json')))
        start = time.monotonic()
        with (root/'entry.log').open('xb') as log:
            command, = plan['commands']
            returncode = subprocess.run(['/usr/bin/timeout','--kill-after=15s',f'{MAX_WALL_SECONDS}s',*command],
                env=cold.clean_env(),cwd=PROJECT/'src',stdout=log,stderr=subprocess.STDOUT).returncode
        return finish(root,plan,inventory,returncode,time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-inputs','check','register','launch','check-early'))
    parser.add_argument('subject',choices=tuple(cold.PREVIOUS))
    parser.add_argument('--commit')
    parser.add_argument('--cuda-proof-sha256')
    args = parser.parse_args()
    if args.action == 'check-inputs':
        data = source_inventory(args.subject)
        print(dict(state='source_preflight_passed',subject=args.subject,files=len(data['files']),
                   selected_frames=data['selected_frames'],reference_frame=data['reference_frame'],
                   retained_holdouts=data['retained_holdouts'],fit_sha256=data['fit_sha256'],
                   candidates=data['candidate_count']),flush=True)
        return
    if args.action == 'check-early':
        root = root_for(args.subject)
        run = root/'run'
        plan = read(root/'preregistration.json')
        inventory = root/'protected-inputs.json'
        pin(inventory,plan['protected_input_inventory_sha256'])
        print(dict(color=check_snapshot(run,1000),
                   native_ply_parity=verify_parity(run,1000,read(inventory)['reference_frame'])),flush=True)
        return
    if not args.commit or not args.cuda_proof_sha256:
        parser.error('--commit and --cuda-proof-sha256 are required for this action.')
    if args.action == 'launch':
        raise SystemExit(launch(args.subject,args.commit,args.cuda_proof_sha256))
    if args.action == 'register':
        register(args.subject,args.commit,args.cuda_proof_sha256)
    else:
        plan, data = plan_for(args.subject,args.commit,args.cuda_proof_sha256)
        print(dict(state='preflight_passed',subject=args.subject,files=len(data['files']),
                   selected_frames=data['selected_frames'],policy=plan['color_policy']),flush=True)


if __name__ == '__main__':
    main()
