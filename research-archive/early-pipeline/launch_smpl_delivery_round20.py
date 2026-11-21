"""Register the head-only fit and heldout pose plan, then run the fixed Gaussian path."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import launch_unified_cold_016_006 as cold
import launch_smpl_delivery_round19 as prior


need, read, pin, sha, write = cold.need, cold.read, cold.pin, cold.sha, cold.write
check_snapshot, color_stats = prior.check_snapshot, prior.color_stats
stage, check_binding = prior.stage, prior.check_binding
snapshot_failure, verify_parity = prior.snapshot_failure, prior.verify_parity
PYTHON = cold.ENV/'bin/python'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-align-20261111')
PARENT_PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-protected-face-20261111')
PARENT = '3415c5cff4a54d84effbdc82611a7bb96eeb4705'
QUALITY_BASELINE = PARENT
MAIN_BASELINE = prior.MAIN_BASELINE
STABLE_TAG_COMMIT = 'a6166a5ffd439362ab2c17cda30f9ec408c688c4'
OUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs')
DIRECTOR_ROOT = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs')
DIRECTOR = DIRECTOR_ROOT/'round19_director_review_20261111.json'
DIRECTOR_SHA = '3456109c68deb55b9ce3029d634e8514d1ab3981e56c6a81814f09e4d18e0f6d'
DELIVERY = OUT/'smpl-delivery-round19/receipt.json'
DELIVERY_SHA = '6b936ff2abc550701b719c3c7fc8c88ea665a5b655a88e619587a0059d564628'
CUDA_PROOF = prior.CUDA_PROOF
CUDA_PROOF_SHA = 'c0e51026fd183f2f2993ed1aab49d49004b627c2d07cdf586f6b3c70639d1544'
HELDOUT_CUDA_PROOF = OUT/'smpl-delivery-round20/input-audit/heldout-replay-cuda.json'
HELDOUT_CUDA_PROOF_SHA = '241bb49738257ab57c333407fb91cae19dd9a908094c7b3512899eec477e8bac'
POLICY = 'projected_ldr_dc'
POLICY_KEY = 'sh0_color_constraint'
COLOR_C0 = .28209479177387814
MAX_WALL_SECONDS = 7200
TRAINING_STEPS = 10000
SCHEMA = 'shuolee-round20-frozen-camera-head-alignment-v1'
REQUIRED = {'src/shuolee_portrait/'+name for name in ('training.py','head_alignment.py','head_fit_stage.py',
            'heldout_pose.py','sequence_pipeline.py','protected_face.py','face_supplement.py')}
MUTABLE_STAGED = {'fit/fitted.npz','fit/joint-candidates.npz','fit/official-fit-result.json'}
FIT_STAGE_FILES = ('fit/head-alignment/result.json','fit/head-alignment/holdout-poses.npz',
                   'fit/head-alignment/holdout-plan.json','fit/head-alignment/holdout-policy.json')


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round20-{subject}'


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
                     and Path(name).name.startswith(('test_head_fit','test_native_head','test_holdout','test_bound_'))
                     and Path(name).suffix == '.py'}
    need(REQUIRED <= changed <= REQUIRED | focused_tests,
         f'Unexpected head-alignment code paths: {sorted(changed)}')
    prior_gitdir = subprocess.check_output(
        ['/usr/bin/wslpath','-u',(PARENT_PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')],
        text=True).strip()
    prior_git = ['/usr/bin/git','--no-optional-locks','-c','core.fileMode=false','--git-dir',prior_gitdir,'--work-tree',str(PARENT_PROJECT)]
    prior_head = subprocess.check_output([*prior_git,'rev-parse','HEAD'],text=True).strip()
    prior_status = subprocess.check_output([*prior_git,'status','--porcelain','--untracked-files=all'],
                                            cwd=PARENT_PROJECT,text=True)
    need(prior_head == PARENT and not prior_status.strip(), 'The R19 worktree changed.')
    tracked = set(cold.SOURCE_FILES) | changed | prior.prior.prior.OPT_PATHS | {
        'src/shuolee_portrait/face_supplement.py','src/shuolee_portrait/face_supplement_training.py',
        'src/shuolee_portrait/protected_face.py',
        'src/shuolee_portrait/sh0_main.py','src/shuolee_portrait/world_scale.py',
        'src/shuolee_portrait/coverage_refinement.py','src/shuolee_portrait/render_support.py'}
    for name in tracked-changed:
        pin(PROJECT/name,sha(PARENT_PROJECT/name))
    return {name:sha(PROJECT/name) for name in sorted(tracked)}


def source_inventory(subject):
    """Pin R19 delivery, all original source rows and held-out evaluation-only inputs."""
    import numpy as np
    from PIL import Image

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
    review, delivery = read(DIRECTOR),read(DELIVERY)
    need(review['round'] == delivery['round'] == 19 and review['commit'] == delivery['commit'] == PARENT
         and delivery['round18_baseline_commit'] == prior.PARENT
         and delivery['round17_main_baseline'] == prior.MAIN_BASELINE
         and review['receipt_sha256'] == DELIVERY_SHA
         and review['quality_accepted'] is delivery['quality_accepted'] is False
         and review['next_round']['round'] == 20,
         'R19 director authorization or paired delivery differs.')
    record(DELIVERY.parent/'quality-findings.json',delivery['quality_findings']['sha256'])
    record(DELIVERY.parent/'review-receipt.json',delivery['review_receipt']['sha256'])
    record(DELIVERY.parent/'input-audit/cuda-protected-face-proof.json',delivery['real_cuda_control']['sha256'])
    baseline = prior.root_for(subject)
    run = baseline/'run'
    receipt = delivery['subjects'][subject]
    record(DELIVERY.parent/subject/'receipt.json',receipt['delivery']['sha256'])
    terminal_path = baseline/'wrapper-terminal.json'
    record(terminal_path,receipt['wrapper']['sha256'])
    terminal = read(terminal_path)
    prereg = baseline/'preregistration.json'
    protected = baseline/'protected-inputs.json'
    record(prereg,terminal['preregistration_sha256'])
    record(protected,terminal['protected_input_inventory_sha256'])
    record(baseline/'launch.json',terminal['launch_sha256'])
    old_plan, old_inventory = read(prereg),read(protected)
    old_protected = terminal['protected_face']
    need(terminal['state'] == 'completed' and terminal['candidate_commit'] == old_plan['candidate_commit'] == PARENT
         and terminal['result_sha256'] == receipt['result_sha256']
         and terminal['fit_sha256'] == inventory['fit_sha256']
         and terminal['same_round18_fit_and_training_selection'] is True and terminal['verified_pose_count'] == 28
         and old_protected['state'] == receipt['protected_stage']['result'] == 'rolled_back'
         and old_protected['committed_updates'] == 0 and old_protected['attempted_updates'] == 336
         and old_inventory['candidate_frames'] == inventory['candidate_frames']
         and old_plan['retained_holdouts'] == inventory['retained_holdouts'] == receipt['retained_holdouts']
         and old_plan['selected_frames'] == inventory['selected_frames']
         and set(receipt['selected_frames']) == set(inventory['selected_frames']),
         'The archived R19 formal result, fixed roster or genuine rollback differs.')
    record(run/'result.json',terminal['result_sha256'])
    record(run/'gaussian-run'/old_protected['file'],old_protected['sha256'])
    record(run/'gaussian-run'/old_protected['main_endpoint'],old_protected['main_endpoint_sha256'])
    main = read(run/'gaussian-run'/old_protected['main_endpoint'])
    for name,digest in main['files_sha256'].items():
        record(run/'gaussian-run'/main['snapshot']/name,digest)
    need(old_plan['runtime'] == read(Path(inventory['baseline_run']).parent/'preregistration.json')['runtime'],
         'The archived R19 runtime differs.')
    staged = {}
    for name,entry in inventory['staged_files'].items():
        need(old_inventory['staged_files'][name]['sha256'] == entry['sha256'],
             f'Frozen R19 source payload differs: {name}.')
        staged[name] = record(run/name,entry['sha256'])
    frozen = {'config.json','gaussian-run/config.json','gaussian-run/training-cameras.json',
              'gaussian-run/initialization.json'}
    profiles = {name:record(run/name,entry['sha256'] if name in frozen else None)
                for name,entry in inventory['profiles'].items()}
    for step,parity in zip((1000,TRAINING_STEPS),terminal['native_ply_parity'],strict=True):
        need(verify_parity(run,step,inventory['reference_frame']) == parity,'R19 native parity differs.')
        record(run/parity['file'],parity['sha256'])
    face = terminal['face_supplement']
    need(face['frames'] == inventory['archived_face_supplement']['frames']
         and face['face_loss_updates'] == 0 and face['separate_optimizer_updates'] == 336,
         'The R19 comparison did not keep the archived four face-only identities.')
    for kind in ('admission','training','replay','poses'):
        filename = f'{kind}.npz' if kind == 'poses' else f'{kind}.json'
        record(run/'gaussian-run/face-supplement'/filename,face[f'{kind}_sha256'])

    held = inventory['retained_holdouts']
    held_inputs = {}
    raw = {row['frame']:row for row in read(run/'data/inputs.json')['records']}
    contract = read(run/'data/training-inputs.json')['source_camera_length_contract']
    scale = float(contract['meters_per_camera_unit'])
    with np.load(run/'data/observations.npz',allow_pickle=False) as f:
        indices = {name:i for i,name in enumerate(f['frame_ids'].tolist())}
    need(len(held) == 4 and set(held).isdisjoint(inventory['candidate_frames'])
         and set(held).isdisjoint(inventory['selected_frames']) and set(held).isdisjoint(face['frames']),
         'An evaluation-only frame overlaps the archived fitting or training roster.')
    for frame in held:
        row = raw[frame]
        image,mask = (record(row[key],row[digest]) for key,digest in (('image','source_sha256'),('mask','mask_sha256')))
        with Image.open(image['path']) as rgb, Image.open(mask['path']) as alpha:
            need(list(rgb.size) == list(alpha.size) == row['wh'],'Heldout native source dimensions differ.')
        E = np.asarray(row['w2c'],dtype=np.float64)
        R = E[:3,:3]
        K = np.asarray(row['K_integer_centers'])
        corner = K.copy()
        corner[:2,2] += .5
        need(E.shape == (4,4) and np.isfinite(E).all() and np.allclose(E[3],[0,0,0,1],atol=1e-12,rtol=0)
             and np.allclose(R@R.T,np.eye(3),atol=1e-6,rtol=0) and np.isclose(np.linalg.det(R),1.,atol=1e-6)
             and K.shape == (3,3) and np.isfinite(K).all() and np.array_equal(corner,row['K']),
             f'Heldout {frame} does not have its own frozen native camera.')
        meters = E.copy()
        meters[:3,3] *= scale
        held_inputs[frame] = dict(frame=frame,source_frame_index=row['source_frame_index'],
            observation_index=indices[frame],image=image,mask=mask,native_wh=row['wh'],
            K_integer_centers=K.tolist(),source_w2c=E.tolist(),metric_w2c=meters.tolist())
    return dict(inventory,schema=SCHEMA,baseline_commit=PARENT,baseline_run=str(run),staged_files=staged,
                profiles=profiles,archived_round18_run=inventory['baseline_run'],archived_round19_terminal=record(terminal_path),
                archived_face_supplement=face,archived_face_cpu_preflight=old_plan['cpu_fit_preflight'],
                heldout_pose_sources=held_inputs)


def plan_for(subject,commit,cuda_proof_sha):
    code = check_code(commit)
    inventory = source_inventory(subject)
    need(cuda_proof_sha == CUDA_PROOF_SHA,'The archived protected CUDA fixture SHA-256 is required.')
    pin(CUDA_PROOF,cuda_proof_sha)
    proof = read(CUDA_PROOF)
    fixture_sources = proof['source_sha256']
    need(proof['state'] == 'passed' and all(sha(PARENT_PROJECT/name) == digest
         for name,digest in fixture_sources.items()), 'The archived CUDA proof does not match the R19 source.')
    old_run = Path(inventory['baseline_run'])
    old_plan = read(old_run.parent/'preregistration.json')
    runtime = cold.cpu_runtime(cold.clean_env())
    need(runtime == old_plan['runtime'],'Training runtime differs from R19.')
    prior._verify_cuda_controls(proof,runtime)
    assets = old_plan['asset_sha256']
    for name,digest in assets.items():
        pin(Path(name),digest)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient space for both training results and their independent replay records.')
    run = root_for(subject)/'run'
    command = [str(PYTHON),'-B','-u','-m','shuolee_portrait.cli','train-fitted','--run',str(run)]
    return dict(schema=SCHEMA,subject=subject,candidate_commit=commit,baseline_commit=PARENT,
                quality_baseline_commit=QUALITY_BASELINE,main_baseline_commit=MAIN_BASELINE,
                stable_git_release_commit=STABLE_TAG_COMMIT,baseline_run=inventory['baseline_run'],
                baseline_result_sha256=inventory['profiles']['result.json']['sha256'],output=str(run),
                director_review_sha256=DIRECTOR_SHA,baseline_delivery_sha256=DELIVERY_SHA,
                archived_r15_006_parity_sha256=inventory['r15_006_parity']['proof_sha256'],
                cuda_fixture_sha256=cuda_proof_sha,cuda_fixture_scope='archived_round19_control',
                wrapper_sha256=sha(Path(__file__)),
                verifier_source_sha256={Path(module.__file__).name:sha(Path(module.__file__)) for module in
                    (prior,prior.prior,prior.prior.prior,prior.prior.prior.prior,prior.prior.prior.prior.prior,cold)},
                runtime=runtime,code_sha256=code,asset_sha256=assets,selected_frames=inventory['selected_frames'],
                retained_holdouts=inventory['retained_holdouts'],candidate_count=52,full_body_views=24,
                archived_face_only_views=inventory['archived_face_supplement']['frames'],
                commands=[command],gaussian_steps=TRAINING_STEPS,main_face_aux_updates=0,
                color_policy=POLICY,auto_resolved_profile_fields=['initial_points','image_cache_frames'],
                max_wall_seconds=MAX_WALL_SECONDS),inventory


def prepare_head_fit(run,inventory,log_path):
    script = """
import json,sys
from pathlib import Path
import torch
from shuolee_portrait import runtime as rt
from shuolee_portrait.head_fit_stage import prepare_head_fit

run = Path(sys.argv[1])
assets = Path(rt.read(run/'run.json')['assets'])
torch.set_num_threads(4)
rt.configure(assets,run)
prepare_head_fit(run,assets=assets,auxiliary_frames=json.loads(sys.argv[2]))
if torch.cuda.is_initialized():
    raise RuntimeError('The independent pre-Gaussian fit stage unexpectedly initialized CUDA.')
"""
    env = dict(cold.clean_env(),CUDA_VISIBLE_DEVICES='')
    with log_path.open('xb') as log:
        result = subprocess.run([str(PYTHON),'-B','-u','-c',script,str(run),
                                 json.dumps(inventory['archived_face_supplement']['frames'])],
                                 cwd=PROJECT/'src',env=env,stdout=log,stderr=subprocess.STDOUT)
    need(result.returncode == 0,f'The frozen CPU head fit failed; its full evidence is at {log_path}.')
    return verify_head_fit(run,inventory)


def prepare(subject,commit,cuda_proof_sha,*,verify_existing=False):
    plan,inventory = plan_for(subject,commit,cuda_proof_sha)
    root = root_for(subject)
    run = root/'run'
    if verify_existing:
        need(not (root/'cpu-preparation.json').exists() and (root/'fit-stage.log').is_file()
             and not (root/'launch.json').exists() and not (run/'status.json').exists(),
             'Existing CPU verification requires completed, unregistered fitted outputs.')
        fit = verify_head_fit(run,inventory)
    else:
        root.mkdir(exist_ok=True)
        need(not any(root.iterdir()),'Registration would overwrite an existing R20 execution.')
        stage(root,inventory)
        fit = prepare_head_fit(run,inventory,root/'fit-stage.log')
    write(root/'cpu-preparation.json',dict(candidate_commit=commit,head_fit=fit,
          fit_stage_log_sha256=sha(root/'fit-stage.log'),archive_input_sha256=inventory['fit_sha256'],
          read_only_verification_of_existing_fit=verify_existing,
          original_verifier_error=('Chronological selection was compared as reference-first order.'
                                   if verify_existing else None)))
    print(dict(state='cpu_prepared',subject=subject,fit_sha256=fit['fit_sha256']),flush=True)


def register(subject,commit,cuda_proof_sha):
    plan,inventory = plan_for(subject,commit,cuda_proof_sha)
    root = root_for(subject)
    run = root/'run'
    prep = read(root/'cpu-preparation.json')
    fit = verify_head_fit(run,inventory)
    pin(root/'fit-stage.log',prep['fit_stage_log_sha256'])
    need(prep['candidate_commit'] == commit and prep['head_fit'] == fit
         and prep['archive_input_sha256'] == inventory['fit_sha256'],
         'The prepared CPU result differs from the candidate or immutable archived fit.')
    verify_heldout_cuda_smoke(plan)
    need(not any((root/name).exists() for name in ('preregistration.json','protected-inputs.json','launch.json')),
         'Registration would overwrite an existing R20 execution.')
    cpu = cpu_fit_gate(run)
    old = inventory['archived_face_cpu_preflight']
    need(cpu['fit_sha256'] == fit['fit_sha256']
         and cpu['training_inputs_sha256'] == inventory['staged_files']['data/training-inputs.json']['sha256']
         and cpu['frames'] == len(inventory['selected_frames']) == 24
         and cpu['face_candidate_count'] == inventory['candidate_count'] == 52
         and cpu['face_admission_considered'] == 28
         and all(cpu[key] == old[key] for key in ('expected_sampling_order_sha256','expected_per_frame_updates',
                                                 'expected_source_bin_draws'))
         and len(set(cpu['chosen_face_frames'])) == len(cpu['chosen_face_frames']) <= 4
         and set(cpu['chosen_face_frames']) <= set(inventory['archived_face_supplement']['frames'])
         and set(cpu['chosen_face_frames']).isdisjoint(inventory['selected_frames']),
         'The normal CPU entry rejected the corrected fit or changed the frozen main training roster.')
    protected = root/'protected-inputs.json'
    write(protected,inventory)
    write(root/'preregistration.json',dict(plan,registered_at=cold.now(),head_fit=fit,cpu_fit_preflight=cpu,
        fit_stage_log_sha256=sha(root/'fit-stage.log'),cpu_preparation_sha256=sha(root/'cpu-preparation.json'),
        heldout_cuda_proof_sha256=HELDOUT_CUDA_PROOF_SHA,
        protected_input_inventory_sha256=sha(protected)))
    print(dict(state='registered',subject=subject,files=len(inventory['files']),
               corrected_fit_sha256=fit['fit_sha256'],chosen_face_frames=cpu['chosen_face_frames'],
               inventory_sha256=sha(protected)),flush=True)


def registered(subject,commit,cuda_proof_sha,*,allow_old_wrapper=False):
    root = root_for(subject)
    plan = read(root/'preregistration.json')
    protected = root/'protected-inputs.json'
    pin(protected,plan['protected_input_inventory_sha256'])
    pin(root/'fit-stage.log',plan['fit_stage_log_sha256'])
    pin(root/'cpu-preparation.json',plan['cpu_preparation_sha256'])
    inventory = read(protected)
    current, current_inventory = plan_for(subject,commit,cuda_proof_sha)
    need(inventory == current_inventory and all(plan[key] == value for key,value in current.items()
         if not (allow_old_wrapper and key == 'wrapper_sha256')),
         'Registered R20 candidate or protected sources differ.')
    need(verify_head_fit(root/'run',inventory) == plan['head_fit'],
         'The pre-registered corrected fit or heldout evaluation plan changed.')
    need(plan['heldout_cuda_proof_sha256'] == HELDOUT_CUDA_PROOF_SHA,'The heldout replay smoke proof changed.')
    verify_heldout_cuda_smoke(plan)
    return plan,inventory


def amend_registration(subject,commit,cuda_proof_sha):
    root = root_for(subject)
    need(not (root/'launch.json').exists() and not (root/'run/status.json').exists(),
         'The preregistration can only be amended before formal execution.')
    plan,_ = registered(subject,commit,cuda_proof_sha,allow_old_wrapper=True)
    old_wrapper = OUT/'smpl-delivery-round20/input-audit/launch-wrapper-before-densification-verifier-fix.py'
    pin(old_wrapper,plan['wrapper_sha256'])
    original = root/'preregistration.json'
    archived = root/'preregistration-before-densification-verifier-fix.json'
    with archived.open('xb') as file:
        file.write(original.read_bytes())
    amendment = dict(previous_registration_sha256=sha(archived),previous_wrapper_sha256=sha(old_wrapper),
        reason='Use the original-only R17 topology verifier also used by R19. No inputs or commands changed.')
    amended = root/'preregistration-amended.tmp'
    write(amended,dict(plan,wrapper_sha256=sha(Path(__file__)),registration_amendment=amendment))
    amended.replace(original)
    print(dict(state='registration_verifier_amended',subject=subject,sha256=sha(original)),flush=True)


def verify_heldout_cuda_smoke(plan):
    pin(HELDOUT_CUDA_PROOF,HELDOUT_CUDA_PROOF_SHA)
    proof = read(HELDOUT_CUDA_PROOF)
    source_files = {'head_fit_stage.py','heldout_pose.py','training.py','sequence_pipeline.py'}
    heldout_016 = root_for('016')/'run/fit/head-alignment/holdout-plan.json'
    need(proof['state'] == 'passed' and proof['subject'] == '016'
         and set(proof['source_sha256']) == source_files
         and all(plan['code_sha256']['src/shuolee_portrait/'+name] == digest
                 for name,digest in proof['source_sha256'].items())
         and proof['heldout_plan_sha256'] == sha(heldout_016)
         and proof['native_frames'] == read(heldout_016)['frames']
         and proof['replay_max_abs_m'] < 2e-6 and proof['camera_projection_max_abs_px'] < .001,
         'The actual CUDA heldout replay smoke does not bind this stage source and prepared 016 plan.')


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


def verify_head_fit(run,inventory):
    """Prove which fitted arrays changed while retaining independent source/observation identity."""
    import numpy as np

    stage = run/'fit/head-alignment'
    preimages = stage/'source'
    frozen = inventory['staged_files']
    for name,entry in frozen.items():
        protected = preimages/Path(name).name if name in MUTABLE_STAGED else run/name
        pin(protected,entry['sha256'])
    fit_path, candidates_path, official_path = (run/'fit'/name for name in
                                               ('fitted.npz','joint-candidates.npz','official-fit-result.json'))
    allowed = {'body_pose_axis_angle','vertices_camera','vertices_world','joints_camera','joints_projected_native'}
    pose_columns = np.asarray([*range(33,36),*range(42,45)])
    other_columns = np.setdiff1d(np.arange(63),pose_columns)
    compared = []
    archives = []
    for path in (fit_path,candidates_path):
        with np.load(preimages/path.name,allow_pickle=False) as before_file, np.load(path,allow_pickle=False) as after_file:
            before, after = dict(before_file),dict(after_file)
        need(before.keys() == after.keys() and all(before[key].shape == after[key].shape
             and before[key].dtype == after[key].dtype for key in before),
             f'The corrected archive changed its stable array interface: {path.name}.')
        changed = {key for key in before if not np.array_equal(before[key],after[key],equal_nan=False)}
        need(changed <= allowed and all(np.isfinite(after[key]).all() for key in changed)
             and np.array_equal(before['body_pose_axis_angle'][:,other_columns],after['body_pose_axis_angle'][:,other_columns]),
             f'The corrected archive changed non-head pose, identity, camera, topology or source observations: {path.name}.')
        rows = after['frame_ids'].tolist()
        expected = inventory['selected_frames'] if path == fit_path else sorted(inventory['candidate_frames'],
                      key=lambda frame: inventory['candidate_frames'][frame]['candidate_pose_index'])
        need(rows == expected and set(rows).isdisjoint(inventory['retained_holdouts']),
             f'The corrected archive changed source frame order or included a holdout: {path.name}.')
        joints = after['joints_camera'].astype(np.float64)
        K = after['K_integer_centers'].astype(np.float64)
        projected = joints @ K.transpose(0,2,1)
        need(np.all(projected[...,2] > 0)
             and np.max(np.abs(projected[...,:2]/projected[...,2:]-after['joints_projected_native'])) < .001,
             f'Native fitted projections do not reproduce using the unchanged K once: {path.name}.')
        changed_frames = [name for i,name in enumerate(rows) if not np.array_equal(
            before['body_pose_axis_angle'][i],after['body_pose_axis_angle'][i])]
        compared.append(dict(file=str(path.relative_to(run)),sha256=sha(path),changed_fields=sorted(changed),
                             changed_pose_frames=changed_frames))
        archives.append(after)
    fitted,candidates = archives
    slots = [inventory['candidate_frames'][name]['candidate_pose_index'] for name in inventory['selected_frames']]
    # The selected rows and their frozen camera gauges must coincide with the
    # matching corrected candidates, not be copied from other views.
    need(all(np.array_equal(fitted[key],candidates[key][slots]) for key in
         ('body_pose_axis_angle','vertices_camera','joints_camera','joints_projected_native')),
         'Corrected full-body poses do not match their own frozen candidate rows.')
    E = fitted['w2c']
    world = (fitted['vertices_camera'].astype(np.float64)-E[:,None,:3,3]) @ E[:,:3,:3]
    need(np.isfinite(fitted['vertices_world']).all() and np.max(np.abs(world-fitted['vertices_world'])) < 2e-6,
         'Corrected world geometry applies a different frozen camera transform.')
    before, after = read(preimages/official_path.name),read(official_path)
    need({key:value for key,value in after.items() if key not in ('fit_sha256','native_head_alignment')}
         == {key:value for key,value in before.items() if key != 'fit_sha256'} and after['fit_sha256'] == sha(fit_path),
         'The revised official-fit receipt changed source, frame selection or shared identity.')
    fit_meta = _verify_fit_audit(run,inventory,fitted,candidates,compared)
    holds = _verify_holdout_plan(run,inventory,fitted)
    active_files = {name:sha(run/name) for name in sorted(MUTABLE_STAGED | set(FIT_STAGE_FILES))}
    return dict(fit_sha256=sha(fit_path),candidate_sha256=sha(candidates_path),official_fit_sha256=sha(official_path),
                active_files_sha256=active_files,comparison=compared,correction=fit_meta,holdout_plan=holds)


def _verify_fit_audit(run,inventory,fitted,candidates,compared):
    import numpy as np

    stage = run/'fit/head-alignment'
    result = read(stage/'result.json')
    descriptor = read(run/'fit/official-fit-result.json')['native_head_alignment']
    active = dict(fit_sha256=compared[0]['sha256'],candidate_sha256=compared[1]['sha256'])
    old = inventory['staged_files']
    sources = {Path(name).name:old[name]['sha256'] for name in MUTABLE_STAGED}
    original = inventory['selected_frames']
    aux = inventory['archived_face_supplement']['frames']
    roster = original+aux
    need(descriptor == dict(file='fit/head-alignment/result.json',sha256=sha(stage/'result.json'),**active)
         and result['schema'] == 'shuolee-native-head-fit-stage/1' and result['state'] == 'complete'
         and result['active'] == active and result['source_sha256'] == sources
         and result['frozen_auxiliary_frames'] == aux
         and len(result['selected_frames']) == len(original) and set(result['selected_frames']) == set(original)
         and result['heldout_frames'] == inventory['retained_holdouts']
         and result['changed_main_frames'] == compared[0]['changed_pose_frames']
         and result['changed_candidate_frames'] == compared[1]['changed_pose_frames'],
         'Head-fit audit does not bind the independent archive and active pose changes.')
    audit = result['fitting']
    changed = set(compared[1]['changed_pose_frames'])
    need(audit['schema'] == 'shuolee-native-head-alignment/1' and audit['state'] == 'completed'
         and audit['original_frames'] == original and audit['auxiliary_frames'] == aux
         and audit['fitted_frames'] == roster and audit['retained_holdouts'] == inventory['retained_holdouts']
         and audit['head_columns'] == [33,34,35,42,43,44] and set(audit['changed_frames']) == changed
         and audit['shared_shape_updates'] == audit['per_frame_scale_parameters'] == 0
         and [row['frame'] for row in audit['frames']] == roster and changed <= set(roster),
         'Head alignment used a different roster, nonlocal shape or altered camera scale.')
    slots = np.asarray(audit['evaluated_landmark_slots'])
    with np.load(stage/'source/joint-candidates.npz',allow_pickle=False) as archived:
        for row in audit['frames']:
            frame = row['frame']
            frozen = inventory['candidate_frames'][frame]
            index = frozen['candidate_pose_index']
            expected = {key:np.asarray(value) for key,value in (
                ('before_projected_native',archived['joints_projected_native'][index,slots]),
                ('after_projected_native',candidates['joints_projected_native'][index,slots]),
                ('native_observed',candidates['observed_native_keypoints'][index,slots]))}
            need(row['source_index'] == frozen['source_frame_index']
                 and row['source_image_sha256'] == frozen['image']['sha256']
                 and row['independent_mask_sha256'] == frozen['mask']['sha256']
                 and all(np.array_equal(row[key],value) for key,value in expected.items())
                 and (row['state'] == 'committed') == (frame in changed)
                 and (frame not in changed or any(not trial['failures'] for trial in row['trials'])),
                 f'Head fit native feature or acceptance evidence differs for {frame}.')
    return dict(result_sha256=sha(stage/'result.json'),changed_main_frames=result['changed_main_frames'],
                changed_candidate_frames=result['changed_candidate_frames'],frozen_auxiliary_frames=aux)


def _verify_holdout_plan(run,inventory,fitted):
    import numpy as np

    stage = run/'fit/head-alignment'
    policy, plan = (read(stage/name) for name in ('holdout-policy.json','holdout-plan.json'))
    ids = inventory['retained_holdouts']
    paths = dict(native_inputs_sha256=run/'data/inputs.json',observations_sha256=run/'data/observations.npz',
                 training_inputs_sha256=run/'data/training-inputs.json',policy_sha256=stage/'holdout-policy.json',
                 poses_sha256=stage/'holdout-poses.npz')
    need(plan['state'] == 'frozen_before_gaussian' and all(plan[key] == value for key,value in policy.items())
         and all(plan[key] == sha(path) for key,path in paths.items()) and plan['frames'] == ids
         and all(plan[key] is True for key in ('shared_shape_frozen','camera_frozen','selection_frozen'))
         and all(plan[key] is False for key in ('use_rgb_for_gaussian_fitting','use_final_render_for_pose_fit'))
         and plan['gaussian_updates'] == plan['shared_identity_updates'] == 0
         and [row['frame'] for row in plan['records']] == ids
         and read(stage/'result.json')['holdout_plan_sha256'] == sha(stage/'holdout-plan.json'),
         'The heldout poses were not bound to an evaluation-only frozen plan.')
    with np.load(paths['poses_sha256'],allow_pickle=False) as archive:
        poses = dict(archive)
    need(poses['frame_ids'].tolist() == ids and np.array_equal(poses['shared_betas'],fitted['shared_betas'])
         and np.array_equal(poses['faces'],fitted['faces']) and np.isfinite(poses['vertices_world']).all(),
         'A heldout pose did not use the frozen shape and same official topology.')
    records = []
    for index,row in enumerate(plan['records']):
        frame = row['frame']
        frozen = inventory['heldout_pose_sources'][frame]
        item = row['official_fit']
        source = (run/item['path']).resolve(strict=True)
        K = np.asarray(frozen['K_integer_centers']).copy()
        K[:2,2] += .5
        need(source.is_relative_to(stage/'holdout-official'/frame) and row['pose_index'] == index
             and row['role'] == 'pose_conditioned_holdout' and row['source_frame_index'] == frozen['source_frame_index']
             and row['source_sha256'] == frozen['image']['sha256'] and row['mask_sha256'] == frozen['mask']['sha256']
             and row['wh'] == frozen['native_wh'] and np.array_equal(row['K'],K)
             and np.array_equal(row['K_integer_centers'],poses['K_integer_centers'][index])
             and row['K_integer_centers'] == frozen['K_integer_centers'] and row['source_w2c'] == frozen['source_w2c']
             and np.array_equal(row['w2c'],poses['w2c'][index])
             and np.allclose(row['w2c'],frozen['metric_w2c'],atol=1e-12,rtol=0),
             f'Heldout {frame} changed source, camera or frozen pose identity.')
        pin(source,item['sha256'])
        receipt = read(source)
        need(receipt['state'] == 'complete' and receipt['five_stages'] is True and receipt['frame_id'] == frame
             and receipt['source_frame_index'] == frozen['source_frame_index']
             and receipt['source_image_sha256'] == frozen['image']['sha256'],
             f'Heldout {frame} has no actual completed official per-frame fit.')
        records.append(dict(frame=frame,official_fit_path=item['path'],official_fit_sha256=item['sha256']))
    return dict(frames=ids,plan_sha256=sha(stage/'holdout-plan.json'),poses_sha256=plan['poses_sha256'],records=records)


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
        receipt['preserved_protected_stage'] = [dict(file=str(path.relative_to(run)),sha256=sha(path))
            for path in sorted((run/'gaussian-run/protected-face').glob('*')) if path.is_file()]
        main = run/'gaussian-run/step-010000-main/native-ply-parity.json'
        if main.is_file():
            receipt['preserved_main_snapshot'] = snapshot_failure(main,run)
        receipt['retry_performed'] = False
    write(root/'wrapper-terminal.json',receipt)
    print(dict(state=receipt['state'],subject=plan['subject'],validation_error=receipt.get('validation_error')),flush=True)
    return 0 if receipt['state']=='completed' else returncode or 1


def verify_outputs(run,plan,inventory):
    for path,digest in inventory['files'].items():
        pin(Path(path),digest)
    need(verify_head_fit(run,inventory) == plan['head_fit'],'The terminal fit differs from the protected CPU stage.')
    for path,digest in plan['code_sha256'].items():
        pin(PROJECT/path,digest)

    def baseline(relative):
        return read(Path(inventory['profiles'][relative]['path']))

    cfg = read(run/'config.json')
    effective = read(run/'gaussian-run/config.json')
    prior_effective = baseline('gaussian-run/config.json')
    fitted_profile_keys = {'official_fit_provenance','reference_extent_m','fit_replay_max_abs_m'}
    fixed_profile = {key:value for key,value in effective.items() if key not in fitted_profile_keys}
    need(cfg == baseline('config.json') and fixed_profile == {
        key:value for key,value in prior_effective.items() if key not in fitted_profile_keys}
        and math.isfinite(effective['reference_extent_m']) and effective['reference_extent_m'] > 0
        and 0 <= effective['fit_replay_max_abs_m'] < 2e-6,
        'R20 changed fixed training settings or broke corrected mesh replay.')
    official = effective['official_fit_provenance']
    official_old = prior_effective['official_fit_provenance']
    old_keys = {'receipt_sha256','fit_sha256'}
    need({key:value for key,value in official.items() if key not in old_keys | {'native_head_alignment'}}
         == {key:value for key,value in official_old.items() if key not in old_keys}
         and official['receipt_sha256'] == plan['head_fit']['official_fit_sha256']
         and official['fit_sha256'] == plan['head_fit']['fit_sha256']
         and official['native_head_alignment'] == read(run/'fit/official-fit-result.json')['native_head_alignment'],
         'The optimizer did not consume the active corrected official SMPL-X fit.')
    for name in ('config.json','gaussian-run/training-cameras.json','gaussian-run/initialization.json'):
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
         'R20 changed initialization, optimization schedule, or the consumed pose/sample sequence.')

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
         'Bound replay changed the selected pose sequence.')
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
    protection = verify_protected_stage(run,integration,trained,effective,inventory,
                                        plan['cpu_fit_preflight']['chosen_face_frames'])
    face = verify_face_supplement(run,integration,inventory,bound,plan['cpu_fit_preflight'],protection)
    heldout = verify_heldout_replay(run,result,plan,inventory,bound)
    densification = prior.prior.prior.verify_densification(run,integration,effective,inventory['selected_frames'],result['points'])
    for payload in (read(run/densification['file']),read(run/'gaussian-run/world-scale.json'),integration['coverage_events']):
        need(not _has_main_auxiliary(payload),'An auxiliary face view influenced main topology decisions.')
    selection = read(run/'fit/selection.json')
    joint = read(run/'fit/official-fit-result.json')['joint_body_refinement']
    return dict(state='completed',result_sha256=sha(run/'result.json'),points=result['points'],
        reference_frame=result['reference_frame_id'],selected_frames=selection['selected_frames'],
        coverage=selection['coverage'],joint_refinement_sha256=joint['sha256'],
        joint_candidate_sha256=plan['head_fit']['candidate_sha256'],source_joint_candidate_sha256=joint['candidate_sha256'],
        world_scale_sha256=integration['world_scale']['sha256'],fit_method='native_head_alignment_on_round19_fit',
        fresh_gaussian=True,original_video_to_export=False,native_inputs_reused=True,
        reference_mesh_frozen_from_prior=False,fit_sha256=plan['head_fit']['fit_sha256'],
        source_fit_sha256=inventory['fit_sha256'],head_fit=plan['head_fit'],
        training_inputs_sha256=inventory['staged_files']['data/training-inputs.json']['sha256'],
        same_round19_training_selection=True,face_supplement=face,heldout_replay=heldout,
        protected_face=protection,verified_pose_count=24+len(face['frames']),
        face_guard=read(run/joint['file'])['face_guard'],
        color_constraint=color_proof,binding=binding,densification_audit=densification,native_ply_parity=parity)


_snapshot_parity = prior._snapshot_parity
_has_main_auxiliary = prior._has_main_auxiliary
verify_mesh_and_bound_identity = prior.verify_mesh_and_bound_identity
_verify_fixed_measurement = prior._verify_fixed_measurement
_verify_committed_state = prior._verify_committed_state


def _verify_protected_sweeps(proof,roster):
    from collections import Counter

    opt, config = proof['protected'],proof['config']
    phases, cycles = opt['phases'],opt['cycles']
    groups = {}
    for frame,role,bin_id in roster:
        groups.setdefault((role,bin_id),[]).append(frame)
    grouped = [groups[key] for key in sorted(groups)]
    expected_phases = [('primary',1.),('conservative_restart',.25)][:len(phases)]
    need(24 < len(roster) <= 28 and 1 <= len(phases) <= 2
         and [(row['phase'],row['lr_factor']) for row in phases] == expected_phases
         and config['max_cycles'] == 6 and config['min_cycles'] == 4 and config['stable_cycles'] == 2
         and config['conservative_lr_factor'] == .25 and config['minimum_gain'] == .0001,
         'The protected-face sweep or bounded retry configuration differs.')
    cumulative = 0
    candidates = []
    draws = Counter()
    seen_cycles = 0
    for phase_index,phase in enumerate(phases):
        part = [row for row in cycles if row['phase'] == phase['phase']]
        need(phase['same_endpoint'] is True and 1 <= len(part) <= config['max_cycles'],
             'A protected phase did not start from the same saved endpoint.')
        used = 0
        for offset,cycle in enumerate(part):
            shift = offset % len(grouped)
            rotated = grouped[shift:]+grouped[:shift]
            rotated = [rows[offset%len(rows):]+rows[:offset%len(rows)] for rows in rotated]
            sequence = [rows[i] for i in range(max(map(len,rotated))) for rows in rotated if i < len(rows)]
            order = cycle['draw_order']
            used += len(order)
            cumulative += len(order)
            draws.update(order)
            seen_cycles += 1
            need(cycle['cycle'] == seen_cycles and cycle['phase_cycle'] == offset+1
                 and cycle['lr_factor'] == phase['lr_factor']
                 and cycle['optimizer_updates'] == used and cycle['cumulative_attempted_updates'] == cumulative
                 and order == sequence[:len(order)] and (not cycle['completed'] or len(order) == len(roster)),
                 'A protected optimizer cycle did not use its full balanced source roster.')
            failure_details = cycle.get('failure_details', [])
            roles = {frame:role for frame,role,_ in roster}
            need([row['frame'] for row in failure_details] == cycle.get('failed_frames', [])
                 and all(row['role'] == roles[row['frame']] and row['failures'] for row in failure_details),
                 'Protected per-cycle failures lost source-role evidence.')
            expected = bool(cycle['completed'] and cycle['finite'] and cycle['all_guards_pass']
                        and cycle['gains']['joint'] >= config['minimum_gain']
                        and cycle['gains']['supplemental'] >= config['minimum_gain'])
            need(cycle['feasible'] is expected,'Protected checkpoint feasibility differs from its fixed assessment.')
            if expected:
                candidates.append(cycle)
        need(phase['optimizer_updates'] == used
             and phase['complete_cycles'] == sum(row['completed'] for row in part)
             and (phase_index == len(phases)-1 or not any(row['feasible'] for row in part)),
             'Protected phase totals differ or a lower-rate branch ran after an approved primary.')
        if phase['stop_reason'] == 'cycle_budget':
            need(len(part) == phase['complete_cycles'] == config['max_cycles'],
                 'The optimizer stopped before its registered bounded cycle budget.')
        elif phase['stop_reason'] == 'stable_complete_sweeps':
            last = part[-3:]
            need(len(part) >= config['min_cycles'] and all(row['feasible'] for row in last)
                 and all(abs(a['gains']['joint']-b['gains']['joint']) <= config['convergence_delta']
                         for a,b in zip(last,last[1:])),
                 'Protected convergence lacks the fixed full-roster checkpoint evidence.')
    need(seen_cycles == len(cycles) and opt['view_draws'] == dict(draws)
         and proof['attempted_updates'] == proof['optimizer_updates'] == opt['optimizer_updates'] == cumulative,
         'Protected actual optimizer counts differ from full source sweep records.')
    chosen = proof['selected']
    best = max(candidates,key=lambda row: row['gains']['joint']) if candidates else None
    decision_keys = ('cycle','phase','phase_cycle','lr_factor','optimizer_updates','gains')
    need(chosen == ({key:best[key] for key in decision_keys} if best else None)
         and proof['committed_updates'] == (best['optimizer_updates'] if proof['state'] == 'committed' else 0),
         'The protected selected update is not the best evaluated feasible checkpoint.')
    control = proof['unprotected']
    expected_control = best['phase_cycle'] if best else phases[0]['complete_cycles']
    control_frames = {frame for frame,role,_ in roster if role == 'supplemental'}
    control_draws = control['view_draws']
    need(control['cycle_budget'] == expected_control and control['lr_factor'] == (best['lr_factor'] if best else 1.)
         and set(control_draws) <= control_frames and control['optimizer_updates'] == sum(control_draws.values())
         and (control['issue'] is not None or control_draws == dict.fromkeys(control_frames,expected_control)),
         'The unprotected comparison did not use the matched auxiliary view counts from the same endpoint.')
    if proof['state'] != 'committed':
        return Counter()
    return Counter(frame for cycle in cycles if cycle['phase'] == chosen['phase']
                   and cycle['phase_cycle'] <= chosen['phase_cycle'] for frame in cycle['draw_order'])


def verify_protected_stage(run,integration,trained,effective,inventory,face_frames):
    out = run/'gaussian-run'
    stage = out/'protected-face'
    final = out/'step-010000'
    main = out/'step-010000-main' if face_frames else final
    desc = integration['protected_face']
    audit_path, endpoint_path = (stage/name for name in ('optimization.json','main-endpoint.json'))
    need(desc == trained['protected_face'] and desc['file'] == 'protected-face/optimization.json'
         and desc['main_endpoint'] == 'protected-face/main-endpoint.json'
         and desc['main_snapshot'] == trained['main_snapshot'] == main.name,
         'The normal training result does not bind the saved main and protected stages.')
    pin(audit_path,desc['sha256'])
    pin(endpoint_path,desc['main_endpoint_sha256'])
    proof, endpoint = read(audit_path),read(endpoint_path)
    need(endpoint['state'] == 'saved' and endpoint['snapshot'] == main.name
         and endpoint['main_optimizer_updates'] == TRAINING_STEPS and endpoint['supplemental_updates'] == 0
         and set(endpoint['files_sha256']) == {'world.ply','reference-render.png','reference-mesh.npz',
               'native-ply-parity.json','bound-local.ply','bound-replay.json'},
         'Missing intact 10,000-step original-only endpoint.')
    for name,digest in endpoint['files_sha256'].items():
        pin(main/name,digest)
    parity = _snapshot_parity(main,inventory['reference_frame'],run)
    main_bound, final_bound = (read(snapshot/'bound-replay.json') for snapshot in (main,final))
    for snapshot,replay in ((main,main_bound),(final,final_bound)):
        pin(snapshot/'bound-local.ply',replay['bound_ply_sha256'])
        bound_inputs = {'fitted':run/'fit/fitted.npz','training_rows':run/'data/training-inputs.json',
                        'reference_mesh':snapshot/'reference-mesh.npz','world_ply':snapshot/'world.ply',
                        'native_ply_parity':snapshot/'native-ply-parity.json'}
        need(set(replay['inputs_sha256']) == set(bound_inputs),'Incomplete bound snapshot input manifest.')
        for key,path in bound_inputs.items():
            pin(path,replay['inputs_sha256'][key])
        need(replay['points'] == trained['points'] and replay['reference_frame_id'] == inventory['reference_frame']
             and replay['reference_pose_index'] == 0 and replay['sh_degree'] == 0
             and [row['frame'] for row in replay['frames']] == inventory['selected_frames'],
             'Main or final bound replay does not preserve the 24 original named poses.')
    need(main_bound['frames'] == final_bound['frames'] and all(main_bound['inputs_sha256'][key] ==
         final_bound['inputs_sha256'][key] for key in ('fitted','training_rows')),
         'Main and final bound snapshots used different original pose/camera inputs.')
    if not face_frames:
        need(proof['state'] == desc['state'] == 'skipped' and proof['reason'] == 'no_admitted_auxiliary_views'
             and all(desc[key] == proof[key] == 0 for key in ('attempted_updates','committed_updates'))
             and desc['committed_auxiliary_frame_ids'] == [], 'An empty auxiliary roster performed updates.')
        return dict(desc,main_native_ply_parity=parity,baseline_views=24,unprotected_control_updates=0)
    committed = proof['state'] == 'committed'
    identity = verify_mesh_and_bound_identity(run,main,final,('f_dc','opacity'),committed)
    need(proof['schema'] == 'shuolee-protected-face/1' and proof['state'] in ('committed','rolled_back')
         and all(desc[key] == proof[key] for key in ('state','attempted_updates','committed_updates','seconds'))
         and proof['main_optimizer_updates'] == 0 and proof['head_rows'] == identity['head_rows']
         and proof['rng_unchanged'] is True and proof['unprotected']['same_endpoint'] is True
         and proof['unprotected']['rng_unchanged'] is True
         and {key:value for key,value in proof['endpoint'].items() if key != 'mutable_head'}
             == {key:value for key,value in proof['live_final'].items() if key != 'mutable_head'}
         and proof['endpoint']['points'] == trained['points'],
         'Protected refinement lost the fixed topology, optimizer identity or common RNG endpoint.')
    roster = (list(zip(inventory['selected_frames'],['original']*24,effective['training_source_bins'],strict=True))
              +[(name,'supplemental',i) for i,name in enumerate(face_frames)])
    guard = effective['unified_main']['world_scale']['render_support']
    baseline, final_metrics = proof['baseline'],proof['final']
    _verify_fixed_measurement(baseline,baseline,roster,guard,require_pass=True)
    _verify_fixed_measurement(proof['unprotected']['measurement'],baseline,roster,guard)
    _verify_fixed_measurement(final_metrics,baseline,roster,guard,require_pass=True)
    sweep = _verify_protected_sweeps(proof,roster)
    need(desc['committed_auxiliary_frame_ids'] == [frame for frame in face_frames if sweep[frame] > 0],
         'Committed auxiliary IDs differ from the exact chosen Adam updates.')
    gains = proof['final_gains']
    need(math.isclose(gains['joint'],baseline['objective']-final_metrics['objective'],abs_tol=1e-9)
         and math.isclose(gains['supplemental'],baseline['supplemental_objective']-final_metrics['supplemental_objective'],abs_tol=1e-9),
         'Reported final gains do not correspond to the same main endpoint.')
    if committed:
        need(proof['selected'] is not None and proof['committed_updates'] > 0
             and gains['joint'] >= proof['config']['minimum_gain'] == .0001
             and gains['supplemental'] >= proof['config']['minimum_gain']
             and proof['live_final']['mutable_head'] != proof['endpoint']['mutable_head'],
             'Protected commit has no verified nonzero joint and supplemental gains.')
        state = _verify_committed_state(stage,proof,main,final)
    else:
        need(proof['committed_updates'] == 0 and proof['optimizer_state'] is None
             and proof['endpoint'] == proof['live_final'] and not (stage/'protected-optimizer-state.pt').exists()
             and math.isclose(gains['joint'],0.,abs_tol=1e-7) and math.isclose(gains['supplemental'],0.,abs_tol=1e-7),
             'A rejected protected stage did not fully roll back to the original saved state.')
        state = None
    return dict(desc,main_native_ply_parity=parity,identity=identity,optimizer_state=state,final_gains=gains,
                committed_frame_draws={frame:sweep[frame] for frame in face_frames},
                baseline_views=len(roster),unprotected_control_updates=proof['unprotected']['optimizer_updates'])


def verify_face_supplement(run,integration,inventory,bound,preflight,protection):
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
    need(descriptor['file'] == 'face-supplement/training.json'
         and descriptor['sha256'] == sha(report_path) and descriptor['views'] == len(frames)
         and descriptor['face_loss_updates'] == report['face_loss_updates'] == 0
         and report['schema'] == 'face-supplement-training-v2' and report['state'] == 'completed'
         and report['admission_sha256'] == sha(admission_path)
         and report['main_optimizer_updates'] == TRAINING_STEPS
         and report['separate_optimizer_updates'] == protection['attempted_updates']
         and report['committed_updates'] == protection['committed_updates'] and report['head_binding_faces'] > 0
         and frames == preflight['chosen_face_frames']
         and set(frames) <= set(inventory['archived_face_supplement']['frames'])
         and len(set(frames)) == len(frames) and set(frames) <= candidates.keys()
         and set(frames).isdisjoint(inventory['selected_frames'])
         and set(frames).isdisjoint(inventory['retained_holdouts']),
         'The separate facial supervision changed the protected training roster or optimizer budget.')
    active_inventory = dict(inventory,staged_files={path:dict(record,sha256=sha(run/path))
        if path in MUTABLE_STAGED else record for path,record in inventory['staged_files'].items()})
    def payload_digest(value):
        return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                         separators=(',',':'),allow_nan=False).encode()).hexdigest()

    overlay_digest = read(run/'fit/official-fit-result.json')['native_head_alignment']['sha256']
    current_digests = admission['input_sha256']
    need(current_digests['native_head_alignment'] == overlay_digest
         and payload_digest(admission) == preflight['face_admission_payload_sha256'],
         'Facial admission did not consume the corrected fit or differs from the registered CPU gate.')
    legacy_admission = dict(admission,input_sha256={key:value for key,value in current_digests.items()
                                                   if key != 'native_head_alignment'})
    legacy_preflight = dict(preflight,face_admission_payload_sha256=payload_digest(legacy_admission))
    admission_rows = prior.prior.verify_admission(legacy_admission,active_inventory,frames,legacy_preflight)
    comparison = admission['fixed_identity_comparison']
    old_frames = inventory['archived_face_supplement']['frames']
    need(comparison['original_face_only_frames'] == old_frames and comparison['readmitted_frame_ids'] == frames
         and [row['frame'] for row in comparison['not_readmitted']] == [frame for frame in old_frames if frame not in frames],
         'A rejected original face-only frame was silently replaced or omitted from the audit.')
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
             and row['updates'] == row['density_observations'] == row['max_density_visible_rows'] == 0
             and row['committed_face_updates'] == protection['committed_frame_draws'][frame]
             and not any(key in row for key in ('first_main_step','last_main_step')),
             f'Face-only frame {frame} used a different pose/camera or contributed to the frozen main stage.')
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
                                 committed_face_updates=row['committed_face_updates'],
                                 valid_mask_sha256=row['valid_mask_sha256'],files=files))
    return dict(frames=frames,training_sha256=sha(report_path),admission_sha256=sha(admission_path),
                replay_sha256=sha(replay_path),poses_sha256=replay['poses_sha256'],
                face_loss_updates=0,separate_optimizer_updates=report['separate_optimizer_updates'],
                committed_updates=report['committed_updates'],records=file_records)


def verify_heldout_replay(run,result,plan,inventory,bound):
    import numpy as np
    from PIL import Image

    parent = bound.parent/'heldout-replay'
    path = parent/'result.json'
    need(result['heldout_bound_replay'] == str(path.relative_to(run)),
         'The heldout renders are not attached to the final bound model.')
    pin(path,result['heldout_bound_replay_sha256'])
    report = read(path)
    source = plan['head_fit']['holdout_plan']
    frames = inventory['retained_holdouts']
    prepared_path = run/'fit/head-alignment/holdout-plan.json'
    prepared = read(prepared_path)
    need(report['schema'] == 'shuolee-pose-conditioned-holdout-replay/1' and report['state'] == 'completed'
         and report['holdout_plan_sha256'] == sha(prepared_path) == source['plan_sha256']
         and report['holdout_pose_sha256'] == source['poses_sha256']
         and report['bound_ply_sha256'] == sha(bound) and report['optimizer_updates'] == 0
         and report['fully_unobserved_prediction'] is False and report['uses_frozen_clothing_offset'] is True
         and [row['frame'] for row in report['records']] == frames == source['frames']
         and report['mesh_file'] == 'meshes.npz',
         'The heldout evidence did not use exactly the frozen bound model and evaluation poses.')
    pin(parent/report['mesh_file'],report['mesh_sha256'])
    first_slot = len(inventory['selected_frames'])
    with np.load(parent/report['mesh_file'],allow_pickle=False) as mesh, \
            np.load(run/'fit/head-alignment/holdout-poses.npz',allow_pickle=False) as holdout:
        need(mesh['frame_ids'].tolist() == frames and np.array_equal(mesh['faces'],holdout['faces'])
             and mesh['pose_indices'].tolist() == list(range(first_slot,first_slot+len(frames)))
             and mesh['vertices_world'].shape == holdout['vertices_world'].shape
             and np.isfinite(mesh['vertices_world']).all(),
             'The heldout mesh archive changed frame, topology or evaluation pose order.')
    records = []
    for i,(row,original) in enumerate(zip(report['records'],prepared['records'],strict=True)):
        frame, files = row['frame'], row['files']
        camera = row['camera']
        need(all(row[key] == value for key,value in original.items() if key != 'pose_index')
             and row['fitted_pose_index'] == original['pose_index'] == i
             and row['pose_index'] == camera['pose_index'] == first_slot+i
             and camera['frame'] == frame and camera['wh'] == original['wh']
             and camera['K'] == original['K'] and camera['w2c'] == original['w2c']
             and camera['pixel_convention'] == 'pixel_corners' and 0 <= camera['projection_error_px'] < .001
             and set(files) == {'black','white','alpha','mesh','feature'},
             f'Heldout {frame} used a different native camera, pose or render set.')
        for role,desc in files.items():
            need(desc['file'] == f'{frame}-{role}.png',f'Invalid heldout output path for {frame}.')
            png = parent/desc['file']
            pin(png,desc['sha256'])
            with Image.open(png) as panel:
                x0,y0,x1,y1 = row['crop_xyxy']
                dims = original['wh'] if role in ('black','white','alpha') else [x1-x0,y1-y0]
                need(list(panel.size) == dims,f'Heldout {frame} {role} uses different native pixel dimensions.')
        records.append(dict(frame=frame,pose_index=row['pose_index'],files=files))
    return dict(file=str(path.relative_to(run)),sha256=sha(path),frames=frames,mesh_sha256=report['mesh_sha256'],
                fully_unobserved_prediction=False,optimizer_updates=0,records=records)


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
        need(not any((root/name).exists() for name in ('launch.json','entry.log','wrapper-terminal.json'))
             and not any((root/'run'/name).exists() for name in ('status.json','progress.json','gaussian-run','export','result.json')),
             'Fresh round-twenty output is required.')
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
    parser.add_argument('action',choices=('check-inputs','check','prepare','verify-prepared','register',
                                          'amend-registration','launch','check-early'))
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
    if args.action in ('prepare','verify-prepared'):
        prepare(args.subject,args.commit,args.cuda_proof_sha256,verify_existing=args.action == 'verify-prepared')
    elif args.action == 'register':
        register(args.subject,args.commit,args.cuda_proof_sha256)
    elif args.action == 'amend-registration':
        amend_registration(args.subject,args.commit,args.cuda_proof_sha256)
    else:
        plan, data = plan_for(args.subject,args.commit,args.cuda_proof_sha256)
        print(dict(state='preflight_passed',subject=args.subject,files=len(data['files']),
                   selected_frames=data['selected_frames'],policy=plan['color_policy']),flush=True)


if __name__ == '__main__':
    main()
