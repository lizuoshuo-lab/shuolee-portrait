"""Run the normal 10k main stage and guarded same-model facial refinement for R19."""

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
import launch_smpl_delivery_round18 as prior


need, read, pin, sha, write = cold.need, cold.read, cold.pin, cold.sha, cold.write
check_snapshot, color_stats = prior.check_snapshot, prior.color_stats
stage, check_binding = prior.stage, prior.check_binding
snapshot_failure, verify_parity = prior.snapshot_failure, prior.verify_parity
PYTHON = cold.ENV/'bin/python'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-protected-face-20261111')
PARENT_PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-supplement-20261111')
PARENT = '9f1bd057fe50fb6c9d93ee69a9ddcb31d941fd9d'
QUALITY_BASELINE = PARENT
MAIN_BASELINE = prior.PARENT
STABLE_TAG_COMMIT = 'a6166a5ffd439362ab2c17cda30f9ec408c688c4'
OUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs')
DIRECTOR_ROOT = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs')
DIRECTOR = DIRECTOR_ROOT/'round18_director_review_20261111.json'
DIRECTOR_SHA = '453fb58304f84ee99c4a6366d33df047d0c9a05122b1755cde3e1d69550cbfdd'
DELIVERY = OUT/'smpl-delivery-round18/receipt.json'
DELIVERY_SHA = '2bc141e4d9e017542393a09c68815f97ecfdcc4c7b9ec148c96cfbf5d3e0d1f4'
CUDA_PROOF = OUT/'smpl-delivery-round19/input-audit/cuda-protected-face-proof.json'
CUDA_FIXTURE_SOURCE = Path(__file__).with_name('probe_cuda_protected_face_round19.py')
POLICY = 'projected_ldr_dc'
POLICY_KEY = 'sh0_color_constraint'
COLOR_C0 = .28209479177387814
MAX_WALL_SECONDS = 7200
TRAINING_STEPS = 10000
SCHEMA = 'shuolee-round19-fixed-fit-protected-face-v1'
REQUIRED = {'src/shuolee_portrait/training.py','src/shuolee_portrait/protected_face.py'}
ALLOWED = REQUIRED | {'src/shuolee_portrait/protected_face.py','src/shuolee_portrait/protected_face_views.py',
                      'src/shuolee_portrait/face_supplement_training.py',
                      'src/shuolee_portrait/render_support.py','src/shuolee_portrait/world_scale.py',
                      'src/shuolee_portrait/bound_replay.py','src/shuolee_portrait/bound_review_panels.py',
                      'src/shuolee_portrait/export.py','tests/test_render_support.py','tests/test_world_scale.py'}
EVIDENCE_FILES = prior.EVIDENCE_FILES


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round19-{subject}'


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
                     and Path(name).name.startswith(('test_protected_face','test_face_supplement','test_bound_'))
                     and Path(name).suffix == '.py'}
    need(REQUIRED <= changed <= ALLOWED | focused_tests,
         f'Unexpected protected face code paths: {sorted(changed)}')
    prior_gitdir = subprocess.check_output(
        ['/usr/bin/wslpath','-u',(PARENT_PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')],
        text=True).strip()
    prior_git = ['/usr/bin/git','--no-optional-locks','-c','core.fileMode=false','--git-dir',prior_gitdir,'--work-tree',str(PARENT_PROJECT)]
    prior_head = subprocess.check_output([*prior_git,'rev-parse','HEAD'],text=True).strip()
    prior_status = subprocess.check_output([*prior_git,'status','--porcelain','--untracked-files=all'],
                                            cwd=PARENT_PROJECT,text=True)
    need(prior_head == PARENT and not prior_status.strip(), 'The R18 worktree changed.')
    tracked = set(cold.SOURCE_FILES) | changed | prior.prior.OPT_PATHS | {
        'src/shuolee_portrait/face_supplement.py','src/shuolee_portrait/face_supplement_training.py',
        'src/shuolee_portrait/sh0_main.py','src/shuolee_portrait/world_scale.py',
        'src/shuolee_portrait/coverage_refinement.py','src/shuolee_portrait/render_support.py'}
    for name in tracked-changed:
        pin(PROJECT/name,sha(PARENT_PROJECT/name))
    return {name:sha(PROJECT/name) for name in sorted(tracked)}


def source_inventory(subject):
    """Bind the previous paired delivery and its exact candidate, pose and RGB/alpha sources."""
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
    need(review['round'] == 18 and review['commit'] == delivery['commit'] == PARENT
         and review['direct_baseline'] == delivery['round17_baseline_commit'] == prior.PARENT
         and delivery['baseline_commit'] == prior.prior.QUALITY_BASELINE
         and review['receipt_sha256'] == DELIVERY_SHA
         and review['quality_accepted'] is delivery['quality_accepted'] is False
         and review['next_round']['round'] == 19,
         'R18 report or director authorization differs.')
    need(review['findings_sha256'] == delivery['quality_findings']['sha256'],
         'The R18 director reviewed different quality findings.')
    record(DELIVERY.parent/'review-findings.json',review['findings_sha256'])
    record(DELIVERY.parent/'input-audit/face-supplement-admission-archive-proof.json',
           delivery['admission_cpu_proof']['sha256'])
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
    record(evidence['cuda_face_supplement_proof']['path'],delivery['cuda_masked_face_controls']['sha256'])
    for step in (3000,6600):
        descriptor = evidence[f'candidate_event_{step:06d}']
        record(descriptor['path'],descriptor['sha256'])
        record(run/'gaussian-run/size-child-diagnostics'/f'step-{step:06d}'/'event.json',descriptor['sha256'])
    terminal = read(baseline/'wrapper-terminal.json')
    r18_plan = read(baseline/'preregistration.json')
    r18_inventory = read(baseline/'protected-inputs.json')
    need(terminal['state'] == 'completed' and terminal['candidate_commit'] == r18_plan['candidate_commit'] == PARENT
         and terminal['result_sha256'] == delivery_subject['result_sha256']
         and terminal['preregistration_sha256'] == evidence['preregistration']['sha256']
         and terminal['protected_input_inventory_sha256'] == evidence['protected_inputs']['sha256']
         and terminal['same_round17_fit_and_training_selection'] is True
         and terminal['fit_sha256'] == inventory['fit_sha256']
         and terminal['historical_r15_006_parity'] == inventory['r15_006_parity'],
         'The archived R18 formal terminal identity differs.')
    record(run/'result.json',terminal['result_sha256'])
    r17_plan = read(Path(inventory['baseline_run']).parent/'preregistration.json')
    need(r18_plan['runtime'] == r17_plan['runtime'] and r18_plan['asset_sha256'] == r17_plan['asset_sha256']
         and r18_plan['selected_frames'] == inventory['selected_frames'] == r18_inventory['selected_frames']
         and r18_plan['retained_holdouts'] == inventory['retained_holdouts'] == r18_inventory['retained_holdouts']
         and r18_inventory['candidate_frames'] == inventory['candidate_frames']
         and inventory['candidate_count'] == 52,
         'The archived R18 runtime or full-body/face candidate roster differs.')
    staged = {}
    for name,entry in inventory['staged_files'].items():
        need(r18_inventory['staged_files'][name]['sha256'] == entry['sha256'],
             f'Frozen R18 source payload differs: {name}.')
        staged[name] = record(run/name,entry['sha256'])
    frozen_profiles = {'config.json','gaussian-run/config.json','gaussian-run/training-cameras.json',
                       'gaussian-run/initialization.json'}
    profiles = {name:record(run/name,entry['sha256'] if name in frozen_profiles else None)
                for name,entry in inventory['profiles'].items()}
    for step,parity in zip((1000,TRAINING_STEPS),terminal['native_ply_parity'],strict=True):
        checked = verify_parity(run,step,inventory['reference_frame'])
        need(checked == parity,f'The archived R18 step {step} parity differs.')
        path = run/parity['file']
        record(path,parity['sha256'])
        record(path.parent/'world.ply',read(path)['sha256'])
        record(path.parent/'reference-mesh.npz',read(path)['mesh_sha256'])
    face_dir = run/'gaussian-run/face-supplement'
    for key,file in (('admission','admission.json'),('training','training.json'),
                     ('replay','replay.json'),('poses','poses.npz')):
        descriptor = evidence[f'supplement_{key}']
        record(descriptor['path'],descriptor['sha256'])
        record(face_dir/file,descriptor['sha256'])
    for row in read(face_dir/'replay.json')['records']:
        record(face_dir/row['valid_mask'],row['valid_mask_sha256'])
        for descriptor in row['files'].values():
            record(face_dir/descriptor['file'],descriptor['sha256'])
    face = prior.verify_face_supplement(run,read(run/'gaussian-run/integration.json'),
          read(run/'gaussian-run/config.json'),inventory,run/'gaussian-run/step-010000/bound-local.ply',
          r18_plan['cpu_fit_preflight'])
    need(face == terminal['face_supplement'] and len(face['frames']) == 4,
         'The archived R18 native face-only support or driven replay differs.')
    return dict(inventory,schema=SCHEMA,baseline_commit=PARENT,baseline_run=str(run),staged_files=staged,
                profiles=profiles,archived_round17_run=inventory['baseline_run'],
                archived_round18_terminal=evidence['wrapper'],archived_face_supplement=face,
                archived_face_cpu_preflight=r18_plan['cpu_fit_preflight'])


def plan_for(subject, commit, cuda_proof_sha):
    code = check_code(commit)
    inventory = source_inventory(subject)
    need(re.fullmatch('[0-9a-f]{64}',cuda_proof_sha) is not None,'The approved CUDA fixture SHA-256 is required.')
    pin(CUDA_PROOF,cuda_proof_sha)
    proof = read(CUDA_PROOF)
    fixture_sources = proof['source_sha256']
    need(proof['state'] == 'passed' and REQUIRED <= fixture_sources.keys() <= code.keys()
         and all(code[name] == digest for name,digest in fixture_sources.items()),
         'The protected face CUDA fixture did not pass for this exact source tree.')
    old_run = Path(inventory['baseline_run'])
    old_plan = read(old_run.parent/'preregistration.json')
    runtime = cold.cpu_runtime(cold.clean_env())
    need(runtime == old_plan['runtime'],'Training runtime differs from R18.')
    _verify_cuda_controls(proof,runtime)
    assets = old_plan['asset_sha256']
    for name,digest in assets.items():
        pin(Path(name),digest)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient space for the two full renders and their replay records.')
    with TemporaryDirectory(prefix='r19-cpu-fit-',dir=cold.LAB) as temporary:
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
         and fit_preflight['face_admission_payload_sha256'] == inventory['archived_face_cpu_preflight']['face_admission_payload_sha256']
         and fit_preflight['chosen_face_frames'] == inventory['archived_face_supplement']['frames']
         and len(set(fit_preflight['chosen_face_frames'])) == len(fit_preflight['chosen_face_frames']) <= 4
         and set(fit_preflight['chosen_face_frames']) <= inventory['candidate_frames'].keys()
         and set(fit_preflight['chosen_face_frames']).isdisjoint(inventory['selected_frames']),
         'The normal CPU entry changed the staged R18 fit inputs or native face admission.')
    run = root_for(subject)/'run'
    command = [str(PYTHON),'-B','-u','-m','shuolee_portrait.cli','train-fitted','--run',str(run)]
    return dict(schema=SCHEMA,subject=subject,candidate_commit=commit,baseline_commit=PARENT,
                quality_baseline_commit=QUALITY_BASELINE,main_baseline_commit=MAIN_BASELINE,
                archived_round17_run=inventory['archived_round17_run'],
                stable_git_release_commit=STABLE_TAG_COMMIT,
                baseline_run=inventory['baseline_run'],baseline_result_sha256=inventory['profiles']['result.json']['sha256'],
                output=str(run),director_review_sha256=DIRECTOR_SHA,baseline_delivery_sha256=DELIVERY_SHA,
                archived_r15_006_parity_sha256=inventory['r15_006_parity']['proof_sha256'],
                cuda_fixture_sha256=cuda_proof_sha,
                wrapper_sha256=sha(Path(__file__)),
                verifier_source_sha256={Path(module.__file__).name:sha(Path(module.__file__))
                                        for module in (prior,prior.prior,prior.prior.prior,prior.prior.prior.prior,cold)},
                runtime=runtime,code_sha256=code,asset_sha256=assets,cpu_fit_preflight=fit_preflight,
                retained_holdouts=inventory['retained_holdouts'],
                selected_frames=inventory['selected_frames'],commands=[command],gaussian_steps=TRAINING_STEPS,
                candidate_count=inventory['candidate_count'],full_body_views=24,face_only_views=4,
                full_pose_replay_count=28,main_face_aux_updates=0,
                color_policy=POLICY,auto_resolved_profile_fields=['initial_points','image_cache_frames'],
                max_wall_seconds=MAX_WALL_SECONDS),inventory


def _verify_cuda_controls(proof,runtime):
    binary_digest, = [digest for name,digest in runtime['file_sha256'].items()
                       if 'diff_gaussian_rasterization' in name and Path(name).name.startswith('_C')]
    real, rollback = proof['real_same_endpoint'],proof['forced_rollback']
    archive = proof['archive']
    real_path = CUDA_PROOF.parent/'real-016/comparison.json'
    rollback_path = real_path.parent/'forced-rollback/comparison.json'
    pin(real_path,real['comparison_sha256'])
    pin(rollback_path,rollback['comparison_sha256'])
    earlier = prior.prior.root_for('016')/'run'
    pin(earlier/'fit/fitted.npz',archive['fitted_sha256'])
    pin(earlier/'gaussian-run/step-010000/bound-local.ply',archive['bound_ply_sha256'])
    observed,forced = read(real_path),read(rollback_path)
    need(proof['fixture_sha256'] == sha(CUDA_FIXTURE_SOURCE)
         and proof['runtime']['raster_extension_sha256'] == binary_digest
         and archive['subject'] == '016' and archive['fit_replay_max_abs_m'] < 2e-6
         and real['original_views'] == 24 and real['auxiliary_views'] == 4
         and real['baseline_verified'] is True and real['final_verified'] is True
         and real['same_endpoint'] is True and real['rng_unchanged'] is True
         and real['unprotected_failed_frames'] == observed['unprotected']['measurement']['failed_frames']
         and len(real['unprotected_failed_frames']) > 0
         and real['state'] == observed['state'] == 'rolled_back'
         and observed['endpoint'] == observed['live_final']
         and rollback['verified'] is True and forced['state'] == 'rolled_back'
         and forced['endpoint'] == forced['live_final'],
         'The tested protected-face CUDA controls or native renderer differ.')


def register(subject,commit,cuda_proof_sha):
    plan, inventory = plan_for(subject,commit,cuda_proof_sha)
    root = root_for(subject)
    root.mkdir(exist_ok=True)
    need(not any(root.iterdir()), 'Registration would overwrite an existing R19 execution.')
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
         'Registered R19 candidate or protected inputs differ.')
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
    for name,entry in inventory['staged_files'].items():
        pin(run/name,entry['sha256'])
    for path,digest in plan['code_sha256'].items():
        pin(PROJECT/path,digest)

    def baseline(relative):
        return read(Path(inventory['profiles'][relative]['path']))

    cfg = read(run/'config.json')
    effective = read(run/'gaussian-run/config.json')
    need(cfg == baseline('config.json') and effective == baseline('gaussian-run/config.json'),
         'R19 changed the effective run profile.')
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
         'R19 changed initialization, optimization schedule, or the consumed pose/sample sequence.')

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
         'Bound replay changed the exact R18 selected pose sequence.')
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
    protection = verify_protected_stage(run,integration,trained,effective,inventory)
    face = verify_face_supplement(run,integration,inventory,bound,plan['cpu_fit_preflight'],protection)
    densification = prior.prior.verify_densification(run,integration,effective,inventory['selected_frames'],result['points'])
    for payload in (read(run/densification['file']),read(run/'gaussian-run/world-scale.json'),integration['coverage_events']):
        need(not _has_main_auxiliary(payload),'An auxiliary face view influenced main topology decisions.')
    selection = read(run/'fit/selection.json')
    joint = read(run/'fit/official-fit-result.json')['joint_body_refinement']
    return dict(state='completed',result_sha256=sha(run/'result.json'),points=result['points'],
        reference_frame=result['reference_frame_id'],selected_frames=selection['selected_frames'],
        coverage=selection['coverage'],joint_refinement_sha256=joint['sha256'],joint_candidate_sha256=joint['candidate_sha256'],
        world_scale_sha256=integration['world_scale']['sha256'],fit_method='reused_exact_round18_fit',
        fresh_gaussian=True,original_video_to_export=False,native_inputs_reused=True,
        reference_mesh_frozen_from_prior=False,fit_sha256=inventory['fit_sha256'],
        training_inputs_sha256=inventory['staged_files']['data/training-inputs.json']['sha256'],
        same_round15_fit_and_training_selection=True,same_round16_fit_and_training_selection=True,
        same_round17_fit_and_training_selection=True,face_supplement=face,
        same_round18_fit_and_training_selection=True,protected_face=protection,verified_pose_count=24+len(face['frames']),
        face_guard=read(run/joint['file'])['face_guard'],
        color_constraint=color_proof,binding=binding,densification_audit=densification,native_ply_parity=parity)


def _snapshot_parity(snapshot,reference,run):
    path = snapshot/'native-ply-parity.json'
    proof = read(path)
    need(proof['passed'] is True and proof['reference_frame_id'] == reference
         and proof['reference_pose_index'] == 0 and len(proof['checks']) == 1
         and all(0 <= check['mean_abs'] < 2e-6 and 0 <= check['max_abs'] < .005 for check in proof['checks']),
         'The preserved main-state PLY did not pass the same native parity gates.')
    pin(snapshot/'world.ply',proof['sha256'])
    pin(snapshot/'reference-mesh.npz',proof['mesh_sha256'])
    return dict(file=str(path.relative_to(run)),sha256=sha(path),checks=proof['checks'])


def _has_main_auxiliary(value):
    if isinstance(value,dict):
        return bool({'face_supplement_frames','face_supplement_trial'} & value.keys()) or any(
            _has_main_auxiliary(item) for item in value.values())
    if isinstance(value,list):
        return any(_has_main_auxiliary(item) for item in value)
    return False


def verify_mesh_and_bound_identity(run,main,final,mutable_attributes,committed):
    """Prove in CPU that only permitted head properties changed after the main endpoint."""
    import numpy as np
    from plyfile import PlyData

    smplx = Path(read(run/'run.json')['assets'])/'smplx/SMPLX_NEUTRAL.npz'
    with np.load(smplx,allow_pickle=False) as model:
        faces = model['f']
        protected = np.isin(model['weights'].argmax(-1),[12,15,22,23,24])
        landmarks = np.concatenate([model['lmk_faces_idx'].reshape(-1),model['dynamic_lmk_faces_idx'].reshape(-1)])
    protected[faces[landmarks].reshape(-1)] = True
    head_faces = protected[faces].all(1)
    with np.load(main/'reference-mesh.npz',allow_pickle=False) as old, \
         np.load(final/'reference-mesh.npz',allow_pickle=False) as new:
        need(old.files == new.files and np.array_equal(old['faces'],faces)
             and all(np.array_equal(old[key],new[key]) for key in old.files),
             'Protected face refinement changed the frozen reference mesh or clothing offsets.')
    before, after = (PlyData.read(path/'bound-local.ply',mmap=False)['vertex'].data for path in (main,final))
    names = before.dtype.names
    need(names == after.dtype.names and len(before) == len(after)
         and np.array_equal(before['binding_0'],after['binding_0']),
         'Protected face refinement changed the ordered bound Gaussian topology.')
    owners = before['binding_0'].astype(np.int64)
    need(np.array_equal(owners,before['binding_0']) and np.all((owners >= 0) & (owners < len(faces))),
         'The saved main-stage binding roster has invalid owners.')
    head_rows = head_faces[owners]
    fields = {'f_dc':{name for name in names if name.startswith('f_dc_')},
              'opacity':{'opacity'},'xyz':{'x','y','z'},
              'scaling':{name for name in names if name.startswith('scale_')},
              'rotation':{name for name in names if name.startswith('rot_')}}
    need(set(mutable_attributes) <= fields.keys(),'Unknown protected refinement parameter group.')
    allowed = set().union(*(fields[name] for name in mutable_attributes))
    changed = {name for name in names if not np.array_equal(before[name],after[name])}
    changed_rows = np.zeros(len(before),dtype=bool)
    for name in changed:
        changed_rows |= before[name] != after[name]
    need(changed <= allowed and not bool(changed_rows[~head_rows].any())
         and bool(changed) is committed,
         'A protected-face decision changed forbidden parameters or disagrees with the saved output.')
    need(np.all(after['opacity'] >= before['opacity'])
         and all(np.max(np.abs(after[name]-before[name])) <= .060001 for name in fields['f_dc'])
         and np.max(after['opacity']-before['opacity']) <= .160001,
         'Protected face opacity or DC left its preregistered one-sided/trust bounds.')
    return dict(rows=len(before),head_rows=int(head_rows.sum()),changed_head_rows=int(changed_rows.sum()),
                changed_fields=sorted(changed),non_head_rows_changed=0,
                ordered_face_binding_identical=True,mesh_and_clothing_offsets_identical=True)


def _verify_fixed_measurement(measurement,baseline,roster,guard,require_pass=False):
    need(measurement['finite'] is True and len(measurement['frames']) == len(roster)
         and [(row['frame'],row['role'],row['source_bin']) for row in measurement['frames']] == roster,
         'Protected comparison did not measure the same 24+4 original native camera/region roster.')
    failed = []
    for current,initial in zip(measurement['frames'],baseline['frames'],strict=True):
        current_coverage,initial_coverage = current['coverage_guard'],initial['coverage_guard']
        need(current['face_pixels'] == initial['face_pixels']
             and current['region_eligibility'] == initial['region_eligibility']
             and current['regions'].keys() == initial['regions'].keys()
             and current_coverage['baseline_target_deficit_px'] == initial_coverage['baseline_target_deficit_px']
             and current_coverage['original_rgb_mae_before'] == initial_coverage['original_rgb_mae_before'],
             f'Protected assessment redefined the fixed endpoint in {current["frame"]}.')
        for name,region in current['regions'].items():
            original = initial['regions'][name]
            need(region['pixels'] == original['pixels'] == current['region_eligibility'][name]
                 and region['rgb_mae_before'] == original['rgb_mae_before']
                 and region['alpha_mae_before'] == original['alpha_mae_before']
                 and region['alpha_below_095_before'] == original['alpha_below_095_before'],
                 f'Protected assessment redefined the fixed endpoint for {current["frame"]}/{name}.')
            if require_pass:
                tolerance = guard['coverage_rgb_mae_increase']+1.1e-6
                need(region['rgb_mae_after']-region['rgb_mae_before'] <= tolerance
                     and region['alpha_mae_after']-region['alpha_mae_before'] <= tolerance
                     and region['new_alpha_holes_095'] == 0,
                     f'Protected approval violated a fixed RGB/alpha guard at {current["frame"]}/{name}.')
        if current['failures']:
            failed.append(current['frame'])
        if require_pass:
            need(not current['failures'] and current_coverage['new_low_alpha_holes'] == 0
                 and current_coverage['foreground_target_deficit_px'] <= guard['coverage_slack_px']
                 and current_coverage['background_gain_px'] <= guard['coverage_slack_px']
                 and current_coverage['worst_deficit_tile']['value'] <= guard['coverage_local_deficit_px']
                 and current_coverage['worst_rgb_tile']['value'] <= guard['coverage_local_rgb_mae_increase']+1.1e-6,
                 f'Protected approval violated a fixed native full-view guard at {current["frame"]}.')
        if current['role'] == 'supplemental':
            need(current['face_pixels'] > 0 and current_coverage['background_gain_px'] == 0,
                 'An auxiliary view contributed exterior labels or lacked verified face pixels.')
    need(measurement['failed_frames'] == failed and measurement['all_guards_pass'] is (not failed)
         and all(math.isfinite(measurement[key]) and measurement[key] >= 0
                 for key in ('objective','original_objective','supplemental_objective')),
         'The protected native assessment aggregate differs from the recorded per-view guards.')


def _verify_protected_sweeps(proof,roster):
    from collections import Counter

    opt, config = proof['protected'],proof['config']
    phases, cycles = opt['phases'],opt['cycles']
    groups = {}
    for frame,role,bin_id in roster:
        groups.setdefault((role,bin_id),[]).append(frame)
    grouped = [groups[key] for key in sorted(groups)]
    expected_phases = [('primary',1.),('conservative_restart',.25)][:len(phases)]
    need(len(roster) == 28 and 1 <= len(phases) <= 2
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
                 and order == sequence[:len(order)] and (not cycle['completed'] or len(order) == 28),
                 'A protected optimizer cycle did not use its full balanced source roster.')
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


def _parameter_fields(vertices,name):
    import numpy as np
    fields = {'f_dc':('f_dc_0','f_dc_1','f_dc_2'),'opacity':('opacity',)}[name]
    return np.stack([vertices[field] for field in fields],axis=1)


def _verify_committed_state(stage,proof,main,final):
    import numpy as np
    from plyfile import PlyData
    import torch

    desc = proof['optimizer_state']
    path = stage/'protected-optimizer-state.pt'
    need(desc['file'] == path.name and desc['main_adam_clock_preserved'] is True
         and desc['head_rows'] == proof['head_rows'] and set(desc['groups']) == {'f_dc','opacity'},
         'Protected commit lacks the selected state and original Adam clock.')
    pin(path,desc['sha256'])
    saved = torch.load(path,map_location='cpu',weights_only=True)
    before, after = (PlyData.read(path/'bound-local.ply',mmap=False)['vertex'].data for path in (main,final))
    def digest(value):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    ids = saved['head_rows'].numpy()
    need(saved['endpoint'] == proof['endpoint'] and saved['committed'] == proof['live_final']
         and ids.ndim == 1 and ids.dtype == np.int64 and len(ids) == proof['head_rows']
         and np.all((ids >= 0) & (ids < len(after))) and np.all(np.diff(ids) > 0)
         and digest(ids) == proof['head_row_sha256'],
         'Saved Adam sidecar does not bind the exact selected endpoint and head rows.')
    for name,group in saved['groups'].items():
        local = saved['local_groups'][name]
        initial,live = (proof[phase]['mutable_head'][name] for phase in ('endpoint','live_final'))
        need(int(group['step']) == initial['clock'] == live['clock']
             and int(local['step']) == initial['clock']+proof['committed_updates']
             and np.array_equal(group['parameter'].numpy().reshape(len(ids),-1),_parameter_fields(after,name)[ids])
             and np.array_equal(group['parameter'].numpy(),local['parameter'].numpy())
             and digest(_parameter_fields(before,name)[ids]) == initial['parameter']
             and digest(_parameter_fields(after,name)[ids]) == live['parameter']
             and all(digest(group[key].numpy()) == live[key] and torch.isfinite(group[key]).all()
                     and torch.isfinite(local[key]).all() for key in ('exp_avg','exp_avg_sq')),
             f'The saved selected Adam/head state differs from the rendered PLY for {name}.')
    return dict(file=str(path.relative_to(stage.parent.parent)),sha256=desc['sha256'],head_rows=len(ids))


def verify_protected_stage(run,integration,trained,effective,inventory):
    out = run/'gaussian-run'
    stage = out/'protected-face'
    main, final = (out/name for name in ('step-010000-main','step-010000'))
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
        need(replay['points'] == trained['points'] and replay['reference_frame_id'] == inventory['reference_frame']
             and replay['reference_pose_index'] == 0 and replay['sh_degree'] == 0
             and [row['frame'] for row in replay['frames']] == inventory['selected_frames'],
             'Main or final bound replay does not preserve the 24 original named poses.')
    need(main_bound['frames'] == final_bound['frames'] and main_bound['inputs_sha256'] == final_bound['inputs_sha256'],
         'Main and final bound snapshots used different original pose/camera inputs.')
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
              +[(name,'supplemental',i) for i,name in enumerate(inventory['archived_face_supplement']['frames'])])
    guard = effective['unified_main']['world_scale']['render_support']
    baseline, final_metrics = proof['baseline'],proof['final']
    _verify_fixed_measurement(baseline,baseline,roster,guard,require_pass=True)
    _verify_fixed_measurement(proof['unprotected']['measurement'],baseline,roster,guard)
    _verify_fixed_measurement(final_metrics,baseline,roster,guard,require_pass=True)
    _verify_protected_sweeps(proof,roster)
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
         and frames == inventory['archived_face_supplement']['frames']
         and len(set(frames)) == len(frames) and set(frames) <= candidates.keys()
         and set(frames).isdisjoint(inventory['selected_frames'])
         and set(frames).isdisjoint(inventory['retained_holdouts']),
         'The separate facial supervision changed the protected training roster or optimizer budget.')
    admission_rows = prior.verify_admission(admission,inventory,frames,preflight)
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
                                 valid_mask_sha256=row['valid_mask_sha256'],files=files))
    return dict(frames=frames,training_sha256=sha(report_path),admission_sha256=sha(admission_path),
                replay_sha256=sha(replay_path),poses_sha256=replay['poses_sha256'],
                face_loss_updates=0,separate_optimizer_updates=report['separate_optimizer_updates'],records=file_records)


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
             'Fresh round-nineteen output is required.')
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
