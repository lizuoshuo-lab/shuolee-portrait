"""Train both R16 subjects on their exact completed R15 fit inputs."""

import argparse
import fcntl
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
import launch_smpl_delivery_round14 as prior


need, read, pin, sha, write = cold.need, cold.read, cold.pin, cold.sha, cold.write
check_snapshot, color_stats = prior.check_snapshot, prior.color_stats
PYTHON = cold.ENV/'bin/python'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-size-optimization-20261111')
PARENT_PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-densification-20261111')
PARENT = '6785a212b304d38a31f699933b23096d38ceb4a4'
OUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs')
DIRECTOR_ROOT = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs')
DIRECTOR = DIRECTOR_ROOT/'round15_director_review_20261111.json'
DIRECTOR_SHA = '2f8f7d821e81cb02b1d764ee3b8ebb44ec8f87bb5f040ae41bf0110d8312bf4d'
RETRY_PROOF = OUT/'smpl-delivery-round15/input-audit/006-retry-parity.json'
RETRY_PROOF_SHA = 'f1c505a406acd2abc71b5765c423d2c8347282744fdf7131204211afc998d2bf'
DELIVERY = OUT/'smpl-delivery-round15/receipt.json'
DELIVERY_SHA = '617dc00c2cc6fa1e913732ec3713c4ac8624e342ef18bf715608647d985c0eb1'
CUDA_PROOF = OUT/'smpl-delivery-round16/input-audit/cuda-size-optimization-proof.json'
POLICY = 'projected_ldr_dc'
POLICY_KEY = 'sh0_color_constraint'
COLOR_C0 = .28209479177387814
MAX_WALL_SECONDS = 7200
TRAINING_STEPS = 10000
SCHEMA = 'shuolee-round16-fixed-fit-size-optimization-v1'
REQUIRED = {'src/avatar_core/scene/gaussian_model.py', 'src/shuolee_portrait/training.py',
            'src/shuolee_portrait/size_child_optimization.py', 'src/shuolee_portrait/size_child_commit.py',
            'src/shuolee_portrait/render_support.py', 'src/shuolee_portrait/world_scale.py'}
ALLOWED = REQUIRED | {'tests/test_bound_split_units.py', 'tests/test_render_support.py', 'tests/test_world_scale.py'}
STAGED = ('run.json', 'data/inputs.json', 'data/observations.npz', 'data/training-inputs.json',
          'fit/fitted.npz', 'fit/official-fit-result.json', 'fit/shared-body.json', 'fit/selection.json',
          'fit/selection-before-joint.json', 'fit/joint-candidates.npz', 'fit/joint-body-refinement.json')
PROFILE_FILES = ('config.json', 'gaussian-run/config.json', 'gaussian-run/training-cameras.json',
                 'gaussian-run/initialization.json', 'gaussian-run/integration.json',
                 'gaussian-run/result.json', 'result.json')
EVIDENCE_FILES = {
    'official_fit': 'fit/official-fit-result.json',
    'joint': 'fit/joint-body-refinement.json',
    'joint_candidates': 'fit/joint-candidates.npz',
    'selection': 'fit/selection.json',
    'pre_joint_selection': 'fit/selection-before-joint.json',
    'effective_config': 'gaussian-run/config.json',
    'initialization': 'gaussian-run/initialization.json',
    'integration': 'gaussian-run/integration.json',
    'color_constraint': 'gaussian-run/color-constraint.json',
    'world_scale': 'gaussian-run/world-scale.json',
    'densification_audit': 'gaussian-run/densification-audit.json',
}


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round16-{subject}'


def check_code(commit):
    pointer = (PROJECT/'.git').read_text().strip()
    need(pointer.startswith('gitdir: '), 'The candidate must be a linked worktree.')
    git_dir = subprocess.check_output(['/usr/bin/wslpath', '-u', pointer.removeprefix('gitdir: ')],text=True).strip()
    git = ['/usr/bin/git','-c','core.fileMode=false','--git-dir',git_dir,'--work-tree',str(PROJECT)]
    head = subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()
    need(head == commit, f'Candidate HEAD {head} differs from requested commit.')
    status = subprocess.check_output([*git,'status','--porcelain','--untracked-files=all'],
                                     cwd=PROJECT,text=True)
    need(not status.strip(), f'Candidate worktree is dirty: {status}')
    subprocess.run([*git,'merge-base','--is-ancestor',PARENT,head],check=True)
    changed = set(subprocess.check_output([*git,'diff','--name-only',PARENT,head],text=True).splitlines())
    focused_tests = {name for name in changed if Path(name).parent == Path('tests')
                     and Path(name).name.startswith(('test_densif', 'test_size_child'))
                     and Path(name).suffix == '.py'}
    need(REQUIRED <= changed <= ALLOWED | focused_tests,
         f'Unexpected size-child optimization code paths: {sorted(changed)}')
    prior_gitdir = subprocess.check_output(
        ['/usr/bin/wslpath','-u',(PARENT_PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')],
        text=True).strip()
    prior_git = ['/usr/bin/git','-c','core.fileMode=false','--git-dir',prior_gitdir,'--work-tree',str(PARENT_PROJECT)]
    prior_head = subprocess.check_output([*prior_git,'rev-parse','HEAD'],text=True).strip()
    prior_status = subprocess.check_output([*prior_git,'status','--porcelain','--untracked-files=all'],
                                            cwd=PARENT_PROJECT,text=True)
    need(prior_head == PARENT and not prior_status.strip(), 'The R15 worktree changed.')
    tracked = set(cold.SOURCE_FILES) | changed | {
        'src/shuolee_portrait/sh0_main.py','src/shuolee_portrait/world_scale.py',
        'src/shuolee_portrait/coverage_refinement.py','src/shuolee_portrait/render_support.py'}
    for name in tracked-changed:
        pin(PROJECT/name,sha(PARENT_PROJECT/name))
    return {name:sha(PROJECT/name) for name in sorted(tracked)}


def source_inventory(subject):
    """Bind the delivered R15 inputs to the exact trained subject and pose order."""
    files = {}

    def record(path, digest=None):
        path = Path(path)
        if digest is None:
            digest = sha(path)
        else:
            pin(path,digest)
        previous = files.setdefault(str(path),digest)
        need(previous == digest, f'Conflicting source identities for {path}')
        return dict(path=str(path),sha256=digest)

    record(DIRECTOR,DIRECTOR_SHA)
    record(RETRY_PROOF,RETRY_PROOF_SHA)
    record(DELIVERY,DELIVERY_SHA)
    review, retry, delivery = read(DIRECTOR), read(RETRY_PROOF), read(DELIVERY)
    need(review['round'] == 15 and review['commit'] == retry['candidate_commit'] == delivery['commit'] == PARENT
         and review['receipt_sha256'] == DELIVERY_SHA and review['quality_accepted'] is False
         and delivery['quality_accepted'] is False, 'R15 report or director review differs.')
    issue = review['unresolved_replay_issue']
    need(issue['subject'] == retry['subject'] == '006' and issue['first_attempt_failed_step'] == 1000
         and issue['gate'] == retry['max_abs_gate'] == .005 and retry['mean_abs_gate'] == 2e-6
         and issue['root_cause_established'] is retry['root_cause_established'] is False
         and issue['unchanged_retry_passed'] is True and retry['fit_reused'] is True,
         'The archived first 006 replay failure identity differs.')
    for phase,passed in (('first',False),('retry',True)):
        archived = retry[phase]
        for descriptor in archived.values():
            if isinstance(descriptor,dict) and Path(descriptor['path']).is_absolute():
                record(descriptor['path'],descriptor['sha256'])
        parity_path = Path(archived['parity']['path'])
        parity = read(parity_path)
        need(archived['parity']['passed'] is parity['passed'] is passed
             and archived['parity']['checks'] == parity['checks'],
             f'The archived R15 006 {phase} parity no longer matches its proof.')
        record(parity_path.parent/'world.ply',parity['sha256'])
        record(parity_path.parent/'reference-mesh.npz',parity['mesh_sha256'])
        record(parity_path.parent/'reference-render.png')
    parent = cold.LAB/f'shuolee-smpl-delivery-round15-{subject}'
    run = parent/'run'
    delivery_subject = delivery['subjects'][subject]
    evidence = delivery_subject['evidence']
    for label,relative in EVIDENCE_FILES.items():
        descriptor = evidence[label]
        record(descriptor['path'],descriptor['sha256'])
        record(run/relative,descriptor['sha256'])
    record(evidence['wrapper']['path'],evidence['wrapper']['sha256'])
    record(parent/'wrapper-terminal.json',evidence['wrapper']['sha256'])
    record(parent/'preregistration.json',evidence['preregistration']['sha256'])
    old_inventory_path = parent/'protected-inputs.json'
    record(old_inventory_path,evidence['protected_inputs']['sha256'])
    terminal = read(parent/'wrapper-terminal.json')
    old_plan = read(parent/'preregistration.json')
    old_inventory = read(old_inventory_path)
    need(terminal['state'] == 'completed' and terminal['candidate_commit'] == PARENT
         and terminal['result_sha256'] == delivery_subject['result_sha256']
         and terminal['preregistration_sha256'] == evidence['preregistration']['sha256']
         and terminal['protected_input_inventory_sha256'] == evidence['protected_inputs']['sha256']
         and old_plan['candidate_commit'] == PARENT, 'R15 formal terminal identity differs.')
    record(run/'result.json',terminal['result_sha256'])
    result = read(run/'result.json')
    need(result['steps'] == TRAINING_STEPS and result['gaussian_profile'] == 'smpl_sh0'
         and result['sh_degree'] == 0 and result['fit_calls'] == 0
         and result['points'] == terminal['points'] == delivery_subject['points'],
         'R15 baseline is not the delivered SH0 run.')
    color = read(run/'gaussian-run/color-constraint.json')
    need(terminal['color_constraint']['sha256'] == evidence['color_constraint']['sha256']
         and color['policy'] == POLICY and color['terminal']['channels_outside_bounds'] == 0
         and color['optimizer_steps'] == TRAINING_STEPS, 'The R15 terminal LDR proof differs.')

    fit = read(run/'fit/official-fit-result.json')
    selection = read(run/'fit/selection.json')
    ordered_rows = read(run/'data/training-inputs.json')['records']
    selected = [row['frame'] for row in ordered_rows]
    held = selection['retained_holdouts']
    need(fit['state'] == 'complete' and fit['frames'] == len(set(selected)) == len(selected) == 24
         and fit['selection'] == selection
         and selected[0] == selection['reference_frame'] == result['reference_frame_id']
         and set(selected) == set(selection['selected_frames']) == set(terminal['selected_frames'])
         and held == old_plan['retained_holdouts'] and len(set(held)) == len(held) == 4
         and set(selected).isdisjoint(held),
         'R15 training selection or holdout identity differs.')
    if subject == '016':
        need('frame_000051' in selected and 'frame_000055' not in selected and
             'frame_000055' in read(run/'fit/joint-body-refinement.json')['fit_frame_ids'], 'R15 f051/f055 identities differ.')

    record(run/'fit/fitted.npz',fit['fit_sha256'])
    record(run/'fit/shared-body.json',fit['shared_body_sha256'])
    record(run/'data/training-inputs.json',fit['training_inputs_sha256'])
    for name in STAGED:
        entry = old_inventory['staged_files'][name]
        record(entry['path'],entry['sha256'])
        record(run/name,entry['sha256'])
    for name in ('data/inputs.json','data/observations.npz'):
        entry = old_inventory['staged_files'][name]
        matches = [(name_from_fit,digest) for name_from_fit,digest in fit['source']['input_files'].items()
                   if name_from_fit.endswith('/'+name)]
        need(len(matches) == 1 and matches[0][1] == entry['sha256'],
             f'R15 fitted observation source changed: {name}.')
        record(*matches[0])
    for descriptor in fit['sources']+fit['official_frame_results']:
        path = Path(descriptor['path'])
        record(path if path.is_absolute() else run/path,descriptor['sha256'])
    for row in ordered_rows:
        record(row['image'],row['source_sha256'])
        record(row['mask'],row['mask_sha256'])

    prior_cfg = read(run/'config.json')
    effective = read(run/'gaussian-run/config.json')
    old_train = read(run/'gaussian-run/result.json')
    init = read(run/'gaussian-run/initialization.json')
    need(prior_cfg.get(POLICY_KEY) == effective.get(POLICY_KEY) == POLICY
         and 'frozen_surface' not in prior_cfg and effective['seed'] == prior_cfg['seed'] == 0
         and effective['iterations'] == prior_cfg['iterations'] == TRAINING_STEPS
         and effective['sh_degree'] == prior_cfg['sh_degree'] == 0
         and effective['unified_main'] == prior_cfg['unified_main']
         and effective['optimization'] == prior_cfg['optimization']
         and init['initial_points'] == effective['initial_points'] == 20908
         and init['optimizer_initial_state_entries'] == 0 and init['no_prior_gaussian_loaded']
         and len(old_train['per_frame_updates']) == 24
         and sum(old_train['per_frame_updates']) == TRAINING_STEPS,
         'R15 training profile is not the controlled fresh baseline.')
    cameras = read(run/'gaussian-run/training-cameras.json')
    need(cameras['source_sha256'] == fit['training_inputs_sha256']
         and cameras['interpolated_cameras'] == 0 and len(cameras['records']) == len(selected),
         'R15 consumed cameras do not bind the protected training manifest.')
    staged = {name:record(run/name,files.get(str(run/name))) for name in STAGED}
    profiles = {name:record(run/name,files.get(str(run/name))) for name in PROFILE_FILES}
    return dict(schema=SCHEMA,subject=subject,baseline_commit=PARENT,baseline_run=str(run),files=files,
                staged_files=staged,profiles=profiles,selected_frames=selected,retained_holdouts=held,
                reference_frame=selection['reference_frame'],fit_sha256=fit['fit_sha256'],
                seed=effective['seed'],gaussian_steps=TRAINING_STEPS,
                r15_006_parity=dict(proof_sha256=RETRY_PROOF_SHA,first=retry['first']['parity'],
                                    retry=retry['retry']['parity'],root_cause_established=False))


def plan_for(subject, commit, cuda_proof_sha):
    code = check_code(commit)
    inventory = source_inventory(subject)
    need(re.fullmatch('[0-9a-f]{64}',cuda_proof_sha) is not None,'The approved CUDA fixture SHA-256 is required.')
    pin(CUDA_PROOF,cuda_proof_sha)
    proof = read(CUDA_PROOF)
    fixture_sources = proof['source_sha256']
    need(proof['state'] == 'passed' and REQUIRED <= fixture_sources.keys() <= code.keys()
         and all(code[name] == digest for name,digest in fixture_sources.items()),
         'The size-child optimization CUDA fixture did not pass for this exact source tree.')
    old_run = Path(inventory['baseline_run'])
    old_plan = read(old_run.parent/'preregistration.json')
    runtime = cold.cpu_runtime(cold.clean_env())
    need(runtime == old_plan['runtime'],'Training runtime differs from R15.')
    assets = old_plan['asset_sha256']
    for name,digest in assets.items():
        pin(Path(name),digest)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient space for the two full renders and their replay records.')
    with TemporaryDirectory(prefix='r16-cpu-fit-',dir=cold.LAB) as temporary:
        staging = Path(temporary)
        stage(staging,inventory)
        fit_preflight = cpu_fit_gate(staging/'run')
    old_train = read(old_run/'gaussian-run/result.json')
    need(fit_preflight['fit_sha256'] == inventory['fit_sha256']
         and fit_preflight['training_inputs_sha256'] == inventory['staged_files']['data/training-inputs.json']['sha256']
         and fit_preflight['frames'] == len(inventory['selected_frames'])
         and fit_preflight['expected_per_frame_updates'] == old_train['per_frame_updates']
         and fit_preflight['expected_source_bin_draws'] == old_train['source_bin_draws'],
         'The normal CPU entry rejected the staged R15 fit inputs.')
    run = root_for(subject)/'run'
    command = [str(PYTHON),'-B','-u','-m','shuolee_portrait.cli','train-fitted','--run',str(run)]
    return dict(schema=SCHEMA,subject=subject,candidate_commit=commit,baseline_commit=PARENT,
                baseline_run=inventory['baseline_run'],baseline_result_sha256=inventory['profiles']['result.json']['sha256'],
                output=str(run),director_review_sha256=DIRECTOR_SHA,baseline_delivery_sha256=DELIVERY_SHA,
                archived_r15_006_parity_sha256=RETRY_PROOF_SHA,cuda_fixture_sha256=cuda_proof_sha,
                wrapper_sha256=sha(Path(__file__)),
                verifier_source_sha256=sha(Path(prior.__file__)),
                runtime=runtime,code_sha256=code,asset_sha256=assets,cpu_fit_preflight=fit_preflight,
                retained_holdouts=inventory['retained_holdouts'],
                selected_frames=inventory['selected_frames'],commands=[command],gaussian_steps=TRAINING_STEPS,
                color_policy=POLICY,auto_resolved_profile_fields=['initial_points','image_cache_frames'],
                max_wall_seconds=MAX_WALL_SECONDS),inventory


def register(subject,commit,cuda_proof_sha):
    plan, inventory = plan_for(subject,commit,cuda_proof_sha)
    root = root_for(subject)
    root.mkdir(exist_ok=True)
    need(not any(root.iterdir()), 'Registration would overwrite an existing R16 execution.')
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
         'Registered R16 candidate or protected inputs differ.')
    return plan,inventory


def stage(root,inventory):
    run = root/'run'
    with TemporaryDirectory(prefix='stage-',dir=root) as temporary:
        destination = Path(temporary)/'run'
        destination.mkdir()
        for name,entry in inventory['staged_files'].items():
            path = destination/name
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(entry['path'],path)
            pin(path,entry['sha256'])
        cfg = read(Path(inventory['profiles']['config.json']['path']))
        # The public entry resolves these fields. The persisted profile is pinned
        # to the delivered R15 bytes after the same resolution has taken place.
        write(destination/'config.json',dict(cfg,initial_points=None,image_cache_frames=None))
        destination.rename(run)


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

run = Path(sys.argv[1])
rt.configure(Path(rt.read(run/'run.json')['assets']),run)
fit = verify_official_fit(run)
cfg = rt.read(run/'config.json')
effective = smpl_sh0_effective_config()
if any(cfg.get(key,value) != value for key,value in effective.items()):
    raise ValueError('Staged config conflicts with the stable SH0 entry.')
rows = rt.read(run/'data/training-inputs.json')['records']
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
    expected_source_bin_draws=per_bin)))
"""
    env = dict(cold.clean_env(),CUDA_VISIBLE_DEVICES='')
    return json.loads(subprocess.check_output([str(PYTHON),'-B','-c',code,str(run)],
                                              cwd=PROJECT/'src',env=env,text=True))


def finish(root,plan,inventory,returncode,elapsed):
    run = root/'run'
    receipt = dict(state='failed' if returncode else 'incomplete',subject=plan['subject'],
        returncode=returncode,stage='train_fitted',candidate_commit=plan['candidate_commit'],
        baseline_commit=PARENT,command_to_end_seconds=elapsed,preregistration_sha256=sha(root/'preregistration.json'),
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
        receipt['retry_performed'] = False
    write(root/'wrapper-terminal.json',receipt)
    print(dict(state=receipt['state'],subject=plan['subject'],validation_error=receipt.get('validation_error')),flush=True)
    return 0 if receipt['state']=='completed' else returncode or 1


def snapshot_failure(path,run):
    proof = read(path)
    files = {item.name:sha(item) for item in sorted(path.parent.iterdir()) if item.is_file() and item != path}
    return dict(file=str(path.relative_to(run)),sha256=sha(path),passed=proof['passed'],checks=proof['checks'],
                snapshot_files_sha256=files)


def verify_parity(run,step,reference):
    snapshot = run/'gaussian-run'/f'step-{step:06d}'
    path = snapshot/'native-ply-parity.json'
    proof = read(path)
    need(proof['passed'] is True and proof['reference_frame_id'] == reference
         and proof['reference_pose_index'] == 0 and len(proof['checks']) == 1
         and all(0 <= check['mean_abs'] < 2e-6 and 0 <= check['max_abs'] < .005 for check in proof['checks']),
         f'Step {step} did not pass the original native PLY parity gates.')
    pin(snapshot/'world.ply',proof['sha256'])
    pin(snapshot/'reference-mesh.npz',proof['mesh_sha256'])
    return dict(file=str(path.relative_to(run)),sha256=sha(path),checks=proof['checks'])


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
         'R16 changed the effective run profile.')
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
         'R16 changed initialization, optimization schedule, or the consumed pose/sample sequence.')

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
         'Bound replay changed the exact R15 selected pose sequence.')
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
    densification = verify_densification(run,integration,effective,inventory['selected_frames'],result['points'])
    selection = read(run/'fit/selection.json')
    joint = read(run/'fit/official-fit-result.json')['joint_body_refinement']
    return dict(state='completed',result_sha256=sha(run/'result.json'),points=result['points'],
        reference_frame=result['reference_frame_id'],selected_frames=selection['selected_frames'],
        coverage=selection['coverage'],joint_refinement_sha256=joint['sha256'],joint_candidate_sha256=joint['candidate_sha256'],
        world_scale_sha256=integration['world_scale']['sha256'],fit_method='reused_exact_round15_fit',
        fresh_gaussian=True,original_video_to_export=False,native_inputs_reused=True,
        reference_mesh_frozen_from_prior=False,fit_sha256=inventory['fit_sha256'],
        training_inputs_sha256=inventory['staged_files']['data/training-inputs.json']['sha256'],
        same_round15_fit_and_training_selection=True,face_guard=read(run/joint['file'])['face_guard'],
        color_constraint=color_proof,binding=binding,densification_audit=densification,native_ply_parity=parity)


def check_binding(ply,fit,points):
    """Independently count terminal point ownership in the exported local PLY."""
    import numpy as np
    from plyfile import PlyData

    with np.load(fit,allow_pickle=False) as arrays:
        triangles = len(arrays['faces'])
    ownership = PlyData.read(ply,mmap=False)['vertex'].data['binding_0']
    need(len(ownership) == points and triangles > 0
         and np.all(np.isfinite(ownership) & (ownership == np.floor(ownership))
                    & (ownership >= 0) & (ownership < triangles)),
         'The exported local PLY has invalid triangle ownership.')
    counts = np.bincount(ownership.astype(np.int64),minlength=triangles)
    need(int(counts.min()) >= 1,'The exported local PLY lost a triangle.')
    return dict(points=points,triangles=triangles,minimum_points_per_triangle=int(counts.min()))


def check_local_evidence(run,step,local,ordered_frames):
    views = local['views']
    indices = [view['view_index'] for view in views]
    fitted = local['status'] == 'fitted'
    need(local['optimizer_update_budget'] == 24 and local['optimizer_view_budget'] == 6
         and len(set(indices)) == len(indices) <= local['optimizer_view_budget']
         and all(type(i) is int and 0 <= i < len(ordered_frames) and ordered_frames[i] == view['frame']
                 for i,view in zip(indices,views,strict=True))
         and (not fitted or (bool(views) and local['optimizer_updates'] == local['optimizer_update_budget']))
         and math.isfinite(local['verification_seconds']) and local['verification_seconds'] >= 0,
         f'Size-child fitting used invalid optimizer inputs or budgets at step {step}.')
    subdir = f'step-{step:06d}'
    need(local['diagnostic_subdir'] == subdir,f'Invalid size-child diagnostic directory at step {step}.')
    directory = run/'gaussian-run/size-child-diagnostics'/subdir
    reps, decisions = (local[key] for key in ('representatives','decision_representatives'))
    need([row['view_index'] for row in reps] == indices[:2]
         and [row['view_index'] for row in decisions] == (indices[:2] if fitted else []),
         f'The local diagnostic crops do not cover the declared optimizer views at step {step}.')
    for rows,prefix in ((reps,'representative'),(decisions,'decision')):
        for row in rows:
            i = row['view_index']
            x0,y0,x1,y1 = row['box']
            need(row['path'] == f'{prefix}-view-{i:02d}.png' and row['frame'] == ordered_frames[i]
                 and 0 <= x0 < x1 and 0 <= y0 < y1,
                 f'Invalid native crop provenance at step {step}.')
            pin(directory/row['path'],row['sha256'])
    need(all(first['box'] == second['box'] for first,second in zip(reps,decisions)),
         f'The saved candidate and decision crops differ at step {step}.')


def verify_densification(run,integration,effective,ordered_frames,points):
    """Verify real stock cycles, candidate-only optimizer work and exact native trials."""
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
    need(audit['state'] == 'completed'
         and 'native pixels' in audit['radius_units'] and 'zero' in audit['radius_window']
         and 'face-local' in audit['local_cap_units']
         and audit['stock_events'] == integration['stock_densifications'] == len(expected)
         and [event['step'] for event in events] == expected,
         'Densification audit does not cover the expected stock schedule and units.')
    keys = ('screen_candidates_before_stock','screen_candidates_after_stock',
            'screen_parents_replaced_by_gradient_split','size_candidates','world_candidates','assessed',
            'foreground_supported','protected','size_pruned','screen_pruned','size_replaced',
            'replacement_children','opacity_only_pruned')
    need(audit['totals'] == {key:sum(event.get(key,0) for event in events) for key in keys},
         'Densification audit totals differ from its real cycle records.')
    optimization_events = []

    def paired_trial(trial,count):
        return (trial['accepted'] is True and trial['proposed'] == count
                and [frame['frame'] for frame in trial['frames']] == ordered_frames
                and all(not frame['failures'] for frame in trial['frames']))

    for event in events:
        step = event['step']
        need(all(type(event.get(key,0)) is int and event.get(key,0) >= 0 for key in keys),
             f'Invalid densification counters at step {step}.')
        limit = 20 if step > options['opacity_reset_interval'] else None
        before, after, split = (event[key] for key in keys[:3])
        need(event['screen_limit_px'] == limit and before == after+split,
             f'Original screen-radius lifecycle differs at step {step}.')
        if limit is None:
            need(not any(event.get(key,0) for key in keys[:-1]) and 'local_optimization' not in event,
                 f'Size-triggered work ran before its schedule at step {step}.')
            continue
        candidates, world = event['size_candidates'], event['world_candidates']
        removed, replaced = event['size_pruned'], event['size_replaced']
        assessed, supported = event['assessed'], event['foreground_supported']
        need(event['screen_candidates'] == after and max(after,world) <= candidates <= after+world
             and supported <= assessed <= candidates
             and removed <= assessed-supported and replaced <= supported
             and removed+replaced+event['protected'] == candidates
             and event['screen_pruned'] <= min(after,removed),
             f'Size-triggered actions do not match their measured candidates at step {step}.')
        if removed:
            need(paired_trial(event['removal_trial'],removed),
                 f'Size removal lacks an accepted exact native trial at step {step}.')
        local = event.get('local_optimization')
        if local is None:
            need(not replaced and not event['replacement_children'],
                 f'Size children at step {step} lack local optimizer evidence.')
            continue
        parents, committed, rejected = (local[key] for key in
            ('stable_parent_ids','committed_parent_ids','rejected_parent_ids'))
        children = local['children_per_parent']
        need(step in schedule['world_scale']['split_steps'] and 1 <= len(parents) <= min(8,supported)
             and all(type(parent) is int and parent >= 0 for parent in parents)
             and len(set(parents)) == len(parents) and set(committed).isdisjoint(rejected)
             and set(committed) | set(rejected) == set(parents) and len(committed)+len(rejected) == len(parents)
             and len(committed) == replaced and type(children) is int and children >= 2
             and event['replacement_children'] == children*replaced
             and type(local['optimizer_updates']) is int and 0 <= local['optimizer_updates'] <= 24
             and math.isfinite(local['seconds']) and local['seconds'] >= 0,
             f'Invalid candidate optimizer or stable lineage proof at step {step}.')
        check_local_evidence(run,step,local,ordered_frames)
        optimization_events.append(local)
        trial = event['replacement_trial']
        if local['status'] != 'fitted':
            need(trial is None and not replaced and 'commit' not in local
                 and local['decision_reason'] == 'local_optimization_did_not_finish',
                 f'An incomplete optimization was submitted to replacement trials at step {step}.')
            continue
        need(type(trial['accepted']) is bool and trial['accepted'] == bool(replaced)
             and trial['initial_proposed'] == len(parents) and trial['trial_count'] == len(trial['trials'])
             and 1 <= trial['trial_count'] <= trial['max_trials']
             and all([frame['frame'] for frame in probe['frames']] == ordered_frames for probe in trial['trials']),
             f'Size replacement trial did not evaluate the protected 24 native training views at step {step}.')
        if not replaced:
            need('commit' not in local and not trial['selected_candidate_ids'],
                 f'Rejected size children were published at step {step}.')
            continue
        final_trial = trial['final_trial']
        radius = final_trial['native_radius_check']
        commit = local['commit']
        need(len(trial['selected_candidate_ids']) == replaced
             and trial['selected_candidate_ids'] == final_trial['candidate_ids']
             and paired_trial(final_trial,replaced) and radius['all_decreased'] is True
             and radius['limit_px'] == limit and len(radius['before_px']) == len(radius['child_max_px']) == replaced
             and all(before > limit and 0 <= after < before for before,after in zip(
                 radius['before_px'],radius['child_max_px'],strict=True))
             and commit['children'] == len(commit['child_ids']) == len(set(commit['child_ids']))
                 == event['replacement_children']
             and commit['parent_ids'] == committed and set(commit['child_ids']).isdisjoint(parents)
             and set(commit['moment_groups']) == {'xyz','scaling','opacity','f_dc'}
             and commit['unchanged_adam_rows'] >= 0 and local['optimizer_updates'] > 0,
             f'Size replacement lacks exact coverage, radius, or optimizer migration evidence at step {step}.')
    candidate = audit['candidate_optimization']
    optimized_ids = [parent for local in optimization_events for parent in local['stable_parent_ids']]
    unique_size = candidate['unique_size_candidate_parents']
    need(desc['candidate_optimization'] == candidate and candidate['main_optimizer_updates'] == TRAINING_STEPS
         and candidate['events'] == len(optimization_events)
         and candidate['size_candidate_events'] == audit['totals']['size_candidates']
         and len(set(optimized_ids)) <= unique_size <= candidate['size_candidate_events']
         and candidate['repeated_size_candidate_events'] == candidate['size_candidate_events']-unique_size
         and candidate['optimized_parent_events'] == len(optimized_ids)
         and candidate['unique_optimized_parents'] == len(set(optimized_ids))
         and candidate['repeated_optimized_parent_events'] == len(optimized_ids)-len(set(optimized_ids))
         and candidate['optimizer_updates'] == sum(local['optimizer_updates'] for local in optimization_events)
         and math.isclose(candidate['seconds'],sum(local['seconds'] for local in optimization_events),rel_tol=1e-9),
         'The candidate optimizer aggregate disagrees with the native event or independent-parent evidence.')
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
                radius_units=audit['radius_units'],stock_events=len(events),totals=audit['totals'],
                candidate_optimization=candidate,
                candidate_verification_seconds=sum(local['verification_seconds'] for local in optimization_events))


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
             'Fresh round-sixteen output is required.')
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
                   retained_holdouts=data['retained_holdouts'],fit_sha256=data['fit_sha256']),flush=True)
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
