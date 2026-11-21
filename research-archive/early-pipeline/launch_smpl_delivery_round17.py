"""Run the R17 local optimizer with the fixed R15/R16 input and main training profile."""

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
import launch_smpl_delivery_round16 as prior


need, read, pin, sha, write = cold.need, cold.read, cold.pin, cold.sha, cold.write
check_snapshot, color_stats = prior.check_snapshot, prior.color_stats
stage, check_binding = prior.stage, prior.check_binding
snapshot_failure, verify_parity = prior.snapshot_failure, prior.verify_parity
PYTHON = cold.ENV/'bin/python'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-multiview-quality-20261111')
PARENT_PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-size-optimization-20261111')
PARENT = 'af19f3a36880115472398c22396194143c0970c9'
QUALITY_BASELINE = prior.PARENT
STABLE_TAG_COMMIT = 'a6166a5ffd439362ab2c17cda30f9ec408c688c4'
OUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs')
DIRECTOR_ROOT = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs')
DIRECTOR = DIRECTOR_ROOT/'round16_director_review_20261111.json'
DIRECTOR_SHA = 'f7068409e5832d39f956dee47da189f1e9fa60ee6a5bb1b3f21071e35675e1c8'
DELIVERY = OUT/'smpl-delivery-round16/receipt.json'
DELIVERY_SHA = '99593659dc9ab660b983aa71eeff280d9c5760e7a02167c02ac91e1bfc32c9bb'
CUDA_PROOF = OUT/'smpl-delivery-round17/input-audit/cuda-multiview-quality-proof.json'
CUDA_FIXTURE_SOURCE = Path(__file__).with_name('probe_cuda_size_optimization_round17.py')
POLICY = 'projected_ldr_dc'
POLICY_KEY = 'sh0_color_constraint'
COLOR_C0 = .28209479177387814
MAX_WALL_SECONDS = 7200
TRAINING_STEPS = 10000
SCHEMA = 'shuolee-round17-fixed-fit-multiview-quality-v1'
REQUIRED = {'src/shuolee_portrait/training.py','src/shuolee_portrait/size_child_optimization.py',
            'src/shuolee_portrait/world_scale.py'}
OPT_PATHS = REQUIRED | {'src/shuolee_portrait/render_support.py','src/shuolee_portrait/size_child_commit.py'}
ALLOWED = REQUIRED | {'tests/test_render_support.py','tests/test_world_scale.py'}
EVIDENCE_FILES = prior.EVIDENCE_FILES
CHECKOUT_NORMALIZED_SOURCE = 'src/shuolee_portrait/size_child_commit.py'


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round17-{subject}'


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
                     and Path(name).name.startswith(('test_densif','test_size_','test_multiview'))
                     and Path(name).suffix == '.py'}
    need(REQUIRED <= changed <= ALLOWED | focused_tests,
         f'Unexpected multi-view child fitting code paths: {sorted(changed)}')
    prior_gitdir = subprocess.check_output(
        ['/usr/bin/wslpath','-u',(PARENT_PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')],
        text=True).strip()
    prior_git = ['/usr/bin/git','--no-optional-locks','-c','core.fileMode=false','--git-dir',prior_gitdir,'--work-tree',str(PARENT_PROJECT)]
    prior_head = subprocess.check_output([*prior_git,'rev-parse','HEAD'],text=True).strip()
    prior_status = subprocess.check_output([*prior_git,'status','--porcelain','--untracked-files=all'],
                                            cwd=PARENT_PROJECT,text=True)
    need(prior_head == PARENT and not prior_status.strip(), 'The R16 worktree changed.')
    tracked = set(cold.SOURCE_FILES) | changed | OPT_PATHS | {
        'src/shuolee_portrait/sh0_main.py','src/shuolee_portrait/world_scale.py',
        'src/shuolee_portrait/coverage_refinement.py','src/shuolee_portrait/render_support.py'}
    for name in tracked-changed:
        if name == CHECKOUT_NORMALIZED_SOURCE:
            # R16's old working copy has one CRLF line. The committed file
            # and the R17 checkout are LF, and CUDA proof pins the latter bytes.
            old = (PARENT_PROJECT/name).read_bytes()
            need((PROJECT/name).read_bytes() == old.replace(b'\r\n',b'\n'),
                 'The unchanged commit transaction differs after checkout line-ending normalization.')
        else:
            pin(PROJECT/name,sha(PARENT_PROJECT/name))
    return {name:sha(PROJECT/name) for name in sorted(tracked)}


def source_inventory(subject):
    """Bind both delivered baselines and stage the unchanged R15 input bytes."""
    inventory = prior.source_inventory(subject)
    files = inventory['files']

    def record(path, digest=None):
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
    need(review['round'] == 16 and review['commit'] == delivery['commit'] == PARENT
         and review['baseline_commit'] == delivery['baseline_commit'] == QUALITY_BASELINE
         and review['receipt_sha256'] == DELIVERY_SHA and review['quality_accepted'] is False
         and delivery['quality_accepted'] is False and review['next']['round'] == 17,
         'The R16 report or director authorization differs.')
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
    record(evidence['cuda_size_optimization_proof']['path'],review['verification']['cuda_proof_hash_verified'])
    terminal = read(baseline/'wrapper-terminal.json')
    r16_plan = read(baseline/'preregistration.json')
    r16_inventory = read(baseline/'protected-inputs.json')
    need(terminal['state'] == 'completed' and terminal['candidate_commit'] == r16_plan['candidate_commit'] == PARENT
         and terminal['result_sha256'] == delivery_subject['result_sha256']
         and terminal['preregistration_sha256'] == evidence['preregistration']['sha256']
         and terminal['protected_input_inventory_sha256'] == evidence['protected_inputs']['sha256']
         and terminal['same_round15_fit_and_training_selection'] is True
         and terminal['fit_sha256'] == inventory['fit_sha256']
         and terminal['densification_audit']['totals']['size_replaced'] == 0
         and terminal['historical_r15_006_parity'] == inventory['r15_006_parity'],
         'The archived R16 formal terminal identity differs.')
    record(run/'result.json',terminal['result_sha256'])
    r15_plan = read(Path(inventory['baseline_run']).parent/'preregistration.json')
    need(r16_plan['runtime'] == r15_plan['runtime'] and r16_plan['asset_sha256'] == r15_plan['asset_sha256']
         and r16_plan['selected_frames'] == inventory['selected_frames'] == r16_inventory['selected_frames']
         and r16_plan['retained_holdouts'] == inventory['retained_holdouts'] == r16_inventory['retained_holdouts'],
         'The archived R16 runtime or frame roster differs from R15.')
    for name,entry in inventory['staged_files'].items():
        need(r16_inventory['staged_files'][name]['sha256'] == entry['sha256'],
             f'R15 and R16 source payload differs: {name}.')
        record(run/name,entry['sha256'])
    for name in ('config.json','gaussian-run/config.json','gaussian-run/training-cameras.json',
                 'gaussian-run/initialization.json'):
        record(run/name,inventory['profiles'][name]['sha256'])
    r16_training = run/'gaussian-run/result.json'
    record(r16_training)
    r15_training = read(Path(inventory['profiles']['gaussian-run/result.json']['path']))
    r16_profile = read(r16_training)
    need(all(r16_profile[key] == r15_training[key] for key in ('per_frame_updates','source_bin_draws','pose_layout'))
         and r16_profile['steps'] == TRAINING_STEPS,
         'R16 sampling and pose consumption differs from the fixed R15 main training profile.')
    for step,parity in zip((1000,TRAINING_STEPS),terminal['native_ply_parity'],strict=True):
        checked = verify_parity(run,step,inventory['reference_frame'])
        need(checked == parity,f'The archived R16 step {step} parity differs.')
        path = run/parity['file']
        record(path,parity['sha256'])
        record(path.parent/'world.ply',read(path)['sha256'])
        record(path.parent/'reference-mesh.npz',read(path)['mesh_sha256'])
    return dict(inventory,schema=SCHEMA,baseline_commit=PARENT,quality_baseline_commit=QUALITY_BASELINE,
                archived_round16_run=str(run),archived_round16_terminal=evidence['wrapper'])


def plan_for(subject, commit, cuda_proof_sha):
    code = check_code(commit)
    inventory = source_inventory(subject)
    need(re.fullmatch('[0-9a-f]{64}',cuda_proof_sha) is not None,'The approved CUDA fixture SHA-256 is required.')
    pin(CUDA_PROOF,cuda_proof_sha)
    proof = read(CUDA_PROOF)
    fixture_sources = proof['source_sha256']
    need(proof['state'] == 'passed' and REQUIRED <= fixture_sources.keys() <= code.keys()
         and all(code[name] == digest for name,digest in fixture_sources.items()),
         'The multi-view optimization CUDA fixture did not pass for this exact source tree.')
    old_run = Path(inventory['baseline_run'])
    old_plan = read(old_run.parent/'preregistration.json')
    runtime = cold.cpu_runtime(cold.clean_env())
    need(runtime == old_plan['runtime'],'Training runtime differs from R15.')
    _verify_cuda_controls(proof,runtime)
    assets = old_plan['asset_sha256']
    for name,digest in assets.items():
        pin(Path(name),digest)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient space for the two full renders and their replay records.')
    with TemporaryDirectory(prefix='r17-cpu-fit-',dir=cold.LAB) as temporary:
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
                quality_baseline_commit=QUALITY_BASELINE,archived_round16_run=inventory['archived_round16_run'],
                stable_git_release_commit=STABLE_TAG_COMMIT,
                baseline_run=inventory['baseline_run'],baseline_result_sha256=inventory['profiles']['result.json']['sha256'],
                output=str(run),director_review_sha256=DIRECTOR_SHA,baseline_delivery_sha256=DELIVERY_SHA,
                archived_r15_006_parity_sha256=prior.RETRY_PROOF_SHA,cuda_fixture_sha256=cuda_proof_sha,
                wrapper_sha256=sha(Path(__file__)),
                checkout_line_ending_normalization=dict(path=CHECKOUT_NORMALIZED_SOURCE,
                    baseline_sha256=sha(PARENT_PROJECT/CHECKOUT_NORMALIZED_SOURCE),
                    candidate_sha256=code[CHECKOUT_NORMALIZED_SOURCE]),
                verifier_source_sha256={Path(module.__file__).name:sha(Path(module.__file__))
                                        for module in (prior,prior.prior,cold)},
                runtime=runtime,code_sha256=code,asset_sha256=assets,cpu_fit_preflight=fit_preflight,
                retained_holdouts=inventory['retained_holdouts'],
                selected_frames=inventory['selected_frames'],commands=[command],gaussian_steps=TRAINING_STEPS,
                local_optimization=dict(max_parent_attempts_per_event=8,source_views_per_cycle=24,
                                        min_complete_cycles_for_convergence=4,max_complete_cycles=8,
                                        max_adam_updates_per_parent=192,exact_guard_source_views=24),
                color_policy=POLICY,auto_resolved_profile_fields=['initial_points','image_cache_frames'],
                max_wall_seconds=MAX_WALL_SECONDS),inventory


def _verify_cuda_controls(proof,runtime):
    binary_digest, = [digest for name,digest in runtime['file_sha256'].items()
                       if 'diff_gaussian_rasterization' in name and Path(name).name.startswith('_C')]
    positive, negative = proof['optimized'], proof['rejected']
    fallback = proof['fallback']
    need(proof['fixture_sha256'] == sha(CUDA_FIXTURE_SOURCE)
         and proof['runtime']['raster_extension_sha256'] == binary_digest
         and positive['views'] == negative['views'] == fallback['trial']['views'] == 24
         and positive['accepted'] is True and positive['native_radius_check']['all_decreased'] is True
         and negative['accepted'] is False and fallback['injected_optimized_rejection']['accepted'] is False
         and fallback['path'] == 'r15_fallback' and fallback['trial']['accepted'] is True
         and len(fallback['parents']) == 1 and len(fallback['children']) == 2
         and proof['post_prune_rollback']['restored_model_and_optimizer'] is True
         and proof['commit']['children'] == 4
         and set(proof['commit']['moment_groups']) == {'xyz','scaling','opacity','f_dc'},
         'The tested CUDA controls, native renderer or migrated optimizer state differ.')
    optimization = proof['optimization']
    views = optimization['views']
    need(_supervision_is_valid(views) and any(row['foreground_equivalent_pixels'] == 0
         and not row['foreground_supervision'] and row['exterior_supervision'] and row['boundary_supervision']
         for row in views),'The masked exterior CUDA control lacks a zero-foreground observation.')
    _cycle_evidence(optimization,[row['frame'] for row in views],[row['angular_bin'] for row in views])


def register(subject,commit,cuda_proof_sha):
    plan, inventory = plan_for(subject,commit,cuda_proof_sha)
    root = root_for(subject)
    root.mkdir(exist_ok=True)
    need(not any(root.iterdir()), 'Registration would overwrite an existing R17 execution.')
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
         'Registered R17 candidate or protected inputs differ.')
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
         'R17 changed the effective run profile.')
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
         'R17 changed initialization, optimization schedule, or the consumed pose/sample sequence.')

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
        same_round15_fit_and_training_selection=True,same_round16_fit_and_training_selection=True,
        face_guard=read(run/joint['file'])['face_guard'],
        color_constraint=color_proof,binding=binding,densification_audit=densification,native_ply_parity=parity)


def _ids(rows):
    return all(type(value) is int and value >= 0 for value in rows) and len(set(rows)) == len(rows)


def _native_trial(trial,ordered_frames,screen_limit=None):
    frames = trial['frames']
    count = trial['proposed']
    need(type(count) is int and count > 0 and [row['frame'] for row in frames] == ordered_frames,
         'The native trial did not cover the ordered protected 24-source view roster.')
    radius = trial['native_radius_check']
    reduced = True
    if screen_limit is not None:
        before, after = radius['before_px'], radius['child_max_px']
        reduced = all(parent > screen_limit and 0 <= child < parent
                      for parent,child in zip(before,after,strict=True))
        need(radius['limit_px'] == screen_limit and len(before) == len(after) == count
             and radius['all_decreased'] is reduced,'The radius check differs from the native measured peaks.')
    else:
        need(radius is None,'A pure removal should not contain a child radius proof.')
    need(trial['accepted'] is (reduced and all(not row['failures'] for row in frames)),
         'The topology decision disagrees with original native appearance or coverage protection.')
    return trial['accepted']


def _cycle_evidence(attempt,ordered_frames,source_bins):
    count = len(ordered_frames)
    budget = 8
    cycles = attempt['cycles']
    proposal = attempt['proposal_check']
    completed = len(cycles)
    finite = []
    converged = 'two_stable_feasible_full_sweeps'
    reason = attempt['stop_reason']
    need(len(source_bins) == count == 24 and attempt['budget_cycles'] == budget
         and attempt['optimizer_view_budget'] == count and attempt['optimizer_update_budget'] == budget*count
         and 0 <= completed <= budget and attempt['convergence_min_cycles'] == 4
         and attempt['convergence_source_objective_delta'] == 1e-5
         and completed*count <= attempt['optimizer_updates'] <= budget*count,
         'The local optimization budget differs from the registered complete-source sweeps.')
    groups = [[frame for frame,value in zip(ordered_frames,source_bins,strict=True) if value == bin_id]
              for bin_id in sorted(set(source_bins))]
    for i,record in enumerate([proposal,*cycles]):
        rejected = record['rejected_views']
        need(record['cycle'] == i and record['updates'] == i*count and record['views_checked'] == count
             and len(set(rejected)) == len(rejected) and set(rejected) <= set(ordered_frames)
             and type(record['guard_feasible']) is bool,
             'The local proposal/checkpoint did not evaluate exactly all protected source views.')
        if i:
            offset = (i-1) % len(groups)
            rotated = groups[offset:]+groups[:offset]
            rotated = [group[(i-1)%len(group):]+group[:(i-1)%len(group)] for group in rotated]
            sequence = [group[j] for j in range(max(map(len,rotated))) for group in rotated if j < len(group)]
            need(record['view_order'] == sequence,'An optimizer cycle did not visit the expected source-bin order.')
        metrics = [record[key] for key in ('worst_normalized_guard_excess','total_normalized_guard_excess',
                                           'source_objective')]
        measured = all(value is not None and math.isfinite(value) and value >= 0 for value in metrics)
        need(measured or (i == completed and reason == 'nonfinite_full_cycle_checkpoint'),
             'Invalid fixed objective or guard metrics in the source sweep history.')
        if measured and i:
            score = (not record['guard_feasible'],metrics[0],metrics[1],len(rejected),metrics[2])
            finite.append((score,record))
    best = min(finite,key=lambda row: row[0])[1] if finite else proposal
    need(attempt['checkpoint_cycle'] == best['cycle'] and attempt['checkpoint_updates'] == best['updates']
         and attempt['checkpoint_feasible'] is (bool(finite) and best['guard_feasible']),
         'The saved optimizer checkpoint is not the best complete native guard/objective result.')
    if reason in (converged,'cycle_budget'):
        need(attempt['optimizer_updates'] == completed*count
             and attempt['status'] == ('fitted' if best['guard_feasible'] else 'no_feasible_checkpoint'),
             'A complete local optimization did not preserve its actual update count or status.')
        if reason == converged:
            last = cycles[-3:]
            need(completed >= 4 and all(row['guard_feasible'] for row in last)
                 and all(abs(a['source_objective']-b['source_objective']) <= 1e-5
                         for a,b in zip(last,last[1:])),
                 'Local optimization stopped without two stable, feasible complete sweeps.')
        else:
            need(completed == budget,'Local optimization stopped before its full cycle budget.')
    else:
        need(reason == attempt['status'] and reason in ('no_supported_optimization_view',
              'nonfinite_candidate_objective_or_gradient','nonfinite_full_cycle_checkpoint'),
             'The local optimizer stopped for an unregistered reason.')
        if reason == 'no_supported_optimization_view':
            need(completed == attempt['optimizer_updates'] == 0,'Unsupported local fitting recorded Adam work.')


def _supervision_is_valid(views):
    flags = ('foreground_supervision','exterior_supervision','boundary_supervision')
    return all(math.isfinite(row['foreground_equivalent_pixels']) and row['foreground_equivalent_pixels'] >= 0
               and all(type(row[key]) is bool for key in flags)
               and (row['foreground_equivalent_pixels'] > 0 or not row['foreground_supervision']) for row in views)


def _local_evidence(run,step,attempt,parent,ordered_frames,source_bins):
    views = attempt['views']
    indices = list(range(len(ordered_frames)))
    need([view['view_index'] for view in views] == indices
         and [view['frame'] for view in views] == ordered_frames
         and [view['angular_bin'] for view in views] == source_bins
         and _supervision_is_valid(views)
         and type(attempt['optimizer_updates']) is int and attempt['optimizer_updates'] >= 0
         and all(math.isfinite(attempt[key]) and attempt[key] >= 0 for key in ('seconds','verification_seconds')),
         f'Size-child fitting used invalid optimizer inputs or timing at step {step}.')
    _cycle_evidence(attempt,ordered_frames,source_bins)
    subdir = f'step-{step:06d}/parent-{parent:07d}'
    need(attempt['diagnostic_subdir'] == subdir,f'Invalid size-child diagnostics directory at step {step}.')
    directory = run/'gaussian-run/size-child-diagnostics'/subdir
    reps, decisions = (attempt[key] for key in ('representatives','decision_representatives'))
    need([row['view_index'] for row in reps] == [row['view_index'] for row in decisions]
         and len(set(row['view_index'] for row in reps)) == len(reps) <= 2,
         f'The original and decision diagnostic crops disagree at step {step}.')
    for rows,prefix in ((reps,'representative'),(decisions,'decision')):
        for row in rows:
            i = row['view_index']
            x0,y0,x1,y1 = row['box']
            need(i in indices and row['path'] == f'{prefix}-view-{i:02d}.png'
                 and row['frame'] == ordered_frames[i] and 0 <= x0 < x1 and 0 <= y0 < y1,
                 f'Invalid native crop provenance at step {step}.')
            pin(directory/row['path'],row['sha256'])
    need(all(first['box'] == second['box'] for first,second in zip(reps,decisions,strict=True)),
         f'The saved candidate and decision boxes differ at step {step}.')


def _fallback_evidence(event,ordered_frames,limit,options,optimized):
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
            _native_trial(probe,ordered_frames,limit)
        selected = trial['selected_candidate_ids']
        need(trial['accepted'] is bool(committed) and len(selected) == len(committed),
             'The original fallback commit count differs from its exact trial.')
        if committed:
            final = trial['final_trial']
            need(final in trials and final['candidate_ids'] == selected
                 and final['proposed'] == len(committed) and _native_trial(final,ordered_frames,limit),
                 'The R15 fallback published a set without full native coverage proof.')
    return committed,children


def verify_densification(run,integration,effective,ordered_frames,points):
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
            accepted = _native_trial(removal,ordered_frames)
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
            accepted = trial is not None and _native_trial(trial,ordered_frames,limit)
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
        fallback_parents, fallback_children = _fallback_evidence(event,ordered_frames,limit,guard,optimized_parents)
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
             'Fresh round-seventeen output is required.')
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
