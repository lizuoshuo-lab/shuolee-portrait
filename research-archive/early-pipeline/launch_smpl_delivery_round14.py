"""Train one fresh color-constrained SH0 model on each completed R13 fit."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
from tempfile import TemporaryDirectory
import zipfile

import launch_unified_cold_016_006 as cold


need, read, pin, sha, write = cold.need, cold.read, cold.pin, cold.sha, cold.write
PYTHON = cold.ENV/'bin/python'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-ldr-20261111')
PARENT_PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-coupled-fit-20261111')
PARENT = 'a9dbfb03565ad89b29b1bd2b5b22fdcd2009d43a'
OUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs')
DIRECTOR_ROOT = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs')
DIRECTOR = DIRECTOR_ROOT/'round13_director_review_20261111.json'
DIRECTOR_SHA = 'd654785b7f2b603e05b40f8f04c6b7e157b8635072feabaf4ec2bb856fb35864'
COLOR_AUDIT = DIRECTOR_ROOT/'round13_color_audit_20261111.json'
COLOR_AUDIT_SHA = 'ea9e52a0389d41197d6f480940f9e14c8fbac1561d1b2a9e6b6a2eafa69fdf2c'
DELIVERY = OUT/'smpl-delivery-round13/receipt.json'
DELIVERY_SHA = '55ccd67daf8607f75f069a8d17ff9783327097fd56938ab19b57475bffdfbbd8'
POLICY = 'projected_ldr_dc'
POLICY_KEY = 'sh0_color_constraint'
COLOR_C0 = .28209479177387814
MAX_WALL_SECONDS = 7200
TRAINING_STEPS = 10000
SCHEMA = 'shuolee-round14-fixed-fit-sh0-ldr-v1'
ALLOWED = {'src/shuolee_portrait/training.py', 'src/shuolee_portrait/sh0_main.py',
           'tests/test_sh0_color_constraint.py'}
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
}


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round14-{subject}'


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
    need('src/shuolee_portrait/training.py' in changed and changed <= ALLOWED,
         f'Unexpected color-isolation code paths: {sorted(changed)}')
    prior_status = subprocess.check_output(
        ['/usr/bin/git','-c','core.fileMode=false','--git-dir',
         subprocess.check_output(['/usr/bin/wslpath','-u',(PARENT_PROJECT/'.git').read_text().strip()
                                  .removeprefix('gitdir: ')],text=True).strip(),'--work-tree',str(PARENT_PROJECT),
         'status','--porcelain','--untracked-files=all'],cwd=PARENT_PROJECT,text=True)
    need(not prior_status.strip(), 'The prior candidate worktree is dirty.')
    tracked = set(cold.SOURCE_FILES) | changed | {
        'src/shuolee_portrait/sh0_main.py','src/shuolee_portrait/world_scale.py',
        'src/shuolee_portrait/coverage_refinement.py'}
    for name in tracked-changed:
        pin(PROJECT/name,sha(PARENT_PROJECT/name))
    return {name:sha(PROJECT/name) for name in sorted(tracked)}


def source_inventory(subject):
    """Bind the delivered R13 inputs to the exact trained subject and pose order."""
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
    record(COLOR_AUDIT,COLOR_AUDIT_SHA)
    record(DELIVERY,DELIVERY_SHA)
    review, audit, delivery = read(DIRECTOR), read(COLOR_AUDIT), read(DELIVERY)
    need(review['round'] == audit['round'] == 13 and review['commit'] == delivery['commit'] == PARENT
         and review['receipt_sha256_verified'] == DELIVERY_SHA and review['quality_accepted'] is False
         and delivery['quality_accepted'] is False, 'R13 report or director review differs.')
    parent = cold.LAB/f'shuolee-smpl-delivery-round13-{subject}'
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
    record(parent/'protected-inputs.json',evidence['protected_inputs']['sha256'])
    terminal = read(parent/'wrapper-terminal.json')
    old_plan = read(parent/'preregistration.json')
    need(terminal['state'] == 'completed' and terminal['candidate_commit'] == PARENT
         and terminal['result_sha256'] == delivery_subject['result_sha256']
         and terminal['preregistration_sha256'] == evidence['preregistration']['sha256']
         and terminal['protected_input_inventory_sha256'] == evidence['protected_inputs']['sha256']
         and old_plan['candidate_commit'] == PARENT, 'R13 formal terminal identity differs.')
    record(run/'result.json',terminal['result_sha256'])
    result = read(run/'result.json')
    need(result['steps'] == TRAINING_STEPS and result['gaussian_profile'] == 'smpl_sh0'
         and result['sh_degree'] == 0 and result['fit_calls'] == 0
         and result['points'] == terminal['points'] == audit['subjects'][subject]['points'],
         'R13 baseline is not the delivered SH0 run.')
    endpoint = audit['subjects'][subject]
    need(Path(endpoint['ply']) == run/'gaussian-run/step-010000/bound-local.ply',
         'R13 audit binds a different terminal PLY.')
    record(endpoint['ply'],endpoint['sha256'])

    fit = read(run/'fit/official-fit-result.json')
    selection = read(run/'fit/selection.json')
    ordered_rows = read(run/'data/training-inputs.json')['records']
    selected = [row['frame'] for row in ordered_rows]
    held = selection['retained_holdouts']
    need(fit['state'] == 'complete' and fit['frames'] == len(set(selected)) == len(selected) == 24
         and fit['selection'] == selection
         and selected[0] == selection['reference_frame'] == result['reference_frame_id']
         and set(selected) == set(selection['selected_frames']) == set(terminal['selected_frames'])
         and held == old_plan['retained_holdouts'] and set(selected).isdisjoint(held),
         'R13 training selection or holdout identity differs.')
    if subject == '016':
        need('frame_000051' in selected and 'frame_000055' not in selected and
             'frame_000055' in read(run/'fit/joint-body-refinement.json')['fit_frame_ids'], 'R13 f051/f055 identities differ.')

    record(run/'fit/fitted.npz',fit['fit_sha256'])
    record(run/'fit/shared-body.json',fit['shared_body_sha256'])
    record(run/'data/training-inputs.json',fit['training_inputs_sha256'])
    for name in ('data/inputs.json','data/observations.npz'):
        record(run/name,fit['source']['input_files'][str(run/name)])
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
    need(POLICY_KEY not in prior_cfg and POLICY_KEY not in effective
         and 'frozen_surface' not in prior_cfg and effective['seed'] == prior_cfg['seed'] == 0
         and effective['iterations'] == prior_cfg['iterations'] == TRAINING_STEPS
         and effective['sh_degree'] == prior_cfg['sh_degree'] == 0
         and effective['unified_main'] == prior_cfg['unified_main']
         and effective['optimization'] == prior_cfg['optimization']
         and init['initial_points'] == effective['initial_points'] == 20908
         and init['optimizer_initial_state_entries'] == 0 and init['no_prior_gaussian_loaded']
         and len(old_train['per_frame_updates']) == 24
         and sum(old_train['per_frame_updates']) == TRAINING_STEPS,
         'R13 training profile is not the controlled fresh baseline.')
    cameras = read(run/'gaussian-run/training-cameras.json')
    need(cameras['source_sha256'] == fit['training_inputs_sha256']
         and cameras['interpolated_cameras'] == 0 and len(cameras['records']) == len(selected),
         'R13 consumed cameras do not bind the protected training manifest.')
    staged = {name:record(run/name,files.get(str(run/name))) for name in STAGED}
    profiles = {name:record(run/name,files.get(str(run/name))) for name in PROFILE_FILES}
    return dict(schema=SCHEMA,subject=subject,baseline_commit=PARENT,baseline_run=str(run),files=files,
                staged_files=staged,profiles=profiles,selected_frames=selected,retained_holdouts=held,
                reference_frame=selection['reference_frame'],fit_sha256=fit['fit_sha256'],
                seed=effective['seed'],gaussian_steps=TRAINING_STEPS)


def plan_for(subject, commit):
    code = check_code(commit)
    inventory = source_inventory(subject)
    old_plan = read(Path(inventory['baseline_run']).parent/'preregistration.json')
    runtime = cold.cpu_runtime(cold.clean_env())
    need(runtime == old_plan['runtime'],'Training runtime differs from R13.')
    assets = old_plan['asset_sha256']
    for name,digest in assets.items():
        pin(Path(name),digest)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient space for the two full renders and their replay records.')
    run = root_for(subject)/'run'
    command = [str(PYTHON),'-B','-u','-m','shuolee_portrait.cli','train-fitted','--run',str(run)]
    return dict(schema=SCHEMA,subject=subject,candidate_commit=commit,baseline_commit=PARENT,
                baseline_run=inventory['baseline_run'],baseline_result_sha256=inventory['profiles']['result.json']['sha256'],
                output=str(run),director_review_sha256=DIRECTOR_SHA,baseline_delivery_sha256=DELIVERY_SHA,
                baseline_color_audit_sha256=COLOR_AUDIT_SHA,wrapper_sha256=sha(Path(__file__)),
                runtime=runtime,code_sha256=code,asset_sha256=assets,retained_holdouts=inventory['retained_holdouts'],
                selected_frames=inventory['selected_frames'],commands=[command],gaussian_steps=TRAINING_STEPS,
                color_policy=POLICY,auto_resolved_profile_fields=['initial_points','image_cache_frames'],
                max_wall_seconds=MAX_WALL_SECONDS),inventory


def register(subject,commit):
    plan, inventory = plan_for(subject,commit)
    root = root_for(subject)
    root.mkdir(exist_ok=True)
    need(all(path.name == 'endpoint-color-isolation' and path.is_dir() for path in root.iterdir()),
         'Registration would overwrite an existing R14 execution.')
    protected = root/'protected-inputs.json'
    write(protected,inventory)
    write(root/'preregistration.json',dict(plan,registered_at=cold.now(),protected_input_inventory_sha256=sha(protected)))
    print(dict(state='registered',subject=subject,files=len(inventory['files']),inventory_sha256=sha(protected)),flush=True)


def registered(subject,commit):
    root = root_for(subject)
    plan = read(root/'preregistration.json')
    protected = root/'protected-inputs.json'
    pin(protected,plan['protected_input_inventory_sha256'])
    inventory = read(protected)
    current, current_inventory = plan_for(subject,commit)
    need(inventory == current_inventory and all(plan[key] == value for key,value in current.items()),
         'Registered R14 candidate or protected inputs differ.')
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
        # R13 persisted the resolved values after training startup. Its public
        # profile used automatic initialization and cache sizing before startup.
        write(destination/'config.json',dict(cfg,initial_points=None,image_cache_frames=None,
                                            **{POLICY_KEY:POLICY}))
        destination.rename(run)


def color_stats(path):
    """Independently decode the standard f_dc stored in a terminal PLY."""
    import numpy as np
    from plyfile import PlyData

    vertices = PlyData.read(path)['vertex'].data
    dc = np.column_stack([vertices[f'f_dc_{channel}'] for channel in range(3)]).astype(np.float64)
    rgb = COLOR_C0*dc + .5
    need(rgb.size > 0 and np.isfinite(rgb).all(),f'Invalid SH0 DC in {path}.')
    # The stored coefficients are float32, so allow half an ulp in RGB.
    outside = (rgb < -1e-7) | (rgb > 1.+1e-7)
    need(not outside.any(),f'Exported SH0 DC leaves the standard LDR range in {path}.')
    return dict(points=len(vertices),rgb_min=float(rgb.min()),rgb_max=float(rgb.max()),
                channels_outside_bounds=int(outside.sum()),finite=True)


def check_snapshot(run,step):
    location = run/'gaussian-run'/f'step-{step:06d}'
    audit = location/'sh0-color-range.json'
    stats = read(audit)
    ply = location/'world.ply'
    measured = color_stats(ply)
    need(stats['finite'] and stats['channels_outside_bounds'] == 0
         and stats['points'] == measured['points']
         and abs(stats['rgb_min']-measured['rgb_min']) <= 1e-7
         and abs(stats['rgb_max']-measured['rgb_max']) <= 1e-7,
         f'Step {step} training color statistics differ from standard SH0 PLY.')
    return dict(step=step,path=str(audit.relative_to(run)),sha256=sha(audit),ply_sha256=sha(ply),stats=stats)


def finish(root,plan,inventory,returncode,elapsed):
    run = root/'run'
    receipt = dict(state='failed' if returncode else 'incomplete',subject=plan['subject'],
        returncode=returncode,stage='train_fitted',candidate_commit=plan['candidate_commit'],
        baseline_commit=PARENT,command_to_end_seconds=elapsed,preregistration_sha256=sha(root/'preregistration.json'),
        protected_input_inventory_sha256=plan['protected_input_inventory_sha256'],quality_accepted=False)
    if returncode == 0:
        try:
            receipt.update(verify_outputs(run,plan,inventory))
        except Exception as error:
            receipt['validation_error'] = f'{type(error).__name__}: {error}'
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
    need(cfg == dict(baseline('config.json'),**{POLICY_KEY:POLICY})
         and effective == dict(baseline('gaussian-run/config.json'),**{POLICY_KEY:POLICY}),
         'R14 changed effective run settings beyond the SH0 color policy.')
    for name in ('gaussian-run/training-cameras.json','gaussian-run/initialization.json'):
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
         'R14 changed initialization, optimization schedule, or the consumed pose/sample sequence.')

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
         'Bound replay changed the exact R13 selected pose sequence.')
    bound = run/'gaussian-run/step-010000/bound-local.ply'
    stats = dict(bound_local=color_stats(bound),standard_world=color_stats(run/result['ply']))
    need(all(row['points'] == result['points'] for row in stats.values()),'Terminal point count differs across PLYs.')
    early = check_snapshot(run,1000)
    late = check_snapshot(run,TRAINING_STEPS)
    need(late['stats'] == proof['terminal'], 'Terminal snapshot and optimizer color receipt disagree.')
    color_proof = dict(file=str(color_path.relative_to(run)),sha256=sha(color_path),policy=POLICY,
                       ply_stats=stats,snapshots=[early,late])
    selection = read(run/'fit/selection.json')
    joint = read(run/'fit/official-fit-result.json')['joint_body_refinement']
    return dict(state='completed',result_sha256=sha(run/'result.json'),points=result['points'],
        reference_frame=result['reference_frame_id'],selected_frames=selection['selected_frames'],
        coverage=selection['coverage'],joint_refinement_sha256=joint['sha256'],joint_candidate_sha256=joint['candidate_sha256'],
        world_scale_sha256=integration['world_scale']['sha256'],fit_method='reused_exact_round13_fit',
        fresh_gaussian=True,original_video_to_export=False,native_inputs_reused=True,
        reference_mesh_frozen_from_prior=False,fit_sha256=inventory['fit_sha256'],
        training_inputs_sha256=inventory['staged_files']['data/training-inputs.json']['sha256'],
        same_round13_fit_and_training_selection=True,face_guard=read(run/joint['file'])['face_guard'],
        color_constraint=color_proof)


def launch(subject,commit):
    root = root_for(subject)
    plan, inventory = registered(subject,commit)
    with cold.LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subject == '006':
            first = root_for('016')
            terminal = read(first/'wrapper-terminal.json')
            need(terminal['state'] == 'completed' and terminal['candidate_commit'] == commit,
                 'Complete the verified 016 export before starting 006.')
            pin(first/'run/result.json',terminal['result_sha256'])
        need(not any((root/name).exists() for name in ('run','launch.json','entry.log','wrapper-terminal.json')),
             'Fresh round-fourteen output is required.')
        stage(root,inventory)
        write(root/'launch.json',dict(subject=subject,commands=plan['commands'],started=cold.now(),
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
    args = parser.parse_args()
    if args.action == 'check-inputs':
        data = source_inventory(args.subject)
        print(dict(state='source_preflight_passed',subject=args.subject,files=len(data['files']),
                   selected_frames=data['selected_frames'],reference_frame=data['reference_frame'],
                   retained_holdouts=data['retained_holdouts'],fit_sha256=data['fit_sha256']),flush=True)
        return
    if args.action == 'check-early':
        print(check_snapshot(root_for(args.subject)/'run',1000),flush=True)
        return
    if not args.commit:
        parser.error('--commit is required for this action.')
    if args.action == 'launch':
        raise SystemExit(launch(args.subject,args.commit))
    if args.action == 'register':
        register(args.subject,args.commit)
    else:
        plan, data = plan_for(args.subject,args.commit)
        print(dict(state='preflight_passed',subject=args.subject,files=len(data['files']),
                   selected_frames=data['selected_frames'],policy=plan['color_policy']),flush=True)


if __name__ == '__main__':
    main()
