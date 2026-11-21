"""Refine verified round-ten initializers and train one fresh SH0 model per subject."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round10 as previous
from launch_smpl_delivery_round9 import replacement_trial_proof


cold = previous.cold
rebound, need, read, pin, sha, write = (previous.rebound, previous.need, previous.read, previous.pin,
                                       previous.sha, previous.write)
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-guard-20261111')
BASE_PROJECT = previous.PROJECT
BASE = 'ed7755c6a293484c2c83fc642fde43981d88b611'
DELIVERY = previous.DELIVERY.parent.parent/'smpl-delivery-round10/receipt.json'
DELIVERY_SHA = '359da8f7873f9c6f6a2b0ea3a0f022178da027cfc5693d55b7af55bff0b5c301'
DIRECTOR = previous.DIRECTOR.with_name('round10_director_review_20261111.json')
DIRECTOR_SHA = 'fbc2671888e53092dc1fc1a3ffcd052398681b159622c46c61f08daef9e51da1'
FIT_MODULE = previous.FIT_MODULE
CHANGED = {FIT_MODULE, 'src/shuolee_portrait/official_sequence.py'}
SOURCE_FILES = previous.SOURCE_FILES
PYTHON = cold.ENV/'bin/python'
MAX_WALL_SECONDS = 3600


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round11-{subject}'


git_state = rebound(cold.git_state, PROJECT=PROJECT, BASE=BASE)
asset_pins = rebound(cold.official_pins, PROJECT=PROJECT)


def check_code(commit):
    git_state(commit)
    pointer = (PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['/usr/bin/wslpath', '-u', pointer], text=True).strip()
    changed = set(subprocess.check_output(['/usr/bin/git', '--git-dir', git_dir, '--work-tree', str(PROJECT),
        'diff', '--name-only', BASE, commit], text=True).splitlines())
    need(CHANGED <= changed and changed <= CHANGED | {'tests/test_joint_body_fit.py'},
         f'Unexpected round-eleven code paths: {sorted(changed)}')
    for name in set(SOURCE_FILES)-CHANGED:
        pin(PROJECT/name, sha(BASE_PROJECT/name))
    return {name:sha(PROJECT/name) for name in SOURCE_FILES}


def source_inventory(subject):
    pin(DELIVERY, DELIVERY_SHA)
    pin(DIRECTOR, DIRECTOR_SHA)
    review, base = read(DIRECTOR), read(DELIVERY)
    need(review['commit'] == base['commit'] == BASE and review['round'] == 10
         and review['receipt_sha256_verified'] == DELIVERY_SHA, 'Unverified round-ten baseline.')
    info = base['subjects'][subject]
    old = Path(info['result']).parent
    files = {}

    def record(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        need(expected is None or digest == expected, f'Pinned source changed: {path}')
        need(files.get(str(path), digest) == digest, f'Conflicting source digest: {path}')
        files[str(path)] = digest
        return dict(path=str(path), sha256=digest)

    result = record(old/'result.json', info['result_sha256'])
    wrapper = read(old.parent/'wrapper-terminal.json')
    need(wrapper['state'] == 'completed' and wrapper['candidate_commit'] == BASE
         and wrapper['result_sha256'] == result['sha256'], 'Round-ten terminal does not bind the original run.')
    record(old.parent/'wrapper-terminal.json')
    fit_receipt = record(old/'fit/official-fit-result.json', info['evidence']['official_fit']['sha256'])
    fit = read(Path(fit_receipt['path']))
    prior_bound = fit['source']['input_files']
    manifest = old/'data/inputs.json'
    observations = old/'data/observations.npz'
    paths = dict(
        manifest=record(manifest, prior_bound[str(manifest)]),
        observations=record(observations, prior_bound[str(observations)]),
        shared_body=record(old/'fit/shared-body.json', fit['shared_body_sha256']),
        before_selection=record(old/fit['joint_body_refinement']['before_selection_file'],
                                fit['joint_body_refinement']['before_selection_sha256']),
        prior_joint=record(old/fit['joint_body_refinement']['file'], fit['joint_body_refinement']['sha256']),
        prior_selection=record(old/'fit/selection.json', info['evidence']['selection']['sha256']),
        config=record(old/'config.json'),
        gaussian_config=record(old/'gaussian-run/config.json', info['effective_config_sha256']),
        run=record(old/'run.json'),
        initialization=record(old/'gaussian-run/initialization.json', info['evidence']['initialization']['sha256']),
        original_result=result, official_fit=fit_receipt)
    old_meta = read(old/'run.json')
    record(old_meta['video'], old_meta['video_sha256'])
    record(old.parent/'camera.json', old_meta['camera_config_sha256'])
    scope = read(Path(paths['prior_joint']['path']))
    names, held = scope['fit_frame_ids'], scope['retained_holdouts']
    rows = read(manifest)['records']
    need(len(names) == len(set(names)) == 52 and len(rows) == 56
         and {r['frame'] for r in rows} == set(names) | set(held)
         and held == info['retained_holdouts'] and set(names).isdisjoint(held),
         'Protected R10 fit/source roster differs.')
    by_name = {row['frame']:row for row in rows}
    winners = {}
    for row in rows:
        record(row['image'], row['source_sha256'])
        record(row['mask'], row['mask_sha256'])
    for name in names:
        location = old/'fit/frames'/name
        array = record(location/'winner-camera-space.npz')
        receipt = record(location/'result.json')
        metadata = read(Path(receipt['path']))
        need(metadata['frame_id'] == name and metadata['source_frame_index'] == by_name[name]['source_frame_index']
             and metadata['source_image_sha256'] == by_name[name]['source_sha256']
             and metadata['state'] == 'complete' and metadata['freeze_betas'], 'Invalid R10 initializer: '+name)
        winners[name] = dict(npz=array['path'], npz_sha256=array['sha256'],
                             result=receipt['path'], result_sha256=receipt['sha256'])
    return dict(schema='shuolee-round11-fixed-inputs-v1', subject=subject, baseline_commit=BASE,
                baseline_run=str(old), files=files, paths=paths, fit_frames=names, retained_holdouts=held,
                winners=winners)


def check(subject, commit):
    codes = check_code(commit)
    inventory = source_inventory(subject)
    assets = asset_pins()
    env = cold.clean_env()
    root = root_for(subject)
    refinement = [str(PYTHON), '-I', '-B', '-u', str(Path(__file__).resolve()), '_refine', subject, '--commit', commit]
    training = [str(PYTHON), '-B', '-u', '-m', 'shuolee_portrait.cli', 'train-fitted', '--run', str(root/'run')]
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient local disk space.')
    return dict(schema='shuolee-round11-joint-face-guard-v1', subject=subject, candidate_commit=commit, baseline_commit=BASE,
        baseline_run=inventory['baseline_run'], baseline_result_sha256=inventory['paths']['original_result']['sha256'],
        output=str(root/'run'), director_review_sha256=DIRECTOR_SHA, baseline_delivery_sha256=DELIVERY_SHA,
        runtime=cold.cpu_runtime(env), code_sha256=codes, asset_sha256=assets, wrapper_sha256=sha(Path(__file__)),
        retained_holdouts=inventory['retained_holdouts'], commands=[refinement,training],
        gaussian_steps=10000, max_wall_seconds=MAX_WALL_SECONDS), inventory


def register(subject, commit):
    plan, inventory = check(subject,commit)
    root = root_for(subject)
    root.mkdir(exist_ok=False)
    path = root/'protected-inputs.json'
    write(path,inventory)
    write(root/'preregistration.json',dict(plan,registered_at=cold.now(),protected_input_inventory_sha256=sha(path)))
    print(json.dumps(dict(state='registered', subject=subject, files=len(inventory['files']), inventory_sha256=sha(path))),
          flush=True)


def registered(subject, commit, *, full):
    root = root_for(subject)
    plan = read(root/'preregistration.json')
    inventory_path = root/'protected-inputs.json'
    pin(inventory_path,plan['protected_input_inventory_sha256'])
    inventory = read(inventory_path)
    need(plan['candidate_commit'] == commit and plan['subject'] == inventory['subject'] == subject
         and inventory['baseline_commit'] == plan['baseline_commit'] == BASE
         and sha(Path(__file__)) == plan['wrapper_sha256'], 'Registered commit or wrapper differs.')
    if full:
        current, frozen = check(subject,commit)
        need(frozen == inventory and all(plan[key] == value for key,value in current.items()),
             'Registered input identity or configuration differs.')
    else:
        for name, digest in inventory['files'].items():
            pin(Path(name),digest)
    return plan, inventory


def fit_proof(root, plan, inventory):
    import numpy as np

    run = root/'run'
    fit = read(run/'fit/official-fit-result.json')
    joint = fit['joint_body_refinement']
    for key in ('file','candidate_file','before_selection_file'):
        path = (run/joint[key]).resolve()
        need(path.is_relative_to(run/'fit'), 'Joint output path escapes fit root.')
        pin(path,joint[{'file':'sha256','candidate_file':'candidate_sha256',
                        'before_selection_file':'before_selection_sha256'}[key]])
    pin(run/joint['before_selection_file'], inventory['paths']['before_selection']['sha256'])
    metrics = read(run/joint['file'])
    guard = metrics['face_guard']
    threshold = guard['threshold_px']
    need(threshold == .25 and guard['confidence_threshold'] == .5
         and guard['reference'] == 'official_initializer' and guard['exact_reprojection_passed'] is True
         and guard['supported_frame_count'] == len(inventory['fit_frames'])
         and set(metrics['frames']) == set(inventory['fit_frames'])
         and all(region['status'] == 'unobserved' or (
             region['status'] == 'pass' and not region['failures']
             and max(region['delta']['rms_px'],region['delta']['peak_px']) <= threshold+1e-9)
             for frame in metrics['frames'].values() for region in frame['face_guard'].values()),
         'An exact exported pose failed the fixed face protection.')
    selection = fit['selection']
    names, held = inventory['fit_frames'], inventory['retained_holdouts']
    selected = selection['selected_frames']
    rows = read(run/'data/training-inputs.json')['records']
    used = [row['frame'] for row in rows]
    need(fit['state'] == 'complete' and fit['official_initializers_reused'] and not fit['new_fitting_started']
         and fit['fitted_candidate_frames'] == 52 and set(metrics['fit_frame_ids']) == set(names)
         and metrics['retained_holdouts'] == selection['retained_holdouts'] == held
         and set(used) == set(selected) <= set(names) and len(used) == len(set(used)) and 0 < len(used) <= 24
         and used[0] == selection['reference_frame'] and selection == read(run/'fit/selection.json')
         and metrics['native_K_frozen'] and metrics['shared_betas_frozen']
         and metrics['per_frame_scale_parameters'] == 0 and metrics['optimizer_updates'] == 80,
         'Joint candidate identity, actual update budget, or named fit differs.')
    for name,key in (('data/inputs.json','manifest'),('data/observations.npz','observations'),
                     ('fit/shared-body.json','shared_body')):
        pin(run/name,inventory['paths'][key]['sha256'])
    pin(PROJECT/FIT_MODULE,metrics['joint_module_sha256'])
    pin(run/'fit/fitted.npz',fit['fit_sha256'])
    pin(run/'data/training-inputs.json',fit['training_inputs_sha256'])
    with np.load(run/joint['candidate_file'],allow_pickle=False) as candidate, \
            np.load(run/'fit/fitted.npz',allow_pickle=False) as final:
        slots = [names.index(name) for name in used]
        need(candidate['frame_ids'].tolist() == names and final['frame_ids'].tolist() == used
             and np.array_equal(candidate['betas'], final['betas'])
             and np.array_equal(candidate['betas'], np.asarray(read(run/'fit/shared-body.json')['betas'],np.float32))
             and float(final['world_scale']) == 1.
             and all(np.array_equal(candidate[key][slots],final[key]) for key in
                     ('vertices_camera','joints_camera','body_pose_axis_angle','global_orient','camera_translation',
                      'K_integer_centers','observed_native_keypoints','source_frames','image_sha256','mask_sha256')),
             'The final training fit differs from the protected joint candidate geometry.')
        if guard['fallback'] is not None:
            need(guard['fallback'] in ('no_feasible_joint_checkpoint','exact_reprojection_guard'),
                 'Unknown face guard fallback.')
            for i,name in enumerate(names):
                with np.load(inventory['winners'][name]['npz'],allow_pickle=False) as initial:
                    need(all(np.array_equal(candidate[key][i],initial[key][0]) for key in
                         ('vertices_camera','joints_camera','joints_projected_native','body_pose_axis_angle',
                          'global_orient','camera_translation','pose_embedding')), 'Fallback changed official geometry.')
        else:
            need(metrics['chosen_checkpoint']['face_guard_passed'] is True
                 and not metrics['chosen_checkpoint']['face_guard_failures'],
                 'The chosen whole-sequence checkpoint failed face protection.')
    return fit, joint, metrics


def refine(subject, commit):
    plan, inv = registered(subject,commit,full=False)
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.official_sequence import refine_saved_official_sequence
    from shuolee_portrait.training import smpl_sh0_effective_config

    run = Path(plan['output'])
    (run/'data').mkdir(parents=True,exist_ok=False)
    for dest,name in (('inputs.json','manifest'),('observations.npz','observations')):
        source = inv['paths'][name]
        shutil.copyfile(source['path'],run/'data'/dest)
        pin(run/'data'/dest,source['sha256'])
    rt.configure(cold.ASSETS,run)
    base_cfg = read(Path(inv['paths']['config']['path']))
    base_gaussian = read(Path(inv['paths']['gaussian_config']['path']))
    profile = smpl_sh0_effective_config()
    need(all(base_gaussian[key] == value for key,value in profile.items()
             if key not in ('initial_points','image_cache_frames')), 'R10 Gaussian profile differs.')
    cfg = dict(base_cfg, **profile)
    need('frozen_surface' not in cfg, 'The R10 clothing preheat requires a fresh offset.')
    meta = read(Path(inv['paths']['run']['path']))
    rt.save(run/'config.json',cfg)
    rt.save(run/'run.json',dict(meta,created=rt.now(),from_scratch=False,official_initializers_reused=True,
        source_reuse_sha256=plan['protected_input_inventory_sha256']))
    with gpu_lock():
        refine_saved_official_sequence(frames_path=run/'data/inputs.json',observations_path=run/'data/observations.npz',
            root=run,official_root=cold.OFFICIAL,assets=cold.ASSETS,camera_length_contract=cfg['camera_length_contract'],
            capture_layout=cfg['capture_layout'],shared_body=inv['paths']['shared_body'],winners=inv['winners'])
    fit,joint,_ = fit_proof(root_for(subject),plan,inv)
    print(dict(state='joint_fit_verified',subject=subject,selected=fit['frames'],joint_refinement_sha256=joint['sha256']),flush=True)


def finish(root, plan, inv, rc, phase, elapsed):
    run = root/'run'
    receipt = dict(state='failed' if rc else 'incomplete',subject=plan['subject'],returncode=rc,stage=phase,
        candidate_commit=plan['candidate_commit'],baseline_commit=BASE, command_to_end_seconds=elapsed,
        preregistration_sha256=sha(root/'preregistration.json'),
        protected_input_inventory_sha256=plan['protected_input_inventory_sha256'],quality_accepted=False)
    if rc == 0:
        try:
            for path,digest in inv['files'].items():
                pin(Path(path),digest)
            fit,joint,metrics = fit_proof(root,plan,inv)
            selection = fit['selection']
            result = read(run/'result.json')
            integration = read(run/'gaussian-run/integration.json')
            cfg = read(run/'gaussian-run/config.json')
            trained = read(run/'gaussian-run/result.json')
            old_cfg = read(Path(inv['paths']['gaussian_config']['path']))
            names = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
            keys = ('optimization','unified_main','alpha_loss_weight','straight_rgb_alpha_floor','sampling_policy',
                    'snapshot_steps','opacity_reset_steps','sh_degree','shared_clothing_offset_lr')
            need(all(cfg[k] == old_cfg[k] for k in keys)
                 and integration['optimizer_updates'] == result['steps'] == plan['gaussian_steps'] == 10000
                 and integration['optimizer_initial_state_entries'] == 0 and integration['no_prior_gaussian_loaded']
                 and integration['initial_points'] == 20908 and integration['frozen_surface_sha256'] is None
                 and integration['raw_native_rgb_target'] and not integration['prior_render_used_as_target']
                 and len(trained['per_frame_updates']) == len(names) and min(trained['per_frame_updates']) > 0
                 and sum(trained['per_frame_updates']) == 10000,
                 'Gaussian behavior differs from the fixed round-ten profile.')
            pin(run/'gaussian-run/initialization.json',inv['paths']['initialization']['sha256'])
            status = read(run/'status.json')
            need(status['state'] == 'completed' and status['static_export_passed']
                 and result['state'] == 'completed' and result['format_passed'] and result['gaussian_from_scratch']
                 and result['frames'] == len(names) and result['reference_frame_id'] == selection['reference_frame']
                 and result['final_snapshot'] == 'step-010000', 'Incomplete public training and export.')
            for key,digest in (('ply','sha256'),('scene','scene_sha256'),('bound_replay','bound_replay_sha256'),
                               ('bound_panels','bound_panels_sha256')):
                target = (run/result[key]).resolve()
                need(target.is_relative_to(run), 'Exported artifact escapes its run.')
                pin(target,result[digest])
            with zipfile.ZipFile(run/result['scene']) as bundle:
                with bundle.open('cameras.json') as camera, bundle.open('point_cloud/iteration_10000/point_cloud.ply') as ply:
                    need(cold.stream_sha(camera) == sha(run/result['camera']) and cold.stream_sha(ply) == result['sha256'],
                         'Bundled camera or world PLY differs.')
            replay, panels = read(run/result['bound_replay']), read(run/result['bound_panels'])
            need([row['frame'] for row in replay['records']] == [row['frame'] for row in panels['panels']] == names
                 and replay['retained_holdouts'] == inv['retained_holdouts'], 'Bound replay identity differs.')
            scale = run/'gaussian-run/world-scale.json'
            pin(scale,integration['world_scale']['sha256'])
            world = read(scale)
            need(world['terminal'] == integration['world_scale']['terminal']
                 and world['terminal']['all_poses_max']['all']['over_limit'] == 0, 'World-scale audit differs.')
            for event in world['splits']:
                support, options = event['render_support'], world['config']['render_support']
                replacement_trial_proof(support['coverage_trial'],event['needle']['support_replaced'],names,options)
                replacement_trial_proof(support['near_limit_trial'],event['split_parents'],names,options)
            for name,digest in plan['code_sha256'].items():
                pin(PROJECT/name,digest)
            receipt.update(state='completed',result_sha256=sha(run/'result.json'),points=result['points'],
                reference_frame=result['reference_frame_id'],selected_frames=selection['selected_frames'],
                coverage=selection['coverage'],joint_refinement_sha256=joint['sha256'],
                joint_candidate_sha256=joint['candidate_sha256'],world_scale_sha256=sha(scale),
                fit_method=fit['execution_kind'],face_guard=metrics['face_guard'],fresh_gaussian=True,
                original_video_to_export=False,
                native_inputs_reused=True,reference_mesh_frozen_from_prior=False)
        except Exception as error:
            receipt['validation_error'] = str(error)
    write(root/'wrapper-terminal.json',receipt)
    print(json.dumps(dict(state=receipt['state'],subject=plan['subject'],validation_error=receipt.get('validation_error'))),
          flush=True)
    return 0 if receipt['state']=='completed' else rc or 1


def launch(subject, commit):
    root = root_for(subject)
    plan, inv = registered(subject,commit,full=True)
    with cold.LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subject == '006':
            earlier = root_for('016')
            terminal = read(earlier/'wrapper-terminal.json')
            need(terminal['state'] == 'completed' and terminal['candidate_commit'] == commit,
                 'Complete the verified 016 run before 006.')
            pin(earlier/'run/result.json',terminal['result_sha256'])
        need(not any((root/name).exists() for name in ('run','launch.json','entry.log','wrapper-terminal.json')),
             'Fresh round output required.')
        write(root/'launch.json',dict(subject=subject,commands=plan['commands'],started=cold.now(),
                                     preregistration_sha256=sha(root/'preregistration.json')))
        start = time.monotonic()
        with (root/'entry.log').open('xb') as log:
            for phase,command in zip(('joint_fit','train_fitted'),plan['commands'],strict=True):
                budget = max(1,MAX_WALL_SECONDS-int(time.monotonic()-start))
                rc = subprocess.run(['/usr/bin/timeout','--kill-after=15s',f'{budget}s',*command],
                    env=cold.clean_env(),cwd=PROJECT/'src',stdout=log,stderr=subprocess.STDOUT).returncode
                if rc:
                    break
        return finish(root,plan,inv,rc,phase,time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','register','launch','_refine'))
    parser.add_argument('subject',choices=tuple(cold.PREVIOUS))
    parser.add_argument('--commit',required=True)
    args = parser.parse_args()
    if args.action == 'launch':
        raise SystemExit(launch(args.subject,args.commit))
    if args.action == '_refine':
        refine(args.subject,args.commit)
    elif args.action == 'register':
        register(args.subject,args.commit)
    else:
        plan,inv = check(args.subject,args.commit)
        print(dict(state='preflight_passed',subject=args.subject,files=len(inv['files']),
                   retained_holdouts=plan['retained_holdouts']),flush=True)


if __name__ == '__main__':
    main()
