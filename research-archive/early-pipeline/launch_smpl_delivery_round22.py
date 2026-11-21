"""Stage, launch and independently verify the two R22 fixed-surface runs."""

import argparse
from collections import Counter
import fcntl
import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import launch_smpl_delivery_round21 as prior
from launch_smpl_delivery_round10 import rebound


cold, r20 = prior.cold, prior.previous
need, read, pin, sha, write = prior.need, prior.read, prior.pin, prior.sha, prior.write
PYTHON = prior.PYTHON
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-splat-distance-20261111')
PARENT = 'ba42474f6c2d1a8826f8bab21e98dcb8752d1bf2'
SCHEMA = 'shuolee-round22-frozen-garment-surface-distribution-v1'
STEPS, MIDPOINT = prior.STEPS, prior.MIDPOINT
DELIVERY = prior.OUT/'smpl-delivery-round22'
OLD_DELIVERY = prior.OUT/'smpl-delivery-round21'
REVIEW_SHA = '84a94651d0f2709f4e0a6e9844913b26419754097869a14e5254df59fef12f20'
DIRECTOR = prior.DIRECTOR.with_name('round21_director_review_20261111.json')
DIRECTOR_SHA = '7dda80dafc5b23ceacab4a334269526f1e3d614e6523917861a7e9f2d7449f3b'
POLICY_SHA = 'ede5a8a1a92545169314aacf44507f9cea265ba6a298e1fee8259892ab4591c9'
TERMINALS = {
    '016': ('wrapper-terminal-frozen-policy-verified.json',
            '35bb87c349650d1b8a92c0dc6a1c8c463f68ac4ee9bfbc4c2f134a38f490ba88'),
    '006': ('wrapper-terminal.json',
            'aab2281039d0e9c90eb7592c3773b515839186684535bca643c66e205820092f'),
}
GARMENT = {
    'mesh': ('fit/r21-shared-clothing-offset.npz', 'shared-clothing-offset.npz'),
    'clothing_support': ('fit/r21-shared-clothing-support.npz', 'shared-clothing-support.npz'),
    'clothing_receipt': ('fit/r21-clothing-surface.json', 'clothing-surface.json'),
}


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round22-{subject}'


def frozen_policy():
    frozen = DELIVERY/'input-audit/frozen-r22-opacity-policy.json'
    if not frozen.is_file():
        pin(prior.POLICY_PATH, POLICY_SHA)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        with frozen.open('xb') as stream:
            stream.write(prior.POLICY_PATH.read_bytes())
    pin(frozen, POLICY_SHA)
    need(read(frozen)['user_acceptance']['opacity']['status'] == 'accepted_by_user',
         'The registered R22 policy must preserve the user-accepted opacity.')
    return frozen


def source_inventory(subject):
    root = prior.root_for(subject)
    run = root/'run'
    name, digest = TERMINALS[subject]
    terminal_path = root/name
    pin(terminal_path, digest)
    terminal = read(terminal_path)
    archive_path, plan_path = root/'protected-inputs.json', root/'preregistration.json'
    pin(archive_path, terminal['protected_input_inventory_sha256'])
    pin(plan_path, terminal['preregistration_sha256'])
    archive, plan = read(archive_path), read(plan_path)
    prior.verify_frozen_fit(run, archive)
    inventory = dict(archive, schema=SCHEMA, baseline_commit=PARENT, baseline_run=str(run))
    files = inventory['files'] = dict(archive['files'])

    def record(path, digest=None):
        path = Path(path)
        digest = digest or sha(path)
        pin(path, digest)
        need(files.setdefault(str(path), digest) == digest, f'Archive identities disagree at {path}.')
        return dict(path=str(path), sha256=digest)

    record(terminal_path, TERMINALS[subject][1])
    record(archive_path, terminal['protected_input_inventory_sha256'])
    record(plan_path, terminal['preregistration_sha256'])
    record(DIRECTOR, DIRECTOR_SHA)
    director = read(DIRECTOR)
    review_path = OLD_DELIVERY/'review-receipt.json'
    record(review_path, REVIEW_SHA)
    review = read(review_path)
    need(director['round'] == 21 and director['receipt_sha256'] == REVIEW_SHA
         and review['commit'] == director['commit'] == terminal['candidate_commit'] == PARENT
         and terminal['state'] == 'completed' and review['quality_accepted'] is False,
         'The R21 candidate, independent review and director review disagree.')
    record(frozen_policy(), POLICY_SHA)
    record(run/'result.json', terminal['result_sha256'])
    staged = inventory['staged_files'] = dict(archive['staged_files'])
    # Inherit the original entry configuration and the exact corrected pose assets.
    # The old R21 prefit garment is superseded by its accepted active garment.
    del staged[archive['frozen_surface']['file']]
    for path, item in staged.items():
        record(run/path, item['sha256'])
    guide_dir = run/'gaussian-run'
    guide_path = guide_dir/GARMENT['clothing_receipt'][1]
    guide = read(guide_path)
    duration = terminal['duration_comparison']
    need(guide['state'] == 'accepted_visible_surface'
         and guide['active_clothing_surface'] == GARMENT['mesh'][1]
         and len(terminal['selected_frames']) == 24 and len(terminal['face_supplement']['frames']) == 4
         and len(terminal['retained_holdouts']) == 4
         and len(set(terminal['selected_frames'] + terminal['face_supplement']['frames']
                     + terminal['retained_holdouts'])) == 32,
         'The accepted R21 mesh and the frozen disjoint 24+4+4 observations are required.')
    hashes = {
        'mesh': guide['active_mesh_sha256'],
        'clothing_support': guide['support_file_sha256'],
        'clothing_receipt': duration['terminal']['surface_attachment']['clothing_guide_sha256'],
    }
    descriptors = {}
    for key, (destination, name) in GARMENT.items():
        staged[destination] = record(guide_dir/name, hashes[key])
        descriptors[key] = dict(file=destination, sha256=hashes[key])
    for step, branch in ((MIDPOINT, run/'duration-10000'), (STEPS, run)):
        result = read(branch/'result.json')
        record(branch/'result.json', duration['intermediate' if step == MIDPOINT else 'terminal']['result_sha256'])
        record(branch/result['scene'], result['scene_sha256'])
        record(branch/result['ply'], result['sha256'])
        record(branch/result['camera'])
        record(branch/f'gaussian-run/step-{step:06d}/reference-mesh.npz', hashes['mesh'])
        record(run/f'gaussian-run/raw-step-{step:06d}/reference-mesh.npz', hashes['mesh'])
    view = prior.VIEW_DIR/f'head-support-views-{subject}.json'
    prior._pin_view_protocol(view, prior.VIEW_SHA[subject], subject, inventory)
    record(view, prior.VIEW_SHA[subject])
    profiles = dict(archive['profiles'])
    for name in ('gaussian-run/config.json', 'gaussian-run/result.json', 'gaussian-run/integration.json',
                 'gaussian-run/training-cameras.json', 'gaussian-run/initialization.json', 'result.json'):
        profiles[name] = record(run/name)
    return dict(inventory, profiles=profiles, frozen_surface=descriptors.pop('mesh'),
                splat_surface=descriptors, r21_terminal=record(terminal_path, TERMINALS[subject][1]),
                r21_plan=record(plan_path, terminal['preregistration_sha256']),
                r21_cpu_fit_preflight=plan['cpu_fit_preflight'],
                views=record(view, prior.VIEW_SHA[subject]))


def verify_staged(run, inventory):
    prior.verify_frozen_fit(run, inventory)
    cfg = read(run/'config.json')
    need(all(cfg[key] == inventory[key] for key in ('frozen_surface', 'splat_surface')),
         'The staged configuration lost its accepted R21 clothing descriptors.')
    pin(run/'data/head-support-views.json', inventory['views']['sha256'])


def stage(subject):
    root = root_for(subject)
    inventory = source_inventory(subject)
    root.mkdir(exist_ok=True)
    run = root/'run'
    need(not any((root/name).exists() for name in ('preregistration.json', 'launch.json', 'entry.log')),
         'Preserve the registered or launched R22 run.')
    if not run.exists():
        r20.stage(root, inventory)
    prior.verify_frozen_fit(run, inventory)
    cfg = read(run/'config.json')
    expected = dict(cfg, frozen_surface=inventory['frozen_surface'], splat_surface=inventory['splat_surface'])
    if cfg != expected:
        temporary = run/'config.r22.tmp'
        write(temporary, expected)
        temporary.replace(run/'config.json')
    view = run/'data/head-support-views.json'
    if not view.exists():
        shutil.copyfile(inventory['views']['path'], view)
    verify_staged(run, inventory)
    cpu = rebound(prior.cpu_preflight, PROJECT=PROJECT)(run, inventory)
    old = inventory['r21_cpu_fit_preflight']
    need(all(cpu[key] == value for key, value in old.items() if not key.startswith('frozen_surface')),
         'The actual CPU loader changed the frozen R21 sampler or auxiliary admission.')
    write(root/'protected-inputs.json', inventory)
    write(root/'cpu-preflight.json', cpu)
    print(dict(state='staged', subject=subject, fit=inventory['fit_sha256'],
               garment=inventory['frozen_surface']['sha256'], views=inventory['views']['sha256'],
               inventory_sha256=sha(root/'protected-inputs.json')), flush=True)


def code_identity(commit, baseline_plan):
    need(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'Use the exact full committed R22 identity.')
    pointer = (PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['/usr/bin/wslpath', '-u', pointer], text=True).strip()
    command = ['/usr/bin/git', '--no-optional-locks', '-c', 'core.fileMode=false',
               '--git-dir', git_dir, '--work-tree', str(PROJECT)]
    def git(*args):
        return subprocess.check_output([*command, *args], cwd=PROJECT, text=True).strip()

    need(git('rev-parse', 'HEAD') == commit and not git('status', '--porcelain', '--untracked-files=all')
         and git('rev-parse', 'refs/tags/smpl-stable-20261111^{commit}') == r20.STABLE_TAG_COMMIT,
         'The clean candidate worktree or fixed stable release tag differs.')
    subprocess.run([*command, 'merge-base', '--is-ancestor', PARENT, commit], cwd=PROJECT, check=True)
    changed = set(git('diff', '--name-only', PARENT, commit).splitlines())
    need({'src/shuolee_portrait/training.py', 'src/shuolee_portrait/surface_attachment.py',
          'src/shuolee_portrait/surface_densification.py'} <= changed
         and all(name.startswith(('src/shuolee_portrait/', 'tests/')) for name in changed),
         f'Unexpected R22 source scope: {sorted(changed)}')
    for name, digest in baseline_plan['code_sha256'].items():
        if name not in changed:
            source, candidate = prior.PROJECT/name, PROJECT/name
            pin(source, digest)
            need(source.read_bytes().replace(b'\r\n', b'\n') == candidate.read_bytes().replace(b'\r\n', b'\n'),
                 f'Unchanged source bytes differ beyond checkout line endings: {name}.')
    return {name: sha(PROJECT/name) for name in sorted(set(baseline_plan['code_sha256']) | changed)}


def register(subject, commit, cuda_proof, cuda_proof_sha):
    root = root_for(subject)
    run = root/'run'
    protected = root/'protected-inputs.json'
    inventory = read(protected)
    need(inventory == source_inventory(subject), 'R22 source bytes changed after CPU staging.')
    verify_staged(run, inventory)
    old = read(Path(inventory['r21_plan']['path']))
    code = code_identity(commit, old)
    runtime = cold.cpu_runtime(cold.clean_env())
    need(runtime == old['runtime'], 'The dedicated Gaussian runtime differs from R21.')
    for name, digest in old['asset_sha256'].items():
        pin(Path(name), digest)
    verify_cuda_proof(cuda_proof, cuda_proof_sha, code)
    need(min(shutil.disk_usage(cold.LAB).free, shutil.disk_usage('/mnt/c').free) > 8*1024**3,
         'Insufficient space for paired snapshots, packages and render evidence.')
    command = [str(PYTHON), '-B', '-u', '-m', 'shuolee_portrait.cli', 'train-fitted', '--run', str(run)]
    plan = dict(schema=SCHEMA, subject=subject, candidate_commit=commit, baseline_commit=PARENT,
                stable_git_release_commit=r20.STABLE_TAG_COMMIT, runtime=runtime, code_sha256=code,
                asset_sha256=old['asset_sha256'], output=str(run), commands=[command],
                cpu_fit_preflight=read(root/'cpu-preflight.json'), selected_frames=inventory['selected_frames'],
                retained_holdouts=inventory['retained_holdouts'], comparison_steps=[MIDPOINT, STEPS],
                pitch_camera_manifest=inventory['views'], quality_policy_sha256=POLICY_SHA,
                protected_input_inventory_sha256=sha(protected), cuda_proof=dict(path=str(cuda_proof), sha256=cuda_proof_sha),
                head_fit=inventory['head_fit'], staged_config_sha256=sha(run/'config.json'),
                max_wall_seconds=r20.MAX_WALL_SECONDS)
    with (root/'preregistration.json').open('x', encoding='utf-8') as stream:
        import json
        json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(dict(state='registered', subject=subject, commit=commit, sha256=sha(root/'preregistration.json')), flush=True)


def verify_cuda_proof(path, digest, code):
    pin(path, digest)
    proof = read(path)
    tested = proof['source_sha256']
    need(proof['state'] == 'passed' and tested
         and all(code.get(name) == value for name, value in tested.items()),
         'The registered CUDA smoke proof must match the tested candidate source bytes.')


def _branch_effective(run, inventory, step):
    effective = read(run/'gaussian-run/config.json')
    baseline = read(Path(inventory['profiles']['gaussian-run/config.json']['path']))
    for key in ('training_source_bins', 'seed', 'opacity_reset_steps', 'unified_main', 'training_duration_policy'):
        need(effective[key] == baseline[key], f'The fixed R21 schedule changed at {key}.')
    need(effective['iterations'] == effective['optimization']['iterations'] == step
         and dict(effective['optimization'], iterations=STEPS) == baseline['optimization']
         and effective['frozen_surface_sha256'] == inventory['frozen_surface']['sha256'],
         'The paired branch changed its fixed optimizer or R21 shared surface.')
    verify_staged(run, inventory)
    return effective


def verify_raw(run, endpoint, step, plan, inventory, optimization, bins):
    import numpy as np
    import torch

    path = run/f'gaussian-run/raw-step-{step:06d}'
    need(endpoint['step'] == step and run/endpoint['raw'] == path, 'Raw checkpoint path differs.')
    pin(path/'state.json', endpoint['raw_receipt_sha256'])
    meta = read(path/'state.json')
    need(meta['state_sha256'] == endpoint['raw_state_sha256'], 'The raw state hash differs.')
    pin(path/'state.pt', meta['state_sha256'])
    blob = torch.load(path/'state.pt', map_location='cpu', weights_only=True)
    need(set(blob) == {'training', 'state', 'sampled_rows', 'optimization', 'surface_distribution'}
         and blob['training'] == {key: value for key, value in meta.items() if key not in ('state_sha256', 'files_sha256')}
         and blob['optimization'] == optimization
         and meta['main_optimizer_updates'] == meta['step'] == meta['scheduler_step'] == step
         and meta['schedule_steps'] == STEPS and meta['seed'] == 0
         and meta['reference_frame_id'] == inventory['reference_frame'],
         'The actual raw optimizer/sampler checkpoint differs from its saved identity.')
    draws = bytes(blob['sampled_rows'].tolist())
    cpu = plan['cpu_fit_preflight']
    expected = cpu if step == MIDPOINT else cpu['continuous_18000']
    counts, by_bin = Counter(draws), Counter(str(bins[index]) for index in draws)
    digest = hashlib.sha256(draws).hexdigest()
    prefix = hashlib.sha256(draws[:MIDPOINT]).hexdigest()
    need(len(draws) == step and digest == meta['source_draw_sha256'] == expected['expected_sampling_order_sha256']
         and prefix == meta['source_draw_prefix_10000_sha256'] == cpu['expected_sampling_order_sha256']
         and [counts[index] for index in range(len(bins))] == meta['per_frame_updates'] == expected['expected_per_frame_updates']
         and dict(by_bin) == meta['source_bin_draws'] == expected['expected_source_bin_draws'],
         'The actual original-only sampler differs from the registered uninterrupted sequence.')
    state = blob['state']
    groups = dict(xyz='_xyz', scaling='_scaling', rotation='_rotation', opacity='_opacity',
                  f_dc='_features_dc', f_rest='_features_rest')
    need(state['schema'] == 'shuolee-duration-checkpoint/1'
         and set(state['optimizer']['groups']) == set(groups)
         and state['color_projection']['optimizer_steps']['value'] == step
         and state['fixed']['num_timesteps'] == 24 and state['fixed']['max_sh_degree'] == 0,
         'The optimizer state or DC projection counter differs from the original source-only training.')
    for group, key in groups.items():
        value = state['parameters'][key]['value']['tensor']
        item = state['optimizer']['states'][group]
        updates = int(item['step']['tensor']) if item is not None else 0
        lr = state['optimizer']['groups'][group]['lr']['value']
        need(len(value) == meta['points'] and bool(torch.isfinite(value).all())
             and updates == meta['optimizer_group_steps'][group] and lr == meta['optimizer_group_lrs'][group]
             and (group == 'f_rest' or updates == step),
             f'The actual {group} Adam rows, schedule or update counter differs.')
        if item is not None:
            for field in ('exp_avg', 'exp_avg_sq'):
                moment = item[field]['tensor']
                need(moment.shape == value.shape and bool(torch.isfinite(moment).all()),
                     f'Invalid actual {group} optimizer moment rows.')
    t = step/STEPS
    scale = state['model']['spatial_lr_scale']['value']
    lr = math.exp((1-t)*math.log(optimization['position_lr_init']*scale)
                  +t*math.log(optimization['position_lr_final']*scale))
    need(math.isclose(meta['optimizer_group_lrs']['xyz'], lr, rel_tol=1e-12),
         'The actual XYZ LR differs from the single registered 18k schedule.')
    bindings = state['model']['binding']['tensor'].to(torch.int64)
    owners = state['model']['binding_counter']['tensor'].to(torch.int64)
    ids = state['model']['densification_ids']['tensor'].to(torch.int64)
    need(len(bindings) == len(ids) == meta['points'] and bool((ids[1:] > ids[:-1]).all())
         and bool((owners > 0).all()) and torch.equal(torch.bincount(bindings, minlength=len(owners)), owners),
         'The bound topology, parent counts or stable identities differ.')
    offset = state['parameters']['static_offset']
    with np.load(run/inventory['frozen_surface']['file'], allow_pickle=False) as garment:
        need(not offset['requires_grad'] and np.array_equal(offset['value']['tensor'].numpy(), garment['static_offset']),
             'The actual frozen optimizer offset differs from the accepted R21 bytes.')
    need(set(meta['files_sha256']) == {'world.ply', 'reference-mesh.npz', 'native-ply-parity.json',
                                      'reference-render.png', 'attachment.json', 'skin-pairs.npz'}
         and meta['files_sha256']['reference-mesh.npz'] == inventory['frozen_surface']['sha256'],
         'The raw checkpoint lost its frozen mesh or required native skin/covariance evidence.')
    for name, value in meta['files_sha256'].items():
        pin(path/name, value)
    extra = blob['surface_distribution']
    need(extra['report'] == read(path/'attachment.json')['support'],
         'Actual distribution coverage differs between the raw checkpoint and attachment audit.')
    split = read(run/'gaussian-run/splat-surface-splits.json')
    need(extra['split_events'] == split['events'] and extra['attempted_stable_ids'] == split['attempted_stable_ids'],
         'The distribution split lifecycle differs between the raw snapshot and final audit.')
    portable_path = run/endpoint['raw_portable_result']
    pin(portable_path, endpoint['raw_portable_result_sha256'])
    portable = read(portable_path)
    need(portable_path == path/'portable-result.json' and portable['step'] == step and portable['format_passed']
         and portable['source_world_sha256'] == meta['files_sha256']['world.ply'] and portable['points'] == meta['points'],
         'The raw portable result does not originate from the unchanged endpoint.')
    for name, key in (('ply', 'sha256'), ('camera', 'camera_sha256'), ('scene', 'scene_sha256')):
        target = (path/portable[name]).resolve(strict=True)
        need(target.is_relative_to(path), 'Raw portable output escaped the run.')
        pin(target, portable[key])
    pin(run/endpoint['result'], endpoint['result_sha256'])
    parity = r20._snapshot_parity(path, inventory['reference_frame'], run)
    return dict(raw=str(path.relative_to(run)), state_sha256=meta['state_sha256'], step=step, points=meta['points'],
                files_sha256=meta['files_sha256'], scheduler_step=meta['scheduler_step'], schedule_steps=STEPS,
                optimizer_group_steps=meta['optimizer_group_steps'], optimizer_group_lrs=meta['optimizer_group_lrs'],
                source_draw_sha256=digest, source_draw_prefix_10000_sha256=prefix,
                postwarmup_source_draws=[draws[2400:].count(index) for index in range(len(bins))],
                native_ply_parity=parity, raw_portable_result_sha256=endpoint['raw_portable_result_sha256'],
                main_seconds=meta['main_seconds'],
                binding_sha256=hashlib.sha256(bindings.numpy().tobytes()).hexdigest(),
                stable_id_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                frozen_offset_sha256=hashlib.sha256(offset['value']['tensor'].numpy().tobytes()).hexdigest()), draws


def verify_protection(run, integration, trained, effective, inventory, face_frames, step):
    out = run/'gaussian-run'
    final = out/f'step-{step:06d}'
    main = out/f'step-{step:06d}-main'
    desc = integration['protected_face']
    audit_path = out/'protected-face/optimization.json'
    endpoint_path = out/'protected-face/main-endpoint.json'
    need(desc == trained['protected_face'] and desc['file'] == 'protected-face/optimization.json'
         and desc['main_endpoint'] == 'protected-face/main-endpoint.json'
         and desc['main_snapshot'] == trained['main_snapshot'] == main.name,
         'The paired endpoint lost its original-only raw proof.')
    pin(audit_path, desc['sha256'])
    pin(endpoint_path, desc['main_endpoint_sha256'])
    proof, endpoint = read(audit_path), read(endpoint_path)
    need(proof['state'] == desc['state'] == 'skipped' and proof['reason'] == 'surface_geometry_only'
         and all(desc[key] == proof[key] == 0 for key in ('attempted_updates', 'committed_updates'))
         and desc['committed_auxiliary_frame_ids'] == []
         and endpoint['state'] == 'saved' and endpoint['snapshot'] == main.name
         and endpoint['main_optimizer_updates'] == step and endpoint['supplemental_updates'] == 0,
         'An extra appearance or opacity optimizer was activated.')
    for name, digest in endpoint['files_sha256'].items():
        pin(main/name, digest)
    identity = r20.verify_mesh_and_bound_identity(run, main, final, ('f_dc', 'opacity'), False)
    return dict(desc, identity=identity, main_native_ply_parity=r20._snapshot_parity(
        main, inventory['reference_frame'], run), baseline_views=24+len(face_frames),
        committed_frame_draws=dict.fromkeys(face_frames, 0), unprotected_control_updates=0)


def verify_split_lifecycle(document, frames, bins, *, completed=True):
    policy = document['policy']
    events = document['events']
    steps = [event['step'] for event in events]
    need(steps == policy['steps'][:len(events)] and (not completed or steps == policy['steps'])
         and max(policy['steps']) < MIDPOINT, 'Split attempts departed from the bounded preregistered schedule.')
    attempted, retained_children = set(), set()
    commits = updates = 0
    for event in events:
        ids = event['selected_parent_ids']
        trials = event['trials']
        need(event['schema'] == 'shuolee-surface-densification/1' and event['policy'] == policy
             and list(zip(frames, bins, strict=True)) == [
                 (view['frame'], view['source_bin']) for view in event['views']]
             and event['attempted_parents'] == len(ids) == len(set(ids)) == len(trials)
             <= min(policy['parent_budget_per_step'],
                    event['candidates_above_threshold']-event['previously_attempted_candidates'])
             and event['candidates_above_threshold'] <= event['skin_rows_with_repeated_source_support']
             and attempted.isdisjoint(ids),
             'Surface densification changed frozen source evidence, candidate budget or stable parent identity.')
        kept = sum(trial['outcome'] == 'committed' for trial in trials)
        total_updates = sum(trial['local_optimizer_updates'] for trial in trials)
        need(event['committed_parents'] == kept and event['rolled_back_parents'] == len(trials)-kept
             and event['rows_after'] == event['rows_before']+kept
             and event['local_optimizer_updates'] == total_updates,
             'The two-child replacement counts do not correspond to the actual native decisions.')
        for parent, trial in zip(ids, trials, strict=True):
            work = trial['local_optimizer_updates']
            need(trial['parent_id'] == parent
                 and 0 <= work <= policy['local_cycles']*len(frames)
                 and trial['skin_raster_source_bins'] >= policy['min_source_bins']
                 and trial['parent_expected_distance_m'] > policy['expected_distance_threshold_m'],
                 'A local split did not originate from the eligible folded-normal distance budget.')
            if trial['outcome'] == 'rolled_back':
                need(not trial['child_ids'] and 'transaction' not in trial,
                     'A rejected split changed live topology.')
                continue
            child, geometry, transaction = trial['child_ids'], trial['geometry'], trial['transaction']
            need(trial['outcome'] == 'committed' and work > 0 and trial['native_gate']['accepted'] is True
                 and trial['contributing_skin_source_bins'] >= policy['min_source_bins']
                 and geometry['expected_distance_drop_m'] >= geometry['required_drop_m']
                 and transaction['parent_ids'] == [parent] and transaction['child_ids'] == child
                 and len(child) == len(set(child)) == 2 and parent not in child
                 and retained_children.isdisjoint(child),
                 'A retained split lacks verified distribution gain, native gate or exact two-child commit.')
            for w, x, y, z in geometry['child_rotation_local_wxyz']:
                need(abs(x) < 2e-5 and abs(z) < 2e-5 and abs(w*w+x*x+y*y+z*z-1.) < 2e-5,
                     'A retained child does not preserve the bound triangle normal.')
            retained_children.update(child)
        attempted.update(ids)
        commits += kept
        updates += total_updates
    need(sorted(attempted) == document['attempted_stable_ids'], 'The split attempt inventory differs.')
    return dict(steps=steps, attempted_parents=len(attempted), committed_parents=commits,
                rolled_back_parents=len(attempted)-commits, local_optimizer_updates=updates)


def verify_distribution_report(proof, step):
    views = proof['views']
    coverage = proof['coverage']
    eligible = sum(view['supported_row_draws'] for view in views)
    nonsheet = sum(view['supported_row_draws']-view['flattened_row_draws'] for view in views)
    need(proof['schema'] == 'shuolee-visible-skin-surface-attachment/2'
         and proof['config']['uses_sheet_gate_for_distribution'] is False
         and eligible == coverage['eligible_row_draws'] == coverage['distribution_row_draws'] > 0
         and nonsheet == coverage['distribution_nonsheet_row_draws'] > 0 and coverage['fraction'] == 1.
         and all(0 <= view['flattened_row_draws'] <= view['distribution_row_draws'] == view['supported_row_draws']
                 and view['distribution_nonsheet_row_draws'] == view['supported_row_draws']-view['flattened_row_draws']
                 for view in views), 'The actual distribution objective omitted eligible native skin rows.')
    steps = proof['gradient_update_audit']['steps']
    recorded = [item for item in steps if item['state'] == 'recorded']
    fields = ('position', 'log_scale', 'quaternion')
    need(bool(recorded) and [item['step'] for item in steps] == sorted({item['step'] for item in steps})
         and all(2400 < item['step'] <= step for item in steps)
         and all(set(item['parameters']) == set(fields) for item in recorded)
         and all(sum(item['parameters'][field]['prior_gradient_rows'] for item in recorded) > 0
                 and sum(item['parameters'][field]['updated_rows_with_prior_gradient'] for item in recorded) > 0
                 for field in fields),
         'Sampled native backward and actual Adam steps did not exercise all three geometry parameters.')
    return dict(coverage, audited_steps=len(steps), recorded_steps=len(recorded))


def verify_surface(run, plan, inventory, integration, trained, effective, step, raw, raw_info):
    out = run/'gaussian-run'
    desc = integration['surface_attachment']
    path = out/desc['file']
    pin(path, desc['sha256'])
    report = read(path)
    raw_report = read(raw/'attachment.json')
    initial = read(out/'surface-attachment-initial.json')
    need(report['schema'] == 'shuolee-surface-attachment-v1' and report['step'] == step
         and report['policy'] == effective['surface_attachment_policy']
         and report['policy']['postprocess'] == 'render and export only'
         and report['training'] == raw_report and raw_report['initial'] == initial
         and report['reference_mesh_sha256'] == inventory['frozen_surface']['sha256']
         and report['postprocessing_changes_geometry'] is False,
         'The actual endpoint attachment scope or raw audit differs.')
    views = raw_report['support']['views']
    frames = inventory['selected_frames']
    need([row['frame'] for row in views] == frames
         and [row['angular_bin'] for row in views] == effective['training_source_bins']
         and [row['loss_calls'] for row in views] == raw_info['postwarmup_source_draws'],
         'Native distribution work used a different skin support or draw sequence.')
    coverage = verify_distribution_report(raw_report['support'], step)
    for item in (initial, raw_report, report):
        need([row['frame'] for row in item['skin']['views']] == frames,
             'The measured skin roster differs from the frozen original cameras.')
        prior._geometry_measurements(item['skin']['views'], 'skin')
        prior._geometry_measurements([item['clothing']], 'clothing')
    guide = report['clothing_guide']
    guide_path = out/guide['file']
    pin(guide_path, guide['sha256'])
    cloth = read(guide_path)
    expected = inventory['splat_surface']
    need(cloth['reused_frozen_surface'] is True and cloth['optimizer_updates'] == 0
         and cloth['state'] == 'accepted_visible_surface' and cloth['changed_pose_parameters'] == 0
         and cloth['source_receipt_sha256'] == expected['clothing_receipt']['sha256']
         and cloth['active_mesh_sha256'] == cloth['original_mesh_sha256'] == inventory['frozen_surface']['sha256']
         and cloth['support_file_sha256'] == expected['clothing_support']['sha256'],
         'The accepted shared clothing surface or its evidence changed after R21.')
    pin(out/cloth['active_clothing_surface'], cloth['active_mesh_sha256'])
    pin(out/cloth['support_file'], cloth['support_file_sha256'])
    split = integration['surface_distribution_splits']
    need(split['file'] == 'splat-surface-splits.json', 'Unregistered distribution split audit.')
    pin(out/split['file'], split['sha256'])
    events = read(out/split['file'])
    need(events['policy'] == effective['surface_attachment_policy']['densification'],
         'The actual distribution-driven split policy changed.')
    lifecycle = verify_split_lifecycle(events, frames, effective['training_source_bins'])
    return dict(file=str(path.relative_to(run)), sha256=desc['sha256'],
                clothing_guide_sha256=guide['sha256'], clothing_guide_state=cloth['state'],
                clothing_support_sha256=cloth['support_file_sha256'], clothing_adam_updates=0,
                clothing_selected_cycle=cloth['selected_cycle'],
                clothing_shared_offset=dict(offset_sha256=raw_info['frozen_offset_sha256'],
                    protected_vertices_bit_exact=True, changed_vertices=0, maximum_change_m=0.0),
                skin_supported_source_views=raw_report['support']['supported_views'],
                skin_supported_calls=sum(row['supported_calls'] for row in views),
                postprocessing_changes_geometry=False, surface_distribution_splits=split,
                distribution_coverage=coverage, split_lifecycle=lifecycle)


def verify_outputs(run, plan, inventory):
    for path, digest in inventory['files'].items():
        prior._pin_input(Path(path), digest)
    for path, digest in plan['code_sha256'].items():
        pin(PROJECT/path, digest)
    verify_cuda_proof(Path(plan['cuda_proof']['path']), plan['cuda_proof']['sha256'], plan['code_sha256'])
    verify_staged(run, inventory)
    for name in ('gaussian-run/training-cameras.json', 'gaussian-run/initialization.json'):
        pin(run/name, inventory['profiles'][name]['sha256'])
    status = read(run/'status.json')
    need(status['state'] == 'completed' and status['static_export_passed'], 'The paired run did not finish.')
    continuity = rebound(prior.verify_continuity, _branch_effective=_branch_effective,
                         _raw_duration_state=verify_raw)(run, plan, inventory)
    check = rebound(prior.verify_branch, _branch_effective=_branch_effective,
                    verify_protected_stage=verify_protection, verify_surface_attachment=verify_surface)
    # The R20 heldout verifier expects this separately pinned fit in its plan.
    replay_plan = dict(plan, head_fit=inventory['head_fit'])
    # The skin-pair archive is measurement evidence, not a member of the
    # exported same-duration main snapshot.
    def branch(branch_run, step, item):
        raw = run/f'gaussian-run/raw-step-{step:06d}'
        keys = {name: digest for name, digest in item['files_sha256'].items() if name != 'skin-pairs.npz'}
        return check(branch_run, replay_plan, inventory, step, raw, dict(item, files_sha256=keys))

    intermediate, final = (branch(run/'duration-10000' if step == MIDPOINT else run, step, item)
                           for step, item in zip((MIDPOINT, STEPS), continuity['raw_endpoints'], strict=True))
    same = ('clothing_guide_sha256', 'clothing_support_sha256', 'clothing_shared_offset', 'surface_distribution_splits')
    need(all(intermediate['surface_attachment'][key] == final['surface_attachment'][key] for key in same),
         'The two durations used different frozen guidance or split lifecycle.')
    integration = read(run/'gaussian-run/integration.json')
    effective = _branch_effective(run, inventory, STEPS)
    legacy = r20.prior.prior.prior
    densification = rebound(legacy.verify_densification, TRAINING_STEPS=STEPS)(
        run, integration, effective, inventory['selected_frames'], final['points'])
    for value in (read(run/densification['file']), read(run/'gaussian-run/world-scale.json'),
                  integration['coverage_events']):
        need(not r20._has_main_auxiliary(value), 'Auxiliary or heldout frames influenced training.')
    return dict(final, state='completed', reference_frame=inventory['reference_frame'],
                selected_frames=inventory['selected_frames'], retained_holdouts=inventory['retained_holdouts'],
                coverage=read(run/'fit/selection.json')['coverage'], fit_method='reused_exact_round21_fit_and_garment',
                fit_sha256=inventory['fit_sha256'], head_fit=inventory['head_fit'],
                training_inputs_sha256=inventory['staged_files']['data/training-inputs.json']['sha256'],
                fresh_gaussian=True, native_inputs_reused=True, same_round21_fit_and_training_selection=True,
                verified_pose_count=24+len(final['face_supplement']['frames']), densification_audit=densification,
                duration_comparison=dict(continuity=continuity, intermediate=intermediate, terminal=final))


def finish(root, plan, inventory, returncode, elapsed, filename='wrapper-terminal.json'):
    receipt = dict(state='failed' if returncode else 'incomplete', subject=plan['subject'], returncode=returncode,
        stage='train_fitted', candidate_commit=plan['candidate_commit'], baseline_commit=PARENT,
        quality_accepted=False, command_to_end_seconds=elapsed,
        preregistration_sha256=sha(root/'preregistration.json'), launch_sha256=sha(root/'launch.json'),
        protected_input_inventory_sha256=plan['protected_input_inventory_sha256'], entry_log_sha256=sha(root/'entry.log'))
    if returncode == 0:
        try:
            receipt.update(verify_outputs(root/'run', plan, inventory))
        except Exception as error:
            receipt['validation_error'] = f'{type(error).__name__}: {error}'
    with (root/filename).open('x', encoding='utf-8') as stream:
        import json
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(dict(state=receipt['state'], subject=plan['subject'],
               validation_error=receipt.get('validation_error')), flush=True)
    return 0 if receipt['state'] == 'completed' else returncode or 1


def launch(subject):
    root = root_for(subject)
    run = root/'run'
    plan = read(root/'preregistration.json')
    protected = root/'protected-inputs.json'
    pin(protected, plan['protected_input_inventory_sha256'])
    inventory = read(protected)
    need(inventory == source_inventory(subject), 'The protected R22 sources changed before launch.')
    verify_staged(run, inventory)
    pin(run/'config.json', plan['staged_config_sha256'])
    need(code_identity(plan['candidate_commit'], read(Path(inventory['r21_plan']['path']))) == plan['code_sha256'],
         'Candidate source differs from the frozen registration.')
    verify_cuda_proof(Path(plan['cuda_proof']['path']), plan['cuda_proof']['sha256'], plan['code_sha256'])
    with cold.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subject == '006':
            first = root_for('016')
            receipt = read(first/'wrapper-terminal.json')
            if receipt['state'] != 'completed':
                receipt = read(first/'wrapper-terminal-verified.json')
            need(receipt['state'] == 'completed' and receipt['candidate_commit'] == plan['candidate_commit'],
                 'The same candidate 016 must finish and pass independent artifact verification first.')
            pin(first/'run/result.json', receipt['result_sha256'])
        need(not any((root/name).exists() for name in ('launch.json', 'entry.log', 'wrapper-terminal.json'))
             and not any((run/name).exists() for name in ('status.json', 'gaussian-run', 'export', 'result.json')),
             'The formal R22 run root must be fresh.')
        write(root/'launch.json', dict(subject=subject, candidate_commit=plan['candidate_commit'], commands=plan['commands'],
            cwd=str(PROJECT/'src'), started=cold.now(), preregistration_sha256=sha(root/'preregistration.json')))
        start = time.monotonic()
        with (root/'entry.log').open('xb') as log:
            command, = plan['commands']
            code = subprocess.run(['/usr/bin/timeout', '--kill-after=15s', f'{plan["max_wall_seconds"]}s', *command],
                env=cold.clean_env(), cwd=PROJECT/'src', stdout=log, stderr=subprocess.STDOUT).returncode
        return finish(root, plan, inventory, code, time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'register', 'launch', 'verify'))
    parser.add_argument('subject', choices=('016', '006'))
    parser.add_argument('--commit')
    parser.add_argument('--cuda-proof', type=Path)
    parser.add_argument('--cuda-proof-sha256')
    args = parser.parse_args()
    if args.action == 'stage':
        stage(args.subject)
    elif args.action == 'register':
        if not all((args.commit, args.cuda_proof, args.cuda_proof_sha256)):
            parser.error('register needs --commit, --cuda-proof and --cuda-proof-sha256')
        register(args.subject, args.commit, args.cuda_proof.resolve(strict=True), args.cuda_proof_sha256)
    elif args.action == 'launch':
        raise SystemExit(launch(args.subject))
    else:
        root = root_for(args.subject)
        plan = read(root/'preregistration.json')
        pin(root/'protected-inputs.json', plan['protected_input_inventory_sha256'])
        original = read(root/'wrapper-terminal.json')
        need(original['returncode'] == 0, 'Only a completed trainer can receive read-only verification.')
        raise SystemExit(finish(root, plan, read(root/'protected-inputs.json'), original['returncode'],
                                original['command_to_end_seconds'], 'wrapper-terminal-verified.json'))


if __name__ == '__main__':
    main()
