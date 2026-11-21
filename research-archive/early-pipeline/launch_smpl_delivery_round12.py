"""Run one verified round-twelve constrained fit and fresh SH0 reconstruction per subject."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round11 as previous


cold = previous.cold
rebound, need, read, pin, sha = (previous.rebound, previous.need, previous.read,
                                previous.pin, previous.sha)
BASE = previous.BASE
BASE_PROJECT = previous.BASE_PROJECT
CODE_PARENT = '509657c17692dd3eca50cf3ad139d8b660f04ce1'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-block-fit-20261111')
FIT_MODULE = previous.FIT_MODULE
FIT_PATHS = {FIT_MODULE, 'src/shuolee_portrait/official_sequence.py'}
ALLOWED_TEST_PATHS = {'tests/test_joint_body_fit.py', 'tests/test_official_sequence.py'}
PYTHON = previous.PYTHON
MAX_WALL_SECONDS = 7200
DIRECTOR = previous.DIRECTOR.with_name('round11_director_review_20261111.json')
DIRECTOR_SHA = '50c57b40c523636cabdbcd132acb55a34fc711ef6112825477baa5dde47fd4e8'
R11_DELIVERY = previous.DELIVERY.parent.parent/'smpl-delivery-round11/receipt.json'
R11_DELIVERY_SHA = '723ddcc377b834835154373339ba77678dd663131a43b64fae9997a64faa2a43'
R11_SOURCE_SHA = {
    '016': 'a225973a8abfd2ed62707f33785f9895bf0c80c9cabc54919cb31aeb3c8d6c1f',
    '006': '85a45dee92852a883d51cf68cd279161989e7e829734ca5f6701b2e6f92931f4',
}
INHERITED_WRAPPERS = tuple(sorted(Path(module.__file__) for name,module in sys.modules.items()
                                 if module is cold or (name.startswith('launch_smpl_delivery_round')
                                                       and name != __name__)))


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round12-{subject}'


git_state = rebound(cold.git_state, PROJECT=PROJECT, BASE=CODE_PARENT)
asset_pins = rebound(cold.official_pins, PROJECT=PROJECT)


def source_inventory(subject):
    pin(DIRECTOR, DIRECTOR_SHA)
    pin(R11_DELIVERY, R11_DELIVERY_SHA)
    review, delivery = read(DIRECTOR), read(R11_DELIVERY)
    need(review['round'] == 11 and review['commit'] == delivery['commit'] == CODE_PARENT
         and review['receipt_sha256_verified'] == R11_DELIVERY_SHA
         and review['quality_accepted'] is False and delivery['quality_accepted'] is False,
         'Unverified round-eleven review and code parent.')
    binding = delivery['subjects'][subject]['evidence']['protected_inputs']
    prior_inventory = R11_DELIVERY.parent/subject/'evidence/protected_inputs/protected-inputs.json'
    need(binding['sha256'] == R11_SOURCE_SHA[subject] and Path(binding['path']) == prior_inventory,
         'Archived round-eleven protected input identity differs.')
    pin(prior_inventory, R11_SOURCE_SHA[subject])
    inventory = previous.source_inventory(subject)
    need(read(prior_inventory) == inventory and len(inventory['files']) == 231,
         'Recomputed inventory differs from the previously protected round-ten inputs.')
    return inventory


def check_code(commit):
    git_state(commit)
    pointer = (PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['/usr/bin/wslpath', '-u', pointer], text=True).strip()
    changed = set(subprocess.check_output(['/usr/bin/git', '--git-dir', git_dir, '--work-tree', str(PROJECT),
        'diff', '--name-only', CODE_PARENT, commit], text=True).splitlines())
    need(FIT_MODULE in changed and changed <= FIT_PATHS | ALLOWED_TEST_PATHS,
         f'Unexpected round-twelve code paths: {sorted(changed)}.')
    for name in set(previous.SOURCE_FILES)-FIT_PATHS:
        pin(PROJECT/name, sha(BASE_PROJECT/name))
    return {name:sha(PROJECT/name) for name in sorted(set(previous.SOURCE_FILES) | changed)}


def check(subject, commit):
    codes = check_code(commit)
    inventory = source_inventory(subject)
    root = root_for(subject)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient local disk space.')
    refinement = [str(PYTHON), '-I', '-B', '-u', str(Path(__file__).resolve()), '_refine', subject, '--commit', commit]
    training = [str(PYTHON), '-B', '-u', '-m', 'shuolee_portrait.cli', 'train-fitted', '--run', str(root/'run')]
    return dict(schema='shuolee-round12-constrained-fit-v1', subject=subject, candidate_commit=commit,
        code_parent_commit=CODE_PARENT, baseline_commit=BASE, baseline_run=inventory['baseline_run'],
        baseline_result_sha256=inventory['paths']['original_result']['sha256'], output=str(root/'run'),
        director_review_sha256=DIRECTOR_SHA, baseline_delivery_sha256=previous.DELIVERY_SHA,
        parent_delivery_sha256=R11_DELIVERY_SHA, round11_input_inventory_sha256=R11_SOURCE_SHA[subject],
        runtime=cold.cpu_runtime(cold.clean_env()), code_sha256=codes, asset_sha256=asset_pins(),
        wrapper_sha256=sha(Path(__file__)), inherited_wrapper_sha256={str(path):sha(path) for path in INHERITED_WRAPPERS},
        retained_holdouts=inventory['retained_holdouts'], commands=[refinement,training],
        gaussian_steps=10000, max_wall_seconds=MAX_WALL_SECONDS), inventory


def verify_solver_audit(metrics, names):
    blocks = metrics['block_solver']
    config = metrics['config']
    sweeps, rows, frames = blocks['completed_sweeps'], blocks['sweeps'], blocks['per_frame']
    need(sweeps == metrics['optimizer_updates'] == len(rows) and 0 < sweeps <= config['max_sweeps'] == blocks['max_sweeps'] == 120
         and blocks['stop_reason'] in ('maximum_sweeps','small_feasible_progress','no_feasible_descent')
         and (blocks['stop_reason'] != 'maximum_sweeps' or sweeps == 120)
         and config['shared_every'] == 4 and set(frames) == set(names)
         and [row['sweep'] for row in rows] == list(range(1,sweeps+1)),
         'The block solver did not report its fixed budget and complete sweep trace.')
    need(all(frame['local_accepted'] + frame['no_feasible_descent'] + frame['joint_objective_or_bound_rejection'] == sweeps
             and max(frame['projected_accepted'],frame['backtracked_accepted'],frame['body_repair_accepted'])
             <= frame['local_accepted'] for frame in frames.values())
         and sum(frame['local_accepted'] for frame in frames.values()) == blocks['local_accepted']
         == sum(row['local_accepted'] for row in rows)
         and sum(frame['projected_accepted'] for frame in frames.values()) == blocks['local_projected_accepted']
         == sum(row['local_projected_accepted'] for row in rows),
         'The block solver does not reconcile per-frame and per-sweep local decisions.')
    need(blocks['shared_attempted'] == sweeps//config['shared_every']
         and blocks['shared_accepted'] == sum(row['shared_accepted'] for row in rows),
         'The shared block update counts do not reconcile.')
    for row in rows:
        checks = row['shared_checks']
        need(bool(checks) == (row['sweep'] % config['shared_every'] == 0)
             and all(check['checked_frames'] == len(names) and 0 <= check['failed_frames'] <= len(names)
                     for check in checks)
             and (not row['shared_accepted'] or (checks and checks[-1]['failed_frames'] == 0 and checks[-1]['descent'])),
             'A shared proposal lacked an all-candidate feasibility check.')
    need(metrics['face_guard']['fallback'] is None
         and metrics['objective_final_step'] == sweeps == metrics['chosen_checkpoint']['step']
         and metrics['chosen_checkpoint']['face_guard_passed']
         and not metrics['chosen_checkpoint']['face_guard_failures'],
         'The final delivered pose is not the audited feasible block state.')
    return blocks


def fit_proof(root, plan, inventory):
    import numpy as np

    run = root/'run'
    fit = read(run/'fit/official-fit-result.json')
    joint = fit['joint_body_refinement']
    for name,digest in (('file','sha256'), ('candidate_file','candidate_sha256'),
                        ('before_selection_file','before_selection_sha256')):
        path = (run/joint[name]).resolve()
        need(path.is_relative_to(run/'fit'), 'Joint output path escapes fit root.')
        pin(path,joint[digest])
    pin(run/joint['before_selection_file'], inventory['paths']['before_selection']['sha256'])
    metrics = read(run/joint['file'])
    names, held = inventory['fit_frames'], inventory['retained_holdouts']
    verify_solver_audit(metrics,names)
    guard = metrics['face_guard']
    threshold = guard['threshold_px']
    need(threshold == .25 and guard['confidence_threshold'] == .5
         and guard['reference'] == 'official_initializer' and guard['exact_reprojection_passed'] is True
         and guard['supported_frame_count'] == len(names) and set(metrics['frames']) == set(names)
         and all(region['status'] == 'unobserved' or (
             region['status'] == 'pass' and not region['failures']
             and max(region['delta']['rms_px'],region['delta']['peak_px']) <= threshold+1e-9)
             for frame in metrics['frames'].values() for region in frame['face_guard'].values()),
         'An exact exported pose failed the fixed face protection.')
    selection = fit['selection']
    selected = selection['selected_frames']
    rows = read(run/'data/training-inputs.json')['records']
    used = [row['frame'] for row in rows]
    need(fit['state'] == 'complete' and fit['official_initializers_reused'] and not fit['new_fitting_started']
         and fit['fitted_candidate_frames'] == 52 and metrics['fit_frame_ids'] == names
         and metrics['retained_holdouts'] == selection['retained_holdouts'] == held
         and set(used) == set(selected) <= set(names) and len(used) == len(set(used)) and 0 < len(used) <= 24
         and used[0] == selection['reference_frame'] and selection == read(run/'fit/selection.json')
         and metrics['native_K_frozen'] and metrics['shared_betas_frozen']
         and metrics['per_frame_scale_parameters'] == 0,
         'Joint candidate identity, update evidence, or actual frame selection differs.')
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
             and np.array_equal(candidate['faces'], final['faces'])
             and float(final['world_scale']) == 1.
             and all(np.array_equal(candidate[key][slots],final[key]) for key in
                     ('vertices_camera','joints_camera','joints_projected_native','body_pose_axis_angle',
                      'global_orient','camera_translation','pose_embedding','expression','jaw_pose',
                      'left_hand_pose','right_hand_pose','leye_pose','reye_pose',
                      'K_integer_centers','observed_native_keypoints','source_frames','image_sha256','mask_sha256')),
             'The final training fit differs from the protected joint candidate geometry.')
        for i,name in enumerate(names):
            with np.load(inventory['winners'][name]['npz'],allow_pickle=False) as initial:
                need(np.array_equal(candidate['K_integer_centers'][i],initial['K_integer_centers']),
                     'A local fit changed protected native intrinsics.')
    return fit, joint, metrics


# Reuse the existing execution/export lifecycle with the new manifest and fit verification.
register = rebound(previous.register, check=check, root_for=root_for)
_registered = rebound(previous.registered, check=check, root_for=root_for, __file__=__file__)


def registered(subject, commit, *, full):
    plan,inventory = _registered(subject,commit,full=full)
    need(plan['schema'] == 'shuolee-round12-constrained-fit-v1' and plan['code_parent_commit'] == CODE_PARENT,
         'Not the registered round-twelve execution.')
    for path,digest in plan['inherited_wrapper_sha256'].items():
        pin(Path(path),digest)
    if not full:
        for name,digest in plan['code_sha256'].items():
            pin(PROJECT/name,digest)
    return plan,inventory


refine = rebound(previous.refine, registered=registered, root_for=root_for, fit_proof=fit_proof, PROJECT=PROJECT)
finish = rebound(previous.finish, fit_proof=fit_proof, PROJECT=PROJECT)
launch = rebound(previous.launch, registered=registered, root_for=root_for, finish=finish,
                 PROJECT=PROJECT, MAX_WALL_SECONDS=MAX_WALL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-inputs','check','register','launch','_refine'))
    parser.add_argument('subject',choices=tuple(R11_SOURCE_SHA))
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'check-inputs':
        inventory = source_inventory(args.subject)
        print(dict(state='source_preflight_passed', subject=args.subject, files=len(inventory['files']),
                   fit_frames=len(inventory['fit_frames']), retained_holdouts=inventory['retained_holdouts'],
                   prior_input_inventory_sha256=R11_SOURCE_SHA[args.subject]), flush=True)
        return
    if not args.commit:
        parser.error('--commit is required for this action.')
    if args.action == 'launch':
        raise SystemExit(launch(args.subject,args.commit))
    if args.action == '_refine':
        refine(args.subject,args.commit)
    elif args.action == 'register':
        register(args.subject,args.commit)
    else:
        plan,inventory = check(args.subject,args.commit)
        print(dict(state='preflight_passed',subject=args.subject,files=len(inventory['files']),
                   retained_holdouts=plan['retained_holdouts']),flush=True)


if __name__ == '__main__':
    main()
