"""Run the round-thirteen coupled fit and fresh SH0 export on fixed native inputs."""

import argparse
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round12 as previous


cold = previous.cold
rebound, need, read, pin, sha = (previous.rebound, previous.need, previous.read,
                                previous.pin, previous.sha)
BASE = previous.BASE
BASE_PROJECT = previous.BASE_PROJECT
CODE_PARENT = '3d3ef8d7e64f0c87f1686003fda26f9d5f672b62'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-coupled-fit-20261111')
FIT_MODULE = previous.FIT_MODULE
FIT_PATHS = previous.FIT_PATHS | {'src/shuolee_portrait/joint_shared_step.py'}
ALLOWED_TEST_PATHS = previous.ALLOWED_TEST_PATHS | {'tests/test_joint_shared_step.py'}
PYTHON = previous.PYTHON
MAX_WALL_SECONDS = 7200
SCHEMA = 'shuolee-round13-coupled-fit-v1'
DIRECTOR = previous.DIRECTOR.with_name('round12_director_review_20261111.json')
DIRECTOR_SHA = 'b25e2ad9e8b034e9e15d054f42ed7d228111dd8494cda741a7f594364e6a4fcf'
R12_DELIVERY = previous.R11_DELIVERY.parent.parent/'smpl-delivery-round12/receipt.json'
R12_DELIVERY_SHA = 'f49424095c42227fe1329cee020bd899fcdd988eb6c07438164460ee42581fd5'
R12_SOURCE_SHA = previous.R11_SOURCE_SHA
INHERITED_WRAPPERS = tuple(sorted(Path(module.__file__) for name,module in sys.modules.items()
                                 if module is cold or (name.startswith('launch_smpl_delivery_round')
                                                       and name != __name__)))


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round13-{subject}'


git_state = rebound(cold.git_state, PROJECT=PROJECT, BASE=CODE_PARENT)
asset_pins = rebound(cold.official_pins, PROJECT=PROJECT)
check_code = rebound(previous.check_code, PROJECT=PROJECT, CODE_PARENT=CODE_PARENT,
                     git_state=git_state, FIT_PATHS=FIT_PATHS, ALLOWED_TEST_PATHS=ALLOWED_TEST_PATHS)


def source_inventory(subject):
    pin(DIRECTOR, DIRECTOR_SHA)
    pin(R12_DELIVERY, R12_DELIVERY_SHA)
    review, delivery = read(DIRECTOR), read(R12_DELIVERY)
    need(review['round'] == 12 and review['commit'] == delivery['commit'] == CODE_PARENT
         and review['receipt_sha256_verified'] == R12_DELIVERY_SHA
         and review['quality_accepted'] is False and delivery['quality_accepted'] is False,
         'Unverified round-twelve review and code parent.')
    binding = delivery['subjects'][subject]['evidence']['protected_inputs']
    prior_inventory = R12_DELIVERY.parent/subject/'evidence/protected_inputs/protected-inputs.json'
    need(binding['sha256'] == R12_SOURCE_SHA[subject] and Path(binding['path']) == prior_inventory,
         'Archived round-twelve protected input identity differs.')
    pin(prior_inventory, R12_SOURCE_SHA[subject])
    inventory = previous.source_inventory(subject)
    need(read(prior_inventory) == inventory and len(inventory['files']) == 231,
         'Recomputed inventory differs from the previously protected round-ten inputs.')
    return inventory


def check(subject, commit):
    codes = check_code(commit)
    inventory = source_inventory(subject)
    root = root_for(subject)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Insufficient local disk space.')
    refinement = [str(PYTHON), '-I', '-B', '-u', str(Path(__file__).resolve()), '_refine', subject, '--commit', commit]
    training = [str(PYTHON), '-B', '-u', '-m', 'shuolee_portrait.cli', 'train-fitted', '--run', str(root/'run')]
    return dict(schema=SCHEMA, subject=subject, candidate_commit=commit,
        code_parent_commit=CODE_PARENT, baseline_commit=BASE, baseline_run=inventory['baseline_run'],
        baseline_result_sha256=inventory['paths']['original_result']['sha256'], output=str(root/'run'),
        director_review_sha256=DIRECTOR_SHA, baseline_delivery_sha256=previous.previous.DELIVERY_SHA,
        parent_delivery_sha256=R12_DELIVERY_SHA, round12_input_inventory_sha256=R12_SOURCE_SHA[subject],
        runtime=cold.cpu_runtime(cold.clean_env()), code_sha256=codes, asset_sha256=asset_pins(),
        wrapper_sha256=sha(Path(__file__)), inherited_wrapper_sha256={str(path):sha(path) for path in INHERITED_WRAPPERS},
        retained_holdouts=inventory['retained_holdouts'], commands=[refinement,training],
        gaussian_steps=10000, max_wall_seconds=MAX_WALL_SECONDS), inventory


def verify_shared_preflight(preflight, config, names):
    directions = preflight['directions']
    need(preflight['checked_frames'] == len(names) == 52
         and preflight['smooth_derivative_passed'] is True
         and preflight['constraint_linearization_passed'] is True
         and preflight['full_gradient_includes_antialiased_silhouette_and_extrema'] is True
         and preflight['smooth_components'] == [
             'reprojection', 'face_native_reprojection', 'shared_pose', 'pose_trust']
         and len(directions) == 2
         and {direction['kind'] for direction in directions} == {'common_only', 'coupled'},
         'Shared numerical preflight did not pass on the complete real candidate roster.')
    for direction in directions:
        samples = direction['samples']
        need(direction['nonzero'] is True and direction['direction_norm'] > 0
             and all(math.isfinite(direction[key]) for key in
                     ('analytic_derivative', 'smooth_analytic_derivative', 'direction_norm'))
             and [sample['h'] for sample in samples] == list(config['shared_fd_steps'])
             and len(samples) >= 2,
             'Invalid shared directional derivative or finite-difference schedule.')
        smooth = direction['smooth_analytic_derivative']
        for sample in samples:
            error = abs(sample['smooth_finite_difference'] - smooth)
            need(math.isfinite(sample['finite_difference']) and math.isfinite(error)
                 and math.isclose(sample['smooth_gradient_abs_error'], error, abs_tol=1e-12)
                 and sample['smooth_gradient_passed'] is (error <= max(5e-5, .08*abs(smooth)))
                 and sample['worst_frame'] in names and sample['worst_metric'],
                 'Reported shared directional-derivative agreement does not recompute.')
            detail = sample['linearization']
            need(set(detail) == {'face_px', 'body_fraction_of_screen_limit', 'depth_m'}
                 and detail['depth_m']['observations'] == len(names)
                 and 0 < detail['body_fraction_of_screen_limit']['observations'] <= len(names)
                 and 0 < detail['face_px']['observations'] <= 14*len(names)
                 and all(math.isfinite(row[key]) and row[key] >= 0 for row in detail.values() for key in
                         ('max_abs_error', 'p90_abs_error', 'rms_abs_error', 'relative_l2', 'predicted_slope_norm',
                          'actual_slope_norm', 'max_forward_prediction_error')),
                 'Missing regional constraint linearization on the complete real candidate roster.')
            noise = {'face_px': .04, 'body_fraction_of_screen_limit': .0002, 'depth_m': .00003}
            for region, row in detail.items():
                threshold = max(noise[region], .15 * row['predicted_slope_norm']/math.sqrt(row['observations']))
                need(row['linearization_passed'] is (row['rms_abs_error'] <= threshold),
                     'Reported constraint linearization agreement does not recompute.')
        need(any(sample['smooth_gradient_passed'] for sample in samples)
             and any(all(value['linearization_passed'] for value in sample['linearization'].values())
                     for sample in samples),
             'A shared direction lacks a passing smooth gradient or constraint linearization probe.')


def verify_shared_trial(row, names, config):
    details, checks = row['shared_diagnostics'], row['shared_checks']
    need(bool(details) == bool(checks) == (row['sweep'] % config['shared_every'] == 0),
         'Shared diagnostics and trial schedule differ.')
    if not details:
        need(not row['shared_accepted'], 'A shared step was marked accepted without a trial.')
        return
    keys = ('trust_radius_before', 'trust_radius_after', 'gradient_norm',
            'projected_gradient_norm', 'tangent_gradient_norm')
    need(all(math.isfinite(details[key]) and details[key] >= 0 for key in keys)
         and all(config['shared_trust_min'] <= details[key] <= config['shared_trust_max']
                 for key in keys[:2])
         and details['outcome'] and details['active_constraints']
         and all(0 <= count <= len(names) for count in details['active_constraints'].values()),
         'A shared trial lacked a finite trust region, projected residual, or active constraints.')
    proof_types = ('projection','tangent_projection','stationarity_projection','common_guided_projection')
    for key in proof_types:
        proof = details[key]
        residual = max(proof['primal_residual'],proof['dual_residual'])
        need(math.isfinite(residual) and residual >= 0 and proof['certified'] is (residual <= 2e-5),
             'A shared projection certification disagrees with its recorded residual.')
    objective_direction_prediction = details['common_predicted_decrease'] + details['compensation_predicted_decrease']
    tried = {'common_guided': [], 'objective_guided': []}
    candidates_with_descent = []
    for check in checks:
        kind = check['direction_kind']
        need(kind in tried and (not tried['objective_guided'] or kind == 'objective_guided'),
             'The shared trial direction order differs.')
        tried[kind].append(check['fraction'])
        predicted = check['common_predicted_decrease'] + check['compensation_predicted_decrease']
        need(check['checked_frames'] == len(names)
             and 0 <= check['failed_frames'] <= len(names)
             and all(0 <= check[key] <= check['failed_frames'] for key in
                     ('failed_face_frames','failed_body_frames','failed_depth_frames'))
             and math.isclose(check['predicted_decrease'], predicted, abs_tol=1e-11)
             and (kind != 'objective_guided' or math.isclose(
                  predicted, objective_direction_prediction*check['fraction'], abs_tol=1e-11))
             and math.isfinite(check['actual_decrease'])
             and check['descent'] is (check['actual_decrease'] > 1e-9)
             and all(math.isfinite(check[key]) and check[key] >= 0 for key in
                     ('max_face_linearization_error_px','max_body_linearization_error',
                      'max_depth_linearization_error_m','common_latent_step_l2','compensation_max_step')),
             'Shared trial feasibility, objective prediction, or linearization is incomplete.')
        ratio = check['ratio']
        if check['predicted_decrease'] > 1e-12:
            need(math.isclose(ratio, check['actual_decrease']/check['predicted_decrease'], abs_tol=1e-9),
                 'The shared trust ratio does not recompute.')
        else:
            need(ratio is None, 'The reported trust ratio uses a degenerate prediction.')
        if (check['failed_frames'] == 0 and check['residual_bound_passed'] is True
                and check['descent'] and ratio is not None and ratio >= config['shared_min_gain_ratio']):
            candidates_with_descent.append(check)
    schedule = list(config['trial_fractions'])
    need(all(fractions == schedule[:len(fractions)] for fractions in tried.values())
         and (tried['common_guided'] or details['common_guided_skipped_predicted_decrease'] <= 1e-12)
         and (not tried['objective_guided'] or not tried['common_guided'] or tried['common_guided'] == schedule)
         and (candidates_with_descent == [checks[-1]] if row['shared_accepted']
              else not candidates_with_descent and tried['objective_guided'] == schedule),
         'The shared trial sequence or acceptance differs from exact feasibility and descent.')
    if details['stationarity_check_passed']:
        stationarity = details['stationarity_projection']
        need(not row['shared_accepted']
             and all(details[key]['certified'] for key in ('projection','tangent_projection'))
             and max(stationarity['primal_residual'], stationarity['dual_residual']) <= 1e-6
             and details['projected_gradient_norm'] <= config['shared_projected_stop']
             and all(trial['actual_decrease'] <= 1e-9 for trial in checks),
             'The shared stationarity claim lacks projection or exact descent evidence.')


def verify_solver_audit(metrics, names):
    blocks = metrics['block_solver']
    config = metrics['config']
    sweeps, rows, frames = blocks['completed_sweeps'], blocks['sweeps'], blocks['per_frame']
    stop = blocks['stop_reason']
    need(sweeps == metrics['optimizer_updates'] == len(rows)
         and 0 < sweeps <= config['max_sweeps'] == blocks['max_sweeps'] == 144
         and stop in ('maximum_sweeps_unconverged', 'shared_stationary_local_progress')
         and blocks['converged'] is (stop == 'shared_stationary_local_progress')
         and (stop != 'maximum_sweeps_unconverged' or sweeps == 144)
         and 'Shared projected stationarity' in blocks['convergence_scope']
         and config['shared_every'] == 4 and set(frames) == set(names)
         and [row['sweep'] for row in rows] == list(range(1, sweeps+1)),
         'The coupled solver did not report its budget, trace, or convergence status.')
    need(all(frame['local_accepted'] + frame['no_feasible_descent'] + frame['joint_objective_or_bound_rejection']
             == sweeps and max(frame['projected_accepted'], frame['backtracked_accepted'], frame['body_repair_accepted'])
             <= frame['local_accepted'] for frame in frames.values())
         and sum(frame['local_accepted'] for frame in frames.values()) == blocks['local_accepted']
         == sum(row['local_accepted'] for row in rows)
         and sum(frame['projected_accepted'] for frame in frames.values()) == blocks['local_projected_accepted']
         == sum(row['local_projected_accepted'] for row in rows),
         'The per-frame and per-sweep local decisions differ.')
    verify_shared_preflight(blocks['shared_preflight'], config, names)
    attempts = [row for row in rows if row['sweep'] % config['shared_every'] == 0]
    need(blocks['shared_attempted'] == len(attempts)
         and blocks['shared_accepted'] == sum(row['shared_accepted'] for row in rows),
         'The shared block update counts do not reconcile.')
    for row in rows:
        verify_shared_trial(row,names,config)
    radius = config['shared_trust_initial']
    for row in attempts:
        details = row['shared_diagnostics']
        need(math.isclose(radius, details['trust_radius_before'], abs_tol=1e-12),
             'The shared trust radius differs between consecutive trials.')
        radius = details['trust_radius_after']
    if blocks['converged']:
        stable, count = config['stable_sweeps'], config['shared_stationary_checks']
        need(len(rows) >= stable and len(attempts) >= count
             and all(not row['new_eligible'] and abs(row['objective_relative_change']) <= config['objective_relative_stop']
                     and row['screen_change'] <= config['screen_stop'] for row in rows[-stable:])
             and all(row['shared_diagnostics']['stationarity_check_passed'] for row in attempts[-count:]),
             'The final convergence claim disagrees with the recorded sweep history.')
    need(metrics['face_guard']['fallback'] is None
         and metrics['objective_final_step'] == sweeps == metrics['chosen_checkpoint']['step']
         and metrics['chosen_checkpoint']['face_guard_passed']
         and not metrics['chosen_checkpoint']['face_guard_failures'],
         'The final delivered pose is not the audited feasible block state.')
    return blocks


fit_proof = rebound(previous.fit_proof, verify_solver_audit=verify_solver_audit, PROJECT=PROJECT)
register = rebound(previous.register, check=check, root_for=root_for)
_registered = rebound(previous._registered, check=check, root_for=root_for, __file__=__file__)


def registered(subject, commit, *, full):
    plan,inventory = _registered(subject,commit,full=full)
    need(plan['schema'] == SCHEMA and plan['code_parent_commit'] == CODE_PARENT,
         'Not the registered round-thirteen execution.')
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
    parser.add_argument('subject',choices=tuple(R12_SOURCE_SHA))
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'check-inputs':
        inventory = source_inventory(args.subject)
        print(dict(state='source_preflight_passed', subject=args.subject, files=len(inventory['files']),
                   fit_frames=len(inventory['fit_frames']), retained_holdouts=inventory['retained_holdouts'],
                   prior_input_inventory_sha256=R12_SOURCE_SHA[args.subject]), flush=True)
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
