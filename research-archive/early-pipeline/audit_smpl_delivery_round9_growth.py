"""Audit finished round-nine replacement decisions and raw PLY scales on CPU."""

from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_smpl_delivery_round6_world_scale as raw_scale
import launch_smpl_delivery_round9 as launch


COMMIT = '7602f31d16dc13ba68c27ca63895d72a07d3a1dd'
OUTPUT = launch.ENDPOINT.parent.parent/'growth-audit.json'
SUBJECTS = ('016', '006')
need, read, sha, pin = launch.need, launch.read, launch.sha, raw_scale.pin
measure_world = launch.rebound(raw_scale.measure, COMMITS={**raw_scale.COMMITS, 9: COMMIT})


def compact_trial(trial, replaced, frames, options):
    launch.replacement_trial_proof(trial, replaced, frames, options)
    if trial is None:
        return dict(proposed=0, accepted=0, selected_ids=[], rejected=[], unresolved=[], trial_count=0,
                    initial_failure_counts={}, rejection_failure_counts={}, final=None)

    trials = trial['trials']
    reasons = Counter()
    rejected = []
    for item in trial['rejected_candidates']:
        evaluation = trials[item['trial_index']]
        failures = [dict(frame=row['frame'], **failure)
                    for row in evaluation['frames'] for failure in row['failures']]
        need(failures, 'Rejected candidate has no failed view and protection item.')
        for failure in failures:
            if failure['item'] != 'nonfinite_native_observation':
                need(failure['value'] > failure['budget'], 'Failure is within its recorded protection budget.')
        reasons.update(row['item'] for row in failures)
        rejected.append(dict(item, accepted_prefix_ids=evaluation['accepted_prefix_ids'], failures=failures))

    proposed = trials[0]['candidate_ids']
    selected, unresolved = trial['selected_candidate_ids'], trial['unresolved_candidate_ids']
    rejected_ids = [item['candidate_id'] for item in rejected]
    classified = selected + rejected_ids + unresolved
    need(len(classified) == len(set(classified)) and set(classified) <= set(proposed),
         'Candidate outcome categories overlap or refer to another shortlist.')
    combination_failed = [item for item in proposed if item not in classified]
    final = trial['final_trial']
    need(not combination_failed or (final and not final['accepted'] and
         set(combination_failed) == set(final['candidate_ids'])),
         'Missing final-combination evidence for evaluated candidates.')
    final_summary = None
    if final:
        final_rows = final['frames']
        final_summary = dict(phase=final['phase'], candidate_ids=final['candidate_ids'],
                             accepted=final['accepted'], source_views=len(final_rows),
                             failures=[dict(frame=row['frame'], **failure) for row in final_rows
                                       for failure in row['failures']])
        if final['accepted']:
            for key in ('foreground_target_deficit_px', 'redundant_alpha_loss_px', 'new_target_crossings',
                        'new_low_alpha_holes', 'background_gain_px', 'rgb_mae_increase'):
                final_summary['max_'+key] = max(row[key] for row in final_rows)

    initial_failures = Counter(failure['item'] for row in trials[0]['frames'] for failure in row['failures'])
    return dict(proposed=len(proposed), accepted=len(selected), selected_ids=selected, rejected=rejected,
                unresolved=unresolved, final_combination_failed_ids=combination_failed,
                trial_count=trial['trial_count'], max_trials=trial['max_trials'],
                initial_accepted=trials[0]['accepted'], initial_failure_counts=dict(initial_failures),
                rejection_failure_counts=dict(reasons), final=final_summary)


def event_summary(split, frames, options):
    support, needle = split['render_support'], split['needle']
    near = compact_trial(support['near_limit_trial'], split['split_parents'], frames, options)
    displaced = compact_trial(support['coverage_trial'], needle['support_replaced'], frames, options)
    near_records, displaced_records = support['near_limit_interventions'], support['interventions']
    actions = Counter(row['action'] for row in near_records + displaced_records)
    need(actions['near_limit_replace'] == near['accepted'] and actions['replace'] == displaced['accepted']
         and actions['trim'] == needle['width_trimmed'] and set(actions) <= {'near_limit_replace', 'replace', 'trim'},
         'Applied replacement or trim count differs from the event log.')
    need(split['children'] == 2*near['accepted'] and needle['children'] == 2*displaced['accepted'],
         'Child count differs from the accepted parent count.')
    return dict(step=split['step'], near_limit=near, displaced=displaced, width_trimmed=actions['trim'],
                children=split['children']+needle['children'],
                center_unobserved_with_foreground=support['center_unobserved_with_foreground'],
                interventions=near_records + displaced_records)


def endpoint_summary(subject, frames, options, receipt):
    expected = receipt['subjects'][subject]
    source = pin(Path(expected['file']), expected['sha256'])
    detail = read(source)
    controls = detail['real_endpoint_controls']
    need(detail['subject'] == subject and detail['state'] == 'verified' and detail['optimizer_updates'] == 0
         and detail['limits'] == expected['limits'] and not detail['fresh_training_initialization_used']
         and controls['frame'] in frames and not controls['unchanged']['failures']
         and not controls['redundant_alpha']['failures']
         and controls['redundant_alpha']['redundant_alpha_loss_px'] > 0
         and controls['redundant_alpha']['foreground_target_deficit_px'] == 0
         and controls['induced_hole']['new_low_alpha_holes'] == 1
         and any('rgb' in item['item'] for item in controls['induced_rgb_error']['failures']),
         'Endpoint control acceptance differs.')
    event = event_summary(detail['replacement_event'], frames, options)
    need(detail['points_after']-detail['points_before'] == event['near_limit']['accepted']+event['displaced']['accepted'],
         'Endpoint topology differs from the validated replacements.')
    return dict(source=str(source), sha256=expected['sha256'], optimizer_updates=0,
                reference_frame=controls['frame'], protected_roi_view_counts=detail['protected_roi_view_counts'],
                event=event, limits=detail['limits'])


def lineage_summary(history, events, terminal, limit_max):
    expected_order = [(event['step'], phase) for event in events for phase in ('before_repair', 'after_repair')]
    expected_order.append((10000, 'terminal'))
    need([(row['step'], row['phase']) for row in history] == expected_order, 'Growth snapshot order differs.')
    tracks = {(row['step'], row['phase']): row['tracks'] for row in history}
    registered, prior = set(), {}
    totals = Counter()
    for event in events:
        before = tracks[event['step'], 'before_repair']
        after = tracks[event['step'], 'after_repair']
        need(set(before) <= set(prior), 'A new lineage appeared between scheduled repairs.')
        next_counts = {key: value['points'] for key, value in before.items()}
        for change in event['interventions']:
            key = str(change['lineage'])
            replaced = change['action'] != 'trim'
            if key in next_counts:
                next_counts[key] += int(replaced)
            else:
                need(key not in registered, 'A previously retired lineage appeared again.')
                registered.add(key)
                next_counts[key] = 2 if replaced else 1
        need(next_counts == {key: value['points'] for key, value in after.items()},
             'Tracked row counts after repair do not match replacements and trims.')
        totals.update(near_limit_replacements=event['near_limit']['accepted'],
                      displaced_replacements=event['displaced']['accepted'], width_trims=event['width_trimmed'])
        prior = after
    final = tracks[10000, 'terminal']
    replacements = totals['near_limit_replacements']+totals['displaced_replacements']
    need(set(final) <= set(prior) and len(registered) == terminal['lineages_registered']
         and set(map(int, registered)) == set(range(terminal['lineages_registered']))
         and sum(row['points'] for row in final.values()) == terminal['live_points']
         == terminal['lineages_registered']+terminal['inherited_children']-terminal['pruned_points']
         and terminal['inherited_children'] >= 2*replacements and terminal['pruned_points'] >= replacements,
         'Terminal live rows and lineage accounting differ.')
    peak = terminal['max_preprojection_growth_ratio']
    need(math.isfinite(peak) and peak >= 1., 'Invalid cumulative growth ratio.')
    for snapshot in history:
        for row in snapshot['tracks'].values():
            need(row['points'] > 0 and 1. <= row['max_preprojection_growth_ratio'] <= peak*(1.+2e-7)
                 and 0 < row['max_sigma_m'] <= limit_max+1e-6
                 and all(math.isfinite(row[key]) and row[key] >= 0 for key in ('foreground', 'background', 'total')),
                 'Invalid footprint growth snapshot.')
    return dict(applied=dict(totals), terminal=terminal, growth_observed=terminal['projected_point_updates'] > 0,
                history=history)


def audit(subject, endpoint):
    root = launch.root_for(subject)
    run = root/'run'
    wrapper = read(root/'wrapper-terminal.json')
    plan = read(pin(root/'preregistration.json', wrapper['preregistration_sha256']))
    integrated = read(pin(run/'gaussian-run/integration.json', wrapper['integration_sha256']))
    need(wrapper['state'] == 'completed' and wrapper['candidate_commit'] == plan['candidate_commit'] == COMMIT
         and wrapper['unified_main_training'] == integrated and integrated['optimizer_updates'] == 10000,
         'Expected the completed public training wrapper and exact candidate commit.')
    endpoint_sha = sha(launch.ENDPOINT)
    need(plan['replacement_endpoint_check']['sha256'] == endpoint_sha, 'Frozen endpoint receipt differs.')
    world_path, world_sha = launch.support_proof(run, integrated, plan)
    world = read(pin(world_path, world_sha))
    scale = measure_world(subject, 9, None, False)
    measured = scale['metrics']
    need(scale['world_scale_log']['sha256'] == world_sha and measured['reference']['all']['over_005m'] == 0
         and measured['all_poses_max']['all']['over_005m'] == 0
         and measured['all_poses_max']['all']['over_limit'] == 0,
         'Independently read PLY violates the all-pose world cap.')
    frames = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
    options = world['config']['render_support']
    events = [event_summary(split, frames, options) for split in world['splits']]
    summary = lineage_summary(world['support_history'], events, world['terminal']['render_support'], world['limits_m']['max'])
    return dict(points=scale['points'], world_scale_path=str(world_path), world_scale_sha256=world_sha,
                wrapper_sha256=sha(root/'wrapper-terminal.json'), native_ply_sha256=scale['world_ply_sha256'],
                bound_ply_sha256=scale['bound_ply_sha256'], portable_ply_sha256=scale['portable_ply_sha256'],
                local_to_world_max_log_sigma_error=scale['local_to_world_max_log_sigma_error'],
                measured_world_scale=measured, candidate_options=options, events=events,
                endpoint_check=endpoint_summary(subject, frames, options, endpoint),
                support_prunes=world['support_prunes'], **summary)


def main():
    endpoint = read(launch.ENDPOINT)
    need(endpoint['candidate_commit'] == COMMIT and endpoint['state'] == 'completed',
         'Expected the exact two-subject endpoint audit.')
    subjects = {subject: audit(subject, endpoint) for subject in SUBJECTS}
    evidence = dict(state='verified', commit=COMMIT, cpu_only=True, quality_accepted=False,
                    endpoint_receipt_sha256=sha(launch.ENDPOINT), subjects=subjects,
                    audit_script_sha256=sha(Path(__file__).resolve()),
                    raw_ply_audit_script_sha256=sha(Path(raw_scale.__file__).resolve()),
                    growth='Maximum local scale ratio immediately before persistent projection. Initial intended shrink is excluded.',
                    scale_scope='Raw native, bound-local and portable PLY with all 24 poses. Per-lineage limits are verified from training history.',
                    candidate_identity_scope='IDs refer to one repair event and branch. Counts across events are occurrences.')
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(file=str(OUTPUT), sha256=sha(OUTPUT),
                          applied={key: value['applied'] for key, value in subjects.items()})))


if __name__ == '__main__':
    main()
