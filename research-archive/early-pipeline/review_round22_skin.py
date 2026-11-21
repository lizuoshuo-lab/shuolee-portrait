"""Read-only audit of the frozen native-view R22 skin measurements."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from review_round21_matched_geometry import _array, _pin, _read, _require, _sha

PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-splat-distance-20261111')
LAB = Path('/opt/gs-slam-lab')
STEPS = (10000, 18000)
METRICS = ('signed_normal_center_mm', 'abs_normal_center_mm', 'normal_sigma_mm')

sys.path.insert(0, str(PROJECT/'src'))
from shuolee_portrait.surface_attachment import compare_skin_pairs


def view_scope(old, new):
    evidence = {}
    for name in ('data/training-inputs.json', 'fit/fitted.npz', 'gaussian-run/training-cameras.json'):
        first, last = old/name, new/name
        digest = _sha(first)
        _pin(last, digest)
        evidence[name] = dict(r21=str(first), r22=str(last), sha256=digest)
    return evidence


def attached(root, step):
    raw = root/f'gaussian-run/raw-step-{step:06d}'
    state = _read(raw/'state.json')
    _require(state['step'] == state['main_optimizer_updates'] == step, 'The saved optimizer step differs.')
    attachment = _pin(raw/'attachment.json', state['files_sha256']['attachment.json'])
    return raw, state, attachment, _read(raw/'attachment.json')


def verify_pair_sidecar(raw, state, report):
    path = raw/'skin-pairs.npz'
    identity = _pin(path, state['files_sha256'][path.name])
    descriptor = report['skin']['stable_pairs']
    _require(descriptor['file'] == path.name and descriptor['sha256'] == identity['sha256'],
             'The active skin sidecar differs from the attachment receipt.')
    checkpoint_path = raw/'state.pt'
    _pin(checkpoint_path, state['state_sha256'])
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    training = checkpoint['surface_distribution']
    _require(all(state[key] == value for key, value in checkpoint['training'].items())
             and training['report'] == report['support'], 'The raw state differs from its endpoint skin receipt.')
    with np.load(path, allow_pickle=False) as data:
        rows, frames, masks = data['pairs'], data['frames'], data['skin_mask_sha256']
    support = report['support']['views']
    actual = report['skin']['views']
    _require(frames.tolist() == [item['frame'] for item in support] == [item['frame'] for item in actual]
             and masks.tolist() == [item.get('native_material_mask_sha256', '') for item in support]
             and len(rows) == descriptor['rows'] == sum(item['supported_rows'] for item in actual)
             and np.array_equal(np.bincount(rows['view'], minlength=len(frames)),
                                [item['supported_rows'] for item in actual]),
             'The active skin sidecar has a different native view, mask, or row count.')
    model = checkpoint['state']['model']
    stable_ids, binding = _array(model['densification_ids']), _array(model['binding'])
    _require(stable_ids.shape == binding.shape == (state['points'],) and (np.diff(stable_ids) > 0).all(),
             'The raw state has invalid stable IDs or owners.')
    slots = np.searchsorted(stable_ids, rows['stable_id'])
    _require((slots < len(stable_ids)).all() and np.array_equal(stable_ids[slots], rows['stable_id'])
             and np.array_equal(binding[slots], rows['face']), 'A skin row or owner differs from the saved state.')
    children = {}
    for event in training['split_events']:
        for trial in event['trials']:
            if trial['outcome'] == 'committed':
                for child in trial['child_ids']:
                    _require(child not in children, 'A committed child ID was reused.')
                    children[child] = trial['owner_face']
    alive = np.intersect1d(stable_ids, list(children), assume_unique=True)
    _require(np.array_equal(binding[np.searchsorted(stable_ids, alive)], [children[id_] for id_ in alive]),
             'A surviving split child changed its binding face.')
    visible_children = rows[np.isin(rows['stable_id'], alive)]
    child_evidence = dict(committed_child_ids=sorted(children), surviving_child_ids=alive.tolist(),
        observed_skin_child_ids=np.unique(visible_children['stable_id']).tolist(),
        observed_skin_child_row_draws=len(visible_children))
    return identity, child_evidence, visible_children


def paired_children(first, last):
    def rows_by_id(rows):
        return {(item['view'], item['stable_id']): item for item in rows}

    left, right = [rows_by_id(rows) for rows in (first, last)]
    common = sorted(left.keys() & right.keys())
    metrics = {}
    if common:
        for field in ('normal_sigma_mm', 'distance_mm'):
            start, end = [np.array([rows[key][field] for key in common]) for rows in (left, right)]
            metrics[field] = {label: dict(zip(('p50', 'p95'), np.quantile(value, (.5, .95)).tolist()))
                              for label, value in (('first', start), ('last', end), ('paired_delta', end-start))}
    return dict(paired_visible_child_rows=len(common), paired_visible_child_ids=sorted({int(key[1]) for key in common}),
                metrics=metrics, scope='Exact committed child IDs retained at both endpoints. Later stock offspring are outside this count.')


def native_view_comparison(old, new):
    old_support, new_support = [report['support']['views'] for report in (old, new)]
    old_values, new_values = [report['skin']['views'] for report in (old, new)]
    comparisons, empty = [], []
    for source, candidate, first, last in zip(old_support, new_support, old_values, new_values, strict=True):
        _require(source['frame'] == candidate['frame'] == first['frame'] == last['frame']
                 and all(source.get(name) == candidate.get(name) for name in
                         ('native_material_mask_sha256', 'box_xyxy', 'skin_pixels', 'repeated_skin_triangles')),
                 'The native skin mask, crop, or supported triangles differ across rounds.')
        if not min(first['supported_rows'], last['supported_rows']):
            empty.append(dict(frame=first['frame'], r21_rows=first['supported_rows'], r22_rows=last['supported_rows']))
            continue
        metrics = {}
        for name in METRICS:
            metrics[name] = dict(r21=first[name], r22=last[name], r22_minus_r21={
                quantile: last[name][quantile]-first[name][quantile] for quantile in ('p50', 'p95')})
        comparisons.append(dict(frame=first['frame'], r21_rows=first['supported_rows'],
            r22_rows=last['supported_rows'], native_material_mask_sha256=source['native_material_mask_sha256'],
            box_xyxy=source['box_xyxy'], metrics=metrics))
    _require(bool(comparisons), 'No common measured native skin view.')
    return dict(scope='Same native camera, image and skin mask. Quantiles compare each run’s visible population.',
        common_measured_views=len(comparisons), empty_measurement_views=empty, views=comparisons,
        median_view_delta_mm={name: {quantile: float(np.median([
            view['metrics'][name]['r22_minus_r21'][quantile] for view in comparisons]))
            for quantile in ('p50', 'p95')} for name in METRICS})


def training_coverage(old, candidate, step):
    old_rows = sum(item['supported_row_draws'] for item in old['views'])
    old_sheets = sum(item['flattened_row_draws'] for item in old['views'])
    coverage = candidate['coverage']
    entries = sum(item['distribution_row_draws'] for item in candidate['views'])
    eligible = sum(item['supported_row_draws'] for item in candidate['views'])
    nonsheet = sum(item['distribution_nonsheet_row_draws'] for item in candidate['views'])
    _require(entries == eligible == coverage['eligible_row_draws'] == coverage['distribution_row_draws']
             and nonsheet == coverage['distribution_nonsheet_row_draws'] and coverage['fraction'] == 1.
             and all(view['distribution_row_draws'] == view['supported_row_draws']
                     and view['distribution_nonsheet_row_draws'] == view['supported_row_draws']-view['flattened_row_draws']
                     for view in candidate['views']),
             'R22 missed eligible native rows in the distribution loss.')
    audit = candidate['gradient_update_audit']
    sampled = audit['steps']
    numbers = [item['step'] for item in sampled]
    _require(bool(numbers) and numbers == sorted(set(numbers)) and numbers[-1] <= step,
             'Sampled skin audit records have invalid optimizer steps.')
    real = [item for item in sampled if item['state'] == 'recorded']
    empty = [item for item in sampled if item['state'] == 'no_visible_skin']
    _require(len(sampled) == len(real)+len(empty) and all(item['eligible_rows'] > 0 for item in real)
             and all(item['eligible_rows'] == 0 for item in empty), 'A skin audit sample has inconsistent eligibility.')
    fields = {}
    for name in ('position', 'log_scale', 'quaternion'):
        rows = [item['parameters'][name] for item in real]
        for item, parameter in zip(real, rows, strict=True):
            _require(0 <= parameter['updated_rows_with_prior_gradient'] <= min(
                parameter['prior_gradient_rows'], parameter['updated_rows']) <= item['eligible_rows']
                and max(parameter['prior_gradient_rows'], parameter['updated_rows']) <= item['eligible_rows']
                and all(np.isfinite(parameter[key]) and parameter[key] >= 0 for key in
                        ('prior_gradient_l2', 'update_mean', 'update_max')), 'Invalid sampled gradient or Adam update.')
        fields[name] = dict(prior_gradient_steps=sum(row['prior_gradient_rows'] > 0 for row in rows),
            prior_gradient_row_draws=sum(row['prior_gradient_rows'] for row in rows),
            updated_row_draws=sum(row['updated_rows'] for row in rows),
            updated_row_draws_with_prior_gradient=sum(row['updated_rows_with_prior_gradient'] for row in rows),
            prior_gradient_l2_peak=max((row['prior_gradient_l2'] for row in rows), default=0.),
            main_adam_update_max=max((row['update_max'] for row in rows), default=0.))
    return dict(r21=dict(eligible_row_draws=old_rows, thickness_row_draws=old_sheets,
                        thickness_fraction=old_sheets/old_rows if old_rows else None), r22=coverage,
        sampled_audit=dict(sampled_steps=len(sampled), recorded_steps=len(real), empty_steps=len(empty),
            audited_eligible_row_draws=sum(item['eligible_rows'] for item in real), fields=fields,
            **{key: audit[key] for key in ('gradient_scope', 'update_scope', 'update_units')}))


def audit(subject, output):
    old_root, new_root = [LAB/f'shuolee-smpl-delivery-round{number}-{subject}' for number in (21, 22)]
    registered = _read(new_root/'preregistration.json')
    old, new = old_root/'run', new_root/'run'
    _require(registered['subject'] == subject and registered['output'] == str(new)
             and registered['comparison_steps'] == list(STEPS), 'This R22 subject or schedule was not registered.')
    scope = view_scope(old, new)
    steps, pairs, children = [], [], []
    for step in STEPS:
        _, old_state, old_attachment, baseline = attached(old, step)
        raw, state, attachment, candidate = attached(new, step)
        _pin(raw/'reference-mesh.npz', old_state['files_sha256']['reference-mesh.npz'])
        pair, child_evidence, visible_children = verify_pair_sidecar(raw, state, candidate)
        pairs.append(pair['file'])
        children.append(visible_children)
        steps.append(dict(step=step, r21_attachment=old_attachment, r22_attachment=attachment,
            r22_skin_pairs=pair, r22_split_children=child_evidence,
            native_view_comparison=native_view_comparison(baseline, candidate),
            training_coverage=training_coverage(baseline['support'], candidate['support'], step)))
    report = dict(schema='shuolee-round22-native-skin-audit/1', subject=subject,
        candidate_commit=registered['candidate_commit'], preregistration_sha256=_sha(new_root/'preregistration.json'),
        source_code_sha256=_sha(Path(__file__)), native_scope=scope, steps=steps,
        same_r22_stable_id_and_native_skin_region=compare_skin_pairs(*pairs),
        same_r22_visible_committed_children=paired_children(*children))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'{subject} {output} sha256={_sha(output)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', choices=('016', '006'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.subject, args.output)
