"""Rescore a complete frozen fit using source-confirmed missing upper-body observations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

import review_capture_formal_holdouts as review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'project', 'source-sheet', 'labels', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run, project = args.run.resolve(), args.project.resolve()
    sheet_path, label_path, output = (p.resolve() for p in (args.source_sheet, args.labels, args.output))
    review.need(not output.exists() and not output.is_relative_to(run.parent)
                and not output.is_relative_to(project), 'Use a fresh external output.')
    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import official_sequence as seq, stable_selection as select

    read, sha, need = review.read, review.sha, review.need
    sheet, labels = read(sheet_path), read(label_path)
    need(sha(sheet_path) == labels['source_sheet_receipt_sha256']
         and sha(sheet_path.parent/'source-blind-contact.jpg') == labels['source_sheet_image_sha256']
         and labels['review_scope'] == 'source_only_before_residuals'
         and labels['excluded_joints'], 'Independent source review identity differs.')
    cfg = read(run/'config.json')
    layout = cfg['capture_layout']
    source, observation = run/'data/inputs.json', run/'data/observations.npz'
    frozen, fit, shape = (run/'fit'/name for name in ('selection.json', 'official-fit-result.json', 'shared-body.json'))
    pins = {p: sha(p) for p in (source, observation, frozen, fit, shape, sheet_path, label_path,
                                Path(__file__).resolve())}
    prior, receipt = read(frozen), read(fit)
    need(receipt['selection'] == prior and pins[shape] == receipt['shared_body_sha256']
         and receipt['source']['input_files'][str(observation)] == pins[observation],
         'Frozen official fit receipt differs.')
    inputs = seq.load_inputs(source, observation, camera_length_contract=cfg['camera_length_contract'],
                             reference_frame=read(source)['records'][0]['frame'], capture_layout=layout)
    scope = inputs['scope']
    protected = set(scope['retained_holdouts'])
    front = {frame for group in scope['source_view_bins'][:scope['view_targets']['front']]
             for frame in group} - protected
    slots = {frame: i for i, frame in enumerate(inputs['ids'])}
    seen = {record['frame']: record for record in sheet['records']}
    need(set(seen) == front and set(labels['excluded_joints']) <= front
         and all(seen[frame]['source_sha256'] == inputs['rows'][slots[frame]]['source_sha256']
                 for frame in front), 'Review differs from the non-holdout front fit roster.')
    mapping = dict(left_shoulder=5, right_shoulder=2, left_elbow=6, right_elbow=3,
                   left_wrist=7, right_wrist=4)
    need(dict(zip(sheet['point_names'], sheet['smplifyx_slots'], strict=True)) == mapping,
         'The reviewed point-to-slot mapping differs.')
    observed = inputs['targets']
    controlled = observed.copy()
    changes, winners, reports = {}, {}, {}
    for frame in scope['fit_frame_ids']:
        folder = run/'fit/frames'/frame
        npz, result = folder/'winner-camera-space.npz', folder/'result.json'
        pins.update({npz: sha(npz), result: sha(result)})
        with np.load(npz, allow_pickle=False) as data:
            winners[frame] = dict(data)
        reports[frame] = read(result)
        i = slots[frame]
        saved_targets = winners[frame].get('observed_native_keypoints')
        need(saved_targets is None or np.array_equal(saved_targets, observed[i:i+1]),
             f'Frozen target receipt differs: {frame}')
        joints = labels['excluded_joints'].get(frame, [])
        need(len(joints) == len(set(joints)) and set(joints) <= set(mapping), f'Invalid joints: {frame}')
        changed = [mapping[joint] for joint in joints if observed[i, mapping[joint], 2] > 0]
        if changed:
            controlled[i, changed, 2] = 0
            changes[frame] = changed

    beta = np.asarray(read(shape)['betas'], dtype=np.float32)
    existing = select.select_frames(inputs, winners, scope=scope, reports=reports, shared_betas=beta)
    need(existing == prior, 'The original selection did not reproduce exactly.')
    need(np.array_equal(controlled[..., :2], observed[..., :2])
         and np.count_nonzero(controlled[..., 2] != observed[..., 2]) == sum(map(len, changes.values())),
         'Only source-confirmed confidence slots may differ.')
    alternate = {frame: dict(pose, observed_native_keypoints=controlled[slots[frame]:slots[frame]+1])
                 if frame in changes and 'observed_native_keypoints' in pose else pose
                 for frame, pose in winners.items()}
    counterfactual = select.select_frames(dict(inputs, targets=controlled), alternate, scope=scope,
                                          reports=reports, shared_betas=beta)
    need(counterfactual['coverage']['rear'] == existing['coverage']['rear']
         and set(counterfactual['selected_frames']).isdisjoint(protected), 'Rear or holdout scope changed.')

    support = {}
    for frame, joints in labels['excluded_joints'].items():
        i = slots[frame]
        original, candidate = existing['reasons'][frame], counterfactual['reasons'][frame]
        coords = winners[frame]['joints_projected_native'][0]
        residual = np.linalg.norm(coords - observed[i, :, :2], axis=-1)
        remaining = [int(j) for j in select.BODY if controlled[i, j, 2] > 0]
        worst = max(remaining, key=lambda j: residual[j])
        _, top, _, bottom = inputs['rows'][i]['initial_box_xyxy']
        support[frame] = dict(joints=joints, disabled_slots=changes.get(frame, []),
            before_flags=original.get('screen_flags', []), after_flags=candidate.get('screen_flags', []),
            retained_max_slot=worst, retained_max_px=float(residual[worst]),
            retained_max_ratio=float(residual[worst])/(bottom-top),
            removed_residual_px={joint: float(residual[mapping[joint]]) for joint in joints})
    report = dict(state='frozen_fit_visibility_rescoring_complete', fit_calls=0, gpu_calls=0,
        quality_accepted=False, original_selection_reproduced=True, unchanged_geometry=True,
        protected_frames=scope['retained_holdouts'], reviewed_front_frames=len(front),
        original_front=existing['coverage']['front'], counterfactual_front=counterfactual['coverage']['front'],
        existing=existing['selected_frames'], counterfactual=counterfactual['selected_frames'],
        changed_slots=changes, support=support, input_sha256={str(p): d for p,d in pins.items()})
    review.check_pins(pins)
    with output.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({key: report[key] for key in ('state', 'original_front', 'counterfactual_front')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
