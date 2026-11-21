"""Run the bounded 008 training-side, confidence-only CPU fit control."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial.transform import Rotation

import review_008_occlusion_control as evidence
import review_capture_formal_holdouts as base


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def archive(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(values)


def worst(projected, target, indices, height):
    valid = indices[target[indices, 2] > 0]
    if not len(valid):
        return dict(points=0, worst_slot=None, max_px=None, max_over_box_height=None, rms_px=None)
    delta = np.linalg.norm(projected[valid]-target[valid, :2], axis=1)
    weights = target[valid, 2].astype(np.float64)**2
    return dict(points=len(valid), worst_slot=int(valid[delta.argmax()]), max_px=float(delta.max()),
                max_over_box_height=float(delta.max()/height),
                rms_px=float(np.sqrt(np.average(delta**2, weights=weights))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'project', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run, project, out = (path.resolve() for path in (args.run, args.project, args.output))
    base.need(not out.is_relative_to(run.parent) and not out.is_relative_to(project), 'Use a separate output.')
    base.need(base.read(run/'status.json')['state'] == 'failed', 'A must be the immutable failed stock run.')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    evaluator, cfg, rows, absent, observations, receipt = evidence.setup(run, project)
    from shuolee_portrait import official_sequence as seq, stable_selection as selection

    layout = cfg['capture_layout']
    manifest = base.read(run/'data/inputs.json')
    scope = selection.selection_scope(dict(rows=rows, source_manifest=manifest), arc=layout['arc'], capture_layout=layout)
    all_candidates = scope['fit_frame_ids']
    eligible = [row for row in rows if row['frame'] in all_candidates
                and set(absent[row['frame']]['absent_groups']) == set(evidence.GROUPS)]
    protected = scope['retained_holdouts']
    selected = {}
    for side in ('front', 'rear'):
        intervals = layout[side+'_source_ranges']
        ordered = sorted((row for row in eligible if any(start <= row['source_frame_index'] <= stop
                         for start, stop in intervals)), key=lambda row: row['source_frame_index'])
        chunks = np.array_split(np.arange(len(ordered)), 3)
        base.need(all(len(chunk) for chunk in chunks), f'Insufficient independent hidden-foot evidence: {side}')
        selected[side] = [ordered[chunk[(len(chunk)-1)//2]] for chunk in chunks]
    chosen = [row for group in selected.values() for row in group]
    base.need(len(chosen) == 6 and all(r['frame'] not in protected for r in chosen), 'Invalid six-frame control roster.')
    with np.load(observations, allow_pickle=False) as raw:
        ids = raw['frame_ids'].astype(str).tolist()
        train = [ids.index(frame) for frame in all_candidates]
        a, confidence = seq.official.adapt_rtmw_observations(*(raw[k][train] for k in
            ('keypoints', 'scores', 'boxes', 'flip_disagreement')))
    shared_path = run/'fit/shared-body.json'
    shared = base.read(shared_path)
    beta = np.asarray(shared['betas'], dtype=np.float32)
    latent = np.asarray(shared['pose_latent'], dtype=np.float32)
    official_root = Path(cfg['official_smplifyx']['source_root'])
    assets = Path(base.read(run/'run.json')['assets'])
    pins = {path: base.sha(path) for path in (Path(__file__).resolve(), Path(evidence.__file__).resolve(),
        Path(base.__file__).resolve(), run/'status.json', run/'config.json', run/'run.json', run.parent/'preregistration.json',
        run/'data/inputs.json', shared_path, assets/'smplx/SMPLX_NEUTRAL.npz')}
    pins.update({observations: receipt['observations_sha256'], evidence.EVIDENCE: evidence.EVIDENCE_SHA,
                 evidence.ARCHIVED_INPUTS: receipt['observation_manifest_sha256']})
    pins.update({project/name: digest for name, digest in base.read(run.parent/'preregistration.json')['code_sha256'].items()})
    pins.update({official_root/name: digest for name, digest in seq.official.PINNED.items()})
    joint_slots = np.asarray(sorted(sum(evidence.GROUPS.values(), ())))
    info, inputs = {}, []
    for side, group in selected.items():
        for row in group:
            frame = row['frame']
            dir_a = run/'fit/frames'/frame
            old_path, old_result = dir_a/'winner-camera-space.npz', dir_a/'result.json'
            old, report = archive(old_path), base.read(old_result)
            before = a[all_candidates.index(frame)].copy()
            x, y = before[:, :2].T
            w, h = row['wh']
            before[(x < -.5) | (x >= w-.5) | (y < -.5) | (y >= h-.5), 2] = 0
            after = before.copy()
            after[joint_slots, 2] = 0.
            changed = np.argwhere(before != after)
            base.need(len(changed) and set(changed[:, 1]) == {2} and set(changed[:, 0]) <= set(joint_slots),
                      'Expected an actual confidence-only change.')
            k = np.asarray(row['K_integer_centers'])
            selection._candidate(row, old, before, k, beta, None, report, True)
            pins.update({old_path: base.sha(old_path), old_result: base.sha(old_result),
                         Path(row['image']): row['source_sha256'], Path(row['mask']): row['mask_sha256']})
            info[frame] = dict(side=side, source_frame_index=row['source_frame_index'], old_winner=str(old_path),
                old_receipt=str(old_result), source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'],
                K_integer_centers=k.tolist(), before_targets_sha256=evidence.target_sha(before),
                after_targets_sha256=evidence.target_sha(after), removed_nonzero_slots=changed[:, 0].tolist())
            inputs.append((row, k, before, after, old, report))
    base.check_pins(pins)
    target_hashes = {item[-1]['shared_body_target_sha256'] for item in inputs}
    base.need(len(target_hashes) == 1, 'A shared target differs.')
    out.mkdir(parents=True, exist_ok=False)
    (out/'B').mkdir()
    write(out/'plan.json', dict(experiment='008 prefit source-confirmed foot visibility only',
        selection='Chronological lower middle of three equal bins per side among confirmed hidden-both training candidates.',
        input_sha256={str(path): digest for path, digest in pins.items()}, selected=info, protected_frames=protected,
        original_camera_pose_seed=True, frozen_beta=beta.tolist(), frozen_pose_latent=latent.tolist(),
        shared_target_sha256=next(iter(target_hashes)), frozen_q90=confidence['per_joint_q90'],
        max_cpu_workers=3, gaussian_updates=0))
    np.savez_compressed(out/'targets.npz', frame_ids=np.asarray(list(info)),
                        A=np.stack([i[2] for i in inputs]), B=np.stack([i[3] for i in inputs]))
    print('CONTROL_REGISTERED', [(frame, row['side'], row['source_frame_index']) for frame, row in info.items()], flush=True)
    started = time.monotonic()
    try:
        import torch
        torch.set_num_threads(1)
        body, vposer, _ = seq.official.create_models(official_root=official_root, assets=assets,
                                                     device='cpu', static_topology=False)
        pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
        base.need(hashlib.sha256(pose.tobytes()).hexdigest() in target_hashes, 'Decoded B shared target differs from A.')
        del body, vposer
        jobs = [(row, out/'B'/row['frame'], after[None], k, beta, True, pose) for row, k, _, after, _, _ in inputs]
        results = []
        with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn'),
                                 initializer=seq._init_cpu_fit, initargs=(official_root, assets)) as pool:
            for (row, k, before, after, old, report_a), (new, desc) in zip(inputs, pool.map(seq._fit_cpu_job, jobs), strict=True):
                frame = row['frame']
                report_b = base.read(Path(desc['path']))
                seq.check_result(report_b, row)
                base.need(all(report_b[k] == report_a[k] for k in ('shared_body_target_sha256',
                    'body_placement_initial_m', 'body_placement_parameters', 'orientation_count', 'source_hashes',
                    'initialization', 'camera_initialization', 'runtime_optimization')), f'B upstream camera or source parity failed: {frame}')
                h = row['initial_box_xyxy'][3]-row['initial_box_xyxy'][1]
                raw = {}
                for label, arrays, report, target in (('A', old, report_a, before), ('B', new, report_b, after)):
                    scores, flags, _, _ = selection._candidate(row, arrays, target, k, beta, None, report, True)
                    projected = arrays['joints_projected_native'][0]
                    raw[label] = dict(common_body=worst(projected, after, selection.BODY, h),
                        common_face=worst(projected, after, selection.FACE, h), own_screen_flags=flags,
                        own_screen_scores=scores, hidden_feet='unobserved',
                        min_depth_m=float(arrays['vertices_camera'][...,2].min()),
                        placement_initial_m=report['body_placement_initial_m'],
                        placement_final_m=report['body_placement_final_m'])
                rotation = Rotation.from_rotvec(new['global_orient'].reshape(3)) * Rotation.from_rotvec(old['global_orient'].reshape(3)).inv()
                results.append(dict(frame=frame, **info[frame], **raw,
                    B_receipt_sha256=desc['sha256'], B_winner_sha256=base.sha(out/'B'/frame/'winner-camera-space.npz'),
                    global_orientation_delta_deg=float(np.degrees(rotation.magnitude())),
                    camera_translation_delta_m=float(np.linalg.norm(new['camera_translation']-old['camera_translation']))))
                print('CONTROL_FIT_COMPLETE', frame, raw['A']['common_body']['max_over_box_height'],
                      raw['B']['common_body']['max_over_box_height'], flush=True)
        base.check_pins(pins)
        write(out/'result.json', dict(state='completed', plan_sha256=base.sha(out/'plan.json'),
            targets_sha256=base.sha(out/'targets.npz'), records=results, protected_frames=protected,
            gpu_calls=0, gaussian_updates=0, quality_accepted=False, elapsed_s=time.monotonic()-started,
            scope='B tests sensitivity to unsupported feet in the original fitter. Common detector residuals '
                  'describe these six training-side frames. Hidden geometry remains unevaluated.'))
    except BaseException as error:
        write(out/'failure.json', dict(state='failed', error=str(error), elapsed_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
