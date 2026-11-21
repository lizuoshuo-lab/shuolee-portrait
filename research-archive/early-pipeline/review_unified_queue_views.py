"""Compare each queued unified terminal with its exact frozen-fit source."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_unified_reused_fit_queue as queue
import review_unified_cold_views as views


common = queue.common
read, pin, require, sha = views.read, views.pin, views.require, views.sha
OLD_PROJECTS = {
    '001': 'Shuolee-smpl-view-coverage',
    '005': 'Shuolee-smpl-sixworker-ffprobe-005',
    '008': 'Shuolee-smpl-source-visibility',
}
SCALE_001 = 'd96b3ce7f8e1b3202f9486600e12d6df8f3bd73961f72e91100d7fec05f79923'


def stock_closeup(old, reference):
    root = Path(old.provenance['run'])
    result = read(root/'result.json')
    replay_path = root/result['bound_replay']
    bound_path = replay_path.parents[1]/'bound-replay.json'
    bound = read(bound_path)
    world = bound_path.parent/'world.ply'
    pin(world, bound['inputs_sha256']['world_ply'])
    vertices = PlyData.read(world, mmap=False)['vertex'].data
    xyz = np.column_stack([vertices[field] for field in ('x', 'y', 'z')])
    slot = old.slots[reference]
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as fit:
        close = views.face_dolly(old.rows[reference], fit['observed_native_keypoints'][slot, 67:118],
                                fit['joints_camera'][slot, 67:118], xyz)
    return dict(w2c=close[0]['w2c'], crop=list(close[2])) if close else None


def make_pairs(subject, old, baseline_sha):
    rows = old.rows
    reference = old.provenance['reference_frame_id']
    chronology = sorted((row for frame, row in rows.items() if frame != reference),
                        key=lambda row: (row['source_frame_index'], row['frame']))
    recorded_close = None
    audit_sha = None
    if subject == '001':
        path = views.LAB/'shuolee-scale-coverage-001/result.json'
        audit_sha = pin(path, SCALE_001)
        audit = read(path)
        require(audit['terminal_sha256'] == baseline_sha and len(audit['records']) == 3
                and audit['records'][0]['frame'] == reference, '001 scale audit is for a different source terminal.')
        chosen = []
        for record in audit['records']:
            frame = record['frame']
            row = rows[frame]
            require(all(record[key] == row[key] for key in ('source_frame_index', 'source_sha256', 'mask_sha256')),
                    f'Archived native camera input changed: {frame}')
            chosen.append(frame)
        record = audit['records'][0]
        if record['closeup_1x']:
            close = record['closeup_1x']['camera']
            require(all(close[key] == rows[reference][key] for key in ('frame', 'K', 'wh')),
                    'Archived dolly changed focal length or native raster.')
            panel, = [item for item in record['panels'] if item['file'].endswith('-dolly.png')]
            pin(path.parent/panel['file'], panel['sha256'])
            recorded_close = dict(w2c=close['w2c'], crop=panel['crop_xyxy'])
    else:
        require(len(chronology) >= 2, 'Need distinct matched early and late source frames.')
        chosen = [reference, chronology[0]['frame'], chronology[-1]['frame']]
        recorded_close = stock_closeup(old, reference)

    require(len(set(chosen)) == 3 and set(chosen).isdisjoint(old.provenance['protected_holdouts']),
            'Native comparisons must be three distinct training frames.')
    pairs = []
    for role, frame in zip(('reference', 'early', 'late'), chosen, strict=True):
        row = rows[frame]
        x0, y0, x1, y1 = row['initial_box_xyxy']
        spec = dict(frame=frame, crop=views.native_box([x0-64, y0-64, x1+64, y1+64], row['wh']),
                    score_mask=row['mask'], score_mask_sha256=row['mask_sha256'],
                    own_mask_sha256=row['mask_sha256'], pose_sha256=views.pose_digest(old, frame))
        pin(Path(row['image']), row['source_sha256'])
        pin(Path(row['mask']), row['mask_sha256'])
        if role == 'reference' and row.get('face_box_xyxy'):
            a, b, c, d = row['face_box_xyxy']
            spec['head_crop'] = views.native_box([a-24, b-24, c+24, d+24], row['wh'])
        if role == 'reference' and recorded_close:
            spec['dolly'] = recorded_close
        pairs.append(dict(label=role, archived_preferred_frame=frame if audit_sha else None,
                          exact_native_source_and_camera=True, exact_fitted_pose=True,
                          old=dict(spec), new=dict(spec)))
    return pairs, audit_sha


def preflight(subject):
    root = common.root_for(subject)
    run = root/'run'
    plan_path = root/'preregistration.json'
    registration = read(plan_path)
    registration_sha = sha(plan_path)
    spec = queue.SUBJECTS[subject]
    old_run = views.LAB/spec['parent']/'run'
    old_project = Path('/mnt/d/Projects')/OLD_PROJECTS[subject]
    require(registration['schema'] == 'shuolee-unified-reused-fit-v1'
            and registration['subject'] == subject and registration['candidate_commit'] == queue.COMMIT
            and registration['baseline_run'] == registration['copied_fitting_input_from'] == str(old_run)
            and registration['baseline_result_sha256'] == spec['result']
            and registration['fit_reused'] is True and registration['fit_calls'] == 0
            and registration['reference_frame'] == spec['reference']
            and registration['source_video_sha256'] == spec['video']
            and registration['camera_config_sha256'] == spec['camera']
            and registration['retained_holdouts'] == list(spec['holdouts']),
            'Unexpected reused-fit source registration.')
    if (root/'launch.json').is_file():
        launched = read(root/'launch.json')
        require(launched['candidate_commit'] == queue.COMMIT
                and launched['preregistration_sha256'] == registration_sha,
                'Launch does not bind this preregistration.')
    for name in queue.base.COPY:
        digest = registration['staged_sha256'][name]
        pin(run/name, digest)
        pin(old_run/name, digest)
    old = views.planned_model(old_run, old_project, spec['result'])
    require(set(old.rows) == set(registration['selected_frames']) and len(old.rows) == spec['frames']
            and old.provenance['protected_holdouts'] == list(spec['holdouts']),
            'Frozen native rows differ from the registered fit.')
    pairs, audit_sha = make_pairs(subject, old, spec['result'])
    plan = dict(schema='shuolee-unified-cold-views-v1', subject=subject, asset=str(views.ASSET),
                baseline_label='Exact frozen-fit source terminal', new_label='Unified / frozen fit, fresh Gaussians',
                old=old.provenance, reused_fit_preregistration_sha256=registration_sha,
                baseline_camera_audit_sha256=audit_sha,
                view_selection='archived_exact_terminal_camera_audit' if audit_sha else
                               'frozen_reference_and_chronological_endpoints_excluding_reference',
                closeup_policy='fixed recorded source-terminal dolly when safe, otherwise native crop',
                exact_fitted_archive=True, exact_native_common_training_frames=sorted(old.rows), pairs=pairs)
    status_path = run/'status.json'
    status = read(status_path) if status_path.is_file() else dict(state='staged')
    wrapper_path = root/'wrapper-terminal.json'
    if status['state'] != 'completed' or not wrapper_path.is_file():
        return dict(plan, state='terminal_not_ready', reconstruction_state=status['state'])

    wrapper = read(wrapper_path)
    require(wrapper['state'] == 'completed' and wrapper['candidate_commit'] == queue.COMMIT
            and wrapper['fit_reused'] is True and wrapper['fit_calls'] == 0
            and wrapper['selected_frames'] == registration['selected_frames'],
            'Wrong completed unified terminal.')
    terminal_path = run/'result.json'
    pin(terminal_path, wrapper['result_sha256'])
    terminal = read(terminal_path)
    require(terminal['reused_fit_preregistration_sha256'] == registration_sha
            and terminal['fit_reused'] is True and terminal['fit_calls'] == 0
            and terminal['gaussian_from_scratch'] is True and terminal['from_scratch'] is False
            and terminal['sh_degree'] == 0 and terminal['frames'] == spec['frames']
            and terminal['points'] == wrapper['points'] and status['sha256'] == terminal['sha256'],
            'Unified model does not bind this completed frozen-fit run.')
    new = views.planned_model(run, common.PROJECT, wrapper['result_sha256'])
    require(new.rows == old.rows and all(new.provenance[key] == old.provenance[key]
            for key in ('fitted_sha256', 'training_inputs_sha256', 'reference_frame_id', 'protected_holdouts')),
            'Candidate source pixels, cameras or fitted poses changed.')
    return dict(plan, state='ready', new=new.provenance)


def render(plan, output):
    require(plan['state'] == 'ready', 'Wait for the registered terminal to finish.')
    with views.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        plan_path = output/'plan.json'
        views.save(plan_path, plan)
        for side in ('old', 'new'):
            with (output/f'{side}.log').open('x', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-B', views.__file__, '--worker', str(plan_path), side, str(lease.fileno())],
                               pass_fds=(lease.fileno(),), stdout=log, stderr=subprocess.STDOUT, check=True)
    old, new = (read(output/f'{side}-result.json') for side in ('old', 'new'))
    comparisons = []
    for entry, left, right in zip(plan['pairs'], old['records'], new['records'], strict=True):
        require(left['native_camera'] == right['native_camera'] and left['frame'] == right['frame'],
                'Native cameras differ in actual replay.')
        role = entry['label']
        native = views.comparison(output, entry, left, right, 'native', plan['new_label'])
        first, second = left['closeup'], right['closeup']
        if first and second and (first['kind'] != second['kind'] or first['camera'] != second['camera']):
            # If only one terminal passes the dolly depth guard, use both real native head crops.
            fallback = []
            for side, record in (('old', left), ('new', right)):
                crop = entry[side].get('head_crop')
                head = output/f'{role}-{side}-head.png'
                if crop and head.is_file():
                    fallback.append(dict(record, closeup=dict(kind='native_head_crop',
                        camera=record['native_camera'], panel=dict(file=head.name, sha256=sha(head), crop_xyxy=crop))))
        else:
            fallback = [left, right]
        closeup = (views.comparison(output, entry, *fallback, 'closeup', plan['new_label'])
                   if len(fallback) == 2 else None)
        comparisons.append(dict(label=role, native=native, closeup=closeup))
    path = output/'result.json'
    views.save(path, dict(state='completed', subject=plan['subject'], plan_sha256=sha(plan_path),
                         reviewer_sha256=sha(Path(__file__)), worker_sha256=sha(Path(views.__file__)),
                         old=old, new=new, comparisons=comparisons, optimizer_updates=0, quality_accepted=False,
                         scope='Matched training cameras, native crops and synthetic fixed-focal dolly when valid.'))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    parser.add_argument('subject', choices=queue.SUBJECTS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan = preflight(args.subject)
    if args.action == 'check':
        print(json.dumps(dict(plan=plan, gpu_calls=0)))
        return
    output = (args.output or views.LAB/f'shuolee-smpl-unified-{args.subject}-matched-views').resolve()
    require(not output.exists() and output.parent == views.LAB, 'Use a fresh separate directory under the lab.')
    path = render(plan, output)
    print(json.dumps(dict(comparison_receipt=str(path), sha256=sha(path))))


if __name__ == '__main__':
    main()
