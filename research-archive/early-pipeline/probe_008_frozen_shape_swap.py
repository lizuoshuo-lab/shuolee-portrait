"""Isolate the effect of saved SMPL-X shape on archived 008 shoulder projections."""
import argparse
import fcntl
from importlib.util import find_spec
from pathlib import Path

from probe_local_articulation_feasibility import LAB, LOCK, PROJECTS, SOURCES, FIELDS
from probe_local_articulation_feasibility import project, replay_error, require, screen, seq

import numpy as np


RUN = LAB / 'shuolee-smpl-visibility-008/run'
COMPARE = LAB / 'shuolee-smpl-visibility-008-upperbody/shoulder-anchor-frozen-comparison.json'
COMPARE_SHA = 'f1dd377e9df2c6929d341ac6312cd701b8bef5092768d28c5df89fc47917f5d8'
OUTPUT = LAB / 'shuolee-frozen-shape-swap-008'


def load_pose(path, row, expected_sha, frozen):
    seq.check_bound({path: expected_sha})
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    report_path = path.with_name('result.json')
    report = seq.rt.read(report_path)
    seq.check_result(report, row)
    require(report['freeze_betas'] is frozen, f'Unexpected shape policy: {path}')
    return arrays, report_path


def prepare():
    spec = SOURCES['008']
    pins = {(RUN / name).resolve(): digest for name, digest in zip(FIELDS, spec['digests'], strict=True)}
    for path in (Path(__file__).resolve(), Path(project.__code__.co_filename).resolve()):
        pins[path] = seq.rt.sha(path)
    pins[COMPARE] = COMPARE_SHA
    seq.check_bound(pins)
    terminal, cfg, fit, selection, shared, run, launch, camera = (seq.rt.read(RUN / name) for name in FIELDS)
    require(fit['selection'] == selection and terminal['state'] == 'completed'
            and cfg['capture_layout'] == camera['capture_layout'], 'Saved 008 contract differs.')
    official = Path(cfg['official_smplifyx']['source_root'])
    assets = Path(run['assets'])
    source = PROJECTS / 'Shuolee-smpl-source-visibility'
    pins.update({source / name: digest for name, digest in launch['code_sha256'].items()})
    pins.update({official / name: digest for name, digest in seq.official.PINNED.items()})
    pins.update({Path(name): digest for name, digest in fit['source']['input_files'].items()})
    pins.update({assets / row['path']: row['sha256'] for row in seq.rt.read(source / 'configs/server-assets.json')['files']
                 if row['path'].startswith(('smplx/', 'vposer/'))})
    smplx = Path(find_spec('smplx').origin).parent
    for path in (smplx/'body_models.py', smplx/'lbs.py', smplx/'vertex_joint_selector.py',
                 source/'src/avatar_core/shuolee_vposer.py',
                 source/'third_party/human_body_prior/human_body_prior/models/vposer_model.py'):
        pins[path] = seq.rt.sha(path)
    seq.check_bound(pins)
    inputs = seq.load_inputs(RUN/'data/inputs.json', RUN/'data/observations.npz',
                             camera_length_contract=cfg['camera_length_contract'],
                             reference_frame=selection['reference_frame'], capture_layout=cfg['capture_layout'])
    holdouts = selection['retained_holdouts']
    require(holdouts == inputs['scope']['retained_holdouts'] == launch['expected_holdouts']
            and inputs['confidence'] == fit['source']['confidence']
            and all(not inputs['targets'][inputs['ids'].index(f)].any() for f in holdouts),
            'Archived 008 heldout or confidence differs.')
    compare = seq.rt.read(COMPARE)
    require(compare['overlap'] == len(compare['rows']) == 15
            and not compare['pose_updated'], 'Expected 15 saved anchor and frozen pairs.')
    beta = np.asarray(shared['betas'], dtype=np.float32).reshape(1, 10)
    cases = []
    for saved in compare['rows']:
        frame = saved['frame']
        require(frame not in holdouts and frame in inputs['scope']['fit_frame_ids']
                and selection['reasons'][frame]['status'] == 'pose_screen_rejection',
                f'Unexpected overlap frame {frame}')
        i = inputs['ids'].index(frame)
        row, target, K = inputs['rows'][i], inputs['targets'][i], inputs['K'][i]
        paired = {}
        for arm, directory in (('anchor', 'shape-anchors'), ('frozen', 'frames')):
            path = RUN/'fit'/directory/frame/'winner-camera-space.npz'
            digest = saved[f'{arm}_winner_sha256']
            arrays, report = load_pose(path, row, digest, arm == 'frozen')
            require(np.array_equal(arrays['K_integer_centers'], K), f'Native K differs: {path}')
            pins.update({path: digest, report: seq.rt.sha(report)})
            paired[arm] = arrays
        require(np.array_equal(paired['frozen']['betas'], beta)
                and not np.array_equal(paired['anchor']['betas'], beta)
                and target[2, 2] > 0 and target[5, 2] > 0, f'Invalid shape or shoulder observation: {frame}')
        pins[Path(row['image'])] = row['source_sha256']
        pins[Path(row['mask'])] = row['mask_sha256']
        cases.append((row, target, K, paired, saved, selection['reasons'][frame]))
    seq.check_bound(pins)
    plan = dict(
        frames=[row['frame'] for row, *_ in cases], protected_holdouts=holdouts,
        source_pins={str(path): digest for path, digest in pins.items()},
        perturbation='Cross the saved independent anchor beta and shared beta on each frozen saved pose.',
        comparison_scope='Each pair holds latent, root, expression, hands, translation, K and topology fixed. '
                         'Independent anchor beta is diagnostic and will never be handed to training.',
        acceptance_scope='Report native supported body, foot and face residuals under the original screen; '
                         'clothed shoulder centers and camera calibration remain uncertain.',
        fit_calls=0, gpu_calls=0, optimizer_updates=0)
    return plan, cases, official, assets


def evaluate(cases, official, assets):
    import torch

    seq.rt.configure(assets)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    body, vposer, _ = seq.official.create_models(
        official_root=official, assets=assets, device='cpu', static_topology=False)
    body.eval()
    records = []
    for row, target, K, paired, saved, admission in cases:
        native = {}
        shoulders = target[[2, 5], :2]
        for pose_name, arrays in paired.items():
            params = {name: arrays[name] for name, _ in body.named_parameters()}
            beta_order = (pose_name, 'anchor' if pose_name == 'frozen' else 'frozen')
            for beta_name in beta_order:
                beta = paired[beta_name]['betas']
                body.reset_params(**dict(params, betas=beta))
                with torch.no_grad():
                    values = tuple(x.numpy() for x in project(body, vposer, torch.from_numpy(arrays['K_compute_float32']),
                        torch.from_numpy(arrays['camera_translation']), torch.from_numpy(arrays['pose_embedding']).reshape(32)))
                angle, vertices, joints, uv = values
                require(np.array_equal(angle, arrays['body_pose_axis_angle']), 'Pose changed during beta swap.')
                replay = None
                if beta_name == pose_name:
                    replay = replay_error(values, arrays)
                    require(replay['uv_px'] <= .005 and max(replay['vertex_m'], replay['joint_m']) <= 2e-6,
                            f'Original archive did not replay: {row["frame"]}/{pose_name}')
                trial = dict(arrays, betas=beta, vertices_camera=vertices, joints_camera=joints,
                             joints_projected_native=uv)
                scores, flags, _, _ = screen._candidate(row, trial, target, K, beta, None, None, True)
                error = np.linalg.norm(uv[0] - target[:, :2], axis=1)
                maximum = {}
                for label, slots in (('body', screen.BODY), ('face', screen.FACE), ('feet', screen.FEET)):
                    active = slots[target[slots, 2] > 0]
                    maximum[label] = float(error[active].max()) if len(active) else None
                joint_span = abs(float(uv[0, 2, 0] - uv[0, 5, 0]))
                shoulder_joints = joints[0, [2, 5]]
                native[f'{pose_name}_pose_{beta_name}_beta'] = dict(
                    shoulder_span_px=joint_span, shoulder_max_px=float(error[[2, 5]].max()),
                    shoulder_native_xy=uv[0, [2, 5]].tolist(),
                    shoulder_distance_m=float(np.linalg.norm(shoulder_joints[0] - shoulder_joints[1])),
                    shoulder_center_depth_m=float(shoulder_joints[:, 2].mean()),
                    max_supported_error_px=maximum, region_scores=scores, screen_flags=flags, replay=replay)
        for arm in ('anchor', 'frozen'):
            stock = native[f'{arm}_pose_{arm}_beta']
            require(abs(stock['shoulder_span_px'] - saved[f'{arm}_width']) < .005
                    and abs(stock['shoulder_max_px'] - saved[f'{arm}_max_shoulder_error']) < .005,
                    f'Shoulder comparison changed: {row["frame"]}/{arm}')
        stock = native['frozen_pose_frozen_beta']
        require(stock['screen_flags'] == admission['screen_flags'] and stock['region_scores'] == admission['scores'],
                f'Original frozen screen did not replay: {row["frame"]}')
        records.append(dict(frame=row['frame'], source_frame_index=row['source_frame_index'],
                            detector_span_px=abs(float(shoulders[0, 0] - shoulders[1, 0])),
                            detector_native_xy=shoulders.tolist(), cases=native))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'probe'))
    args = parser.parse_args()
    require(not OUTPUT.exists(), 'Use a fresh output and preserve existing evidence.')
    plan, cases, official, assets = prepare()
    if args.mode == 'check':
        print(dict(frames=len(cases), pins=len(plan['source_pins']), body_forwards=0))
        return
    with LOCK.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUTPUT.mkdir()
        seq.rt.save(OUTPUT/'preregistration.json', plan)
        try:
            records = evaluate(cases, official, assets)
            seq.check_bound(plan['source_pins'])
        except Exception as exc:
            seq.rt.save(OUTPUT/'failure.json', dict(state='failed', error=str(exc)))
            raise
        seq.rt.save(OUTPUT/'result.json', dict(state='shape_effect_measured', records=records,
            preregistration_sha256=seq.rt.sha(OUTPUT/'preregistration.json'), fit_calls=0, gpu_calls=0,
            optimizer_updates=0, quality_accepted=False))
        print(dict(state='shape_effect_measured', frames=len(records)))


if __name__ == '__main__':
    main()
