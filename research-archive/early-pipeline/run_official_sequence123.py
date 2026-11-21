"""Thin named-frame orchestration of the pinned official SMPLify-X entry.

One body/VPoser instance is loaded. Every fit resets its own zero/mean pose,
uses one explicitly supplied frozen beta, and runs the upstream five stages.
No sequence solver, temporal loss, GVHMR initialization or Gaussian training.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback

import numpy as np

ENTRY_SHA256 = '9dd537a60bffb00263214c6e27a3a132136a8b588662596f83681c05569e213f'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_inputs(args):
    """CPU input join only; never open the prior/crop/optimized-fit entries."""
    sys.path.insert(0, str(Path(args.project_root).resolve() / 'src'))
    from shuolee_portrait.camera_world_units import normalize_camera_w2c
    from smplify_score_contract123 import relative_confidence
    import official_smplifyx_entry123 as entry
    need(sha(entry.__file__) == args.entry_sha256, 'Official adapter source changed.')
    data = Path(args.data_root).resolve()
    inputs_path, obs_path = data / 'data/inputs.json', data / 'data/observations.npz'
    length_path = Path(args.length_contract_json).resolve()
    source_manifest = read(length_path)
    for name, path in [('frames', inputs_path), ('observations', obs_path)]:
        binding = source_manifest[name]
        need(Path(binding['path']).resolve() == path and sha(path) == binding['sha256'],
             'Camera-length manifest does not bind these ' + name + '.')
    rows = read(inputs_path)['records']
    ids = [r['frame'] for r in rows]
    need(ids and len(set(ids)) == len(ids) and
         all(isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in ids),
         'Input frame names must be unique safe identifiers.')
    need(args.reference_frame_id in ids, 'Reference frame must belong to the original roster.')
    selected = list(args.frame_ids) if args.frame_ids else ids
    need(selected and len(set(selected)) == len(selected) and set(selected) <= set(ids),
         'Requested frame subset must contain unique known identities.')
    with np.load(obs_path, allow_pickle=False) as f:
        obs = {key: f[key].copy() for key in ('frame_ids', 'keypoints', 'scores', 'boxes', 'flip_disagreement')}
    need(obs['frame_ids'].astype(str).tolist() == ids, 'Observation frame order differs from input identities.')
    need(obs['keypoints'].shape == (len(ids), 133, 2) and np.isfinite(obs['keypoints']).all(),
         'Expected finite native RTMW133 observations.')
    confidence, q90 = relative_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'])
    need(confidence.shape == (len(ids), 133), 'Expected RTMW133 confidence.')
    # The same full-roster confidence adapter as the accepted single-frame entry.
    targets = np.zeros((len(ids), 118, 3), np.float32)
    dest = np.r_[entry.COCO, np.arange(19, 25), np.arange(25, 67), np.arange(67, 118)]
    source = np.r_[np.arange(23), np.arange(91, 133), np.arange(40, 91)]
    targets[:, dest, :2] = obs['keypoints'][:, source]
    targets[:, dest, 2] = confidence[:, source]
    slots = [ids.index(frame) for frame in selected]
    chosen = [rows[i] for i in slots]
    source_E = np.asarray([r['w2c'] for r in chosen], np.float64)
    units = normalize_camera_w2c(source_E, source_manifest['camera_length_contract'])
    K = np.asarray([r['K_integer_centers'] for r in chosen], np.float64)
    need(K.shape == (len(chosen), 3, 3) and np.isfinite(K).all(), 'Invalid complete native K.')
    for i, row in enumerate(chosen):
        expected = K[i].copy(); expected[0, 2] += .5; expected[1, 2] += .5
        need(np.array_equal(expected, row['K']), 'Input integer/corner K disagrees: ' + row['frame'])
        need(Path(row['image']).is_absolute() and Path(row['mask']).is_absolute(), 'Image/mask paths must be absolute.')
    bound = {str(p): sha(p) for p in (inputs_path, obs_path, length_path, Path(entry.__file__), Path(__file__),
              Path(__file__).with_name('smplify_score_contract123.py'),
              Path(args.project_root) / 'src/shuolee_portrait/camera_world_units.py')}
    return entry, chosen, targets[slots], K, units, bound, dict(
        frame_ids=selected, input_roster_frames=len(ids), reference_frame_id=args.reference_frame_id,
        reference_in_subset=args.reference_frame_id in selected,
        per_joint_score_q90=q90.tolist(), score_semantics='Relative RTMW/flip heuristic, not calibrated probability or physical visibility',
        input_files=bound, source_camera_length_contract=dict(units.contract),
        old_prior_or_optimized_fit_loaded=False, gvhmr_used=False, quality_accepted=False)


def checked_image(row):
    from PIL import Image
    image = None
    for key, digest in [('image', 'source_sha256'), ('mask', 'mask_sha256')]:
        raw = Path(row[key]).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == row[digest], key + ' changed: ' + row['frame'])
        with Image.open(io.BytesIO(raw)) as im:
            need(list(im.size) == row['wh'], key + ' native dimensions differ: ' + row['frame'])
            im.load()
            if key == 'image':
                image = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
    return image


def root_to_world(arrays, E, root_template, root_shape_dirs, root_expr_dirs):
    """LBS pelvis pivot, not a guessed identity or origin translation.

    SMPL-X rotates its articulated body about shape-dependent J0. The old
    export field world_translation denotes that pelvis in world coordinates:
    world=(root_zero_vertices-J0) @ root_rotation.T + world_translation.
    """
    from scipy.spatial.transform import Rotation
    beta = arrays['betas'].reshape(10)
    expression = arrays['expression'].reshape(10)
    J0 = root_template + root_shape_dirs @ beta + root_expr_dirs @ expression
    global_R = Rotation.from_rotvec(arrays['global_orient'].reshape(3)).as_matrix()
    camera_t = arrays['camera_translation'].reshape(3)
    R, t = E[:3, :3], E[:3, 3]
    return (R.T @ global_R, (J0 + camera_t - t) @ R, J0)


def run(args):
    started = time.monotonic()
    entry, rows, targets, K, units, bound, provenance = load_inputs(args)
    if args.inspect_only:
        for row in rows:
            checked_image(row)
        print(json.dumps(dict(provenance, state='cpu_inputs_checked', model_loaded=False, gpu_started=False), indent=2))
        return
    need(args.shared_betas_npz and args.output_dir, 'A shared-beta NPZ and a new output directory are required.')
    need(os.name == 'posix', 'Run fitting in the existing Linux/WSL environment.')
    import fcntl
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never restart an existing run.
    for name in ('frames', 'fit', 'data'):
        (output / name).mkdir()
    active = None
    progress = dict(state='preparing', pid=os.getpid(), started=now(), frames_total=len(rows), frames_completed=0)
    save(output / 'progress.json', progress)
    try:
        beta_path = Path(args.shared_betas_npz).resolve()
        beta_hash = sha(beta_path)
        with np.load(beta_path, allow_pickle=False) as f:
            shared = np.asarray(f[args.shared_betas_key], dtype=np.float32).copy()
        need(shared.shape in [(10,), (1, 10)] and np.isfinite(shared).all(), 'One finite official shared beta vector is required.')
        shared = shared.reshape(1, 10)
        bound[str(beta_path)] = beta_hash
        provenance.update(shared_betas_path=str(beta_path), shared_betas_key=args.shared_betas_key,
                          shared_betas_sha256=beta_hash, freeze_betas=True,
                          initialization='Independent official zero/mean pose each frame; explicit shared shape only')
        save(output / 'input-provenance.json', provenance)
        entry.adapted_entry_source(args.official_root)
        import torch
        import smplx
        utils, _, _, _ = entry.load_official(args.official_root)
        mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                                       use_face_contour=False, openpose_format='coco25')
        need(len(mapping) == 118, 'Official mapping must contain 118 anchors.')
        runtime, assets = Path(args.runtime_root), Path(args.assets_root)
        loader = entry.module(runtime / 'src/avatar_core/shuolee_vposer.py', '_sequence123_vposer_loader')
        loader.MODEL = assets / 'vposer/V02_05'; loader.CODE = runtime / 'third_party/human_body_prior'
        vposer, _, vposer_meta = loader.load_vposer(args.device)
        body = smplx.SMPLX(str(assets / 'smplx/SMPLX_NEUTRAL.npz'), batch_size=1,
            num_betas=10, num_expression_coeffs=10, use_pca=True, num_pca_comps=12,
            flat_hand_mean=False, use_face_contour=False, create_body_pose=False, create_transl=False,
            joint_mapper=utils.JointMapper(mapping), dtype=torch.float32).to(args.device)
        # LBS joint regression is linear in beta/expression; no extra body forward.
        cpu = lambda value: value.detach().cpu().numpy().astype(np.float64)
        reg = cpu(body.J_regressor)[0]
        root_template = reg @ cpu(body.v_template)
        root_shape = np.einsum('v,vck->ck', reg, cpu(body.shapedirs))
        root_expr = np.einsum('v,vck->ck', reg, cpu(body.expr_dirs))
        results, reports, training_rows = [], [], []
        faces = None
        for i, row in enumerate(rows):
            active = row['frame']
            for path, digest in bound.items():
                need(sha(path) == digest, 'Bound input/source changed: ' + path)
            image = checked_image(row)
            frame_out = output / 'frames' / active; frame_out.mkdir()
            save(output / 'progress.json', dict(progress, state='official_frame_fitting', frame_id=active,
                 frame_index=i, frames_completed=i, elapsed_seconds=time.monotonic()-started))
            frame_started = time.monotonic()
            arrays, report = entry.fit_one(official_root=args.official_root, image=image, keypoints=targets[i:i+1],
                K_integer_centers=K[i], body_model=body, vposer2=vposer,
                result_path=frame_out / 'official-result.pkl', shared_betas=shared, freeze_betas=True,
                device=args.device, diff_path=frame_out / 'input-output-adapter.diff')
            need(np.array_equal(arrays['betas'], shared), 'Official fit changed the frozen shared beta.')
            if faces is None:
                faces = arrays['faces'].copy()
            need(np.array_equal(faces, arrays['faces']), 'Frame topology changed.')
            E = units.w2c_meters[i]
            vc = arrays['vertices_camera'][0]
            vw = (vc.astype(np.float64) - E[:3, 3]) @ E[:3, :3]
            root_R, root_t, pivot = root_to_world(arrays, E, root_template, root_shape, root_expr)
            arrays.update(vertices_world=vw[None], root_rotation=root_R[None], world_translation=root_t[None],
                          root_pivot_model=pivot[None])
            save_npz(frame_out / 'winner-camera-space.npz', **arrays, frame_id=np.asarray(active),
                source_frame=np.asarray(row['source_frame_index']), image_sha256=np.asarray(row['source_sha256']),
                mask_sha256=np.asarray(row['mask_sha256']), w2c=E, source_w2c=units.source_w2c[i])
            report.update(state='complete', frame_id=active,
                          source_frame_index=row['source_frame_index'], source_image_sha256=row['source_sha256'],
                          elapsed_seconds=time.monotonic()-frame_started,
                          image_quality_accepted=False, physical_quality_accepted=False)
            save(frame_out / 'result.json', report)
            results.append(arrays); reports.append(report)
            corner_K = K[i].copy(); corner_K[0, 2] += .5; corner_K[1, 2] += .5
            training_rows.append(dict(row, K=corner_K.tolist(), K_integer_centers=K[i].tolist(),
                                      w2c=E.tolist(), source_w2c=units.source_w2c[i].tolist()))
            print(json.dumps(dict(frame_id=active, frames_completed=i+1, frames_total=len(rows),
                                  elapsed_seconds=report['elapsed_seconds'], winning_loss=report['winning_loss'])), flush=True)
        for path, digest in bound.items():
            need(sha(path) == digest, 'Bound input/source changed before export: ' + path)
        normalized_contract = dict(translation_unit='meters', meters_per_camera_unit=1.,
            basis=units.contract['basis'], evidence='Already converted once from explicit source camera convention. ' + units.contract['evidence'])
        arrays = {key: np.concatenate([value[key] for value in results], axis=0) for key in (
            'vertices_camera', 'vertices_world', 'joints_camera', 'joints_projected_native', 'root_rotation',
            'world_translation', 'root_pivot_model', 'global_orient', 'camera_translation', 'body_pose_axis_angle',
            'pose_embedding', 'expression', 'jaw_pose', 'left_hand_pose', 'right_hand_pose', 'leye_pose', 'reye_pose')}
        arrays.update(faces=faces, frame_ids=np.asarray([r['frame'] for r in rows]),
            reference_frame_id=np.asarray(args.reference_frame_id), shared_betas=shared, betas=shared,
            source_frames=np.asarray([r['source_frame_index'] for r in rows], np.int64),
            image_sha256=np.asarray([r['source_sha256'] for r in rows]), mask_sha256=np.asarray([r['mask_sha256'] for r in rows]),
            K_integer_centers=K, w2c=units.w2c_meters, source_w2c=units.source_w2c,
            observed_native_keypoints=targets, camera_length_contract_json=np.asarray(json.dumps(normalized_contract)),
            source_camera_length_contract_json=np.asarray(json.dumps(dict(units.contract))),
            world_scale=np.asarray(1.), root_translation_semantics=np.asarray('shape-dependent LBS pelvis in world; not affine origin'))
        save_npz(output / 'fit/fitted.npz', **arrays)
        save(output / 'data/training-inputs.json', dict(records=training_rows,
            pixel_domain='undistorted_integer_centers',
            intrinsic_origin='Provided calibration with conditional output-size/resize/center-crop mapping; input K unchanged, not COLMAP-estimated',
            K_convention='K_integer_centers plus exactly 0.5 on cx/cy; renderer subtracts 0.5',
            camera_length_contract=normalized_contract, source_camera_length_contract=dict(units.contract),
            pose_mode='named_official_per_frame_shared_shape', formal_calibration_accepted=False, quality_accepted=False))
        official_results = [dict(frame_id=r['frame'], path=str(output / 'frames' / r['frame'] / 'result.json'),
                                sha256=sha(output / 'frames' / r['frame'] / 'result.json')) for r in rows]
        receipt = dict(state='complete', frames=len(rows),
            elapsed_seconds=time.monotonic()-started, source=provenance, frame_results=reports, vposer=vposer_meta,
            fit_sha256=sha(output / 'fit/fitted.npz'), training_inputs_sha256=sha(output / 'data/training-inputs.json'),
            official_frame_results=official_results,
            sources=[dict(path=str(Path(p).resolve()), sha256=bound[str(Path(p))]) for p in (entry.__file__, __file__)],
            full_roster=len(rows)==provenance['input_roster_frames'], reference_in_subset=provenance['reference_in_subset'],
            world_export='(camera_vertices - normalized_E.translation) @ E.rotation; one source-E conversion only',
            shared_identity=True, physical_quality_accepted=False, quality_accepted=False, gaussian_training_started=False)
        save(output / 'fit/official-fit-result.json', receipt)
        save(output / 'progress.json', dict(state='complete', pid=None, frames_completed=len(rows), finished=now(),
                                           elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        save(output / 'failure.json', dict(frame_id=active, error_type=type(error).__name__, error=str(error),
             traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started))
        save(output / 'progress.json', dict(state='failed', pid=None, frame_id=active, finished=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('official-root', 'data-root', 'runtime-root', 'assets-root', 'project-root',
                'length-contract-json', 'reference-frame-id'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--shared-betas-npz'); parser.add_argument('--shared-betas-key', default='betas')
    parser.add_argument('--output-dir'); parser.add_argument('--frame-ids', nargs='+')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--entry-sha256', default=ENTRY_SHA256)
    parser.add_argument('--inspect-only', action='store_true', help='CPU file/identity validation only, without loading models.')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
