"""Export one completed VHAP59 mesh to the existing Gaussian input contract.

Interface only: GPU VPoser batch1 decoding and frozen SMPL-X replay, without
an optimizer, geometry admission, Gaussian training, or a new point cloud.
The public GaussianAvatars_SMPLX entry is incomplete; its existing local
full-body adapter is used explicitly. No official-fit receipt is fabricated.
"""
from pathlib import Path
import argparse
import ast
import fcntl
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
from run_official_gaussians123 import PROJECT, ASSETS, read, save, sha, now


HERE = Path(__file__).resolve().parent
DEMO = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
DEFAULT_RUN = DEMO / 'smplify-vhap-continuous'
DEFAULT_INPUTS = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/inputs.json')
FRAME_KEYS = ('transl', 'global_orient', 'jaw_pose', 'leye_pose', 'reye_pose',
              'expression', 'pose_embedding', 'left_hand_pose', 'right_hand_pose')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def reused_fixture():
    """Reuse the existing pure fixture function without importing its GPU probe."""
    path = HERE / 'preflight_vhap_direct_body123.py'
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'fixture']
    need(len(functions) == 1, 'Existing fixture function is absent or ambiguous.')
    node = functions[0]
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['fixture'], dict(path=str(path), sha256=sha(path),
        function='fixture', function_source_sha256=hashlib.sha256(
            ast.get_source_segment(source, node).encode()).hexdigest())


def checked_rows(inputs, saved):
    from PIL import Image
    document = read(inputs)
    rows = document['records']
    ids = np.asarray(saved['frame_ids']).astype(str).tolist()
    roster = [f'frame_{i:06d}' for i in range(1, 60)]
    need(len(ids) == len(set(ids)) == 59 and set(ids) == set(roster),
         'Checkpoint must contain exactly the complete59 named frames.')
    row_ids = [row['frame'] for row in rows]
    need(len(row_ids) == len(set(row_ids)) == 59 and set(row_ids) == set(ids),
         'Input manifest must contain those same59 identities exactly once.')
    by_id = {row['frame']: row for row in rows}
    checked = []
    for slot, frame in enumerate(ids):
        row = dict(by_id[frame])
        need(row['wh'] == [2704, 1520], 'This interface preserves native2704x1520 pixels.')
        E = np.asarray(row['w2c'], np.float64)
        Ki = np.asarray(row['K_integer_centers'], np.float64)
        Kc = np.asarray(row['K'], np.float64)
        expected = Ki.copy()
        expected[:2, 2] += .5
        need(E.shape == (4, 4) and np.array_equal(E, np.eye(4)),
             'Current VHAP checkpoint is camera-relative with identity w2c only.')
        need(Ki.shape == (3, 3) and np.isfinite(Ki).all()
             and np.array_equal(Kc, expected), 'Native integer/corner K contract differs.')
        need(np.array_equal(saved['K_integer_centers'][slot], Ki)
             and np.array_equal(saved['w2c'][slot], E),
             'Saved checkpoint and original input cameras differ: ' + frame)
        for key, digest_key in (('image', 'source_sha256'), ('mask', 'mask_sha256')):
            need(sha(row[key]) == row[digest_key], 'Changed source pixels: ' + frame + '/' + key)
            with Image.open(row[key]) as image:
                need(list(image.size) == row['wh'], 'Native pixel dimensions differ: ' + frame + '/' + key)
        for key in ('source_frame_index', 'time_seconds'):
            need(key in row, 'Missing source frame/PTS identity: ' + frame)
        checked.append(row)
    return document, checked


def run(args):
    need(os.name == 'posix', 'Use the existing local WSL CUDA environment.')
    source = Path(args.source_run).resolve()
    checkpoint_epoch = getattr(args, 'checkpoint_epoch', 30)
    checkpoint = source / f'epoch-{checkpoint_epoch:02d}/tracked-smplx-params.npz'
    inputs, out = Path(args.inputs).resolve(), Path(args.output_root).resolve()
    need(not out.exists(), 'Never overwrite or resume an existing interface export.')
    result = read(source / 'result.json')
    progress = read(source / 'progress.json')
    need(result.get('state') == 'complete_pending_visual_review'
         and progress.get('state') == 'complete_pending_visual_review'
         and progress.get('pid') is None and result.get('frames') == 59,
         'VHAP must record terminal all59 completion before interface export.')
    need(result.get('gaussian_training_started') is False,
         'Source result does not establish a fresh Gaussian handoff.')
    need(result.get('smplifyx_fitting_used') is True,
         'The completed source does not establish this SMPLify-X-to-VHAP route.')
    latest = read(source / 'latest-checkpoint.json')
    need(Path(latest['folder']).resolve() == checkpoint.parent
         and latest.get('epoch') == checkpoint_epoch, 'Requested checkpoint is not the saved final one.')
    source_paths = [checkpoint, inputs, source / 'result.json', source / 'progress.json',
                    source / 'config.json', source / 'latest-checkpoint.json', Path(__file__)]
    source_hashes = {str(path): sha(path) for path in source_paths}
    with np.load(checkpoint, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    config = read(source / 'config.json')
    need(config['model']['use_vposer'] is True
         and config['model']['use_hand_pca'] is True
         and config['model']['num_hand_pca_comps'] == 12
         and config['model']['use_dynamic_offset'] is False,
         'This exporter is only for the existing VPoser32/PCA12 shared-static-offset branch.')
    expected_shapes = dict(transl=(59, 3), global_orient=(59, 3), jaw_pose=(59, 3),
        leye_pose=(59, 3), reye_pose=(59, 3), expression=(59, 10), pose_embedding=(59, 32),
        left_hand_pose=(59, 12), right_hand_pose=(59, 12), betas=(10,),
        static_offset=(1, 10475, 3), vertices_camera=(59, 10475, 3), faces=(20908, 3),
        K_integer_centers=(59, 3, 3), w2c=(59, 4, 4))
    for key, shape in expected_shapes.items():
        need(key in saved and saved[key].shape == shape and np.isfinite(saved[key]).all(),
             'Checkpoint shape/finite contract differs: ' + key)
    need(saved['faces'].dtype.kind in 'iu', 'Face topology must contain integer indices.')
    document, source_rows = checked_rows(inputs, saved)
    ids = [row['frame'] for row in source_rows]
    need(args.reference_frame in ids, 'Reference must identify an existing original frame.')
    reference = ids.index(args.reference_frame)
    order = [reference] + [i for i in range(59) if i != reference]
    rows = [source_rows[i] for i in order]
    ordered = dict(saved)
    for key in (*FRAME_KEYS, 'frame_ids', 'vertices_camera', 'K_integer_centers', 'w2c'):
        ordered[key] = saved[key][order].copy()
    if 'joints_projected_native' in saved:
        need(saved['joints_projected_native'].shape[0] == 59,
             'Saved joint projection roster differs.')
        ordered['joints_projected_native'] = saved['joints_projected_native'][order].copy()
    fixture, fixture_source = reused_fixture()
    source_hashes[fixture_source['path']] = fixture_source['sha256']
    began = time.monotonic()

    def budget():
        need(time.monotonic() - began < args.max_seconds, 'Bounded interface-export time exhausted.')

    with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=False)
        fit_dir = out / 'fit'
        fit_dir.mkdir()
        try:
            import torch
            from scipy.spatial.transform import Rotation
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            need(torch.cuda.is_available(), 'CUDA is required; never substitute CPU latent decoding.')
            sys.path.insert(0, str(PROJECT / 'src'))
            from shuolee_portrait import runtime as rt
            rt.configure(ASSETS)
            import shuolee_vposer as vp
            from shuolee_portrait.training import named_pose_layout
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            source_hashes.update({str(path): sha(path) for path in (
                PROJECT / 'src/avatar_core/shuolee_vposer.py',
                PROJECT / 'src/avatar_core/scene/smplx_gaussian_model.py',
                PROJECT / 'src/shuolee_portrait/training.py',
                ASSETS / 'smplx/SMPLX_NEUTRAL.npz')})
            vposer, _, vposer_meta = vp.load_vposer('cuda')
            vposer.eval().requires_grad_(False)
            need(not vposer.training, 'Official VPoser must remain in eval mode.')
            poses = []
            with torch.no_grad():
                for i, row in enumerate(rows):
                    budget()
                    latent = torch.from_numpy(ordered['pose_embedding'][i:i+1].copy()).cuda()
                    pose = vposer.decode(latent)['pose_body'].reshape(1, 63)
                    need(bool(torch.isfinite(pose).all()), 'Nonfinite decoded pose: ' + row['frame'])
                    poses.append(pose[0].cpu().numpy())
            poses = np.asarray(poses, np.float32)
            del vposer
            checking = fit_dir / 'fitted.checking.npz'
            arrays = fixture(ordered, poses, checking)
            arrays.update(pose_embedding=ordered['pose_embedding'],
                vertices_camera=ordered['vertices_camera'],
                source_frames=np.asarray([row['source_frame_index'] for row in rows], np.int64),
                image_sha256=np.asarray([row['source_sha256'] for row in rows]),
                mask_sha256=np.asarray([row['mask_sha256'] for row in rows]),
                K_integer_centers=np.asarray([row['K_integer_centers'] for row in rows], np.float64),
                reference_frame_id=np.asarray(args.reference_frame), betas=ordered['betas'][None],
                world_scale=np.asarray(1.),
                camera_length_contract_json=np.asarray(json.dumps(dict(translation_unit='meters',
                    meters_per_camera_unit=1., basis='model_conditioned',
                    evidence='Same SMPL-X camera-relative gauge and identity w2c; not a measured scene trajectory.'))))
            if 'joints_projected_native' in ordered:
                arrays['joints_projected_native'] = ordered['joints_projected_native']
            # The existing consumer first replays the complete absolute offset
            # from its immutable original template, never from an inflated mesh.
            np.savez_compressed(checking, **arrays)
            layout = named_pose_layout(arrays, rows)
            need(layout['initialization_frame_id'] == layout['reference_frame_id'] == args.reference_frame,
                 'Reference-first array order or named mapping differs.')
            model = SMPLXGaussianModel(3)
            model.load_fresh_meshes(checking)
            replay = model.enable_shared_static_offset(checking, ASSETS / 'smplx/SMPLX_NEUTRAL.npz')
            need(np.array_equal(model.static_offset.detach().cpu().numpy(), ordered['static_offset'][0]),
                 'Fitted canonical offset was not loaded exactly once.')
            cpu64 = lambda tensor: tensor.detach().cpu().numpy().astype(np.float64)
            body = model.offset_body
            reg = cpu64(body.J_regressor)[0]
            template = cpu64(body.v_template) + ordered['static_offset'][0].astype(np.float64)
            root_template = reg @ template
            root_shape = np.einsum('v,vck->ck', reg, cpu64(body.shapedirs))
            root_expr = np.einsum('v,vck->ck', reg, cpu64(body.expr_dirs))
            pivots, root_R, root_t, records = [], [], [], []
            with torch.no_grad():
                for i, row in enumerate(rows):
                    budget()
                    model.select_mesh_by_timestep(i)
                    actual = model.shared_offset_vertices(i)
                    expected = torch.as_tensor(ordered['vertices_camera'][i], device='cuda')
                    error = float((actual - expected).abs().max())
                    need(error < 2.e-6, 'Frozen replay differs from saved VHAP vertices: ' + row['frame'])
                    params = {key: value if key == 'betas' else value[i:i+1]
                              for key, value in model.offset_parameters.items()}
                    response = model._offset_forward(body,
                        {'v_template': body.v_template + model.static_offset}, (),
                        dict(params, return_verts=True, return_shaped=False))
                    J0 = (root_template + root_shape @ ordered['betas'].astype(np.float64)
                          + root_expr @ ordered['expression'][i].astype(np.float64))
                    pivot_error = float(np.max(np.abs(cpu64(response.joints)[0, 0] - J0)))
                    need(pivot_error < 2.e-6, 'Canonical J0 does not match actual SMPL-X pelvis: ' + row['frame'])
                    E = np.asarray(row['w2c'], np.float64)
                    global_R = Rotation.from_rotvec(ordered['global_orient'][i]).as_matrix()
                    pivots.append(J0)
                    root_R.append(E[:3, :3].T @ global_R)
                    root_t.append((J0 + ordered['transl'][i] - E[:3, 3]) @ E[:3, :3])
                    records.append(dict(frame=row['frame'], source_checkpoint_slot=order[i],
                        replay_max_abs_m=error, canonical_pelvis_max_abs_m=pivot_error))
            need(getattr(model, 'optimizer', None) is None, 'Interface export must not create an optimizer.')
            arrays.update(root_pivot_model=np.asarray(pivots), root_rotation=np.asarray(root_R),
                world_translation=np.asarray(root_t),
                root_translation_semantics=np.asarray('shape/expression/offset-dependent LBS pelvis in world; not affine origin'))
            named_pose_layout(arrays, rows)
            for path, digest in source_hashes.items():
                need(sha(path) == digest, 'Bound source changed during interface export: ' + path)
            for row in rows:
                need(sha(row['image']) == row['source_sha256'] and sha(row['mask']) == row['mask_sha256'],
                     'Source image/mask changed during interface export: ' + row['frame'])
            np.savez_compressed(checking, **arrays)
            fitted = fit_dir / 'fitted.npz'
            checking.rename(fitted)
            training_inputs = out / 'data/training-inputs.json'
            save(training_inputs, dict(records=rows,
                pixel_domain='already_rectified_MP4_native_integer_centers',
                K_convention='Saved K_integer_centers plus exactly0.5 on cx/cy for the existing renderer.',
                pose_mode='named_vhap_local_SMPLX_camera_relative',
                intrinsic_origin=document.get('intrinsic_origin'),
                source_manifest_sha256=source_hashes[str(inputs)],
                camera_trajectory_measured=False, formal_calibration_accepted=False,
                quality_accepted=False))
            torch.cuda.synchronize()
            receipt = dict(updated=now(), state='interface_verified_quality_pending',
                source_kind='local full-body VHAP_smplx adaptation initialized from official SMPLify-X',
                upstream_complete_reproduction=False, source_run=str(source),
                source_checkpoint=str(checkpoint), source_checkpoint_sha256=source_hashes[str(checkpoint)],
                source_hashes=source_hashes, fixture_function=fixture_source, vposer=vposer_meta,
                decode_device='cuda:0', decode_batch_size=1, decode_eval=True,
                decode_CPU_fallback=False, frames=59, original_checkpoint_frame_ids=ids,
                exported_frame_ids=[row['frame'] for row in rows], source_slots=order,
                export_order_is_not_temporal_order=True,
                reference_frame_id=args.reference_frame, initialization_frame_id=args.reference_frame,
                fit_sha256=sha(fitted), training_inputs_sha256=sha(training_inputs),
                saved_field_shapes={key: list(np.asarray(value).shape) for key, value in arrays.items()},
                replay=replay, records=records,
                replay_max_abs_m=max(row['replay_max_abs_m'] for row in records),
                canonical_pelvis_max_abs_m=max(row['canonical_pelvis_max_abs_m'] for row in records),
                offset_semantics='One absolute shared canonical vertex offset added once before official SMPL-X LBS.',
                root_pivot_includes=['template', 'shared betas', 'per-frame expression', 'shared canonical offset'],
                optimizer_created=False, optimizer_updates=0, gaussian_training_started=False,
                geometry_gate_written=False, official_fit_receipt_written=False,
                quality_accepted=False, new_point_cloud_published=False,
                seconds=time.monotonic() - began,
                limits=['Interface equality does not establish physically correct pose, clothing surface or camera calibration.',
                        'All59 extrinsics are identity in the camera-relative gauge; body roots vary by frame.',
                        'A final static reference-frame PLY is not all59 articulated poses or a measured59-view camera trajectory.',
                        'Gaussian training still requires a separate actual all59 quality review and explicit caller admission.'])
            save(out / 'vhap-gaussian-interface-receipt.json', receipt)
            print(json.dumps({key: receipt[key] for key in ('state', 'frames', 'replay_max_abs_m',
                'canonical_pelvis_max_abs_m', 'seconds', 'quality_accepted', 'gaussian_training_started')}), flush=True)
        except BaseException as error:
            save(out / 'failure.json', dict(updated=now(), error=repr(error),
                traceback=traceback.format_exc(), optimizer_updates=0,
                gaussian_training_started=False, quality_accepted=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default=str(DEFAULT_RUN))
    parser.add_argument('--checkpoint-epoch', type=int, choices=[30, 120], default=30)
    parser.add_argument('--inputs', default=str(DEFAULT_INPUTS))
    parser.add_argument('--output-root', required=True, help='New interface-only directory; never overwritten.')
    parser.add_argument('--reference-frame', default='frame_000033')
    parser.add_argument('--max-seconds', type=float, default=120.)
    args = parser.parse_args()
    need(np.isfinite(args.max_seconds) and 0 < args.max_seconds <= 180., 'Interface budget must be positive and bounded at180sec.')
    need(args.reference_frame == 'frame_000033', 'This prepared handoff retains the already designated frame33 reference.')
    run(args)


if __name__ == '__main__':
    main()
