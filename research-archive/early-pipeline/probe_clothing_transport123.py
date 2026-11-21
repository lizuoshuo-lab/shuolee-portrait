"""One registered synthetic native-GPU clothing-transport control, never train.

The parent registers native-plan.json and runs this script only after checking
the real GPU process list. Re-running a directory with progress/result refuses.
No person archive, body model, fitted camera, optimizer or PLY is consumed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
import traceback


SCHEMA = 'shuolee-clothing-native-probe/1'
FIXTURE_ID = 'six-free-offset-gaussians-one-directed-triangle-v1'
LIMITS = dict(image_mean_abs=1e-5, image_max_abs=.002, covariance_max_abs=2e-6,
              finite_difference_eps=1e-3, finite_difference_abs=1e-4,
              finite_difference_rel=.05, max_forwards=11, max_backwards=1,
              projection_max_abs_px=2e-5, pixel_origin_offset_abs=1e-12,
              gpu_wall_seconds=180)
FD_COMPONENTS = dict(xyz=[0, 0], quaternion=[0, 2], log_scale=[0, 0], sh=[0, 2, 0])


def need(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def local_path(value):
    value = str(value)
    if os.name != 'nt' and len(value) > 2 and value[1] == ':' and value[2] in '/\\':
        return Path('/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/'))
    return Path(value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=strict_object,
                      parse_constant=reject)


def write_json(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=local_path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    plan_path = output / 'native-plan.json'
    progress_path, result_path = output / 'native-progress.json', output / 'native-result.json'
    need(output.is_dir(), 'Registered output directory must already exist.')
    need(not progress_path.exists() and not result_path.exists(),
         'This native probe already started; inspect PID/evidence instead of repeating it.')
    plan = read_json(plan_path)
    need(plan.get('schema') == SCHEMA and plan.get('purpose') == 'synthetic_contract_test',
         'Explicit registered synthetic native-probe plan required.')
    need(plan.get('fixture_id') == FIXTURE_ID and plan.get('limits') == LIMITS and
         plan.get('scale_factors') == [1., 2.], 'The fixed fixture, scales or budget changed.')
    paths = plan.get('paths', {})
    need(set(paths) == {'project_src', 'gsplat_repo'}, 'Explicit local source directories required.')
    project_src, repo = (local_path(paths[k]).resolve() for k in ('project_src', 'gsplat_repo'))
    helper = Path(__file__).resolve().parent
    registered = plan.get('source_sha256')
    need(isinstance(registered, dict) and registered, 'Registered source SHA256 map required.')
    bound = {}
    for name, digest in registered.items():
        path = local_path(name).resolve()
        need(path not in bound, 'Two registered names alias one source file.')
        need(isinstance(digest, str) and len(digest) == 64 and
             all(c in '0123456789abcdef' for c in digest), 'Invalid source SHA256.')
        need(sha(path) == digest, 'Source changed before execution: ' + str(path))
        bound[path] = digest
    required = {
        Path(__file__).resolve(),
        project_src / 'shuolee_portrait/free_clothing_transport.py',
        project_src / 'shuolee_portrait/camera_world_units.py',
        project_src / 'avatar_core/utils/sh_utils.py',
        helper / 'native_portrait_renderer122_v3.py',
        helper / 'native_residual_camera122.py',
        helper / 'native_scene_package122.py',
        helper / 'native_gaussian_ply122.py',
        helper / 'native_portrait_renderer122.py',
        repo / 'gsplat/rendering.py', repo / 'gsplat/cuda/_wrapper.py', repo / 'gsplat/utils.py',
    }
    need(required <= set(bound), 'Required probe, transport, camera, SH or backend source hash missing.')
    started = time.monotonic()
    result = dict(schema=SCHEMA, fixture_id=FIXTURE_ID, passed=False, started=utc(),
                  pid=os.getpid(), plan_sha256=sha(plan_path), source_sha256=registered,
                  checks=[], forward_count=0, backward_count=0, body_model_forwards=0,
                  fitting_steps=0, gaussian_optimizer_steps=0, person_images_loaded=0,
                  ply_exports=0, training_admitted=False, physical_camera_accepted=False,
                  synthetic_only=True, finite_difference_components=FD_COMPONENTS,
                  finite_difference_selection='fixed before execution; no derivative or step-size search')
    write_json(progress_path, dict(pid=os.getpid(), phase='import', started=result['started'],
                                  plan_sha256=result['plan_sha256']), exclusive=True)
    watchdog = None
    gpu_started = None

    def progress(phase):
        write_json(progress_path, dict(pid=os.getpid(), phase=phase, updated=utc(),
                                      seconds=time.monotonic() - started,
                                      forward_count=result['forward_count'], backward_count=result['backward_count']))

    def check(name, passed, **metrics):
        result['checks'].append(dict(name=name, passed=bool(passed), **metrics))
        need(passed, 'Fixed native control failed: ' + name)

    def timeout():
        # A watchdog terminates the process even if Python is inside a blocking
        # native/CUDA call. Root must still inspect the GPU PID before new work.
        result.update(passed=False, error='TimeoutError', message='180-second GPU wall budget exhausted',
                      finished=utc(), seconds=time.monotonic() - started,
                      gpu_seconds=time.monotonic() - gpu_started)
        try:
            write_json(result_path, result)
            progress('failed_timeout')
        finally:
            os._exit(124)

    try:
        sys.path[:0] = [str(project_src), str(repo), str(helper)]
        os.environ.setdefault('MAX_JOBS', '2')
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.0')
        import numpy as np
        import torch
        import gsplat
        from scipy.spatial.transform import Rotation
        from shuolee_portrait.free_clothing_transport import FreeClothingTransport
        from shuolee_portrait.camera_world_units import normalize_camera_w2c
        from gsplat.utils import normalized_quat_to_rotmat
        from native_portrait_renderer122_v3 import render_native
        from native_residual_camera122 import ResidualRenderContract
        sh_path = project_src / 'avatar_core/utils/sh_utils.py'
        spec = importlib.util.spec_from_file_location('transport123_existing_sh', sh_path)
        sh_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sh_module)
        eval_sh = sh_module.eval_sh
        need(str(gsplat.__version__) == '1.5.3', 'Only the registered gsplat 1.5.3 backend is allowed.')
        need(Path(gsplat.__file__).resolve().is_relative_to(repo), 'Unexpected imported gsplat repository.')

        # Analytic rotations are independently defined by elementary matrices;
        # the direct-camera baseline never reads the transport output or Q.
        def rx(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])

        def ry(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

        def rz(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

        triangle = np.array([[-.9, -.7, 3.], [.9, -.7, 3.], [0., 1., 3.]], dtype=np.float64)
        reference_centroid = triangle.mean(0)
        pose_rotation, pose_scale = rz(.23) @ rx(-.17), 1.15
        posed_centroid = reference_centroid + np.array([.07, -.03, .08])
        posed_triangle = (triangle - reference_centroid) @ pose_rotation.T * pose_scale + posed_centroid
        reference_xyz = np.array([[-.50, -.35, 3.16], [-.10, -.38, 3.28], [.36, -.30, 3.19],
                                  [-.42, .19, 3.23], [.03, .28, 3.14], [.46, .24, 3.30]])
        q_values = np.array([[.97, .12, -.18, .05], [.95, -.20, .08, .19], [.96, .10, .19, -.12],
                             [.94, -.18, -.13, .25], [.98, .08, -.10, -.14], [.93, .24, .18, .09]])
        s_values = np.array([[.095, .050, .035], [.070, .045, .028], [.085, .040, .031],
                             [.080, .055, .030], [.090, .042, .026], [.075, .050, .033]])
        sh_values = np.zeros((6, 16, 3), dtype=np.float32)
        sh_values[:, 0, :] = np.array([[.08, -.12, .16], [-.20, .14, .04], [.12, .04, -.15],
                                       [.20, -.10, -.08], [-.12, .15, .11], [.05, -.03, .20]])
        sh_values[:, 1:, :] = (.025 * np.sin(np.arange(6 * 15 * 3).reshape(6, 15, 3) * .37)).astype(np.float32)
        opacity_values = np.array([.62, .68, .59, .65, .71, .63], dtype=np.float32)
        camera_rotation = rz(-.045) @ ry(.08)
        source_w2c = np.eye(4)
        source_w2c[:3, :3], source_w2c[:3, 3] = camera_rotation, [.06, -.04, .20]
        width, height = 192, 144
        K_integer = np.array([[124., 0., 86.25], [0., 119., 78.75], [0., 0., 1.]])
        pixel_shift = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
        K_corners = pixel_shift @ K_integer
        gaussian_ids = tuple('synthetic_gaussian_' + str(i) for i in range(6))
        frame_ids = ('synthetic_reference', 'synthetic_pose')
        parameters = dict(xyz=torch.tensor(reference_xyz, dtype=torch.float32, requires_grad=True),
                          quaternion=torch.tensor(q_values, dtype=torch.float32, requires_grad=True),
                          log_scale=torch.tensor(np.log(s_values), dtype=torch.float32, requires_grad=True),
                          sh=torch.tensor(sh_values, dtype=torch.float32, requires_grad=True))
        result['fixture'] = dict(frame_ids=list(frame_ids), reference_frame_id=frame_ids[0],
                                 evaluated_frame_id=frame_ids[1], gaussian_ids=list(gaussian_ids),
                                 triangle=triangle.tolist(), posed_triangle=posed_triangle.tolist(),
                                 reference_xyz=reference_xyz.tolist(), q_values=q_values.tolist(),
                                 scales=s_values.tolist(), sh=sh_values.tolist(), opacity=opacity_values.tolist(),
                                 pose_rotation=pose_rotation.tolist(), pose_scale=pose_scale,
                                 source_w2c=source_w2c.tolist(), K_integer_centers=K_integer.tolist(),
                                 K_renderer_corners=K_corners.tolist(), pixel_origin_conversions=1,
                                 width=width, height=height, skew=0, radial=[0.] * 6,
                                 tangential=[0.] * 2, thin_prism=[0.] * 4,
                                 covariance_parameterization='normalized wxyz quaternion and exp(log_scale)',
                                 unit_policy='same declared global factor multiplies camera translations, both meshes, free XYZ and Gaussian scales',
                                 camera_baseline='direct analytic pose followed by Rcam,tcam; identity rendering camera')
        transport_contracts = {}
        for factor in (1., 2.):
            vertices = np.stack([triangle, posed_triangle]) * factor
            hashes = {frame: hashlib.sha256(vertices[i].tobytes()).hexdigest()
                      for i, frame in enumerate(frame_ids)}
            transport_contracts[factor] = FreeClothingTransport(
                frame_ids=frame_ids, reference_frame_id=frame_ids[0], gaussian_ids=gaussian_ids,
                frame_source_sha256=hashes, vertices_world=vertices, faces=np.array([[0, 1, 2]]),
                motion_support_triangle_ids=np.zeros(6, dtype=np.int64), translation_unit='meters')
        result['source_hash_semantics'] = 'known synthetic frame mesh bytes, not claimed hashes of video images'

        gpu_started = time.monotonic()
        watchdog = threading.Timer(LIMITS['gpu_wall_seconds'], timeout)
        watchdog.daemon = True
        watchdog.start()
        need(torch.cuda.is_available(), 'CUDA is unavailable; no CPU renderer substitution.')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        result['runtime'] = dict(torch=torch.__version__, gsplat=gsplat.__version__,
                                 gsplat_path=str(Path(gsplat.__file__).resolve()),
                                 gpu=torch.cuda.get_device_name(device), device=str(device))
        contract = ResidualRenderContract('synthetic_known_camera', 'pixel_corners')
        K_gpu = torch.tensor(K_corners[None], dtype=torch.float32, device=device)
        zeros = [torch.zeros((1, n), dtype=torch.float32, device=device) for n in (6, 2, 4)]
        opacity_gpu = torch.tensor(opacity_values, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                torch.arange(width, device=device, dtype=torch.float32), indexing='ij')
        weight = (1. + .5 * xx / width + .3 * yy / height)[None, :, :, None]
        channels = torch.tensor([.7, 1.1, 1.3], dtype=torch.float32, device=device)

        def loss_of(rgb, alpha):
            # Smooth fixed spatial/channel weighting exposes translation,
            # anisotropic orientation, size and SH without selecting derivatives.
            return ((rgb * weight * channels).sum() + .17 * (alpha * weight).sum()) / 6.

        def quaternion_product(a, b):
            aw, ax, ay, az = a.unbind(-1)
            bw, bx, by, bz = b.unbind(-1)
            return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                                aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)

        def native(means, quats, scales, colors, w2c, label):
            need(result['forward_count'] < LIMITS['max_forwards'], 'Forward budget exhausted.')
            result['forward_count'] += 1
            progress('native_forward_' + label)
            # Tensor.to retains autograd through the CPU transport/GPU consumer.
            gpu = [v.to(device=device, dtype=torch.float32) for v in (means, quats, scales, colors)]
            view = torch.tensor(np.asarray(w2c)[None].copy(), dtype=torch.float32, device=device)
            rgb, alpha, meta = render_native(*gpu[:3], opacity_gpu, gpu[3], view, K_gpu, width, height,
                                            contract=contract, sh_degree=None, radial_coeffs=zeros[0],
                                            tangential_coeffs=zeros[1], thin_prism_coeffs=zeros[2])
            torch.cuda.synchronize(device)
            need(bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(alpha).all()), 'Nonfinite native image.')
            need(bool((meta['radii'] > 0).all()) and float(alpha.max()) > 0, 'Fixture was clipped or empty.')
            return rgb, alpha, meta

        def transport_render(p, factor, label):
            unit_contract = dict(translation_unit='sfm_units', meters_per_camera_unit=factor,
                                 basis='synthetic_known', evidence=FIXTURE_ID + ': explicit common scale control')
            units = normalize_camera_w2c(source_w2c, unit_contract)
            q_ref = p['quaternion'] / torch.linalg.vector_norm(p['quaternion'], dim=-1, keepdim=True)
            s_ref = p['log_scale'].exp() * factor
            R_ref = normalized_quat_to_rotmat(q_ref)
            covariance_ref = (R_ref * s_ref[:, None, :].square()) @ R_ref.transpose(-1, -2)
            camera_center = -units.w2c_meters[:3, :3].T @ units.w2c_meters[:3, 3]
            moved = transport_contracts[factor].transport_torch(
                frame_ids[1], p['xyz'] * factor, covariance_ref, camera_center, gaussian_ids=gaussian_ids)
            # Q comes from fixed known mesh geometry. Only this constant converts
            # via SciPy; q_ref, XYZ, log_scale and SH never detach before rendering.
            q_motion_np = Rotation.from_matrix(moved['rotations'].numpy().astype(np.float64)).as_quat()[:, [3, 0, 1, 2]]
            q_motion = torch.tensor(q_motion_np, dtype=torch.float32)
            q_pose = quaternion_product(q_motion, q_ref)
            s_pose = moved['scales'][:, None] * s_ref
            R_pose = normalized_quat_to_rotmat(q_pose / q_pose.norm(dim=-1, keepdim=True))
            native_covariance = (R_pose * s_pose[:, None, :].square()) @ R_pose.transpose(-1, -2)
            cov_error = float((native_covariance - moved['covariance']).abs().max())
            need(cov_error <= LIMITS['covariance_max_abs'], 'Quaternion/scale consumer lost full covariance.')
            raw_rgb = eval_sh(3, p['sh'].transpose(1, 2), moved['reference_view_directions']) + .5
            need(bool((raw_rgb > .05).all()), 'Fixture reached color-clamp boundary.')
            rgb, alpha, meta = native(moved['centers'], q_pose, s_pose, raw_rgb.clamp_min(0), units.w2c_meters, label)
            diagnostic = dict(factor=factor, covariance_max_abs=cov_error, unit_contract=units.report(),
                              covariance=native_covariance.detach().numpy().tolist(),
                              centers=moved['centers'].detach().numpy().tolist(),
                              reference_directions=moved['reference_view_directions'].detach().numpy().tolist(),
                              max_support_vertex_residual=float(moved['support_vertex_residual'].max()),
                              means2d=meta['means2d'].detach().cpu().numpy().tolist())
            return rgb, alpha, diagnostic

        progress('scale_one_with_autograd')
        rgb_one, alpha_one, diagnostic_one = transport_render(parameters, 1., 'scale_one')
        loss = loss_of(rgb_one, alpha_one)
        with torch.no_grad():
            rgb_two, alpha_two, diagnostic_two = transport_render(parameters, 2., 'scale_two')
            # Independent baseline: known analytic pose/camera transforms, no
            # call to transport, no moved centers/covariance/directions reuse.
            direct_world = (reference_xyz - reference_centroid) @ pose_rotation.T * pose_scale + posed_centroid
            direct_camera = direct_world @ camera_rotation.T + source_w2c[:3, 3]
            q_ref_np = q_values / np.linalg.norm(q_values, axis=-1, keepdims=True)
            R_ref_np = Rotation.from_quat(q_ref_np[:, [1, 2, 3, 0]]).as_matrix()
            direct_rotation = camera_rotation[None] @ pose_rotation[None] @ R_ref_np
            direct_quats = Rotation.from_matrix(direct_rotation).as_quat()[:, [3, 0, 1, 2]]
            direct_scales = s_values * pose_scale
            direct_dirs = direct_camera / np.linalg.norm(direct_camera, axis=-1, keepdims=True)
            reference_dirs = direct_dirs @ camera_rotation @ pose_rotation
            direct_rgb = eval_sh(3, torch.tensor(sh_values).transpose(1, 2),
                                 torch.tensor(reference_dirs, dtype=torch.float32)) + .5
            need(bool((direct_rgb > .05).all()), 'Baseline reached color-clamp boundary.')
            tensor = lambda v: torch.tensor(v, dtype=torch.float32)
            rgb_direct, alpha_direct, direct_meta = native(tensor(direct_camera), tensor(direct_quats),
                                                          tensor(direct_scales), direct_rgb.clamp_min(0),
                                                          np.eye(4), 'independent_camera_space')
        result['scale_diagnostics'] = [diagnostic_one, diagnostic_two]
        center_camera_one = np.asarray(diagnostic_one['centers']) @ camera_rotation.T + source_w2c[:3, 3]
        expected_pixels = direct_camera[:, :2] / direct_camera[:, 2:] * [K_corners[0, 0], K_corners[1, 1]] + K_corners[:2, 2]
        actual_pixels = center_camera_one[:, :2] / center_camera_one[:, 2:] * [K_corners[0, 0], K_corners[1, 1]] + K_corners[:2, 2]
        # Project directly with the original integer-center K, rather than
        # subtracting 0.5 from a shared corner projection as the oracle.
        expected_integer = direct_camera[:, :2] / direct_camera[:, 2:] * [K_integer[0, 0], K_integer[1, 1]] + K_integer[:2, 2]
        actual_integer = center_camera_one[:, :2] / center_camera_one[:, 2:] * [K_integer[0, 0], K_integer[1, 1]] + K_integer[:2, 2]
        integer_error = float(np.max(np.abs(actual_integer - expected_integer)))
        expected_origin_error = float(np.max(np.abs(expected_pixels - expected_integer - .5)))
        actual_origin_error = float(np.max(np.abs(actual_pixels - actual_integer - .5)))
        result['analytic_projection'] = dict(expected_pixel_corners=expected_pixels.tolist(),
                                             transported_pixel_corners=actual_pixels.tolist(),
                                             max_abs=float(np.max(np.abs(actual_pixels - expected_pixels))),
                                             expected_integer_centers=expected_integer.tolist(),
                                             transported_integer_centers=actual_integer.tolist(),
                                             integer_max_abs_px=integer_error,
                                             expected_corner_minus_integer_error=expected_origin_error,
                                             transported_corner_minus_integer_error=actual_origin_error,
                                             ut_means2d_direct=direct_meta['means2d'].detach().cpu().numpy().tolist(),
                                             note='UT Gaussian mean is not assumed equal to the projection of its finite-sized center')
        check('independent_integer_center_projection', integer_error <= LIMITS['projection_max_abs_px'],
              max_abs_px=integer_error, limit_px=LIMITS['projection_max_abs_px'])
        check('exactly_one_half_pixel_origin_conversion',
              max(expected_origin_error, actual_origin_error) <= LIMITS['pixel_origin_offset_abs'],
              expected_offset_error=expected_origin_error, transported_offset_error=actual_origin_error,
              tolerance=LIMITS['pixel_origin_offset_abs'])
        for name, rgb, alpha in [('global_scale_invariance', rgb_two, alpha_two),
                                  ('independent_camera_space', rgb_direct, alpha_direct)]:
            rgb_error, alpha_error = (rgb.detach() - rgb_one.detach()).abs(), (alpha.detach() - alpha_one.detach()).abs()
            metrics = dict(rgb_mean_abs=float(rgb_error.mean()), rgb_max_abs=float(rgb_error.max()),
                           alpha_mean_abs=float(alpha_error.mean()), alpha_max_abs=float(alpha_error.max()))
            check(name, metrics['rgb_mean_abs'] <= LIMITS['image_mean_abs'] and
                  metrics['alpha_mean_abs'] <= LIMITS['image_mean_abs'] and
                  metrics['rgb_max_abs'] <= LIMITS['image_max_abs'] and
                  metrics['alpha_max_abs'] <= LIMITS['image_max_abs'], **metrics)
        check('full_covariance_native_factorization', max(diagnostic_one['covariance_max_abs'],
              diagnostic_two['covariance_max_abs']) <= LIMITS['covariance_max_abs'],
              per_scale_max_abs=[diagnostic_one['covariance_max_abs'], diagnostic_two['covariance_max_abs']])

        progress('single_backward')
        need(result['backward_count'] == 0, 'Only one native backward is authorized.')
        result['backward_count'] += 1
        loss.backward()
        torch.cuda.synchronize(device)
        gradients = {}
        for name, value in parameters.items():
            gradient = value.grad
            valid = gradient is not None and bool(torch.isfinite(gradient).all()) and float(gradient.abs().max()) > 0
            gradients[name] = dict(finite_nonzero=valid,
                                   max_abs=float(gradient.abs().max()) if gradient is not None and bool(torch.isfinite(gradient).all()) else None)
            check('native_gradient_' + name, valid, **gradients[name])
        result['loss'] = float(loss.detach())
        result['gradient_summary'] = gradients
        result['finite_differences'] = []
        epsilon = LIMITS['finite_difference_eps']
        for name, location in FD_COMPONENTS.items():
            idx = tuple(location)
            analytical = float(parameters[name].grad[idx])
            finite_losses = []
            for sign in (-1., 1.):
                perturbed = {key: value.detach().clone() for key, value in parameters.items()}
                perturbed[name][idx] += sign * epsilon
                with torch.no_grad():
                    r, a, _ = transport_render(perturbed, 1., name + ('_minus' if sign < 0 else '_plus'))
                    finite_losses.append(float(loss_of(r, a)))
            numerical = (finite_losses[1] - finite_losses[0]) / (2 * epsilon)
            error = abs(numerical - analytical)
            tolerance = LIMITS['finite_difference_abs'] + LIMITS['finite_difference_rel'] * abs(numerical)
            entry = dict(parameter=name, index=location, epsilon=epsilon, analytical=analytical,
                         numerical=numerical, absolute_error=error, allowed_error=tolerance,
                         minus_loss=finite_losses[0], plus_loss=finite_losses[1])
            result['finite_differences'].append(entry)
            check('fixed_finite_difference_' + name, math.isfinite(numerical) and error <= tolerance, **entry)
        check('fixed_execution_counts', result['forward_count'] == 11 and result['backward_count'] == 1,
              forwards=result['forward_count'], backwards=result['backward_count'])
        # Synthetic float images retain exact numerical evidence; no person
        # images, labels burned into pixels, preview PLY or training checkpoint.
        np.savez_compressed(output / 'native-synthetic-images.npz',
                            rgb_scale_one=rgb_one.detach().cpu().numpy(), alpha_scale_one=alpha_one.detach().cpu().numpy(),
                            rgb_scale_two=rgb_two.cpu().numpy(), alpha_scale_two=alpha_two.cpu().numpy(),
                            rgb_camera_space=rgb_direct.cpu().numpy(), alpha_camera_space=alpha_direct.cpu().numpy())
        need(sha(plan_path) == result['plan_sha256'], 'Registered plan changed during execution.')
        for path, digest in bound.items():
            need(sha(path) == digest, 'Registered source changed during execution: ' + str(path))
        result['passed'] = True
    except BaseException as error:
        result.update(passed=False, error=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        if watchdog is not None:
            watchdog.cancel()
        result.update(finished=utc(), seconds=time.monotonic() - started,
                      gpu_seconds=None if gpu_started is None else time.monotonic() - gpu_started)
        if result['gpu_seconds'] is not None and result['gpu_seconds'] > LIMITS['gpu_wall_seconds']:
            result.update(passed=False, error='TimeoutError', message='GPU wall budget exceeded before finalization.')
        write_json(result_path, result)
        progress('completed' if result['passed'] else 'failed')
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
