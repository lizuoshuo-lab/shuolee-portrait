"""Compare stock camera H2D copies against identical resident GPU camera tensors."""

import argparse
import ast
import fcntl
from hashlib import sha256
import inspect
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
LAB = Path('/opt/gs-slam-lab')
RUN = LAB / 'shuolee-smpl-sixworker-opacity-006/run'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-sixworker-opacity')
ENV = LAB / 'shuolee-portrait-clean-001/env'
RASTER = ENV / 'lib/python3.10/site-packages/diff_gaussian_rasterization'
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
OUTPUT = LAB / 'shuolee-smpl-camera-residency006'
EARLY = RUN / 'gaussian-run/step-001000'
LATE = RUN / 'gaussian-run/step-010000'
CAMERA_FIELDS = ('world_view_transform', 'full_proj_transform', 'camera_center')
RAW_PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
ORDER = (('stock', 'resident'), ('resident', 'stock'), ('stock', 'resident'))
REPEATS = 12
WARMUP = 4
PINS = {
    RUN / 'result.json': '4b228f050170a172997a12fa7e4c2d06361877098c03232fa8011202e0f25683',
    RUN / 'fit/official-fit-result.json': '86a83a1e04cf669784c1f5037a9878bbae5563a7af257c0bbf1e1d10887f3e76',
    RUN / 'gaussian-run/training-cameras.json': '20116a3ca1f873e54a1e0a5b5b2bddb6cd4e3648d8164377008f787ab05d8248',
    RUN / 'gaussian-run/config.json': '68e6af7e75b73b81def26a4647ff43721209235b8a7a63babb136b54a15bc867',
    LATE / 'bound-replay.json': '5d77f6351aa13407f8935045e780e266fcad8c1eeb058a7f5e4d22f08a92975c',
    LATE / 'reference-render.png': '6f1b49873347433d148fd237c1242771ba69040e70ba4857f08c6bb570113d66',
    EARLY / 'native-ply-parity.json': 'a813c48d8fc25cabfbd39e5ce2c555db0245c9e8779e950d6a18e3bf00542c9f',
    EARLY / 'reference-render.png': '48651703bf6d10d0e4bf9a764d9b02defe43f8732c664206dadeb14383e09edd',
    RASTER / '__init__.py': 'eafa9c3f670258b34535b7ee2832e6d8410d6a68762037216f04935ccdafba34',
    RASTER / '_C.cpython-310-x86_64-linux-gnu.so': '5161c09abd032639e582230874df6042a4bdd2b62f5137733386520c2a05e113',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(tensor):
    if tensor is None:
        return None
    value = tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(value.shape), dtype=value.dtype.str, sha256=sha256(value.tobytes()).hexdigest())


def prepare():
    import torch

    git = '/mnt/c/Program Files/Git/cmd/git.exe'
    win = subprocess.check_output(['wslpath', '-w', str(PROJECT)], text=True).strip()
    head = subprocess.check_output([git, '-C', win, 'rev-parse', 'HEAD'], text=True).strip()
    need(head == '213ad16c035ead820947549f2783d6f7cea1f771', 'Expected the archived 006 checkout.')
    sys.path.insert(0, str(PROJECT / 'src'))
    from shuolee_portrait import runtime as rt, training

    pins = {}

    def pin(path, expected=None):
        digest = rt.sha(path)
        need(expected is None or digest == expected, f'Pinned file differs: {path}')
        pins[str(path)] = digest

    for path, expected in PINS.items():
        pin(path, expected)
    pin(__file__)
    terminal = rt.read(RUN / 'result.json')
    bound = rt.read(LATE / 'bound-replay.json')
    fit = rt.read(RUN / 'fit/official-fit-result.json')
    config = rt.read(RUN / 'gaussian-run/config.json')
    need(terminal['state'] == 'completed' and terminal['steps'] == 10_000 and terminal['points'] == bound['points'] == 121680
         and terminal['reference_frame_id'] == bound['reference_frame_id'] == 'frame_000033'
         and terminal['reference_pose_index'] == bound['reference_pose_index'] == 0
         and config['image_cache_frames'] == 24 and config['sh_degree'] == 0, 'Unexpected source run identity.')
    for rel, expected in bound['source_files_sha256'].items():
        pin(PROJECT / rel, expected)
    pin(PROJECT / 'src/avatar_core/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu',
        '068eb4b0c20af25d22f274443e3fa6e24368418f691f099183e84109faccd6f7')
    pin(ASSETS / 'smplx/SMPLX_NEUTRAL.npz', bound['asset_sha256'])
    pin(RUN / 'fit/fitted.npz', bound['inputs_sha256']['fitted'])
    pin(RUN / 'data/training-inputs.json', bound['inputs_sha256']['training_rows'])
    for snap in (EARLY, LATE):
        check = rt.read(snap / 'native-ply-parity.json')
        need(check['passed'] and check['reference_frame_id'] == terminal['reference_frame_id'],
             'Missing archived native PLY readback.')
        pin(snap / 'world.ply', check['sha256'])
        pin(snap / 'reference-mesh.npz', check['mesh_sha256'])
        pin(snap / 'native-ply-parity.json')
    pin(LATE / 'bound-local.ply', bound['bound_ply_sha256'])
    row = rt.read(RUN / 'data/training-inputs.json')['records'][0]
    need(row['frame'] == 'frame_000033' and row['source_frame_index'] == 256 and row['wh'] == [2704, 1520]
         and row['frame'] not in fit['selection']['retained_holdouts']
         and row['source_sha256'] == bound['frames'][0]['source_sha256']
         and row['mask_sha256'] == bound['frames'][0]['mask_sha256'], 'Frozen native observation differs.')
    pin(row['image'], row['source_sha256'])
    pin(row['mask'], row['mask_sha256'])
    rt.configure(ASSETS)
    camera, receipt = training.camera(row, 0, pose_index=0)
    expected = rt.read(RUN / 'gaussian-run/training-cameras.json')['records'][0]
    need(receipt == expected and not torch.cuda.is_initialized()
         and all(getattr(camera, name).device.type == 'cpu' for name in CAMERA_FIELDS),
         'Original CPU camera projection receipt or residency differs.')
    plan = dict(state='preregistered', checkout=head, source_pins=pins, original_camera=receipt,
        camera_fields={name: digest(getattr(camera, name)) for name in CAMERA_FIELDS},
        only_change='Move the three exact camera tensors to the same GPU once after CPU native projection validation.',
        stock_renderer_unchanged=True, stock_rgb_and_white_alpha_calls_per_step=2,
        phases=dict(early='1k world PLY photometric/alpha replay; the archive lacks bound early state',
                    late='10k bound local PLY with frozen reconstructed offset and original regularizers'),
        point_counts=[32553, 121680], warmup_iterations_per_arm=WARMUP, timed_pairs=ORDER,
        repetitions_per_arm_per_pair=REPEATS, parity='Exact byte digests for stock A/A first, then stock/resident. '
        'Preserve both independent screen gradients and report any stock nondeterminism instead of relaxing the gate.',
        max_seconds=65, gpu_memory_limit_bytes=8*2**30, optimizer_updates=0, cold_to_export_measured=False)
    return rt, training, row, camera, config, plan


def models(training, torch):
    from roma import quat_xyzw_to_wxyz, rotmat_to_unitquat
    from scene.gaussian_model import GaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from utils.graphics_utils import compute_face_orientation

    early = GaussianModel(0)
    early.load_ply(str(EARLY / 'world.ply'))
    yield 'early_world', early, None, EARLY
    tree = ast.parse(inspect.getsource(training._train_smpl_sh0))
    node, = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'SharedSurface']
    scope = dict(SMPLXGaussianModel=SMPLXGaussianModel, compute_face_orientation=compute_face_orientation,
                 quat_xyzw_to_wxyz=quat_xyzw_to_wxyz, rotmat_to_unitquat=rotmat_to_unitquat)
    exec(compile(ast.Module(body=[node], type_ignores=[]), inspect.getsourcefile(training), 'exec'), scope)
    with np.load(RUN / 'fit/fitted.npz', allow_pickle=False) as file:
        fit = dict(file)
    base, linear, clothing, _, _ = training._frozen_smpl_surface(fit, ASSETS / 'smplx/SMPLX_NEUTRAL.npz')
    late = scope['SharedSurface'](0)
    late.load_ply(str(LATE / 'bound-local.ply'))
    late.base, late.linear = base.cuda(), linear.cuda()
    late.clothing = torch.as_tensor(clothing, device='cuda')
    late.faces = torch.as_tensor(fit['faces'], dtype=torch.long, device='cuda')
    late.num_timesteps = len(base)
    with np.load(LATE / 'reference-mesh.npz', allow_pickle=False) as file:
        late.static_offset = torch.nn.Parameter(torch.as_tensor(file['static_offset'], device='cuda'))
        with torch.no_grad():
            mesh_error = float((late.posed_vertices(0) - torch.as_tensor(file['vertices_world'], device='cuda')).abs().max())
        need(mesh_error < 2e-6, 'Original trained offset does not replay the reference mesh.')
    indices, values = training._uniform_laplacian(fit['faces'], len(clothing))
    lap = torch.sparse_coo_tensor(indices, values, (len(clothing), len(clothing)), device='cuda').coalesce()
    yield 'late_bound', late, lap, LATE


def audit(prepared, output):
    import torch
    from diff_gaussian_rasterization import _C
    from gaussian_renderer import render

    rt, training, row, stock_camera, config, plan = prepared
    started = perf_counter()
    need(Path(_C.__file__).resolve().parent == RASTER.resolve(), 'Different rasterizer binary imported.')
    need(torch.cuda.device_count() == 1 and torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 5080',
         'Expected the same single local GPU.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cached_camera, receipt = training.camera(row, 0, pose_index=0)
    need(receipt == plan['original_camera'], 'Pre-residency native projection changed.')
    torch.cuda.synchronize()
    copy_start = perf_counter()
    for name in CAMERA_FIELDS:
        value = getattr(cached_camera, name).cuda()
        setattr(cached_camera, name, value)
    torch.cuda.synchronize()
    copy_seconds = perf_counter() - copy_start
    for name in CAMERA_FIELDS:
        value = getattr(cached_camera, name)
        need(value.cuda() is value and digest(value) == plan['camera_fields'][name],
             'Resident camera tensor changed, or stock render would still transfer it.')
    cameras = dict(stock=stock_camera, resident=cached_camera)
    target_rgb, target_alpha, box = training._native_foreground_target(row)
    density = float((target_alpha > .99).count_nonzero()) / target_alpha.numel()
    target_rgb, target_alpha = target_rgb.cuda(), target_alpha.cuda()
    bg = torch.zeros(3, device='cuda')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    options = SimpleNamespace(**config['optimization'])
    torch.cuda.reset_peak_memory_stats()
    records = {}

    def guard():
        need(perf_counter() - started < plan['max_seconds']
             and torch.cuda.max_memory_allocated() < plan['gpu_memory_limit_bytes'],
             'Exceeded fixed diagnostic wall/memory budget.')

    for (phase, model, lap, snapshot), expected in zip(models(training, torch), plan['point_counts'], strict=True):
        names = RAW_PARAMETERS if lap is None else RAW_PARAMETERS + ('static_offset',)
        params = [getattr(model, name) for name in names]
        need(len(model._xyz) == expected and model.optimizer is None, 'Checkpoint state or point density changed.')
        before = [digest(p) for p in params]

        def step(mode, capture=False):
            cam = cameras[mode]
            rgb = training.render_pose(render, cam, model, pipe, bg, 0)
            alpha = render(cam, model, pipe, bg, override_color=torch.ones_like(model._xyz))
            photo, l1, ssim, alpha_loss = training._sh0_foreground_alpha_loss(
                rgb['render'], alpha['render'][:1], target_rgb, target_alpha, box,
                alpha_floor=config['straight_rgb_alpha_floor'])
            loss = photo + config['alpha_loss_weight'] * alpha_loss
            if lap is not None:
                visible = rgb['visibility_filter']
                delta = model.static_offset * model.clothing[:, None]
                xyz = (model._xyz[visible].norm(dim=1)-options.threshold_xyz).relu().mean()*options.lambda_xyz
                scale = (model._scaling[visible].exp()-options.threshold_scale).relu().norm(dim=1).mean()*options.lambda_scale
                smooth = torch.sparse.mm(lap, delta).square().sum(-1).mean()*options.lambda_laplacian
                loss = loss + xyz + scale + smooth
            loss.backward()
            if not capture:
                for param in params:
                    param.grad = None
                return None
            rgb_screen, alpha_screen = rgb['viewspace_points'].grad, alpha['viewspace_points'].grad
            need(rgb_screen is not None and alpha_screen is not None and rgb_screen is not alpha_screen,
                 'Stock separate RGB/alpha screen gradients unavailable.')
            with torch.no_grad():
                visible = rgb['visibility_filter']
                stats = (rgb_screen * density)[visible, :2].norm(dim=-1, keepdim=True)
            values = dict(rgb=rgb['render'], alpha=alpha['render'][:1], rgb_radii=rgb['radii'], alpha_radii=alpha['radii'],
                          rgb_visible=visible, alpha_visible=alpha['visibility_filter'], loss=loss, rgb_loss=photo,
                          l1=l1, ssim=ssim, alpha_loss=alpha_loss, rgb_screen_grad=rgb_screen, alpha_screen_grad=alpha_screen,
                          densification_norm=stats)
            values.update({name + '_grad': p.grad for name, p in zip(names, params, strict=True)})
            signatures = {key: digest(value) for key, value in values.items()}
            for param in params:
                param.grad = None
            return signatures

        with torch.no_grad():
            prediction = training.render_pose(render, stock_camera, model, pipe, bg, 0)['render']
            current = np.asarray(training.image(prediction), dtype=np.int16)
            archived = np.asarray(Image.open(snapshot / 'reference-render.png').convert('RGB'), dtype=np.int16)
            error = np.abs(current - archived)
            need(float(error.mean()) < .02 and int(error.max()) <= 2, 'Checkpoint failed native RGB archive replay.')
        a, repeat, b = step('stock', True), step('stock', True), step('resident', True)
        stock_differences = [name for name in a if a[name] != repeat[name]]
        resident_differences = [name for name in a if a[name] != b[name]]
        phase_result = dict(points=expected, native_rgb_byte_mean=float(error.mean()), native_rgb_byte_max=int(error.max()),
            stock_repeat_exact=not stock_differences, resident_exact=not resident_differences,
            stock_repeat_differing_fields=stock_differences, resident_differing_fields=resident_differences,
            digest_stock=a, digest_stock_repeat=repeat, digest_resident=b, timed_pairs=[])
        need(before == [digest(p) for p in params] and model.optimizer is None, 'Parity check altered fixed model values.')
        records[phase] = phase_result
        rt.save(output / 'progress.json', dict(records=records, optimizer_updates=0))
        if stock_differences or resident_differences:
            continue
        for mode in cameras:
            for _ in range(WARMUP):
                step(mode)
        for pair in ORDER:
            block = {}
            for mode in pair:
                guard()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                host = perf_counter()
                begin.record()
                for _ in range(REPEATS):
                    step(mode)
                end.record()
                end.synchronize()
                block[mode] = dict(host_ms=(perf_counter()-host)*1000/REPEATS, cuda_ms=begin.elapsed_time(end)/REPEATS)
            phase_result['timed_pairs'].append(block)
        phase_result['median_paired_saved_host_ms'] = median(
            block['stock']['host_ms']-block['resident']['host_ms'] for block in phase_result['timed_pairs'])
        phase_result['median_paired_saved_cuda_ms'] = median(
            block['stock']['cuda_ms']-block['resident']['cuda_ms'] for block in phase_result['timed_pairs'])
        need(before == [digest(p) for p in params] and model.optimizer is None, 'Diagnostic altered fixed model values.')
        guard()
    for path, expected in plan['source_pins'].items():
        need(rt.sha(path) == expected, f'Original source changed: {path}')
    return dict(state='fixed_camera_residency_audit_complete', records=records,
        exact_both_phases=all(r['stock_repeat_exact'] and r['resident_exact'] for r in records.values()),
        initial_camera_copy_seconds=copy_seconds, optimizer_updates=0, gaussian_training_started=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=perf_counter()-started,
        cold_to_export_measured=False, quality_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'audit'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == LAB and not output.exists(), 'Use a fresh isolated audit output.')
    prepared = prepare()
    rt, _, _, _, _, plan = prepared
    if args.mode == 'check':
        print(json.dumps(dict(state='006_camera_residency_inputs_verified', pins=len(plan['source_pins']),
                              original_camera=plan['original_camera'], gpu_calls=0, optimizer_updates=0)))
        return
    with open('/tmp/shuolee-native122-gpu.lock', 'a+b') as lock, open('/tmp/shuolee-body-export123.lock', 'a+b') as export:
        for handle in (lock, export):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        need(not apps.strip(), 'GPU must be idle.')
        output.mkdir()
        rt.save(output / 'preregistration.json', plan)
        try:
            result = audit(prepared, output)
            rt.save(output / 'result.json', dict(result, preregistration_sha256=rt.sha(output / 'preregistration.json')))
            print('006_CAMERA_RESIDENCY_AUDIT_COMPLETE', result['exact_both_phases'], flush=True)
        except BaseException as exc:
            rt.save(output / 'failure.json', dict(state='failed', error=repr(exc), optimizer_updates=0))
            raise


if __name__ == '__main__':
    main()
