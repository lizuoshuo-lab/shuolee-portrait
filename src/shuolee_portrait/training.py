"""Fresh mesh-local Gaussian training. This is a local SMPL-X adapter."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from PIL import Image
from . import runtime as rt


def camera(row, index, height=None, *, pose_index=0):
    w, h = row['wh']
    if height is not None:
        w, h = round(w * height / h), height
    K = np.diag([w / row['wh'][0], h / row['wh'][1], 1.]) @ np.asarray(row['K'])
    E = np.asarray(row['w2c'])
    if (K.shape != (3, 3) or E.shape != (4, 4) or
            not np.isfinite(K).all() or not np.isfinite(E).all() or
            not np.array_equal(K[2], [0., 0., 1.]) or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError('Invalid full corner-convention K or w2c: ' + row['frame'])
    if K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError('The existing rasterizer does not support camera skew; K must not be silently changed.')
    from scene.cameras import CameraProj2
    cam = CameraProj2(index, E[:3, :3].T, E[:3, 3], K[0, 0], K[1, 1], 'black',
                     w, None, h, row.get('image', ''), row['frame'], index,
                     cx=K[0, 2], cy=K[1, 2], timestep=pose_index)
    # Test the actual matrix sent to CUDA against pinhole pixel coordinates.
    cp = np.array([[0., 0., 2.], [.1, .2, 3.], [-.4, -.3, 4.]])
    world = (cp - E[:3, 3]) @ E[:3, :3]
    q = cp @ K.T
    expected = q[:, :2] / q[:, 2:] - .5
    clip = np.c_[world, np.ones(3)] @ cam.full_proj_transform.numpy()
    actual = ((clip[:, :2] / clip[:, 3:] + 1) * [w, h] - 1) / 2
    error = float(abs(expected - actual).max())
    if error > .001:
        raise RuntimeError(f'Projection disagreement {row["frame"]}: {error} px')
    return cam, dict(frame=row['frame'], K=K.tolist(), w2c=E.tolist(), wh=[w, h],
                     pose_index=pose_index, pixel_convention='pixel_corners', projection_error_px=error)


def named_pose_layout(fit, rows):
    """Bind the saved mesh sequence to named observations, without sorting.

    Legacy unnamed files remain usable only for their original identical-pose
    case. Named files retain the native source K/E and one final shared beta.
    This verifies interface identity, not fitting or reconstruction quality.
    """
    def need(value, message):
        if not value:
            raise ValueError(message)
    ids = [r['frame'] for r in rows]
    need(len(ids) > 0 and all(isinstance(v, str) and v for v in ids) and len(set(ids)) == len(ids),
         'Observation frame identities must be nonempty and unique.')
    vertices, faces = np.asarray(fit['vertices_world']), np.asarray(fit['faces'])
    need(vertices.ndim == 3 and len(vertices) > 0 and vertices.shape[1] >= 3 and vertices.shape[2] == 3
         and np.isfinite(vertices).all(), 'Invalid common-topology mesh sequence.')
    need(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0 and faces.dtype.kind in 'iu'
         and np.all(faces >= 0) and np.all(faces < vertices.shape[1]), 'Invalid common faces.')
    if 'frame_ids' not in fit:
        need(np.array_equal(vertices, np.broadcast_to(vertices[0], vertices.shape)),
             'Nonstatic meshes require explicit frame_ids; positional ordering is not sufficient.')
        return dict(named=False, frame_ids=ids, pose_indices=[0]*len(ids), pose_by_frame=dict.fromkeys(ids, 0),
                    reference_frame_id=ids[0], reference_pose_index=0, initialization_frame_id=ids[0])
    mesh_ids = np.asarray(fit['frame_ids'])
    need(mesh_ids.shape == (len(vertices),) and mesh_ids.dtype.kind in 'US', 'Invalid mesh frame_ids.')
    mesh_ids = mesh_ids.astype(str).tolist()
    need(len(set(mesh_ids)) == len(mesh_ids) and set(mesh_ids) == set(ids),
         'Mesh and observation frame identities must match exactly once.')
    slots = {name: i for i, name in enumerate(mesh_ids)}
    reference = np.asarray(fit['reference_frame_id'])
    need(reference.shape == () and reference.dtype.kind in 'US' and str(reference.item()) in slots,
         'A named reference frame in this sequence is required.')
    beta = np.asarray(fit['shared_betas'])
    need(beta.shape == (1, 10) and np.isfinite(beta).all(), 'One final shared beta is required.')
    if 'betas' in fit:
        need(np.array_equal(beta, fit['betas']), 'Duplicate shared beta values disagree.')
    for key in ('source_frames', 'image_sha256', 'mask_sha256'):
        need(np.asarray(fit[key]).shape == (len(mesh_ids),), 'Invalid named source field: ' + key)
    for key, shape in (('K_integer_centers', (len(mesh_ids), 3, 3)), ('w2c', (len(mesh_ids), 4, 4))):
        a = np.asarray(fit[key])
        need(a.shape == shape and np.isfinite(a).all(), 'Invalid named camera field: ' + key)
    for row in rows:
        slot = slots[row['frame']]
        need(fit['source_frames'][slot] == row['source_frame_index'] and
             str(fit['image_sha256'][slot]) == row['source_sha256'] and
             str(fit['mask_sha256'][slot]) == row['mask_sha256'], 'Named image/mask/source-frame association differs.')
        expected_K = np.asarray(fit['K_integer_centers'][slot], dtype=np.float64).copy()
        expected_K[0, 2] += .5; expected_K[1, 2] += .5
        need(np.array_equal(expected_K, row['K']), 'Observation K must be the saved native K plus the explicit half pixel.')
        need(np.array_equal(fit['w2c'][slot], row['w2c']), 'Named pose and observation w2c differ.')
    reference = str(reference.item())
    return dict(named=True, frame_ids=mesh_ids, pose_indices=[slots[name] for name in ids], pose_by_frame=slots,
                reference_frame_id=reference, reference_pose_index=slots[reference], initialization_frame_id=mesh_ids[0])


def render_pose(render, cam, model, pipe, background, pose_index, **kwargs):
    """Select the bound mesh for every call; a world-PLY model is already posed."""
    if getattr(model, 'binding', None) is not None:
        model.select_mesh_by_timestep(pose_index)
    return render(cam, model, pipe, background, **kwargs)


def reference_ply_comparison(render, cam, model, reloaded, pipe, background, reference_pose):
    reference = render_pose(render, cam, model, pipe, background, reference_pose)['render']
    return reference, abs(reference-render(cam, reloaded, pipe, background)['render'])


def foreground_loss(im, gt, alpha):
    import torch
    F = torch.nn.functional
    valid = alpha > .99
    support = F.max_pool2d(F.pad((~valid).float()[None], (5, 5, 5, 5), value=1), 11, stride=1)[0] == 0
    if valid.sum() < 1000 or support.sum() < 1000:
        raise RuntimeError('Foreground too small for reliable training at the configured resolution.')
    a, b = torch.where(valid, im, 0), torch.where(valid, gt, 0)
    q = torch.arange(11, device=im.device, dtype=im.dtype) - 5
    kernel = torch.exp(-q.square() / (2 * 1.5 ** 2)); kernel /= kernel.sum()
    win = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 11, 11).contiguous()
    conv = lambda x: F.conv2d(x[None], win, padding=5, groups=3)[0]
    u, v = conv(a), conv(b)
    aa, bb, ab = conv(a*a)-u*u, conv(b*b)-v*v, conv(a*b)-u*v
    ss = ((2*u*v+.01**2)*(2*ab+.03**2))/((u*u+v*v+.01**2)*(aa+bb+.03**2))
    l1 = (im-gt).abs()[:, valid[0]].mean(); ssim = ss[:, support[0]].mean()
    return .8*l1+.2*(1-ssim), l1, ssim


def image(tensor):
    return Image.fromarray(np.rint(tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype('uint8'))


class Views:
    def __init__(self, rows, height, cache_size, *, pose_indices=None):
        self.rows = rows; self.cache = OrderedDict(); self.cache_size = cache_size
        if pose_indices is None:
            pose_indices = [0]*len(rows)
        if len(pose_indices) != len(rows):
            raise ValueError('One named pose slot per observation is required.')
        self.cameras, self.records = zip(*(camera(r, i, height, pose_index=pose_indices[i]) for i, r in enumerate(rows)))
        for r in rows:
            if rt.sha(r['image']) != r['source_sha256'] or rt.sha(r['mask']) != r['mask_sha256']:
                raise RuntimeError(f'Input changed: {r["frame"]}')

    def get(self, i):
        import torch
        if i not in self.cache:
            r = self.rows[i]; size = tuple(self.records[i]['wh'])
            with Image.open(r['image']) as im:
                rgb = torch.from_numpy(np.array(im.convert('RGB').resize(size, Image.Resampling.LANCZOS))).permute(2, 0, 1)
            with Image.open(r['mask']) as im:
                alpha = torch.from_numpy(np.array(im.convert('L').resize(size, Image.Resampling.BILINEAR)))[None]
            self.cache[i] = rgb, alpha
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i); rgb, alpha = self.cache[i]
        return self.cameras[i], rgb.cuda().float()/255, alpha.cuda().float()/255


def train(root, cfg):
    from .admission import require_alignment
    require_alignment(root)
    return _train_static(root, cfg)


def verify_official_fit(root):
    """Read actual official frame receipts; quality approval is not an input.

    Descriptor paths are absolute or relative to the reconstruction root. The
    small aggregate is an index of completed runs, not a new fitting format.
    """
    root = Path(root)
    receipt = rt.read(root/'fit/official-fit-result.json')
    if (receipt.get('state') != 'complete' or
            receipt.get('fit_sha256') != rt.sha(root/'fit/fitted.npz') or
            receipt.get('training_inputs_sha256') != rt.sha(root/'data/training-inputs.json')):
        raise ValueError('Official completed-fit receipt does not bind these actual fit/source inputs.')
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = {key: archive[key] for key in archive.files}
    rows = rt.read(root/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, rows)
    if not layout['named']:
        raise ValueError('The official entry requires the complete named pose output.')
    frames = receipt.get('official_frame_results', [])
    if (not isinstance(frames, list) or len(frames) != len(rows) or
            {item.get('frame_id') for item in frames} != set(layout['frame_ids'])):
        raise ValueError('One completed official receipt per named frame is required.')
    def checked(item):
        path = Path(item['path'])
        if not path.is_absolute():
            path = root/path
        if not path.is_file() or rt.sha(path) != item.get('sha256'):
            raise ValueError('Changed or missing official source: ' + str(path))
        return path
    from .official_sequence import checked_image, check_result
    rows_by_frame = {row['frame']: row for row in rows}
    for item in frames:
        result = rt.read(checked(item))
        row = rows_by_frame[item['frame_id']]
        check_result(result, row)
        checked_image(row)
        if (result.get('state') != 'complete' or result.get('frame_id') != item['frame_id'] or
                result.get('five_stages') is not True or not result.get('official_commit')):
            raise ValueError('Official frame fitting is not complete: ' + item['frame_id'])
        if (result.get('source_frame_index') != row['source_frame_index'] or
                result.get('source_image_sha256') != row['source_sha256']):
            raise ValueError('Official frame result belongs to different source pixels: ' + item['frame_id'])
    sources = receipt.get('sources', [])
    if not isinstance(sources, list) or not sources:
        raise ValueError('Actual official adapter/export source descriptors are required.')
    for item in sources:
        checked(item)
    return dict(receipt_sha256=rt.sha(root/'fit/official-fit-result.json'),
                fit_sha256=receipt['fit_sha256'], training_inputs_sha256=receipt['training_inputs_sha256'],
                official_frame_results=frames, sources=sources, quality_accepted=False,
                formal_training_admitted=False)


def smpl_sh0_effective_config():
    """Resolve the stable profile once from the stock GaussianAvatars options."""
    from arguments import OptimizationParams

    parser = argparse.ArgumentParser()
    optimization = vars(OptimizationParams(parser).extract(parser.parse_args([]))).copy()
    optimization.update(iterations=10000, position_lr_max_steps=18000, densify_until_iter=18000,
        densify_from_iter=300, densification_interval=60, opacity_reset_interval=1800, lambda_laplacian=1e6)
    return dict(gaussian_profile='smpl_sh0', iterations=optimization['iterations'], seed=0,
        training_height=None, initial_points=None, initial_points_policy='one_per_triangle',
        image_cache_frames=None, image_cache_policy='all_selected_native_masked_targets',
        sh_degree=0, shared_clothing_offset_lr=1e-6, snapshot_steps=[1000, optimization['iterations']],
        optimization=optimization)


def train_official(root, cfg):
    """Train on hash-bound official named fits using the requested profile."""
    if cfg.get('gaussian_profile') == 'smpl_sh0':
        effective = smpl_sh0_effective_config()
        if any(cfg.get(key, value) != value for key, value in effective.items()):
            raise ValueError('Run config conflicts with the stable SH0 training profile.')
        return _train_smpl_sh0(Path(root), effective)
    return _train_static(Path(root), cfg, execution_kind='official_smplifyx')


def _uniform_laplacian(faces, vertex_count):
    """Uniform degree-normalized edge Laplacian of the fixed mesh topology."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.concatenate([edges, edges[:, ::-1]]), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    degrees = np.bincount(edges[:, 0], minlength=vertex_count)
    diag = np.flatnonzero(degrees)
    indices = np.concatenate([edges, np.column_stack([diag, diag])]).T
    values = np.concatenate([1. / degrees[edges[:, 0]], -np.ones(len(diag))]).astype(np.float32)
    return indices, values


def _frozen_smpl_surface(fit, model_path):
    """Replay official LBS and capture its fixed per-vertex linear transform."""
    from unittest.mock import patch
    import smplx
    import smplx.lbs as lbs
    import torch

    body = smplx.SMPLX(str(model_path), batch_size=1, num_betas=10, num_expression_coeffs=10,
        use_pca=True, num_pca_comps=12, flat_hand_mean=False, use_face_contour=False,
        create_body_pose=False, create_transl=False, dtype=torch.float32).eval().requires_grad_(False)
    faces = fit['faces']
    if not np.array_equal(body.faces, faces):
        raise ValueError('Official SMPL-X fit and training asset topology differ.')
    dominant = body.lbs_weights.argmax(-1).numpy()
    protected = np.isin(dominant, [12, 15, 22, 23, 24])
    with np.load(model_path, allow_pickle=False) as archive:
        landmarks = np.concatenate([body.lmk_faces_idx.flatten().numpy(), archive['dynamic_lmk_faces_idx'].flatten()])
    if not len(landmarks) or landmarks.min() < 0 or landmarks.max() >= len(faces):
        raise ValueError('SMPL-X facial landmark faces differ from the fitted topology.')
    protected[faces[landmarks].reshape(-1)] = True
    clothing = np.isin(dominant, [0, 1, 2, 3, 4, 5, 6, 9, 13, 14, 16, 17, 18, 19]) & ~protected
    base, linear = [], []
    original = lbs.batch_rigid_transform
    fields = ('global_orient', 'expression', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose')
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for i in range(len(fit['frame_ids'])):
            captured = []

            def capture(*args, **kwargs):
                result = original(*args, **kwargs)
                captured.append(result[1])
                return result

            params = {key: tensor(fit[key][i:i+1]) for key in fields}
            with patch.object(lbs, 'batch_rigid_transform', capture):
                output = body(**params, body_pose=tensor(fit['body_pose_axis_angle'][i:i+1]),
                              betas=tensor(fit['shared_betas']), return_verts=True)
            [transforms] = captured
            transforms = (body.lbs_weights @ transforms[0].reshape(-1, 16)).reshape(-1, 4, 4)
            E = tensor(fit['w2c'][i])
            base.append((output.vertices[0] + tensor(fit['camera_translation'][i]) - E[:3, 3]) @ E[:3, :3])
            linear.append(E[:3, :3].T @ transforms[:, :3, :3])
    base, linear = torch.stack(base), torch.stack(linear)
    error = float((base - tensor(fit['vertices_world'])).abs().max())
    if not np.isfinite(error) or error >= 2e-6 or not torch.isfinite(linear).all():
        raise RuntimeError(f'Frozen official SMPL-X mesh replay differs by {error} m.')
    return base, linear, clothing, protected, error


def _masked_native_target(row):
    import torch
    with Image.open(row['image']) as source, Image.open(row['mask']) as mask:
        if source.size != tuple(row['wh']) or mask.size != source.size:
            raise ValueError('Native RGB or mask size differs: ' + row['frame'])
        rgb = torch.from_numpy(np.array(source.convert('RGB'))).permute(2, 0, 1)
        alpha = torch.from_numpy(np.array(mask.convert('L')))[None]
    # Match the audited CPU composition before caching the float32 target.
    return rgb.float().div(255) * alpha.float().div(255)


def _train_smpl_sh0(root, effective):
    import random
    import torch
    from gaussian_renderer import render
    from roma import quat_xyzw_to_wxyz, rotmat_to_unitquat
    from scene.gaussian_model import GaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from utils.graphics_utils import compute_face_orientation
    from utils.loss_utils import l1_loss, ssim

    official = verify_official_fit(root)
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = dict(archive)
    rows = rt.read(root/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, rows)
    reference = layout['reference_pose_index']
    reference_row = next(i for i, row in enumerate(rows) if row['frame'] == layout['reference_frame_id'])
    cameras, records = zip(*(camera(row, i, pose_index=layout['pose_indices'][i]) for i, row in enumerate(rows)))
    torch.set_num_threads(4)
    base, linear, clothing, protected, replay_error = _frozen_smpl_surface(fit, rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    torch.backends.cuda.matmul.allow_tf32 = False
    seed = effective['seed']
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    options = SimpleNamespace(**effective['optimization'])

    class SharedSurface(SMPLXGaussianModel):
        def posed_vertices(self, slot):
            delta = self.static_offset * self.clothing[:, None]
            return self.base[slot] + (self.linear[slot] @ delta[..., None]).squeeze(-1)

        def select_mesh_by_timestep(self, timestep, original=False):
            self.timestep = int(timestep)
            verts = self.posed_vertices(self.timestep)
            self.face_center = verts[self.faces].mean(1)
            self.face_orien_mat, scaling = compute_face_orientation(verts, self.faces, return_scale=True)
            self.face_scaling = scaling.clamp_min(1e-8)
            self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))

    model = SharedSurface(effective['sh_degree'])
    model.base, model.linear = base.cuda(), linear.cuda()
    del linear
    model.clothing = torch.as_tensor(clothing, device='cuda')
    model.static_offset = torch.nn.Parameter(torch.zeros_like(model.base[0]))
    model.faces = torch.as_tensor(fit['faces'], dtype=torch.long, device='cuda')
    model.num_timesteps = len(base)
    model.binding = torch.arange(len(model.faces), device='cuda')
    model.binding_counter = torch.ones(len(model.faces), dtype=torch.int32, device='cuda')
    with torch.no_grad():
        model.select_mesh_by_timestep(reference)
        model.create_from_pcd(None, 1.)
    model.training_setup(options)
    model.optimizer.add_param_group(dict(params=[model.static_offset],
                                         lr=effective['shared_clothing_offset_lr'], name='static_offset'))
    indices, values = _uniform_laplacian(fit['faces'], len(clothing))
    laplacian = torch.sparse_coo_tensor(indices, values, (len(clothing), len(clothing)), device='cuda').coalesce()
    targets = [_masked_native_target(row).cuda() for row in rows]
    loader = torch.utils.data.DataLoader(range(len(rows)), batch_size=None, shuffle=True, num_workers=0)
    order = iter(loader)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][reference], axis=0)))
    if not 0 < extent < 10:
        raise ValueError('Invalid reference mesh extent in meters.')
    out = root/'gaussian-run'; out.mkdir()
    rt.save(out/'training-cameras.json', dict(records=records, source_sha256=official['training_inputs_sha256'], interpolated_cameras=0))
    resolved = dict(effective, initial_points=len(model.faces), image_cache_frames=len(targets))
    config_path = root/'config.json'
    rt.save(config_path, dict(rt.read(config_path), **resolved))
    rt.save(out/'config.json', dict(**resolved, execution_kind='official_smplifyx',
        official_fit_provenance=official, native_resolution=True, reference_extent_m=extent,
        optimizer_initial_state_entries=len(model.optimizer.state),
        cached_target_bytes=sum(t.numel()*t.element_size() for t in targets), fit_replay_max_abs_m=replay_error,
        active_clothing_vertices=int(clothing.sum()), protected_head_vertices=int(protected.sum()),
        target_composition='Native RGB times soft mask over black; original full-image L1 and DSSIM',
        source_training_options='259/269 10k terminal and 18k schedules; stock renderer and model operations'))
    if (root/'status.json').is_file():
        status = rt.read(root/'status.json')
        status.update(stage='training', training_started=True, updated=rt.now())
        rt.save(root/'status.json', status)
    counts = np.zeros(len(rows), dtype=int)
    start = time.monotonic()
    with (out/'loss.jsonl').open('x') as log:
        for step in range(1, options.iterations+1):
            model.update_learning_rate(step)
            try:
                i = next(order)
            except StopIteration:
                order = iter(loader)
                i = next(order)
            counts[i] += 1
            slot = layout['pose_indices'][i]
            pkg = render_pose(render, cameras[i], model, pipe, bg, slot)
            rgb, visible = pkg['render'], pkg['visibility_filter']
            delta = model.static_offset * model.clothing[:, None]
            photo = ((1-options.lambda_dssim)*l1_loss(rgb, targets[i]) +
                     options.lambda_dssim*(1-ssim(rgb, targets[i])))
            xyz = (model._xyz[visible].norm(dim=1)-options.threshold_xyz).relu().mean()*options.lambda_xyz
            scale = (model._scaling[visible].exp()-options.threshold_scale).relu().norm(dim=1).mean()*options.lambda_scale
            lap = torch.sparse.mm(laplacian, delta).square().sum(-1).mean()*options.lambda_laplacian
            loss = photo+xyz+scale+lap
            loss.backward()
            with torch.no_grad():
                if step == 1:
                    grads = [p.grad for g in model.optimizer.param_groups for p in g['params']]
                    if (model.optimizer.state or any(g is None or not torch.isfinite(g).all() for g in grads)
                            or not model.static_offset.grad.norm() > 0):
                        raise RuntimeError('Fresh SH0 preflight did not produce finite gradients and a shared surface update.')
                    rt.save(out/'preflight.json', dict(finite_backward=True, optimizer_updates=0,
                        static_offset_gradient_norm=float(model.static_offset.grad.norm()),
                        fit_replay_max_abs_m=replay_error))
                if step == 1 or step % 100 == 0 or step == options.iterations:
                    if not torch.isfinite(loss) or torch.count_nonzero(model.static_offset.grad[~model.clothing]):
                        raise RuntimeError(f'Invalid SH0 loss or protected-vertex gradient at step {step}.')
                    progress = dict(updated=rt.now(), stage='training', step=step, target_steps=options.iterations,
                        points=len(model._xyz), loss=float(loss), rgb_loss=float(photo), laplacian_loss=float(lap),
                        seconds=time.monotonic()-start, views_seen=int((counts>0).sum()), quality_accepted=False)
                    rt.save(root/'progress.json', progress); log.write(json.dumps(progress)+'\n'); log.flush()
                if step in effective['snapshot_steps']:
                    snapshot = out/f'step-{step:06d}'; snapshot.mkdir()
                    model.export_world_ply(snapshot/'world.ply', reference)
                    reloaded = GaussianModel(0); reloaded.load_ply(str(snapshot/'world.ply'))
                    native, diff = reference_ply_comparison(render, cameras[reference_row], model, reloaded, pipe, bg, reference)
                    parity = dict(mean_abs=float(diff.mean()), max_abs=float(diff.max()))
                    passed = parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005
                    mesh = model.posed_vertices(reference).cpu().numpy()
                    if not np.isfinite(mesh).all() or not np.array_equal(mesh[protected], base[reference].numpy()[protected]):
                        raise RuntimeError('Nonfinite or displaced protected reference geometry.')
                    np.savez_compressed(snapshot/'reference-mesh.npz', vertices_world=mesh, faces=fit['faces'],
                        reference_frame_id=layout['reference_frame_id'], reference_pose_index=reference,
                        static_offset=delta.cpu().numpy())
                    image(native).save(snapshot/'reference-render.png')
                    rt.save(snapshot/'native-ply-parity.json', dict(passed=passed, checks=[parity],
                        sha256=rt.sha(snapshot/'world.ply'), mesh_sha256=rt.sha(snapshot/'reference-mesh.npz'),
                        reference_frame_id=layout['reference_frame_id'], reference_pose_index=reference))
                    if not passed:
                        raise RuntimeError('SH0 PLY does not reproduce native reference geometry.')
                    model.select_mesh_by_timestep(slot)
                    del reloaded, native, diff
                model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], pkg['radii'][visible])
                model.add_densification_stats(pkg['viewspace_points'], visible)
                if step > options.densify_from_iter and step % options.densification_interval == 0:
                    size = 20 if step > options.opacity_reset_interval else None
                    model.densify_and_prune(options.densify_grad_threshold, .005, extent, size)
                    if len(model._xyz) >= 1200000 or int(model.binding_counter.min()) < 1:
                        raise RuntimeError('Point budget or triangle coverage exceeded the SH0 safety bounds.')
                if step % options.opacity_reset_interval == 0:
                    model.reset_opacity()
                if step < options.iterations:
                    model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
    rt.save(out/'result.json', dict(state='completed', from_scratch=True, steps=options.iterations,
        execution_kind='official_smplifyx', gaussian_profile='smpl_sh0', sh_degree=0, points=len(model._xyz),
        schedule_steps=options.position_lr_max_steps, formal_training_admitted=False, quality_accepted=False,
        per_frame_updates=counts.tolist(), pose_layout=layout, reference_frame_id=layout['reference_frame_id'],
        final_snapshot=f'step-{options.iterations:06d}'))
    del targets, laplacian
    try:
        from .bound_replay import save_bound_snapshot

        with torch.no_grad():
            return save_bound_snapshot(root, out/f'step-{options.iterations:06d}', model)
    except Exception as error:
        # A sidecar failure does not invalidate the already verified static PLY.
        rt.save(out/'bound-snapshot-failure.json', dict(state='failed', step=options.iterations,
                error_type=type(error).__name__, error=str(error), optimizer_updates=0))
        return None


def replay_official_bound(root, replay=None):
    """Render each selected native camera with its saved pose under the caller's GPU lease."""
    from tempfile import TemporaryDirectory
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from .bound_replay import load_bound_snapshot

    root = Path(root)
    out = root/'gaussian-run'
    training = rt.read(out/'result.json')
    effective = smpl_sh0_effective_config()
    terminal_step = effective['iterations']
    if (training.get('state') != 'completed' or training.get('gaussian_profile') != effective['gaussian_profile']
            or training.get('steps') != terminal_step or training.get('sh_degree') != effective['sh_degree']
            or training.get('final_snapshot') != f'step-{terminal_step:06d}'):
        raise ValueError('Bound replay requires the successful terminal stable SH0 training snapshot.')
    failure = out/'bound-snapshot-failure.json'
    if failure.is_file():
        raise RuntimeError(f'Terminal bound snapshot failed. See {failure}')
    snapshot = out/training['final_snapshot']
    destination = snapshot/'posed-replay'
    if destination.exists():
        raise FileExistsError(f'Preserve the existing posed replay: {destination}')
    official = verify_official_fit(root)
    selection = rt.read(root/'fit/official-fit-result.json')['selection']
    rows = rt.read(root/'data/training-inputs.json')['records']
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        layout = named_pose_layout(archive, rows)
    frames = [dict(frame=row['frame'], pose_index=slot, source_frame_index=row['source_frame_index'],
                   source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'])
              for row, slot in zip(rows, layout['pose_indices'], strict=True)]
    ids = [item['frame'] for item in frames]
    held = selection['retained_holdouts']
    counts = training['per_frame_updates']
    if (set(ids) != set(selection['selected_frames']) or set(ids) & set(held)
            or training['pose_layout'] != layout or len(counts) != len(ids)
            or min(counts) <= 0 or sum(counts) != training['steps']
            or any(Path(frame).name != frame or frame in ('.', '..') for frame in ids)):
        raise ValueError('The replay roster must match only the selected, sampled fit and camera identities.')

    manifest_path = snapshot/'bound-replay.json'
    if replay is None:
        run_path = root/'run.json'
        assets = rt.read(run_path).get('assets') if run_path.is_file() else None
        asset_path = Path(assets)/'smplx/SMPLX_NEUTRAL.npz' if assets else None
        replay = load_bound_snapshot(root, snapshot, asset_path=asset_path)
    model, manifest = replay
    ref_frame, ref_slot = layout['reference_frame_id'], layout['reference_pose_index']
    bound_path = snapshot/'bound-local.ply'
    if (manifest != rt.read(manifest_path) or manifest['frames'] != frames
            or manifest['bound_ply_sha256'] != rt.sha(bound_path)
            or manifest['reference_frame_id'] != ref_frame or manifest['reference_pose_index'] != ref_slot
            or manifest['points'] != training['points'] or len(model.binding) != training['points']
            or model.optimizer is not None or model.active_sh_degree != 0):
        raise ValueError('The verified bound model, snapshot and replay roster differ.')
    movement = manifest['geometry_parity']['live_to_reloaded']
    if [(m['frame'], m['pose_index']) for m in movement] != [(f['frame'], f['pose_index']) for f in frames]:
        raise ValueError('The terminal bound motion records differ from the named poses.')

    camera_path = out/'training-cameras.json'
    archived_cameras = rt.read(camera_path)
    camera_records = archived_cameras['records']
    if archived_cameras['source_sha256'] != official['training_inputs_sha256'] or len(camera_records) != len(frames):
        raise ValueError('The terminal cameras do not identify the selected source inputs.')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device=model.binding.device)
    records = []
    with TemporaryDirectory(prefix='.posed-replay-', dir=snapshot) as temporary, torch.no_grad():
        stage = Path(temporary)/'replay'
        stage.mkdir()
        for i, (row, item, actual, motion) in enumerate(zip(rows, frames, camera_records, movement, strict=True)):
            slot = item['pose_index']
            cam, record = camera(row, i, pose_index=slot)
            if any(record[key] != actual[key] for key in ('frame', 'pose_index', 'K', 'w2c', 'wh', 'pixel_convention')):
                raise ValueError('Replayed camera differs from the training camera: ' + item['frame'])
            rgb = render_pose(render, cam, model, pipe, bg, slot)['render']
            if not torch.isfinite(rgb).all():
                raise RuntimeError('Nonfinite bound replay: ' + item['frame'])
            posed_image = image(rgb)
            if item['frame'] == ref_frame:
                world = GaussianModel(0)
                world.load_ply(str(snapshot/'world.ply'))
                diff = (rgb - render(cam, world, pipe, bg)['render']).abs()
                with Image.open(snapshot/'reference-render.png') as reference:
                    archive_diff = np.abs(np.asarray(posed_image, dtype=np.int16)
                                          - np.asarray(reference.convert('RGB'), dtype=np.int16))
                parity = dict(mean_abs=float(diff.mean()), max_abs=float(diff.max()),
                              archived_max_abs_8bit=int(archive_diff.max()))
                if not (parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005
                        and parity['archived_max_abs_8bit'] == 0):
                    raise RuntimeError(f'The reloaded bound reference render differs: {parity}')
            name = item['frame'] + '.png'
            posed_image.save(stage/name)
            records.append(dict(item, image=name, image_sha256=rt.sha(stage/name), camera=record, movement=motion))

        mesh_file = 'meshes.npz'
        slots = layout['pose_indices']
        np.savez_compressed(stage/mesh_file, vertices_world=model.vertices_sequence[slots].cpu().numpy(),
            faces=model.faces.cpu().numpy(), frame_ids=np.asarray(ids), pose_indices=np.asarray(slots),
            source_frames=np.asarray([item['source_frame_index'] for item in frames]),
            image_sha256=np.asarray([item['source_sha256'] for item in frames]),
            mask_sha256=np.asarray([item['mask_sha256'] for item in frames]))
        receipt = dict(state='completed', reference_frame_id=ref_frame, reference_pose_index=ref_slot,
            reference_reload_parity=parity, bound_replay_sha256=rt.sha(manifest_path),
            training_inputs_sha256=official['training_inputs_sha256'], training_camera_sha256=rt.sha(camera_path),
            mesh_file=mesh_file, mesh_sha256=rt.sha(stage/mesh_file), retained_holdouts=held, records=records,
            movement_scope='Gaussian centers in the fitted world, including rigid pose placement.',
            optimizer_updates=0, quality_accepted=False)
        rt.save(stage/'result.json', receipt)
        stage.rename(destination)
    return receipt


def _train_static(root, cfg, *, execution_kind='formal_training'):
    official_receipt = None
    if execution_kind == 'official_smplifyx':
        official_receipt = verify_official_fit(root)
    else:
        gate = rt.read(root/'fit/geometry-gate.json')
        if not gate['coarse_initialization_gate_passed'] or gate['fit_sha256'] != rt.sha(root/'fit/fitted.npz'):
            raise RuntimeError('Fresh geometry gate missing, failed or no longer matches the fit.')
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from arguments import OptimizationParams
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = {key: archive[key] for key in archive.files}
    rows = rt.read(root/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, rows)
    out = root/'gaussian-run'; out.mkdir()
    seed = cfg['seed']; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    views = Views(rows, cfg['training_height'], cfg['image_cache_frames'], pose_indices=layout['pose_indices'])
    rt.save(out/'training-cameras.json', dict(records=list(views.records), source_sha256=rt.sha(root/'data/training-inputs.json'), interpolated_cameras=0))
    model = SMPLXGaussianModel(3); model.load_fresh_meshes(root/'fit/fitted.npz')
    init = model.initialize_surface(cfg['initial_points'], seed)
    parser = argparse.ArgumentParser(); options = OptimizationParams(parser).extract(parser.parse_args([]))
    schedule_steps = cfg['iterations']; steps = schedule_steps
    options.iterations = schedule_steps; options.position_lr_max_steps = schedule_steps
    options.densify_until_iter = schedule_steps; model.training_setup(options)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    rt.save(out/'config.json', dict(**cfg, route='local_named_SMPLX_mesh_bound' if layout['named'] else 'local_static_SMPLX_mesh_bound',
                                  upstream_complete_reproduction=False, pose_layout=layout,
                                  official_fit_provenance=official_receipt,
                                  execution_kind=execution_kind, executed_step_budget=steps,
                                  formal_training_admitted=execution_kind=='formal_training',
                                  from_scratch=True, old_subject_checkpoint_loaded=False, initialization=init,
                                  fit_sha256=rt.sha(root/'fit/fitted.npz'), optimizer_initial_state_entries=len(model.optimizer.state)))
    cam, gt, alpha = views.get(0); pkg = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][0])
    photo, _, _ = foreground_loss(pkg['render'], gt, alpha)
    other = gt.clone(); other[:, alpha[0] <= .99] = .37
    outside_effect = float(abs(photo-foreground_loss(pkg['render'], other, alpha)[0]))
    if outside_effect > 1e-7:
        raise RuntimeError('Background RGB leaked into foreground photometric loss.')
    aa = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][0], override_color=torch.ones_like(model.get_xyz))['render'][:1]
    (photo+.1*(aa-alpha).abs().mean()).backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params']):
        raise RuntimeError('Nonfinite or missing training gradients.')
    model.optimizer.zero_grad(set_to_none=True)
    rt.save(out/'preflight.json', dict(foreground_external_RGB_effect=outside_effect, finite_backward=True, optimizer_updates=0))
    del pkg, photo, aa, other
    if (root/'status.json').is_file():
        status = rt.read(root/'status.json')
        status.update(stage='training', training_started=True, updated=rt.now())
        rt.save(root/'status.json', status)
    counts = np.zeros(len(rows), np.int32); stack = []; rng = np.random.default_rng(seed); start = time.monotonic()
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][0], axis=0)))
    selected = np.unique(np.linspace(0, len(rows)-1, 6).astype(int))
    checkpoints = sorted({v for v in [1000, 6000, 12000, steps] if v <= steps})
    with (out/'loss.jsonl').open('x') as log:
        for step in range(1, steps+1):
            model.update_learning_rate(step)
            if step % 1000 == 0: model.oneupSHdegree()
            if not stack: stack = rng.permutation(len(rows)).tolist()
            i = stack.pop(); counts[i] += 1; cam, gt, alpha = views.get(i)
            pkg = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][i]); photo, l1, ssim = foreground_loss(pkg['render'], gt, alpha)
            aa = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][i], override_color=torch.ones_like(model.get_xyz))['render'][:1]
            silhouette = (aa-alpha).abs().mean(); visible = pkg['visibility_filter']
            if not visible.any(): raise RuntimeError('No visible Gaussians in a registered camera.')
            regularizer = torch.relu(model._xyz[visible].norm(dim=-1)-options.threshold_xyz).mean()*options.lambda_xyz
            regularizer += torch.relu(model._scaling[visible].exp()-options.threshold_scale).norm(dim=-1).mean()*options.lambda_scale
            loss = photo+.1*silhouette+regularizer
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            with torch.no_grad():
                model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], pkg['radii'][visible])
                if step < steps: model.add_densification_stats(pkg['viewspace_points'], visible)
                model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
                if options.densify_from_iter < step < steps and step % options.densification_interval == 0:
                    model.densify_and_prune(options.densify_grad_threshold, .005, extent, None)
                    if len(model._xyz) >= 1200000 or int(model.binding_counter.min()) < 1:
                        raise RuntimeError('Point budget or surface-coverage constraint failed.')
            if step == 1 or step % 100 == 0 or step == steps:
                progress = dict(updated=rt.now(), stage='training', step=step, target_steps=steps, points=len(model.get_xyz),
                                loss=float(loss), foreground_l1=float(l1), foreground_ssim=float(ssim), alpha_mae=float(silhouette),
                                seconds=time.monotonic()-start, views_seen=int((counts>0).sum()), quality_accepted=False,
                                execution_kind=execution_kind)
                rt.save(root/'progress.json', progress); log.write(__import__('json').dumps(progress)+'\n'); log.flush()
                print(progress, flush=True)
            if step in checkpoints:
                with torch.no_grad():
                    snapshot = out/f'step-{step:06d}'; snapshot.mkdir()
                    reference_pose = layout['reference_pose_index']
                    model.export_world_ply(snapshot/'world.ply', reference_pose)
                    reloaded = GaussianModel(3); reloaded.load_ply(str(snapshot/'world.ply')); reloaded.active_sh_degree=model.active_sh_degree
                    checks = []
                    for j in selected:
                        cam, gt, alpha = views.get(int(j))
                        reference, delta = reference_ply_comparison(render, cam, model, reloaded, pipe, bg, reference_pose)
                        checks.append(dict(frame=rows[j]['frame'], pose_frame=layout['reference_frame_id'],
                                           mean_abs=float(delta.mean()), max_abs=float(delta.max())))
                        native = reference if layout['pose_indices'][j] == reference_pose else render_pose(
                            render, cam, model, pipe, bg, layout['pose_indices'][j])['render']
                        image(gt).save(snapshot/f'{rows[j]["frame"]}_input.png')
                        image(gt*alpha).save(snapshot/f'{rows[j]["frame"]}_masked_input.png')
                        image(native).save(snapshot/f'{rows[j]["frame"]}_render.png')
                    passed = all(r['mean_abs'] < 2e-6 and r['max_abs'] < .005 for r in checks)
                    rt.save(snapshot/'native-ply-parity.json', dict(passed=passed, checks=checks, sha256=rt.sha(snapshot/'world.ply'),
                        reference_frame_id=layout['reference_frame_id'], reference_pose_index=reference_pose,
                        observed_render_pose='Each *_render.png uses its matching named input pose; PLY parity uses only the named reference pose.'))
                    if not passed: raise RuntimeError('World PLY does not reproduce native training geometry.')
                    del reloaded
    rt.save(out/'result.json', dict(state='completed', from_scratch=True, steps=steps,
                                   execution_kind=execution_kind, schedule_steps=schedule_steps,
                                   formal_training_admitted=execution_kind=='formal_training',
                                   per_frame_updates=counts.tolist(), pose_layout=layout,
                                   reference_frame_id=layout['reference_frame_id'],
                                   final_snapshot=f'step-{steps:06d}', quality_accepted=False))
