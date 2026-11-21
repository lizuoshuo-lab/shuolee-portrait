"""Read back named SH0 poses at 1× and 0.5× Gaussian standard deviations."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from review_bound_alpha_ab import delivered, detector_rois, mask_metrics, native_box, pin, read, require, sha, write_panel


def scale_pixels(render, render_pose, cam, model, pipe, black, white, pose, scale):
    import torch

    args = (render, cam, model, pipe)
    rgb = render_pose(*args, black, pose, scaling_modifier=scale)['render']
    over_white = render_pose(*args, white, pose, scaling_modifier=scale)['render']
    alpha = render_pose(*args, black, pose, scaling_modifier=scale,
                        override_color=torch.ones_like(model.get_xyz))['render']
    error = float((over_white-rgb-(1-alpha[:1])).abs().max())
    require(all(torch.isfinite(t).all() for t in (rgb, over_white, alpha))
            and float((alpha-alpha[:1]).abs().max()) < 1e-6
            and float(alpha.min()) >= -1e-5 and float(alpha.max()) <= 1.00001 and error < 5e-5,
            'Native black/white composition disagrees with the alpha readback.')
    pixels = (rgb.permute(1, 2, 0).cpu().numpy(), over_white.permute(1, 2, 0).cpu().numpy(), alpha[0].cpu().numpy())
    return pixels, error


def geometry(model):
    from scipy.spatial import cKDTree

    centers = model.get_xyz.detach().cpu().numpy()
    sigma = model.get_scaling.detach().cpu().numpy()
    binding = model.binding.detach().cpu().numpy()
    triangles = model.vertices_sequence[model.timestep][model.faces].detach().cpu().numpy()
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    require(np.isfinite(centers).all() and np.isfinite(sigma).all() and (sigma > 0).all(),
            'Invalid terminal Gaussian geometry.')
    valid = norms[binding] > 1e-12
    face = binding[valid]
    plane = np.einsum('ij,ij->i', centers[valid]-triangles[face, 0], normals[face]) / norms[face]
    nearest = cKDTree(centers).query(centers, k=2)[0][:, 1]
    quantiles = lambda values: np.quantile(values, [.5, .95]).tolist()
    return dict(nearest_center_distance_mm_p50_p95=quantiles(nearest*1000),
                principal_sigma_max_mm_p50_p95=quantiles(sigma.max(axis=1)*1000),
                abs_bound_face_plane_distance_mm_p50_p95=quantiles(np.abs(plane)*1000),
                bound_face_plane_evaluable=int(valid.sum()), occupied_faces=int(np.unique(binding).size))


def interior(points, wh):
    from scipy.ndimage import binary_erosion
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 8:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    raster = Image.new('1', tuple(wh))
    ImageDraw.Draw(raster).polygon([tuple(point) for point in points[hull.vertices]], fill=1)
    return binary_erosion(np.asarray(raster), iterations=2)


def face_alpha(alpha, region):
    if region is None or region.sum() < 32:
        return None
    samples = alpha[region]
    return dict(pixels=len(samples), alpha_p01_p05_p50=np.quantile(samples, [.01, .05, .5]).tolist(),
                fraction_below_095=float((samples < .95).mean()), fraction_below_090=float((samples < .9).mean()),
                mean_transmission=float((1-samples).mean()))


def face_dolly(row, observed, joints_camera, world_xyz):
    """Move along the face-center ray at fixed focal length, above CUDA's 0.2 m cutoff."""
    K, E = np.asarray(row['K']), np.asarray(row['w2c'])
    selected = (observed[:, 2] >= .5) & (joints_camera[:, 2] > .2)
    if selected.sum() < 8:
        return None
    source_proj = joints_camera[selected] @ K.T
    residual = np.linalg.norm(source_proj[:, :2]/source_proj[:, 2:]-.5-observed[selected, :2], axis=1)
    matched = joints_camera[selected][residual <= 6.]
    if len(matched) < 8:
        return None
    center = np.median(matched, axis=0)
    depth = world_xyz @ E[2, :3] + E[2, 3]
    shift = min(center[2]*.5, depth.min()-.30)
    if shift < center[2]*.20:
        return None
    movement = center/center[2]*shift
    new_E = E.copy()
    new_E[:3, 3] -= movement
    near_joints = matched-movement
    if not np.isfinite(near_joints).all() or (near_joints[:, 2] <= .2).any():
        return None
    projected = near_joints @ K.T
    xy = projected[:, :2]/projected[:, 2:]-.5
    if (xy < 0).any() or (xy >= np.asarray(row['wh'])-1).any():
        return None
    region = interior(xy, row['wh'])
    if region is None or region.sum() < 32:
        return None
    box = native_box([*(xy.min(axis=0)-24), *(xy.max(axis=0)+24)], row['wh'])
    return dict(row, w2c=new_E.tolist()), region, box, dict(
        source_face_depth_m=float(center[2]), dolly_camera_space_m=movement.tolist(),
        min_visible_candidate_center_depth_m=float(depth.min()-shift), raster_cutoff_m=.20,
        retained_landmarks=int(len(matched)), native_landmark_match_max_px=6.,
        scope='Projected matched inner-face landmarks. No measured RGB exists at the dolly camera.')


def write_dolly(path, box, pixels):
    x0, y0, x1, y1 = box
    w, h = x1-x0, y1-y0
    panel = Image.new('RGB', (w*4, h+21), '#171717')
    draw = ImageDraw.Draw(panel)
    labels = ('1x near / black', '1x near / white', 'Accumulated alpha', 'Transmission 1-alpha')
    for i, value in enumerate((*pixels, 1-pixels[2])):
        rgba = np.rint(np.clip(value[y0:y1, x0:x1], 0, 1)*255).astype(np.uint8)
        panel.paste(Image.fromarray(rgba).convert('RGB'), (i*w, 21))
        draw.text((i*w+4, 5), labels[i], fill='white')
    panel.save(path)
    return dict(file=path.name, sha256=sha(path), crop_xyxy=box)


def exported_view(row, transform):
    E = np.asarray(row['w2c'])
    Q, origin, scale = np.asarray(transform['rotation']), np.asarray(transform['origin_world']), transform['scale']
    moved = np.eye(4)
    moved[:3, :3] = E[:3, :3] @ Q.T
    moved[:3, 3] = scale * (E[:3, :3] @ origin + E[:3, 3])
    return dict(row, w2c=moved.tolist())


def pixel_parity(left, right):
    delta = [np.abs(a-b) for a, b in zip(left, right, strict=True)]
    result = [dict(mean=float(d.mean()), max=float(d.max())) for d in delta]
    return dict(black_white_alpha=result, passed=all(d['mean'] < 2e-6 and d['max'] < .005 for d in result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'project', 'asset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frames', nargs='+', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    root, project, asset, output = (path.resolve() for path in (args.run, args.project, args.asset, args.output))
    require(not output.is_relative_to(root.parent) and not output.is_relative_to(project),
            'Write the audit separately from the saved run and source.')
    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait.bound_replay import _load_model

    rows = read(root/'data/training-inputs.json')['records']
    selection = read(root/'fit/selection.json')
    saved = delivered(root, asset, rows, selection['retained_holdouts'])
    require(saved.bound['source_files_sha256'] == saved.facts['source_files_sha256'],
            'Use the exact terminal renderer and model source.')
    require(len(set(args.frames)) == len(args.frames) and set(args.frames) <= {row['frame'] for row in rows},
            'Use unique training-frame IDs from the actual terminal roster.')
    fit_receipt = read(root/'fit/official-fit-result.json')
    require(fit_receipt['fit_sha256'] == saved.facts['inputs_sha256']['fitted'],
            'The official fitting receipt no longer matches the bound model.')
    observation_pins = [(Path(path), digest) for path, digest in fit_receipt['source']['input_files'].items()
                        if Path(path).name == 'observations.npz']
    require(len(observation_pins) == 1, 'Expected one observation archive in the official fitting receipt.')
    observations, observation_pin = observation_pins[0]
    observations_sha = pin(observations, observation_pin)
    with np.load(observations, allow_pickle=False) as obs:
        landmarks = dict(zip(obs['frame_ids'].tolist(), obs['keypoints'], strict=True))
    records = []
    for i, row in enumerate(rows):
        if row['frame'] not in args.frames:
            continue
        identity = saved.facts['frames'][i]
        pin(Path(row['image']), identity['source_sha256'])
        pin(Path(row['mask']), identity['mask_sha256'])
        require(identity['frame'] in landmarks, 'Missing raw detector landmarks for the named frame.')
        records.append((i, row, identity))
    if args.preflight:
        print(json.dumps(dict(terminal=saved.terminal_sha256, frames=[identity for _, _, identity in records],
                              renderer_sources_pinned=True, gpu_calls=0)))
        return
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    evidence = []
    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            model, parity = _load_model(saved.snapshot/'bound-local.ply', saved.snapshot, saved.fit,
                                        saved.mesh, saved.facts, asset)
            fixed = GaussianModel(0)
            fixed.load_ply(str(saved.snapshot/'world.ply'))
            export = root/'export'
            transform = read(export/'validation.json')
            capture = read(export/'camera_provenance.json')
            pin(export/'cameras.json', capture['files_sha256']['cameras.json'])
            require(transform['sha256'] == saved.terminal['sha256'] and transform['format_passed'] is True
                    and transform['scale'] == 1 and len(capture['records']) == 1,
                    'The export is not a verified unit-scale reference snapshot.')
            view_model = GaussianModel(0)
            view_model.load_ply(str(root/saved.terminal['ply']))
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            output.mkdir(parents=True, exist_ok=False)
            for i, row, identity in records:
                pose, frame = identity['pose_index'], identity['frame']
                model.select_mesh_by_timestep(pose)
                points = model.get_xyz.clone()
                spread = geometry(model)
                cam, _ = camera(row, i, pose_index=pose)
                with Image.open(row['image']) as im, Image.open(row['mask']) as target:
                    require(list(im.size) == list(target.size) == row['wh'], 'Native frame size changed.')
                    rgb, mask = np.asarray(im.convert('RGB')), np.asarray(target.convert('L'), dtype=np.float32)/255.
                _, dark, head = detector_rois(rgb, mask, landmarks[frame])
                observations_face = saved.fit['observed_native_keypoints'][pose, 67:118]
                roi = interior(observations_face[observations_face[:, 2] >= .5, :2], row['wh'])
                if roi is not None:
                    roi &= mask > .99
                metrics, pixels = {}, []
                for scale in (1., .5):
                    rendered, error = scale_pixels(render, render_pose, cam, model, pipe, black, white, pose, scale)
                    pixels.append(rendered)
                    metrics[str(scale)] = dict(**mask_metrics(rendered[2], mask, dark), composition_max_abs=error,
                                               interior_face_alpha=face_alpha(rendered[2], roi))
                require(torch.equal(model.get_xyz, points) and model.optimizer is None,
                        'Scale pressure unexpectedly changed centers or created an optimizer.')
                archived = saved.replay['records'][i]
                path = saved.snapshot/'posed-replay'/archived['image']
                pin(path, archived['image_sha256'])
                with Image.open(path) as im:
                    prior = np.asarray(im.convert('RGB'), dtype=np.int16)
                difference = np.abs(np.rint(np.clip(pixels[0][0], 0, 1)*255).astype(np.int16)-prior)
                require(difference.max() <= 1, 'The 1× render no longer matches the saved native replay.')
                is_reference = pose == saved.facts['reference_pose_index']
                export_native_parity = None
                if is_reference:
                    viewer_row = exported_view(row, transform)
                    expected = capture['records'][0]
                    require(all(viewer_row[k] == expected[k] for k in ('frame', 'K', 'wh', 'source_frame_index'))
                            and np.allclose(viewer_row['w2c'], expected['w2c'], rtol=0, atol=1e-8),
                            'Rebased reference camera does not match the delivered camera.')
                    view_cam, _ = camera(viewer_row, i, pose_index=pose)
                    view_pixels, _ = scale_pixels(render, render_pose, view_cam, view_model, pipe, black, white, pose, 1.)
                    export_native_parity = pixel_parity(view_pixels, pixels[0])
                x0, y0, x1, y1 = row['initial_box_xyxy']
                full = native_box([x0-64, y0-64, x1+64, y1+64], row['wh'])
                panels = [write_panel(output, frame+'-full.png', rgb, mask, *pixels, full)]
                if head is not None:
                    panels.append(write_panel(output, frame+'-head.png', rgb, mask, *pixels, head))
                closeup = None
                dolly = face_dolly(row, observations_face, saved.fit['joints_camera'][pose, 67:118],
                                   points.cpu().numpy())
                if dolly is not None:
                    dolly_row, region, crop, closeup = dolly
                    near_cam, near_camera = camera(dolly_row, i, pose_index=pose)
                    near, error = scale_pixels(render, render_pose, near_cam, model, pipe, black, white, pose, 1.)
                    closeup.update(camera=near_camera, scale=1., composition_max_abs=error,
                                   interior_face_alpha=face_alpha(near[2], region))
                    if is_reference:
                        static, _ = scale_pixels(render, render_pose, near_cam, fixed, pipe, black, white, pose, 1.)
                        closeup['reference_static_parity'] = pixel_parity(static, near)
                        require(closeup['reference_static_parity']['passed'],
                                'Bound reference and native static PLY disagree at the close-up camera.')
                        viewer_cam, _ = camera(exported_view(dolly_row, transform), i, pose_index=pose)
                        viewer_pixels, _ = scale_pixels(render, render_pose, viewer_cam, view_model, pipe, black, white, pose, 1.)
                        closeup['exported_static_parity'] = pixel_parity(viewer_pixels, near)
                    panels.append(write_dolly(output/(frame+'-dolly.png'), crop, near))
                evidence.append(dict(**identity, metrics=metrics, geometry=spread,
                                     native_1x_archive_max_abs_8bit=int(difference.max()),
                                     exported_static_native_parity=export_native_parity, closeup_1x=closeup, panels=panels))
    pin(observations, observations_sha)
    note = dict(state='completed', terminal_sha256=saved.terminal_sha256,
                audit_source_sha256=sha(Path(__file__)),
                pose_replay_sha256=saved.replay_sha256, bound_ply_sha256=saved.bound['bound_ply_sha256'],
                renderer_source_sha256=saved.bound['source_files_sha256'], observations_sha256=observations_sha,
                model_parity=parity, optimizer_updates=0, records=evidence,
                scale_scope='World-space standard deviation multiplier. The CUDA rasterizer retains its 0.3 px² screen covariance floor.',
                geometry_scope='Assigned face plane and nearest Gaussian center distances; not observed clothing surface error.',
                face_scope='Confident inner RTMW landmark hull, eroded two native pixels and intersected with target mask. '
                           'Dolly uses projected fitted landmarks with <=6 px native agreement and preserves focal length. '
                           'Face interior and eye subsets are audit proxies, not a calibrated physical opacity truth.',
                panel_layout='Top: original, mask, 1× black, 1× white. Bottom: 1× alpha, 0.5× black, 0.5× white, 0.5× alpha.')
    receipt = output/'result.json'
    receipt.write_text(json.dumps(note, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(receipt=str(receipt), sha256=sha(receipt), frames=len(evidence))))


if __name__ == '__main__':
    main()
