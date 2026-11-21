"""Replay the registered opaque shirt-hem hole with temporary opacity controls."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import label

from review_bound_alpha_ab import pin, read, require, sha
from review_bound_scale_coverage import face_alpha, scale_pixels
from review_unified_cold_views import ASSET, GPU_LEASE, LAB, save


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-appearance-20261110')
RUN = LAB/'shuolee-smpl-head-appearance-008/run'
OUT = LAB/'shuolee-smpl-008-hem-opacity-ceiling'
FRAME = 'frame_000058'
OWNERS = [4047, 4198, 4199, 4237, 5313, 14579, 14730, 14731, 15276, 15842]
HITS = [8, 312, 203, 167, 39, 1, 218, 237, 177, 1]
CROP = (1213, 1128, 1421, 1282)


def archived_roi():
    png = LAB/'shuolee-smpl-unified-008-matched-views/late-new-native.png'
    digest = pin(png, 'ee62e9673362406d73e0d3153d983ca0a4e5e9b79f6f3633ececbe41b98be942')
    with Image.open(png) as im:
        score = np.asarray(im.crop((834, 21, 1668, 1199)).convert('L'))
        alpha = np.asarray(im.crop((0, 1220, 834, 2398)).convert('L'))
    source_y = 342 + np.arange(1178)[:, None]
    blobs, _ = label((score >= 253) & (alpha < 128) & (source_y >= 850) & (source_y <= 1250),
                     structure=np.ones((3, 3)))
    index = blobs[1209-342, 1305-940]
    require(index > 0, 'The historical opaque-mask hole is missing.')
    part = blobs == index
    roi = np.zeros((1520, 2704), bool)
    roi[342:1520, 940:1774] = part
    signature = hashlib.sha256(roi.tobytes()).hexdigest()
    require(int(roi.sum()) == 1363 and signature ==
            'd4640409df7cf3756d599ab75c9f489bc31a8f41a6219dfa56db8c1d8be53225',
            'Historical native component differs.')
    return roi, dict(path=str(png), sha256=digest, roi_sha256=signature,
                     archived_alpha8_median=float(np.median(alpha[part])))


def nearest_faces(vertices, faces, K, roi):
    """Compute nearest ray/mesh hits via perspective-correct screen barycentrics."""
    camera = vertices @ np.asarray(K, np.float64).T
    xy = camera[:, :2]/camera[:, 2:] - .5
    y, x = np.nonzero(roi)
    triangles = xy[faces]
    lo, hi = triangles.min(axis=1), triangles.max(axis=1)
    indices = np.flatnonzero((hi >= [x.min(), y.min()]).all(axis=1)
                            & (lo <= [x.max(), y.max()]).all(axis=1)
                            & (camera[faces, 2] > .2).all(axis=1))
    a, b, c = triangles[indices].transpose(1, 0, 2)
    ab, ac = b-a, c-a
    determinant = ab[:, 0]*ac[:, 1] - ab[:, 1]*ac[:, 0]
    require((np.abs(determinant) > 1e-9).any(), 'No finite projected triangle supports this crop.')
    qx, qy = x[:, None]-a[:, 0], y[:, None]-a[:, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        wb = (qx*ac[:, 1] - qy*ac[:, 0])/determinant
        wc = (ab[:, 0]*qy - ab[:, 1]*qx)/determinant
        wa = 1-wb-wc
        za, zb, zc = camera[faces[indices], 2].T
        inverse = wa/za + wb/zb + wc/zc
    inside = (wa >= -1e-8) & (wb >= -1e-8) & (wc >= -1e-8) & (np.abs(determinant) > 1e-9)
    nearest = np.argmax(np.where(inside, inverse, -np.inf), axis=1)
    require(inside.any(axis=1).all(), 'Some fixed ROI pixels have no mesh support.')
    return np.unique(indices[nearest], return_counts=True)


def preflight():
    pin(RUN/'result.json', 'e3e5738bdaaaad0e666b5dc63f0feee559a43f804e0abfa3785e8cf38d70d0af')
    result = read(RUN/'result.json')
    snapshot = RUN/'gaussian-run'/result['final_snapshot']
    replay_path = RUN/result['bound_replay']
    pin(replay_path, result['bound_replay_sha256'])
    replay = read(replay_path)
    pin(snapshot/'bound-replay.json', replay['bound_replay_sha256'])
    slot, replay_row = next((i, row) for i, row in enumerate(replay['records']) if row['frame'] == FRAME)
    rows_path = RUN/'data/training-inputs.json'
    pin(rows_path, replay['training_inputs_sha256'])
    row = read(rows_path)['records'][slot]
    require(slot == replay_row['pose_index'] == 18 and row['source_frame_index'] == 457
            and row['w2c'] == np.eye(4).tolist()
            and all(row[k] == replay_row['camera'][k] for k in ('wh', 'K', 'w2c', 'frame')),
            'The frozen native frame identity or camera changed.')
    pin(Path(row['image']), row['source_sha256'])
    pin(Path(row['mask']), '5aed7cd4a5fdcff95d0b8535ca317fabde494edc38f2fc9d3f45a7471029375d')
    pin(replay_path.parent/replay_row['image'], replay_row['image_sha256'])
    mesh_path = replay_path.parent/replay['mesh_file']
    pin(mesh_path, '73c1945a4f04a3bb928c7d542a454a827b234634d9c030843ed18462ca542928')
    roi, old = archived_roi()
    with Image.open(row['mask']) as im, Image.open(row['image']) as rgb:
        matte = np.asarray(im.convert('L'))
        source = np.asarray(rgb.convert('RGB'))/np.float32(255)
    require((matte[roi] >= 253).all(), 'The historical component leaves the current opaque training mask.')
    with np.load(mesh_path, allow_pickle=False) as mesh:
        require(mesh['frame_ids'][slot] == FRAME, 'Frozen posed mesh frame differs.')
        owners, count = nearest_faces(mesh['vertices_world'][slot], mesh['faces'], row['K'], roi)
    require(owners.tolist() == OWNERS and count.tolist() == HITS, 'Fixed ROI ray support differs.')
    plan = dict(state='frozen_native_control_preflight', frame=FRAME, pose_index=slot,
                source_frame_index=457, source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'],
                mesh_sha256=sha(mesh_path), roi_pixels=int(roi.sum()), ray_hits_per_triangle=dict(zip(OWNERS, HITS)),
                historical_roi=old, camera=replay_row['camera'], crop_xyxy=CROP,
                temporary_opacity_ceiling=.999, scale=1., optimizer_updates=0, refits=0,
                quality_accepted=False, script_sha256=sha(Path(__file__)))
    return plan, snapshot, row, replay_path.parent/replay_row['image'], roi, matte, source


def render(plan, snapshot, row, reference, roi, matte, source):
    import torch
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    require(not OUT.exists(), 'Preserve any existing hem replay.')
    rt.configure(ASSET.parents[1])
    from gaussian_renderer import render as raster

    with GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            model, _ = load_bound_snapshot(RUN, snapshot, asset_path=ASSET)
            pose = plan['pose_index']
            cam, camera_receipt = camera(row, pose, pose_index=pose)
            require(camera_receipt == plan['camera'], 'Current replay camera differs.')
            model.select_mesh_by_timestep(pose)
            owned = torch.isin(model.binding, torch.as_tensor(OWNERS, device='cuda'))
            require(int(owned.sum()) == 340 and model.optimizer is None, 'Triangle Gaussian ownership changed.')
            original = model._opacity.detach().clone()
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            def pixels():
                return scale_pixels(raster, render_pose, cam, model, pipe, black, white, pose, 1.)[0]
            before = pixels()
            uint8 = lambda a: np.rint(np.clip(a, 0, 1)*255).astype(np.uint8)
            with Image.open(reference) as im:
                delta = np.abs(uint8(before[0]).astype(np.int16)-np.asarray(im.convert('RGB')))
            require(int(delta.max()) <= 1, 'Stock frame differs from saved posed replay.')
            arms = {'stock': before}
            for name, group in (('nearest_triangle_owners', owned), ('all_existing_gaussians_upper_bound', torch.ones_like(owned))):
                try:
                    model._opacity[group] = original[group].clamp_min(math.log(999.))
                    require(torch.equal(model._opacity[~group], original[~group]), 'Opacity outside temporary group changed.')
                    arms[name] = pixels()
                finally:
                    model._opacity.copy_(original)
            require(torch.equal(model._opacity, original) and model.optimizer is None
                    and all(np.array_equal(a, b) for a, b in zip(before, pixels(), strict=True)), 'Stock frame did not restore exactly.')
    scores = {key: dict(alpha=face_alpha(v[2], roi), median_alpha=float(np.median(v[2][roi])),
                        black_rgb_mae=float(np.abs(v[0][roi]-source[roi]).mean()),
                        exterior_half=int(((matte < 3) & (v[2] >= .5)).sum())) for key, v in arms.items()}
    OUT.mkdir()
    x0, y0, x1, y1 = CROP
    w, h = x1-x0, y1-y0
    sheet = Image.new('RGB', (4*w, 2*h))
    sheet.paste(Image.fromarray(uint8(source[y0:y1, x0:x1])), (0, 0))
    sheet.paste(Image.fromarray((roi[y0:y1, x0:x1]*255).astype(np.uint8)).convert('RGB'), (0, h))
    for i, (rgb, _, alpha) in enumerate(arms.values(), 1):
        sheet.paste(Image.fromarray(uint8(rgb[y0:y1, x0:x1])), (i*w, 0))
        sheet.paste(Image.fromarray(uint8(alpha[y0:y1, x0:x1])).convert('RGB'), (i*w, h))
    panel = OUT/'native-rgb-alpha.png'
    sheet.save(panel)
    report = OUT/'result.json'
    save(report, dict(plan, state='read_only_replay_completed', scores=scores, changed_saved_parameters=False,
                      restored_exact=True, owned_gaussians=340, panel_sha256=sha(panel)))
    print(json.dumps(dict(result=str(report), sha256=sha(report), scores=scores)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    args = parser.parse_args()
    inputs = preflight()
    if args.action == 'check':
        print(json.dumps(dict(inputs[0], gpu_calls=0)))
    else:
        render(*inputs)
