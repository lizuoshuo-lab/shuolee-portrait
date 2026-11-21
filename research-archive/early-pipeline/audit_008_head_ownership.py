"""Attribute frozen 008 SH0 pixels to anatomical binding groups without changing opacity."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw


LAB = Path('/opt/gs-slam-lab')
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-source-visibility')
RUN = LAB/'shuolee-smpl-visibility-008/run'
EVAL = LAB/'shuolee-smpl-visibility-008-formal'
OUT = LAB/'shuolee-smpl-head-ownership-008'
SNAP = RUN/'gaussian-run/step-010000'
ASSET = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
RASTER_ENV = LAB/'shuolee-portrait-clean-001/env/lib/python3.10/site-packages/diff_gaussian_rasterization'
RASTER_BINARY = RASTER_ENV/'_C.cpython-310-x86_64-linux-gnu.so'
ARCHIVE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
MODELS = ARCHIVE/'shuolee/current/novel-view-face/head-structure/private-runtime/models'
FLAME = MODELS/'flame-masks/FLAME_masks.pkl'
MAP = MODELS/'smplx/correspondences/SMPL-X__FLAME_vertex_ids.npy'
HISTORICAL = (ARCHIVE/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'
              /'indoor-automatic-sh0-254-formal-render262/result.json')
COMMIT = 'd080021d3c9d17a986e9fb60a93a0a8243c4a210'
GROUPS = ('face_interior', 'mixed_face_boundary', 'other_head_neck', 'nonhead')
# Adjacent visible wall, kept separate from the confirmed occiput and narrow beard profile.
HELD_BACKGROUND = ((1468, 417), (1486, 417), (1486, 543), (1468, 543))
SPECS = {
    'frame_000004': ('train_face_positive', 24, 0, (1300, 475, 1510, 755),
                     ((1370, 570), (1463, 570), (1467, 642), (1448, 691), (1411, 707), (1375, 677))),
    'frame_000047': ('train_occiput_control', 368, 7, (1265, 375, 1485, 655),
                     ((1322, 428), (1364, 413), (1410, 435), (1420, 466),
                      (1401, 537), (1362, 551), (1327, 526), (1307, 480))),
    'frame_000043': ('heldout_occiput_evaluation_only', 336, 20, (1260, 350, 1505, 644),
                     ((1326, 402), (1364, 388), (1411, 411), (1418, 453),
                      (1394, 525), (1350, 546), (1317, 520), (1304, 465))),
}
FIXED_PINS = {
    SNAP/'bound-replay.json': 'c2f445f18b57253d24227bbcd14a1eae6d6095ea9c2378d17b10dd39cf7f8357',
    SNAP/'bound-local.ply': 'dc601f1ba8214b1d4909ade7aacb24440810267d60385ce2215bf511e09c53c9',
    SNAP/'posed-replay/result.json': '2bd276191b9d9f1302f741c38f90a09f0f4ed829a72af02d5f4192ca1abd5642',
    RUN/'data/inputs.json': 'a0e101fb4bb546f35b1638d8786f3fde3d4a9ea9d73f797886423940abe98147',
    EVAL/'evaluation-scope.json': 'c081d2a717ddc27e3326e7f72500811bad6b0d16f67e4bca211217cbb3382f44',
    EVAL/'fit-result.json': 'b7d48c7dc64b3be5b3679fb44b1ce107f1fbba9f438baeb1d7da3b9cdcd276f4',
    EVAL/'evaluation-fitted.npz': 'ac20914babf47f05d8e8e20bbf0c94f2e8891a5b8b35468974c096366047b642',
    EVAL/'renders/result.json': '8ae2cddf5b9dee8f9f2a58b826ccc669a70bd9302ec819704dddb4d56c6f519b',
    HISTORICAL: '5488b7ef74f35aac82a7af0c3d06ecb2665ccfa5b57777b05a0865885d9147b9',
    FLAME: 'ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492',
    MAP: '7e70cdc3659aae699b9732e8dd4af49106310c69b90dc83d9f73e96dbf871e49',
    RASTER_BINARY: '5161c09abd032639e582230874df6042a4bdd2b62f5137733386520c2a05e113',
    RASTER_ENV/'__init__.py': 'eafa9c3f670258b34535b7ee2832e6d8410d6a68762037216f04935ccdafba34',
    PROJECT/'src/avatar_core/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h':
        'f529a2ba6ca075793da65c3fc4deb2b3a4aacc7f1618feac658884dd608ba602',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(pins):
    for path, expected in pins.items():
        need(sha(Path(path)) == expected, f'Changed input: {path}')


def save(path, value):
    with path.open('x', encoding='utf-8') as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')


class MaskReader(pickle.Unpickler):
    def find_class(self, module, name):
        return {('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
                ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}[module, name]


def semantics(faces):
    with np.load(ASSET, allow_pickle=False) as asset:
        need(faces.shape == (20908, 3) and np.array_equal(faces, asset['f']), 'Changed SMPL-X topology.')
        weights = asset['weights']
        static = asset['lmk_faces_idx'].ravel()
        dynamic = asset['dynamic_lmk_faces_idx'].ravel()
    need(weights.shape == (10475, 55) and np.isfinite(weights).all(), 'Bad model skinning table.')
    for ids in (static, dynamic):
        need(ids.size > 0 and ids.dtype.kind in 'iu' and ids.min() >= 0 and ids.max() < len(faces),
             'Invalid anatomical landmark triangles.')
    mapping = np.load(MAP, allow_pickle=False)
    need(mapping.shape == (5023,) and mapping.dtype.kind in 'iu' and len(np.unique(mapping)) == len(mapping)
         and mapping.min() >= 0 and mapping.max() < len(weights), 'Bad FLAME-to-SMPL-X mapping.')
    with FLAME.open('rb') as source:
        masks = MaskReader(source, encoding='latin1').load()
    regions = {}
    for name, count in {'face': 1787, 'scalp': 489, 'neck': 253, 'left_ear': 588, 'right_ear': 588}.items():
        ids = masks[name]
        need(ids.shape == (count,) and ids.dtype.kind in 'iu' and len(np.unique(ids)) == count
             and ids.min() >= 0 and ids.max() < len(mapping), f'Invalid FLAME semantic group {name}.')
        regions[name] = mapping[ids]
    need(6 in regions['left_ear'] and 616 in regions['right_ear'], 'Expected SMPL-X ear landmarks absent.')
    # Label an anatomical triangle once. Mixed facial edges cannot count as definite face.
    fv = np.isin(faces, regions['face']).sum(axis=1)
    protected = np.isin(weights.argmax(axis=1), [12, 15, 22, 23, 24])
    protected[faces[np.concatenate([static, dynamic])].ravel()] = True
    semantic_head = np.concatenate([ids for name, ids in regions.items() if name != 'face'])
    other_head = (protected[faces] | np.isin(faces, semantic_head)).any(axis=1)
    labels = np.select([fv == 3, fv > 0, other_head], [0, 1, 2], default=3)
    inner_covered = int(np.count_nonzero(labels[static] == 0))
    need(inner_covered == len(static), f'Face mask covers only {inner_covered}/{len(static)} inner landmarks.')
    return labels, dict(vertices_by_flame_region={k: len(v) for k,v in regions.items()},
        triangles_by_group=dict(zip(GROUPS, np.bincount(labels, minlength=4).tolist(), strict=True)),
        static_face_landmark_triangles=len(static), static_in_face_interior=inner_covered,
        dynamic_contour_triangles=len(np.unique(dynamic)))


def one(rows, frame):
    [row] = [row for row in rows if row['frame'] == frame]
    return row


def polygon(crop, points):
    x0,y0,x1,y1 = crop
    need(all(x0 <= x < x1 and y0 <= y < y1 for x,y in points), 'Source annotation outside crop.')
    mask = Image.new('1', (x1-x0, y1-y0))
    ImageDraw.Draw(mask).polygon([(x-x0,y-y0) for x,y in points], fill=1)
    return np.asarray(mask)


def preflight():
    pointer = (PROJECT/'.git').read_text(encoding='utf-8').strip()
    need(pointer.startswith('gitdir: '), 'Expected the pinned detached Windows worktree.')
    gitdir = subprocess.check_output(['wslpath', '-u', pointer.removeprefix('gitdir: ')], text=True).strip()
    head = subprocess.check_output(['git','--git-dir',gitdir,'rev-parse','HEAD'], text=True).strip()
    need(head == COMMIT, f'Expected stock checkout {COMMIT}, got {head}.')
    pins = dict(FIXED_PINS)
    verify(pins)
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait.bound_replay import inspect_bound_ply, _inputs

    manifest = read(SNAP/'bound-replay.json')
    fit, mesh, _ = _inputs(RUN, SNAP, ASSET, manifest)
    ids, _ = inspect_bound_ply(SNAP/'bound-local.ply', len(fit['faces']))
    labels, anatomical = semantics(fit['faces'])
    need(len(ids) == manifest['points'] == 58253 and manifest['sh_degree'] == 0,
         'Unexpected bound Gaussian point count or SH degree.')
    anatomical['gaussians_by_group'] = dict(zip(GROUPS, np.bincount(labels[ids], minlength=4).tolist(), strict=True))
    need(all(anatomical['gaussians_by_group'].values()), 'An anatomical group is empty.')
    pins.update({PROJECT/name: value for name,value in manifest['source_files_sha256'].items()})
    pins.update({ASSET: manifest['asset_sha256'], RUN/'fit/fitted.npz': manifest['inputs_sha256']['fitted'],
        RUN/'data/training-inputs.json': manifest['inputs_sha256']['training_rows'],
        SNAP/'reference-mesh.npz': manifest['inputs_sha256']['reference_mesh'],
        SNAP/'world.ply': manifest['inputs_sha256']['world_ply'],
        SNAP/'native-ply-parity.json': manifest['inputs_sha256']['native_ply_parity']})

    scope = read(EVAL/'evaluation-scope.json')
    fit_result = read(EVAL/'fit-result.json')
    train_receipt = read(SNAP/'posed-replay/result.json')
    render_receipt = read(EVAL/'renders/result.json')
    need(fit_result['evaluation_only'] is True
         and scope['protected_frames'] == fit_result['protected_frames'] == render_receipt['protected_frames']
         and scope['fitted_sha256'] == fit_result['fitted_sha256'] == pins[EVAL/'evaluation-fitted.npz']
         and set(scope['protected_frames']).isdisjoint(fit['frame_ids'].astype(str))
         and set(scope['protected_frames']) == set(train_receipt['retained_holdouts'])
         and render_receipt['fit_result_sha256'] == pins[EVAL/'fit-result.json'],
         'Evaluation provenance or disjointness failed.')
    with np.load(EVAL/'evaluation-fitted.npz', allow_pickle=False) as z:
        poses = dict(z)
    need(poses['frame_ids'].tolist() == scope['evaluated_frames'] and np.array_equal(poses['faces'], fit['faces'])
         and np.array_equal(poses['shared_betas'], fit['shared_betas']), 'Changed evaluation pose topology or shared body.')
    training_rows = read(RUN/'data/training-inputs.json')['records']
    input_rows = read(RUN/'data/inputs.json')['records']
    views = []
    for frame, (role, index, pose, crop, region) in SPECS.items():
        held = role.startswith('heldout')
        row = one(input_rows if held else training_rows, frame)
        saved = one(render_receipt['records'] if held else train_receipt['records'], frame)
        records_dir = EVAL/'renders' if held else SNAP/'posed-replay'
        expected = saved['posed_camera' if held else 'camera']
        need(row['source_frame_index'] == saved['source_frame_index'] == index
             and expected['pose_index'] == pose and row['wh'] == [2704,1520]
             and all(row[key] == expected[key] for key in ('K', 'w2c', 'wh')), f'Changed camera or source: {frame}')
        if held:
            slot = poses['frame_ids'].tolist().index(frame)
            need(pose == len(manifest['frames'])+slot and scope['protected_frames'][frame] == index
                 and row['source_sha256'] == saved['native_source_sha256'], 'Changed independent evaluation frame.')
            ox,oy,ex,ey = one(read(HISTORICAL)['records'], frame)['crop_xyxy']
            native_crop = (crop[0]-ox,crop[1]-oy,crop[2]-ox,crop[3]-oy)
            need(0 <= native_crop[0] < native_crop[2] <= ex-ox and
                 0 <= native_crop[1] < native_crop[3] <= ey-oy, 'ROI escaped the formal crop.')
            recorded_crop = [native_crop[0]+ex-ox,native_crop[1],native_crop[2]+ex-ox,native_crop[3]]
        else:
            slot = fit['frame_ids'].tolist().index(frame)
            bound = one(manifest['frames'], frame)
            need(pose == slot == bound['pose_index'] and row['source_sha256'] == bound['source_sha256']
                 == saved['source_sha256'], 'Changed training frame, source or pose.')
            recorded_crop = list(crop)
        panel = records_dir/saved['image']
        source = Path(row['image'])
        pins.update({source: row['source_sha256'], panel: saved['image_sha256']})
        with Image.open(source) as im, Image.open(panel) as picture:
            need(list(im.size) == row['wh'] and polygon(crop, region).sum() > 1000, 'Bad source-only region.')
            if held:
                need(picture.size == (3*(ex-ox),ey-oy) and
                     np.array_equal(im.convert('RGB').crop(crop), picture.convert('RGB').crop(native_crop)),
                     'Formal panel source crop does not match the exact native original.')
            else:
                need(picture.size == im.size, 'Archived named replay is not native size.')
        background = HELD_BACKGROUND if held else None
        views.append(SimpleNamespace(row=row, role=role, source=source, panel=panel, panel_crop=recorded_crop,
            expected=expected, crop=crop, region=region, background=background, pose=pose, slot=slot))
    pins[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    verify(pins)
    return pins, anatomical, views, poses, mesh, labels[ids]


def source_only(view):
    with Image.open(view.source) as original:
        source = original.convert('RGB').crop(view.crop)
    shown = source.copy()
    x0,y0,_,_ = view.crop
    points = [(x-x0,y-y0) for x,y in view.region]
    draw = ImageDraw.Draw(shown)
    draw.line([*points, points[0]], fill='#00f9fa', width=2)
    if view.background:
        background = [(x-x0,y-y0) for x,y in view.background]
        draw.line([*background, background[0]], fill='#e9a738', width=2)
    return source, shown


def plan(pins, anatomical, views):
    return dict(schema='shuolee-008-frozen-head-ownership-v1', commit=COMMIT, points=58253, step=10000,
        gpu_calls_at_registration=0, max_render_calls=30, fit_calls=0, optimizer_updates=0,
        quality_accepted=False, training_decisions_from_holdout=False, anatomical=anatomical,
        semantics='Strict FLAME face interior, mixed face boundary, remaining head/neck and nonhead.',
        region_policy='Native source-only face and occiput. Exclude beard on 043. Amber wall strip is separate.',
        method='All original splats and opacity/depth retained. Binary and true SH0 group color overrides.',
        pins={str(path): digest for path,digest in pins.items()},
        views=[dict(frame=v.row['frame'], role=v.role, source=str(v.source), source_index=v.row['source_frame_index'],
            pose=v.pose, crop_xyxy=list(v.crop), polygon_xy=[list(p) for p in v.region], camera=v.expected,
            background_xy=[list(p) for p in v.background] if v.background else None,
            archived_panel=str(v.panel), archived_crop_xyxy=v.panel_crop) for v in views])


def prepare(pins, anatomical, views):
    need(not OUT.exists(), 'Preserve the existing preregistration.')
    OUT.mkdir()
    source_pins = {}
    for view in views:
        path = OUT/(view.row['frame']+'-source-only.png')
        source_only(view)[1].save(path)
        source_pins[str(path)] = sha(path)
    record = plan(pins, anatomical, views)
    save(OUT/'preregistration.json', dict(record, source_annotation_sha256=source_pins))
    print(json.dumps(dict(path=str(OUT/'preregistration.json'), anatomical=anatomical, gpu_calls=0)))


def panel(path, view, rgb, alpha, parts):
    source, outlined = source_only(view)
    w,h = source.size
    sheet = Image.new('RGB', (6*w, 2*h+42), '#171717')
    draw = ImageDraw.Draw(sheet)
    for i,(label, upper, lower) in enumerate(zip(
            ('Source', 'Full stock', 'Face interior', 'Mixed facial edge', 'Other head/neck', 'Nonhead'),
            (source, rgb, *(p[0] for p in parts)), (outlined, alpha, *(p[1] for p in parts)), strict=True)):
        draw.text((i*w+4,5), label, fill='white')
        draw.text((i*w+4,h+26), 'Source-only ROI' if i == 0 else 'Transmittance-weighted alpha', fill='white')
        sheet.paste(upper, (i*w,21))
        sheet.paste(lower, (i*w,h+42))
    sheet.save(path)


def render_one(view, model, group_ids, directory):
    import torch
    from gaussian_renderer import render
    from shuolee_portrait.training import camera, image, render_pose

    cam, actual = camera(view.row, view.slot, pose_index=view.pose)
    need(actual == view.expected, f'Exact archived native camera mismatch: {view.row["frame"]}')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device='cuda')

    def draw(**kwargs):
        return render_pose(render, cam, model, pipe, black, view.pose, **kwargs)['render']

    x0,y0,x1,y1 = view.crop
    area = (..., slice(y0,y1), slice(x0,x1))
    roi = polygon(view.crop, view.region)
    background = polygon(view.crop, view.background) if view.background else None
    dc = (model.get_features[:,0,:]*0.28209479177387814+0.5).clamp_min(0).contiguous()
    stock = draw()
    stock_alpha = draw(override_color=torch.ones_like(dc))[0]
    sum_color, sum_alpha = torch.zeros_like(stock), torch.zeros_like(stock_alpha)
    rgb = image(stock[area])
    with Image.open(view.panel) as archived:
        reference = np.asarray(archived.convert('RGB').crop(view.panel_crop), dtype=np.int16)
    byte_error = int(np.abs(np.asarray(rgb, dtype=np.int16)-reference).max())
    need(byte_error <= 1, f'Archived posed RGB differs by {byte_error} 8-bit levels.')
    part_images, metrics = [], {}
    roi_total = float(stock_alpha[area].cpu().numpy()[roi].sum())
    background_total = float(stock_alpha[area].cpu().numpy()[background].sum()) if background is not None else None
    for group, name in enumerate(GROUPS):
        indicator = (group_ids == group)[:,None].to(dc.dtype)
        color = draw(override_color=(indicator*dc).contiguous())
        alpha = draw(override_color=indicator.expand_as(dc).contiguous())[0]
        sum_color += color
        sum_alpha += alpha
        alpha_crop = alpha[area].cpu().numpy()
        contribution = float(alpha_crop[roi].sum())
        color_mass = color[area].cpu().numpy()[:,roi].sum(axis=1, dtype=np.float64)
        part_images.append((image(color[area]), image(alpha[area].expand(3,-1,-1))))
        metrics[name] = dict(roi_alpha_mass=contribution, fraction_of_roi_alpha=contribution/max(roi_total,1e-10),
                             background_alpha_mass=float(alpha_crop[background].sum()) if background is not None else None,
                             roi_premultiplied_rgb_sum=color_mass.tolist())
    alpha_error = float((sum_alpha-stock_alpha).abs().max())
    color_error = float((sum_color-stock).abs().max())
    need(alpha_error < 2e-5 and color_error < 2e-5, 'Contribution buffers do not recover stock compositing.')
    target = directory/(view.row['frame']+'-head-ownership.png')
    panel(target, view, rgb, image(stock_alpha[area].expand(3,-1,-1)), part_images)
    return dict(frame=view.row['frame'], role=view.role, roi_pixels=int(roi.sum()), roi_alpha_mass=roi_total,
        background_pixels=int(background.sum()) if background is not None else None,
        background_alpha_mass=background_total,
        full_frame_alpha_additivity_max=alpha_error, full_frame_rgb_additivity_max=color_error,
        archived_crop_rgb_byte_max=byte_error, contribution=metrics, image=target.name, image_sha256=sha(target))


def replay(pins, anatomical, views, poses, mesh, labels):
    record = read(OUT/'preregistration.json')
    source_pins = record.pop('source_annotation_sha256')
    need(record == plan(pins, anatomical, views), 'Preregistered source, renderer or experiment changed.')
    verify(source_pins)
    need(not (OUT/'render').exists(), 'Preserve the first complete rendering.')
    lease_path = Path('/tmp/shuolee-native122-gpu.lock')
    with lease_path.open('r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from shuolee_portrait import runtime as rt
        from shuolee_portrait.bound_replay import load_bound_snapshot
        from shuolee_portrait.cli import gpu_lock
        from shuolee_portrait.training import _frozen_smpl_surface

        with gpu_lock(), torch.no_grad(), TemporaryDirectory(prefix='.render-', dir=OUT) as scratch:
            need(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            rt.configure(ASSET.parent.parent)
            import diff_gaussian_rasterization as rasterizer
            from diff_gaussian_rasterization import _C
            need(Path(rasterizer.__file__).resolve() == (RASTER_ENV/'__init__.py').resolve()
                 and Path(_C.__file__).resolve() == RASTER_BINARY.resolve(), 'Unexpected rasterizer runtime path.')
            model, _ = load_bound_snapshot(RUN, SNAP, asset_path=ASSET)
            base, linear, clothing, _, _ = _frozen_smpl_surface(poses, ASSET)
            delta = mesh['static_offset']
            need(not np.any(delta[~clothing]), 'Shared offset crossed the pinned anatomical boundary.')
            offset = torch.from_numpy(delta).cuda()[..., None]
            sequence = torch.stack([v+(a@offset).squeeze(-1) for v,a in zip(base.cuda(),linear.cuda(),strict=True)])
            model.vertices_sequence = torch.cat((model.vertices_sequence, sequence))
            model.num_timesteps = len(model.vertices_sequence)
            group_ids = torch.from_numpy(labels).cuda()
            staging = Path(scratch)/'render'
            staging.mkdir()
            rows = [render_one(view, model, group_ids, staging) for view in views]
            need(model.optimizer is None, 'Read-only replay initialized an optimizer.')
            verify(pins)
            verify(source_pins)
            save(staging/'result.json', dict(state='complete_pending_visual_review', renders=30,
                fit_calls=0, optimizer_updates=0, model_changed=False, quality_accepted=False,
                preregistration_sha256=sha(OUT/'preregistration.json'), records=rows))
            staging.rename(OUT/'render')
    print(OUT/'render/result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'prepare', 'render'))
    mode = parser.parse_args().mode
    if mode != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    state = preflight()
    if mode == 'render':
        replay(*state)
    elif mode == 'prepare':
        prepare(*state[:3])
    else:
        print(json.dumps(dict(state='preflight_passed', anatomical=state[1], gpu_calls=0)))


if __name__ == '__main__':
    main()
