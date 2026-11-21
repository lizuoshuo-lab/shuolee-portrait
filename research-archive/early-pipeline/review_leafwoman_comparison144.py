"""Render pinned same-frame controls for the independent 022 fits."""
import argparse
import ast

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from official_smplifyx_entry123 import sha
from probe_leafwoman_ear139 import annotate_image, crop_visible_faces, masks
from review_full_sequence123 import joined, mesh_renderer, relocate
from run_leafwoman_gvhmr143 import ARCHIVE, CONTACTS, OFFICIAL
from run_leafwoman_gvhmr_visual144 import OUT as VISUAL
from run_leafwoman_smplify_independent144 import OUT
from update_leafwoman129_state import atomic, read


def body_indices():
    path = OFFICIAL/'smplifyx/utils.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if getattr(n, 'name', None) == 'smpl_to_openpose')
    scope = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope['smpl_to_openpose'](use_face_contour=False)[2:8]


def metrics(values):
    return dict(count=len(values), p50_px=float(np.median(values)),
                p90_px=float(np.quantile(values, .9)), max_px=float(np.max(values)))


def ear_comparison(meshes, faces, K, target, frame='frame_000033'):
    ear_dir = OUT.with_name('ear-region139')
    plan = read(ear_dir/'plan.json')
    plan_sha = sha(ear_dir/'plan.json')
    item = next(r for r in plan['frames'] if r['frame'] == frame)
    assert sha(ARCHIVE) == plan['archive_sha256']
    src = relocate(item['original_crop'])
    assert sha(src) == item['crop_sha256']
    with Image.open(src) as im:
        raw = im.convert('RGB')
    origin = np.asarray(item['crop_xyxy'][:2])
    ear_faces = np.isin(faces, masks(plan)[item['side']]).all(axis=1)
    vertex = {'right': 616, 'left': 6}[item['side']]
    polygons, regions, uncertainty = [], [], []
    for label in ('a', 'b'):
        labels = read(ear_dir/f'source-only-annotation-{label}.json')
        assert labels['plan_sha256'] == plan_sha
        row = next(r for r in labels['rows'] if r['frame'] == item['frame'])
        poly = np.asarray(row['polygon_xy'], dtype=np.int32)
        raster = np.zeros((raw.height, raw.width), np.uint8)
        cv2.fillPoly(raster, [poly], 1)
        polygons.append(poly)
        regions.append(raster.astype(bool))
        uncertainty.append(row['uncertainty_px'])
    images, stats = [raw], []
    for v in meshes:
        p = v[0].astype(float)@K.T
        uv = p[:, :2]/p[:, 2:]-origin
        owner = crop_visible_faces(uv, p[:, 2], faces, raw.size)
        visible = (owner >= 0) & ear_faces[np.maximum(owner, 0)]
        images.append(annotate_image(raw, polygons, visible, target-origin, uv[vertex]))
        if not visible.any():
            stats.append(dict(patch_visible=False, coverage=[0., 0.], p90_px=None))
            continue
        distance = cv2.distanceTransform((~visible).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        stats.append(dict(patch_visible=True, coverage=[float(np.mean(distance[r] <= tol)) for r, tol in zip(regions, uncertainty)],
                          p90_px=[float(np.quantile(distance[r], .9)) for r in regions]))
    frozen = next(r for r in read(ear_dir/'result.json')['rows'] if r['frame'] == item['frame'])
    assert np.allclose(stats[0]['coverage'], [r['toleranced_source_region_coverage'] for r in frozen['readings']])
    return joined(images), dict(frame=item['frame'], model_self_occlusion_only=True, source_label_uncertainty_px=uncertainty,
                                methods=stats, source_plan_sha256=plan_sha)


def side_comparison(renderer, meshes, faces):
    wh = (300, 420)
    K = np.array([[900., 0, 150], [0, 900., 210], [0, 0, 1.]])
    blank = Image.new('RGB', wh)
    union = np.concatenate(meshes, axis=1)
    center = (union.min(axis=1)+union.max(axis=1))/2
    z_offset = 3*float(np.ptp(union, axis=1).max())
    panels = []
    for angle in (90., -90.):
        R = Rotation.from_euler('y', angle, degrees=True).as_matrix()
        for v in meshes:
            camera = (v-center[:, None])@R.T
            camera[..., 2] += z_offset
            panels.append(renderer(blank, np.array([0, 0, *wh]), dict(vertices_camera=camera, faces=faces, K_integer_centers=K)))
    return joined(panels)


def review(fit_dir):
    contacts = {r['frame']: r for r in read(CONTACTS)['records']}
    witness = read(OUT/'visible-cuff-source-only.json')
    assert witness['source_contacts_sha256'] == sha(CONTACTS)
    result = read(OUT/'result.json')
    assert result['successful'] and len(result['frames']) == 3
    assert read(OUT/'plan.json')['observation_archive_sha256'] == sha(ARCHIVE)
    renderer, renderer_path = mesh_renderer()
    board = Image.new('RGB', (1080, 1590))
    measures = []
    arm_errors = {}
    suffix = 'smplify-independent' if fit_dir is None else f'gvhmr-{fit_dir.name}'
    gvhmr = None
    ear, side = None, None
    if fit_dir is not None:
        saved = read(fit_dir/'result.json')
        archive = fit_dir/'prediction-incam.npz'
        assert saved['prediction_sha256'] == sha(archive) and saved['real_prediction']
        with np.load(archive, allow_pickle=False) as arrays:
            gvhmr = dict(arrays)
        mapped = body_indices()
    with np.load(ARCHIVE, allow_pickle=False) as old:
        names = old['frame_ids'].tolist()
        K, faces = old['K_integer_centers'], old['faces']
        for ordinal, item in enumerate(result['frames']):
            name = item['frame']
            source = contacts[name]
            labels = witness['data'][name]
            assert sha(relocate(source['paths']['upper']['path'])) == labels['source_upper_sha256']
            entry = source['paths']['body']
            image = relocate(entry['path'])
            assert sha(image) == entry['sha256'] and sha(item['output']) == item['sha256']
            box = np.asarray(entry['crop_original_xyxy'])
            with Image.open(image) as im:
                raw = im.convert('RGB')
            assert raw.size == tuple(box[2:]-box[:2])
            i = names.index(name)
            target = old['observed_native_keypoints'][i, 2:8]
            assert (target[:, 2] > 0).all()
            meshes = [old['vertices_camera'][i:i+1]]
            pixels = {'shared_smplify': old['joints_projected_native'][i, 2:8]}
            with np.load(item['output'], allow_pickle=False) as fresh:
                assert np.array_equal(fresh['observed_native_keypoints'][0, 2:8], target)
                assert np.array_equal(fresh['K_integer_centers'], K) and np.array_equal(fresh['faces'], faces)
                meshes.append(fresh['vertices_camera'])
                pixels['independent_smplify'] = fresh['joints_projected_native'][0, 2:8]
            if gvhmr is not None:
                assert np.array_equal(gvhmr['K_integer_centers'], K) and np.array_equal(gvhmr['faces'], faces)
                j = gvhmr['frame_ids'].tolist().index(name)
                meshes.append(gvhmr['vertices_camera'][j:j+1])
                q = gvhmr['joints_camera'][j, mapped]@K.T
                assert (q[:, 2] > 0).all()
                pixels['gvhmr'] = q[:, :2]/q[:, 2:]
            errors = {method: np.linalg.norm(xy-target[:, :2], axis=1).tolist() for method, xy in pixels.items()}
            for method, e in errors.items():
                arm_errors.setdefault(method, []).extend(e)
            cuff = raw.copy()
            draw = ImageDraw.Draw(cuff)
            for key, color in (('screen_left', '#2af2f0'), ('screen_right', '#ff75d7')):
                p = np.asarray(labels[key]).reshape(2, 2)-box[:2]
                draw.rectangle(tuple(p.ravel()), outline=color, width=3)
            panels = [raw, *[renderer(raw, box, dict(vertices_camera=v, faces=faces, K_integer_centers=K))
                            for v in meshes], cuff]
            out = OUT/f'{name}-{suffix}.png'
            composed = joined(panels)
            composed.save(out)
            composed.thumbnail((1070, 520), Image.Resampling.LANCZOS)
            board.paste(composed, ((1080-composed.width)//2, ordinal*530))
            measures.append(dict(frame=name, arm_errors_px=errors, predicted_wrist_pixels={
                key: value[[2, 5]].tolist() for key, value in pixels.items()},
                image=out.name, image_sha256=sha(out)))
            if name == 'frame_000033':
                panel, ear = ear_comparison(meshes, faces, K, old['observed_native_keypoints'][i, 17, :2])
                file = OUT/f'{name}-{suffix}-ear.png'
                panel.save(file)
                ear.update(image=file.name, image_sha256=sha(file))
            if name == 'frame_000020':
                panel = side_comparison(renderer, meshes, faces)
                file = OUT/f'{name}-{suffix}-side.png'
                panel.save(file)
                side = dict(frame=name, image=file.name, image_sha256=sha(file),
                            camera_is_synthetic=True, same_camera_across_methods=True)
    board.save(OUT/f'{suffix}-preview.jpg', quality=90)
    totals = {method: metrics(values) for method, values in arm_errors.items()}
    atomic(OUT/f'{suffix}-native-review.json', dict(fit_result_sha256=sha(OUT/'result.json'),
        gvhmr_result_sha256=None if fit_dir is None else sha(fit_dir/'result.json'),
        renderer_sha256=sha(renderer_path), independent_cuff_witness_sha256=sha(OUT/'visible-cuff-source-only.json'),
        frames=measures, metrics=totals, ear=ear, side=side, native_pixels=True, metrics_from_detector=True,
        visual_review_pending=True, quality_accepted=False))
    print(totals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving', action='store_true')
    parser.add_argument('--visual', action='store_true')
    args = parser.parse_args()
    assert not (args.visual and args.moving)
    review(VISUAL/'hmr2-vitpose-moving' if args.moving else VISUAL/'hmr2-vitpose' if args.visual else None)
