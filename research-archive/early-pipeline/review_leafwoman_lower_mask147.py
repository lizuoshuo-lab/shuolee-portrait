"""Review all completed lower-envelope controls in their native source camera."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from probe_leafwoman_lower_mask147 import ASSET, H, LOWER_JOINTS, OUT, SOURCE_REVIEW, W, distance, read_mask
from review_full_sequence123 import bound_file, joined, mesh_renderer
from review_leafwoman_comparison144 import ARCHIVE, ear_comparison, side_comparison
from update_leafwoman129_state import atomic, read


METHODS = ('shared_seed', 'original', 'containment')
SIDE = ('frame_000001', 'frame_000004', 'frame_000011', 'frame_000024', 'frame_000027', 'frame_000038')


def load(path, digest):
    assert sha(path) == digest
    with np.load(path, allow_pickle=False) as fit:
        return dict(fit)


def cover(uv, faces):
    canvas = np.zeros((H,W), np.uint8)
    polys = np.rint(uv[faces]*16).astype(np.int32)
    for tri in polys:
        cv2.fillConvexPoly(canvas, tri, 1, shift=4)
    return canvas.astype(bool)


def main():
    cv2.setNumThreads(1)
    plan = read(OUT/'plan.json')
    fit = read(OUT/'result.json')
    assert fit['complete'] and fit['plan_sha256'] == sha(OUT/'plan.json')
    names = [row['frame'] for row in plan['rows']]
    assert names == [row['frame'] for row in fit['frames']] and len(names) == 24
    assert not set(names) & set(plan['heldout_frames'])
    with np.load(ARCHIVE, allow_pickle=False) as old, np.load(ASSET, allow_pickle=False) as model:
        assert names == old['frame_ids'].tolist()
        legacy = old['vertices_camera'].copy()
        weights, faces = model['weights'], old['faces']
        K = old['K_integer_centers']
    lower = weights[:, LOWER_JOINTS].sum(axis=1) > .5
    foot = weights[:, [7,8,10,11]].sum(axis=1) > .5
    lower_faces = faces[lower[faces].sum(axis=1) >= 2]
    foot_faces = faces[foot[faces].sum(axis=1) >= 2]
    assert len(lower_faces) > 2000 and len(foot_faces) > 200
    renderer, renderer_path = mesh_renderer()
    ears = {row['frame']: row['side'] for row in read(OUT.parent.with_name('ear-region139')/'plan.json')['frames']}
    review = OUT/'review'
    review.mkdir()
    rows, sheets = [], []
    for i, (source, rec) in enumerate(zip(plan['rows'], fit['frames'], strict=True)):
        frame = source['frame']
        raw = Image.open(bound_file(source['source'])).convert('RGB')
        box = np.asarray(source['source']['crop_original_xyxy'])
        assert raw.size == tuple(box[2:]-box[:2])
        seed = load(source['baseline'], source['baseline_sha256'])
        variants = [seed, *[load(rec['variants'][k]['output'], rec['variants'][k]['sha256']) for k in METHODS[1:]]]
        mask = read_mask(source)
        safe = distance(mask, source['fit_y_max_exclusive']) > 0
        all_source = safe if source['fit_y_max_exclusive'] == H else distance(mask, H) > 0
        witness = source['shoe_witness']
        parts, panels, foot_panels, measures = [], [raw], [], {}
        if witness:
            crop = witness['review_crop']
            crop_path = SOURCE_REVIEW/f'foot-region-{int(frame[-6:]):02d}.png'
            assert sha(crop_path) == witness['review_crop_sha256']
            raw_witness = Image.open(crop_path).convert('RGB')
            foot_panels = [raw_witness]
            assert raw_witness.size == (crop[2]-crop[0], crop[3]-crop[1]) and source['fit_y_max_exclusive'] <= witness['xyxy'][1]
        for method, candidate in zip(METHODS, variants, strict=True):
            assert np.array_equal(candidate['K_integer_centers'], K) and np.array_equal(candidate['faces'], faces)
            verts = candidate['vertices_camera']
            parts.append(verts)
            q = verts[0].astype(np.float64)@K.T
            uv = q[:, :2]/q[:, 2:]
            raster = cover(uv, lower_faces)
            visible = (uv[lower] >= 0).all(axis=1) & (uv[lower] < [W,H]).all(axis=1)
            change = np.linalg.norm(candidate['joints_projected_native'][0]-seed['joints_projected_native'][0],axis=1)
            m = measures[method] = dict(raster_area_px=int(raster.sum()), exterior_pixels=int((raster & all_source).sum()),
                exterior_fit_pixels=int((raster & safe).sum()), lower_vertices_in_frame=int(visible.sum()),
                face_max_change_px=float(change[67:118].max()), arm_max_change_px=float(change[2:8].max()))
            panels.append(renderer(raw,box,candidate))
            if witness:
                x0,y0,x1,y1 = witness['xyxy']
                foot_cover = cover(uv,foot_faces)
                if foot_cover.any():
                    px = cv2.distanceTransform((~foot_cover).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[y0:y1,x0:x1]
                    m['shoe'] = dict(p50_distance_px=float(np.median(px)), zero_distance_fraction=float(np.mean(px==0)))
                else:
                    m['shoe'] = dict(p50_distance_px=None, zero_distance_fraction=0.)
                foot_panel = renderer(raw_witness, np.asarray(crop), candidate)
                foot_panels.append(foot_panel)
        board = joined(panels)
        image = review/f'{frame}-native.jpg'
        board.save(image,quality=94)
        if i % 4 == 0:
            sheet = Image.new('RGB',(1152,1560),'#14171b')
        board.thumbnail((1152,365),Image.Resampling.LANCZOS)
        y = (i%4)*390
        ImageDraw.Draw(sheet).text((8,y+4),frame,fill='white')
        sheet.paste(board,((1152-board.width)//2,y+17))
        if i % 4 == 3:
            page = review/f'overview-{i//4+1:02d}.jpg'
            sheet.save(page,quality=90)
            sheets.append(dict(name=page.name,sha256=sha(page)))
        row = dict(frame=frame,native_image=image.name,native_image_sha256=sha(image),methods=measures)
        if witness:
            x0,y0,x1,y1 = witness['xyxy']
            for panel in foot_panels:
                ImageDraw.Draw(panel).rectangle((x0-crop[0],y0-crop[1],x1-1-crop[0],y1-1-crop[1]),outline='#5cff5c',width=2)
            path = review/f'{frame}-shoe.png'
            joined(foot_panels).save(path)
            row['shoe_image'] = dict(name=path.name,sha256=sha(path))
        if frame in SIDE:
            path = review/f'{frame}-side.jpg'
            side_comparison(renderer, parts, faces).save(path,quality=94)
            row['side_image'] = dict(name=path.name,sha256=sha(path))
        if frame in ears:
            target = seed['observed_native_keypoints'][0,17 if ears[frame]=='right' else 18,:2]
            ear, stats = ear_comparison([legacy[i:i+1], *parts], faces, K, target, frame=frame)
            path = review/f'{frame}-ear.png'
            ear.save(path)
            row['ears'] = dict(stats,image=path.name,image_sha256=sha(path))
        rows.append(row)
        print('022_MASK147_REVIEW',frame,[m['exterior_pixels'] for m in measures.values()],flush=True)
    summary = {}
    for method in METHODS:
        records = [row['methods'][method] for row in rows]
        summary[method] = dict(median_exterior_pixels=float(np.median([v['exterior_pixels'] for v in records])),
            median_raster_exterior_fraction=float(np.median([v['exterior_pixels']/max(1,v['raster_area_px']) for v in records])),
            total_exterior_pixels=sum(v['exterior_pixels'] for v in records),
            total_lower_vertices_in_frame=sum(v['lower_vertices_in_frame'] for v in records),
            face_max_change_px=max(v['face_max_change_px'] for v in records), arm_max_change_px=max(v['arm_max_change_px'] for v in records))
    atomic(review/'result.json',dict(methods=METHODS, rows=rows,sheets=sheets, summary=summary,
        fit_sha256=sha(OUT/'result.json'), renderer_sha256=sha(renderer_path), lower_faces=len(lower_faces),foot_faces=len(foot_faces),
        evaluation_opencv=cv2.__version__, evaluator_sha256=sha(__file__),
        foot_witness_in_fit=False, metric_scope='One-sided native silhouette proxy, no anatomy or depth truth.',
        all_images_generated=True, actual_visual_review=False, quality_accepted=False, gaussian_admitted=False))


if __name__ == '__main__':
    main()
