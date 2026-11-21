"""Read-only indoor fit review using the existing native mesh renderer."""
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from review_full_sequence123 import mesh_renderer, crop_box, joined
from run_official_gaussians123 import read, save, sha, now
from review_grass_selected_views124 import ROOT, load

OUT = ROOT/'indoor-selected-view-pose124'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cv2.setNumThreads(1)
    plan, result = read(OUT/'plan.json'), read(OUT/'result.json')
    assert read(OUT/'process-exit.json')['exit_code'] == 0
    assert read(OUT/'progress.json')['state'] == 'completed_review_pending'
    assert sha(OUT/'selected-fit.npz') == result['fit_sha256']
    review = OUT/'review'
    review.mkdir(exist_ok=True)
    assert not (review/'result.json').exists()
    record = Path('/mnt/d/00/记录/008')
    renderer, renderer_source = mesh_renderer()
    metrics = {r['frame']: r for r in result['metrics']}
    records, tiles = [], []
    for row in plan['rows']:
        name = row['frame']
        newfile = OUT/'frames'/name/'winner-camera-space.npz'
        arr = load(newfile)
        assert np.array_equal(arr['K_integer_centers'], row['K_integer_centers'])
        assert sha(row['image']) == row['source_sha256']
        assert sha(row['mask']) == row['mask_sha256']
        raw = Image.open(row['image']).convert('RGB')
        mask = Image.open(row['mask']).convert('L')
        box = crop_box(mask.getbbox(), raw.size, 30)
        boxes = dict(body=box,
            head=crop_box(row['face_box_xyxy'], raw.size, 35),
            lower_visible=crop_box([box[0], box[1]+.72*(box[3]-box[1]), box[2], box[3]], raw.size))
        hand = arr['observed_native_keypoints'][0, 25:67]
        hand = hand[hand[:, 2] > 0, :2]
        if len(hand):
            boxes['hands'] = crop_box(np.r_[hand.min(0), hand.max(0)], raw.size, 55)
        images = {}
        for region, bb in boxes.items():
            crop = raw.crop(bb)
            panel = joined([crop, renderer(crop, np.array(bb), arr)])
            path = record/f'smpl-x_室内六帧联合124_{name}_{region}_原图_共享体型姿态.png'
            assert not path.exists()
            panel.save(path)
            images[region] = dict(path=str(path), sha256=sha(path), native_crop=bb, resized=False)
        v0 = arr['vertices_camera'][0]
        center = (v0.min(0)+v0.max(0))/2
        extent = float(np.ptp(v0, axis=0).max())
        K = np.array([[1150., 0, 170], [0, 1150., 240], [0, 0, 1.]])
        views = Image.new('RGB', (1360, 480))
        for j, degree in enumerate([0, 90, 180, -90]):
            rot = Rotation.from_rotvec([0, np.deg2rad(degree), 0]).as_matrix()
            v = (v0-center) @ rot.T
            v[:, 2] += 3*extent
            img = renderer(Image.new('RGB', (340, 480)), np.array([0, 0, 340, 480]),
                dict(vertices_camera=v[None], faces=arr['faces'], K_integer_centers=K))
            views.paste(img, (340*j, 0))
        path = record/f'分析_室内六帧联合124_{name}_相同四方向_共享体型姿态.png'
        assert not path.exists()
        views.save(path)
        images['sides'] = dict(path=str(path), sha256=sha(path), synthetic_not_truth=True)
        tile = Image.new('RGB', (1480, 480))
        native = Image.open(images['body']['path'])
        native.thumbnail((800, 480))
        tile.paste(native, (0, 0))
        tile.paste(views.crop((340, 0, 1020, 480)), (800, 0))
        tiles.append(tile)
        records.append(dict(frame=name, images=images, metrics=metrics[name],
            fit_sha256=sha(newfile), same_K_pixels=True, full_mesh_lower_body_is_prior_not_observed=True))
    pages = []
    for start in range(0, len(tiles), 3):
        page = Image.new('RGB', (1480, 480*min(3, len(tiles)-start)))
        for i, tile in enumerate(tiles[start:start+3]):
            page.paste(tile, (0, i*480))
        path = review/f'page-{start//3+1:02d}.png'
        page.save(path)
        pages.append(str(path))
    save(review/'result.json', dict(updated=now(), records=records, pages=pages,
        columns_native=['raw', 'shared_six_fit'], columns_sides=['same_mesh_rotated_90', 'same_mesh_rotated_180'],
        actual_visual_review=False, quality_accepted=False, gaussian_admitted=False,
        lower_visible_not_feet_truth=True, detector_error_not_truth=True,
        original_inputs_unchanged=True, text_on_images=False,
        renderer_sha256=sha(renderer_source), source_sha256=sha(__file__)))
    print('INDOOR_SIX_NATIVE_REVIEW_READY', flush=True)


if __name__ == '__main__':
    main()
