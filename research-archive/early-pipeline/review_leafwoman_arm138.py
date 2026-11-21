"""Render the same source crop and synthetic cameras for the 022 arm control."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from official_smplifyx_entry123 import sha
from replay_leafwoman_arm138 import ARCHIVE, OUT
from review_full_sequence123 import joined, mesh_renderer, relocate
from update_leafwoman129_state import NATIVE, atomic, read


def review():
    cv2.setNumThreads(1)
    plan = read(OUT/'plan.json')
    result = read(OUT/'arm-result.json')
    contacts = read(NATIVE/'source-pose-review/source-contacts.json')
    source = {row['frame']: row['paths']['body'] for row in contacts['records']}
    renderer, renderer_path = mesh_renderer()
    records = []
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        for i, frame, row in zip(plan['slots'], plan['frames'], result['rows'], strict=True):
            assert frame == row['frame'] == saved['frame_ids'][i]
            old = dict(vertices_camera=saved['vertices_camera'][i:i+1], faces=saved['faces'],
                       K_integer_centers=saved['K_integer_centers'])
            with np.load(OUT/(frame+'.npz'), allow_pickle=False) as arrays:
                candidate = {key: arrays[key] for key in old}
            assert np.array_equal(old['faces'], candidate['faces'])
            assert np.array_equal(old['K_integer_centers'], candidate['K_integer_centers'])
            item = source[frame]
            path = relocate(item['path'])
            assert sha(path) == item['sha256'] and item['native_integer_crop'] and not item['resized']
            box = np.asarray(item['crop_original_xyxy'])
            with Image.open(path) as image:
                raw = image.convert('RGB')
            assert raw.size == tuple(box[2:]-box[:2])
            body = OUT/(frame+'-arm-native.png')
            joined([raw, renderer(raw, box, old), renderer(raw, box, candidate)]).save(body)

            v = old['vertices_camera'][0]
            center = (v.min(0)+v.max(0))/2
            extent = float(np.ptp(v, axis=0).max())
            wh = (340, 480)
            K = np.array([[1150.,0,170],[0,1150.,240],[0,0,1.]])
            blank = Image.new('RGB', wh)
            tiles = []
            for angle in (90, -90):
                rotation = Rotation.from_rotvec([0, np.deg2rad(angle), 0]).as_matrix()
                for snapshot in (old, candidate):
                    vv = (snapshot['vertices_camera']-center)@rotation.T
                    vv[...,2] += 3*extent
                    tiles.append(renderer(blank, np.array([0,0,*wh]),
                                          dict(vertices_camera=vv, K_integer_centers=K, faces=old['faces'])))
            side = OUT/(frame+'-arm-side.png')
            joined(tiles).save(side)
            records.append(dict(frame=frame, source_crop_sha256=item['sha256'],
                                files={p.name: sha(p) for p in (body, side)}))
    atomic(OUT/'review-frames.json', dict(frames=records, renderer=str(renderer_path),
        renderer_sha256=sha(renderer_path), source_panels=['source','frozen','arm_refine'],
        synthetic_panels=['+90 frozen','+90 arm_refine','-90 frozen','-90 arm_refine']))


if __name__ == '__main__':
    review()
