"""Verify the delivered camera convention and record inspected viewer imports."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V = B.parent / 'viewer'
R = B / 'current/person-static-fullscene-camera-fresh-075'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def save(p, data):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    tmp.replace(p)


rows = read(R / 'data/training-inputs.json')['records']
meta = read(V / 'model-whole-body-static-fullscene.json')
receipt = read(R / 'gaussian-run/viewer-export/delivery.json')
T = np.asarray(receipt['world_to_viewer']['rotation'])
origin = np.asarray(receipt['world_to_viewer']['origin_world'])
assert len(rows) == len(meta['cameras']) == 175
assert sha(V / meta['model']) == receipt['sha256'] == meta['sha256']
rng = np.random.default_rng(75)
maximum = 0.
for row, cam in zip(rows, meta['cameras']):
    E, K = np.asarray(row['w2c']), np.asarray(row['K'])
    assert (row['source_frame_index'], row['frame'], row['camera_kind']) == (cam['raw_source_frame'], cam['img_name'], cam['camera_kind'])
    assert row['camera_kind'] == 'fresh_full_scene_sfm'
    Kv = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]])
    np.testing.assert_allclose(K, Kv, atol=1e-10, rtol=0)
    xc = rng.uniform(-1, 1, (64, 3))
    xc[:, 2] = rng.uniform(1, 10, 64)
    xw = (xc - E[:3, 3]) @ E[:3, :3]
    xv = (xw - origin) @ T.T
    nv = (xv - np.asarray(cam['position'])) @ np.asarray(cam['rotation'])
    a, b = xc @ K.T, nv @ Kv.T
    maximum = max(maximum, float(abs(a[:, :2] / a[:, 2:] - b[:, :2] / b[:, 2:]).max()))
assert maximum < 1e-8
save(R / 'gaussian-run/viewer-camera-roundtrip.json', dict(
    checked_at=now, cameras=175, synthetic_rays_per_camera=64,
    max_projection_difference_px=maximum, passed=True,
    camera_counts=dict(fresh_full_scene_sfm=175, interpolated=0),
    claim_boundary='Coordinate/export consistency only, not estimated camera accuracy'))

screens = [R / 'review' / p for p in ['viewer-preset.png', 'viewer-standalone-gaussians.png', 'viewer-standalone-points.png']]
assert all(p.is_file() and p.stat().st_size > 10000 for p in screens)
manual = read(R / 'review/viewer-interaction-review.json')
assert manual['manual_open_file'] and manual['gaussian_mode_seen'] and manual['point_mode_seen']
assert manual['points_displayed'] == receipt['points']
visual_path = R / 'review/visual-decision.json'
visual = read(visual_path)
visual.update(local_independent_ply_import='passed', local_viewer_checked_at=now,
    local_viewer_points=receipt['points'], local_viewer_modes=['gaussian', 'point_cloud'],
    local_viewer_screenshots=[str(p) for p in screens], pc00_opened_this_round=False)
save(visual_path, visual)
public_receipt_path = V / 'models/whole-body-static-fullscene/delivery.json'
public = read(public_receipt_path)
public.update(local_standalone_open_verified=True, local_preset_open_verified=True,
    local_viewer_proof=[dict(path=str(p), sha256=sha(p)) for p in screens],
    camera_export_check=str(R / 'gaussian-run/viewer-camera-roundtrip.json'),
    quality_accepted=visual['quality_accepted'], pc00_opened_this_round=False)
save(public_receipt_path, public)
registry_path = B / 'gaussian-deliveries.json'
registry = read(registry_path)
registry['rounds']['075'].update(local_standalone_open_verified=True,
    local_gaussian_and_centers_verified=True, local_viewer_proof=public['local_viewer_proof'],
    quality_accepted=visual['quality_accepted'])
registry['updated'] = now
save(registry_path, registry)
print(json.dumps(dict(passed=True, points=receipt['points'], projection_max_px=maximum,
    standalone_open_verified=True, remote_00_verified=False), ensure_ascii=False))
