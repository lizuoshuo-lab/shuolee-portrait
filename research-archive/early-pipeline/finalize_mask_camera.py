"""Close reviewed local artifacts without changing models or remote files."""
from pathlib import Path
import datetime, hashlib, json, shutil
import numpy as np

B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V = B.parent / 'viewer'
U = Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
R = B / 'current/person-static-mask-foreground-fresh-070'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')

def save(p, d):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)

# Verify the published 175-camera coordinate conversion numerically.
rows = read(R / 'data/training-inputs.json')['records']
meta = read(V / 'model-whole-body-static-mask-foreground.json')
receipt = read(R / 'gaussian-run/viewer-export/delivery.json')
T = np.asarray(receipt['world_to_viewer']['rotation'])
o = np.asarray(receipt['world_to_viewer']['origin_world'])
assert len(rows) == len(meta['cameras']) == 175
rng = np.random.default_rng(71)
maximum = 0.0
for row, camera in zip(rows, meta['cameras']):
    E, K = np.asarray(row['w2c']), np.asarray(row['K'])
    assert row['source_frame_index'] == camera['raw_source_frame']
    assert row['frame'] == camera['img_name']
    assert row['camera_kind'] == camera['camera_kind']
    # This viewer assumes a centred principal point; check the actual inputs.
    Kv = np.array([[camera['fx'], 0., camera['width']/2],
                   [0., camera['fy'], camera['height']/2], [0., 0., 1.]])
    np.testing.assert_allclose(Kv, K, atol=1e-10, rtol=0)
    xc = rng.uniform(-1., 1., (64, 3)); xc[:, 2] = rng.uniform(1., 10., 64)
    xw = (xc - E[:3, 3]) @ E[:3, :3]
    xv = (xw - o) @ T.T
    xc_view = (xv - np.asarray(camera['position'])) @ np.asarray(camera['rotation'])
    uv = xc @ K.T; uv = uv[:, :2] / uv[:, 2:]
    uv_view = xc_view @ Kv.T; uv_view = uv_view[:, :2] / uv_view[:, 2:]
    maximum = max(maximum, float(np.max(np.abs(uv - uv_view))))
assert maximum < 1e-8, maximum
projection = dict(checked_at=now, cameras=175, synthetic_rays_per_camera=64,
    max_projection_difference_px=maximum, passed=True,
    claim_boundary='Numeric coordinate/metadata consistency only; not camera accuracy.',
    training_camera_counts=dict(calibrated=59, interpolated=116),
    new071_cameras_used_for070_training=False)
save(R / 'gaussian-run/viewer-camera-roundtrip.json', projection)

# UI evidence is from actual independent Open File import, in both modes.
screens = [U/'mask-070-viewer-gaussians.png', U/'mask-070-viewer-points.png']
assert all(p.is_file() and p.stat().st_size > 10000 for p in screens)
visual_path = R/'review/visual-decision.json'; visual = read(visual_path)
visual.update(local_independent_ply_import='passed', local_viewer_checked_at=now,
    local_viewer_points=325508, local_viewer_modes=['gaussian', 'point_cloud'],
    local_viewer_screenshots=[str(p) for p in screens],
    quality_accepted=False, pc00_opened_this_round=False)
save(visual_path, visual)

# Copy only the final rendered/inspected deck over the unchanged canonical.
deck = read(U/'mask-deck-receipt.json')
validation = read(U/'mask-camera-validation.json')
src, dst = Path(deck['finalPath']), Path(deck['source'])
assert sha(src) == deck['final_sha256'] == validation['finalSha256']
assert sha(dst) == deck['source_sha256'], 'Canonical changed after editing; do not overwrite.'
assert validation['packageIntegrity']['status'] == 'pass'
assert validation['firstPartyImport']['passed']
renders = [U/f'mask-final-{s}.png' for s in deck['changed_slides']]
assert all(p.is_file() for p in renders)
visual_deck = dict(reviewed_at=now, final_sha256=sha(src),
    actual_render_inspection=True, slides=52, inspected_changed_slides=deck['changed_slides'],
    render_files=[str(p) for p in renders],
    unchanged_slide_render_hashes=48, passed=True,
    resolved='Reflowed slides51/52 after detecting footer/image overlap; final renders inspected.',
    native_powerpoint_verified=False, pc00_deployed=False)
save(U/'mask-camera-visual-review.json', visual_deck)
shutil.copy2(src, dst)
assert sha(dst) == deck['final_sha256']
deck.update(canonical_replaced=True, canonical_replaced_at=now)
save(U/'mask-deck-receipt.json', deck)
manifest_path = B/'reports/report-manifest.json'; manifest = read(manifest_path)
report = next(x for x in manifest['reports'] if x['key'] == 'reconstruction-optimization')
report.update(sha256=sha(dst), bytes=dst.stat().st_size, slides=52,
    local_validation=str(U/'mask-camera-validation.json'),
    local_visual_review=str(U/'mask-camera-visual-review.json'),
    latest_scope='静态人物mask内部颜色监督及逐帧相机配准；新相机高斯重训尚未执行',
    current_local_report_deployed_to_00=False)
manifest['updated'] = now
manifest['last_local_revision'] = dict(path=str(dst), sha256=sha(dst), bytes=dst.stat().st_size,
    slides=52, remote_delivered=False, transfer_performed=False,
    local_review=str(U/'mask-camera-visual-review.json'), updated=now)
save(manifest_path, manifest)

status_path = B/'work-status.json'; status = read(status_path)
delivery = read(B/'gaussian-deliveries.json')['rounds']['070']
status['updated'] = now
status['latest_completed_static_delivery'] = dict(created=now, round='070',
    state='trained_export_and_local_import_verified_quality_not_accepted',
    steps=18000, initial_points=200000, points=325508, from_scratch=True, frames=175,
    representation='shared_static_world', compatibility_passed=True,
    quality_accepted=False, local_viewer_passed=True, pc00_opened_this_round=False,
    download_url=delivery['download_url'], viewer_url=delivery['viewer_url'],
    local_ply=delivery['download_path'], training_cameras='59 SfM + 116 interpolation; historical070',
    review=str(visual_path), camera_export_check=str(R/'gaussian-run/viewer-camera-roundtrip.json'))
status['latest_report'] = dict(path=str(dst), sha256=sha(dst), changed_slides=deck['changed_slides'],
    slides=52, updated=now, remote_deployed=False)
status['pending_report_delivery'] = dict(status='local_updated_remote_deferred', source=str(dst),
    file=dst.name, sha256=sha(dst), bytes=dst.stat().st_size, remote_delivered=False)
save(status_path, status)
print(json.dumps(dict(camera_projection=projection, report_sha256=sha(dst),
    local_ply_import='passed', gaussian_retraining_started=False), ensure_ascii=False, indent=2))
