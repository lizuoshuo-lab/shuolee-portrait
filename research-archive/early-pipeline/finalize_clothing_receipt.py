import hashlib
import json
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path


def path(value):
    return Path('\\\\?\\' + str(value).replace('/', '\\'))


BASE = path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
PHOTO = BASE / 'current/novel-view-face/head-structure/flame-photometric-registration'
CLOTH = PHOTO / 'whole-body-gaussianavatar/all-views-background/observed-clothing-pilot'
VIEWER = BASE.parent / 'viewer'
BUILD = path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
FINAL = path('C:/Users/Administrator/Documents/New project/shuolee-slide-output/02_有限视角重建优化路线_20261022_衣物修订.pptx')
REPORT = BASE / 'reports/02_有限视角重建优化路线.pptx'
NOW = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
OLD_SHA = 'b11a18d22ecf5c393b06ee79f62c1ccb8d039fa47b4f940c6ff50f774ff55269'
NEW_SHA = '7342d60a349f4cf958043deb0cd7f1d43567249fcfa65aa764a8406b6b5d3114'


def clean(p):
    return str(p).removeprefix('\\\\?\\')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, value):
    temp = p.with_name(p.name + '.revision-tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, p)


assert sha(FINAL) == NEW_SHA
assert sha(REPORT) in (OLD_SHA, NEW_SHA), 'Existing report changed outside this revision; preserve it.'
validation = read(BUILD / 'clothing-final-validation.json')
assert validation['finalSha256'] == NEW_SHA
assert validation['firstPartyImport']['passed'] is True
assert validation['firstPartyImport']['slideCount'] == 53
slide_check = read(BUILD / 'clothing-unchanged-slide-check.json')
assert slide_check['unexpectedDifferences'] == []
visual = {
    'updated': NOW, 'sha256': NEW_SHA, 'slide_count': 53,
    'reviewed': True, 'modified_slides': [2, 53],
    'all_slide_montage': clean(BUILD / 'clothing-final-import-montage.png'),
    'final_import_rendered_slides': [2, 53],
    'final_import_images': [clean(BUILD / f'clothing-final-renders/final-import-{n}.png') for n in (2, 53)],
    'observed': 'Final imported slide53 has readable complete four-column real comparison, method and training-only metric; limitations remain on final slide. Final montage retains53slides and unchanged-slide hash check has no unexpected differences.',
    'native_powerpoint_verified': False,
}
write(BUILD / 'clothing-visual-review.json', visual)

native = read(CLOTH / 'review/result.json')
assert native['quality_accepted'] is False and native['total_points'] == 480820
assert native['main_preserved'] and native['head_preserved'] and native['background_preserved']
model = VIEWER / 'models/whole-body-clothing-candidate/point_cloud.ply'
assert sha(model) == native['scene_ply_sha256']
proofs = [VIEWER / 'body-dynamic/clothing-proof.png', VIEWER / 'body-dynamic/clothing-viewer-proof.png']
assert all(p.is_file() and p.stat().st_size > 10000 for p in proofs)
browser = {
    'verified_at': NOW,
    'page_url': 'http://127.0.0.1:8870/body-dynamic.html#clothing',
    'viewer_url': 'http://127.0.0.1:8870/?model=whole-body-clothing-candidate',
    'visible_point_count': 480820,
    'visible_file_validation': '文件校验通过',
    'actual_canvas_visually_inspected': True,
    'clothing_images_loaded': 5,
    'video_ready_state': [4, 4],
    'video_error': [None, None],
    'video_duration_seconds': [4.916667, 5.083333],
    'time_video_playback_advanced_seconds': 0.649612,
    'horizontal_overflow': False,
    'screenshots': [{'path': clean(p), 'sha256': sha(p)} for p in proofs],
    'static_viewer_snapshot_frame': 23,
    'dynamic_playback_location': 'effect page actual59-state video',
    'quality_accepted': False,
    'claim_boundary': 'Browser load, file validation, visible canvas and video playback verified. This is not independent novel-view ground truth or complete artifact-free temporal acceptance.',
}
write(CLOTH / 'review/browser-verification.json', browser)
native['updated'] = NOW
native['state'] = 'native_and_browser_verified_local_candidate'
native['browser_verification'] = clean(CLOTH / 'review/browser-verification.json')
native['export_compatibility_note'] = {
    'source': clean(PHOTO / 'prepare_dynamic_observed_clothing.py'),
    'current_source_sha256': sha(PHOTO / 'prepare_dynamic_observed_clothing.py'),
    'change_after_training': 'Added Union._opacity property forwarding raw_opacity for SceneModel export compatibility; no parameter, rendering math or checkpoint change.',
    'training_protocol_hash_retained': True,
}
write(CLOTH / 'review/result.json', native)

review = read(CLOTH / 'matched-training/root-review.json')
review['updated'] = NOW
review['viewer_browser_validation_pending'] = False
review['browser_verified'] = True
review['browser_evidence'] = clean(CLOTH / 'review/browser-verification.json')
review['visual_files_reviewed'] = list(dict.fromkeys(review['visual_files_reviewed'] + ['matched-training/000000-clothing.png', 'review/pitch-45.png']))
review['next'] = '保留已改善衣服轮廓的局部候选。下一轮先解决头发/头部表面表示与脚边真实背景支持；继续检查连续时间和侧视稳定性。静态与动态衣物差异不足以宣称动态更优，不加主轮步数，不重启眼镜专项。'
write(CLOTH / 'matched-training/root-review.json', review)

if sha(REPORT) != NEW_SHA:
    temporary = REPORT.with_name(REPORT.name + '.revision-tmp')
    shutil.copy2(FINAL, temporary)
    assert sha(temporary) == NEW_SHA
    assert sha(REPORT) == OLD_SHA, 'Existing report changed during copy.'
    os.replace(temporary, REPORT)
assert sha(REPORT) == NEW_SHA
manifest = read(BASE / 'reports/report-manifest.json')
manifest['updated'] = NOW
for entry in manifest['reports']:
    if entry['key'] == 'reconstruction-optimization':
        entry.update(sha256=NEW_SHA, bytes=REPORT.stat().st_size, slides=53,
                     local_validation=clean(BUILD / 'clothing-final-validation.json'),
                     local_visual_review=clean(BUILD / 'clothing-visual-review.json'))
manifest['last_local_revision'] = {
    'sha256': NEW_SHA, 'bytes': REPORT.stat().st_size, 'slides': 53,
    'remote_delivered': False, 'remote_status': 'local_first_clothing_partial_candidate', 'updated': NOW,
}
write(BASE / 'reports/report-manifest.json', manifest)

status = read(BASE / 'work-status.json')
status.update(updated=NOW, current_phase='observed_clothing_reviewed_local_candidate',
              local_training_running=False, local_inference_running=False,
              latest_local_execution='Retained main all59 dynamic18000-step model. Completed3x885 native2D controls without adoption; completed2x885 observed-clothing controls.30592 added clothing Gaussians improve garment contour. Joint480820-point candidate passes native export and browser load/playback checks, overall novel-view quality not accepted.')
status['observed_clothing_pilot'] = review
status['current_experiment'] = {
    'name': 'Dynamic_all59_body_observed_clothing_static_background',
    'status': review['state'], 'accepted': False, 'main_training_complete': True,
    'main_steps': 18000, 'main_body_points': 191595, 'added_clothing_points': 30592,
    'background_points': 258633, 'scene_points': 480820,
    'pilot_steps_per_arm': 885, 'pilot_actual_updates': 1770,
    'evidence': clean(CLOTH / 'matched-training/root-review.json'),
    'new_best_model': False, 'remote_deployed': False, 'eyewear_specialization': False,
}
status['latest_local_report_revision'] = {
    'path': clean(REPORT), 'sha256': NEW_SHA, 'slides': 53,
    'remote_delivered': False, 'local_import_and_visual_review_passed': True,
}
status['body_surface_gaussian_pilot']['next'] = 'Parameter isolation and3x885 staged controls completed without adoption. Detailed evidence remains in staged-controls/root-review.json; continue independent observed-clothing branch rather than more blind geometry weights.'
write(BASE / 'work-status.json', status)
write(BUILD / 'clothing-final-delivery.json', {
    'updated': NOW, 'report': clean(REPORT), 'sha256': NEW_SHA,
    'slides': 53, 'browser_evidence': clean(CLOTH / 'review/browser-verification.json'),
    'local_quality_accepted': False, 'remote_deployed': False,
})
print(json.dumps({'report_updated': clean(REPORT), 'sha256': sha(REPORT),
                  'browser_verified': True, 'quality_accepted': False,
                  'training_running': False, 'remote_deployed': False}, ensure_ascii=False))
