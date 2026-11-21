import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta

BASE = Path(r'\\?\C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
CROWN = BASE / 'current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/crown-attribution'
BUILD = Path(r'C:\Users\Administrator\Documents\New project\shuolee-slide-update')
VIEWER = BASE.parent / 'viewer'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
ui = read(BUILD / 'motion-ui-observation.json')
assert ui['details_open'] and len(ui['images']) == 3
assert all(i['complete'] and i['width'] > 0 and i['height'] > 0 for i in ui['images'])
assert ui['video']['error'] is None and ui['video']['ended'] and ui['video']['readyState'] == 4
assert abs(ui['video']['duration'] - 61 / 12) < 1e-5
cleanup = read(CROWN / 'motion-cleanup.json')
assert cleanup['state'] == 'completed' and len(cleanup['files']) == 12
proof = VIEWER / 'body-dynamic/motion-review-proof.png'
review = read(CROWN / 'motion-review.json')
receipt = dict(updated=stamp, state='browser_verified', browser='Codex in-app browser',
               observed_ui=ui, playback_observed_after_native_play_click=True,
               complete_comparison_and_caption_visually_reviewed=True,
               screenshot=str(proof), screenshot_sha256=sha(proof),
               screenshot_dimensions=[1265, 712],
               visual_review_boundary='Named source and novel-view stills were inspected; playback verification is not a frame-by-frame review of all 61 frames.',
               quality_accepted=False, model_replaced=False)
save(CROWN / 'motion-browser-verification.json', receipt)
review.update(updated=stamp, browser_verified=True,
              browser_verification='motion-browser-verification.json',
              cleanup=dict(manifest='motion-cleanup.json', files_removed=12,
                           bytes_removed=cleanup['bytes_removed'],
                           failed_local_candidates_removed=True,
                           main_clothing_viewer_sha256_rechecked=True))
review['relocated_review_evidence'] = {
    f['path']: f['retained_duplicate'] for f in cleanup['files']
    if f['retained_duplicate'] and f['reason'] == 'Exact copy published to the local review page.'
}
review['candidate_checkpoints_retained'] = False
save(CROWN / 'motion-review.json', review)
ws = read(BASE / 'work-status.json')
ws.update(updated=stamp, dynamic_measured_motion=review,
          local_training_running=False,
          current_phase='dynamic_real_motion_reviewed_local_patch_not_adopted',
          goal_accepted=False)
save(BASE / 'work-status.json', ws)
visual = read(BUILD / 'motion-visual-review.json')
visual.update(full_size_final_slide_55_visually_reviewed=True,
              browser_evidence_receipt=str(CROWN / 'motion-browser-verification.json'))
save(BUILD / 'motion-visual-review.json', visual)
print(json.dumps(dict(state='review_finalized', browser_verified=True,
                      files_removed=12, bytes_removed=cleanup['bytes_removed'],
                      main_model_replaced=False, training_running=False,
                      goal_accepted=False), ensure_ascii=False))
