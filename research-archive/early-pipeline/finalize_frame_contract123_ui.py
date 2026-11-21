import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
A = T / 'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/training-interface-audit'
COORD = Path('C:/Users/Administrator/Documents/New project/shuolee-coordination')
stamp = datetime.now(timezone.utc).isoformat()

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

review = dict(updated=stamp, url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#frame-pose-contract',
    browser_tab='32', actual_reload=True, actual_accessibility_text_verified=True,
    checked=['27 CPU checks and CLI', 'real-frame identity and explicit reference frame',
             'integer-center to corner half-pixel distinction', 'skew unsupported by old CameraProj2',
             'no model forward, new fit, or Gaussian training', '00 handoff and deployment pending'],
    screenshot_verified=False, screenshot_error='Cannot take screenshot with 0 width.',
    rendered_visual_layout_accepted=False, history_tab15_untouched=True,
    no_new_model=True, physical_quality_accepted=False, company00_deployed=False)
save(A / 'contract-viewer-ui-review.json', review)
pub_path = A / 'contract-publication.json'
pub = json.loads(pub_path.read_text(encoding='utf-8-sig'))
pub.update(updated=stamp, viewer_UI_pending=False, viewer_UI_scope='Reload and accessibility text only; screenshot unavailable, no rendered-layout acceptance.',
           viewer_accessibility_text_verified=True, viewer_screenshot_verified=False,
           viewer_UI_review=str(A / 'contract-viewer-ui-review.json'))
save(pub_path, pub)
for base in [R / 'video10-global-initializer', T / 'viewer/diagnostics/video10-global-initializer', Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    for name in ['contract-publication.json', 'contract-viewer-ui-review.json']:
        shutil.copy2(A / name, base / 'training-interface-audit' / name)
status_path = COORD / 'local-research-status.json'
status = json.loads(status_path.read_text(encoding='utf-8-sig'))
status.update(updated_at=stamp, latest_publication=str(pub_path), latest_viewer_ui_review=str(A / 'contract-viewer-ui-review.json'),
              viewer_text_and_table_verified=True, viewer_screenshot_verified=False)
status['company00_requested_work'].update(state='dispatched_target_active_code_review',
    thread_id='01a17c8c-e70f-7822-ad39-440290b3d1da',
    last_observed_feedback='00 is adding full image decoding validation before commit. Current remote reported public; requested checking direct user authorization before public push.',
    Git_uploaded_confirmed=False, server_deployed_confirmed=False,
    user_authorized_feedback_to_origin=True)
save(status_path, status)
print(json.dumps({'ui_text_verified': True, 'screenshot_verified': False, '00_task_active': True, 'Git_or_server_deployment_confirmed': False}))
