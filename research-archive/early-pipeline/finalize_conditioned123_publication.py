from pathlib import Path
from datetime import datetime, timezone
import json, shutil

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O = T / 'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/conditioned-reference'
stamp = datetime.now(timezone.utc).isoformat()
review = {
    'updated': stamp,
    'url': 'http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#smplify-conditioned-control',
    'existing_tab_reloaded': True,
    'actual_accessibility_review': True,
    'observed': ['three completed fit rows', 'face median and maximum residuals',
                 'duplicate wrist targets and 94-frame gaps', 'all 12 image entries',
                 'synthetic nonregression failures retained', 'Gaussian 0 steps'],
    'screenshot_available': False,
    'screenshot_error': 'Cannot take screenshot with 0 width.',
    'image_review': 'All 12 source comparison panels and three-view montage opened separately; see visual-review.json.',
    'original_tab15_preserved': True,
    'new_gaussian_openfile_applicable': False,
    'new_model_published': False,
    'quality_passed': False,
    'company00_verified': False,
    'cloud_sync_verified': False,
    'running_fit_or_gpu_task_at_final_check': False,
}
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
save(O / 'viewer-ui-review.json', review)
p = O / 'publication.json'
pub = json.loads(p.read_text(encoding='utf-8-sig'))
pub.update(updated=stamp, actual_UI_review_pending=False,
           actual_UI_accessibility_review=True, browser_screenshot_available=False,
           quality_passed=False, gaussian_steps=0)
save(p, pub)
for dest in [T / 'shuolee/current/robust-portrait-122/video10-global-initializer',
             T / 'viewer/diagnostics/video10-global-initializer',
             Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),
             Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    target = dest / 'smplifyx-reference-control/conditioned-reference'
    assert target.is_dir()
    for name in ['viewer-ui-review.json', 'publication.json']:
        shutil.copy2(O / name, target / name)
print(json.dumps({'publication_finalized': True, 'quality_passed': False,
                  'goal_complete': False, 'gaussian_steps': 0}))
