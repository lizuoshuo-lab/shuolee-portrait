from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
O = R / 'native-residual-camera'
C = H.parent / 'shuolee-coordination'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (O / 'viewer-ui-review.json').exists()
receipt = H / 'residualCamera-122-deck-receipt.json'
d = read(receipt)
source, final = Path(d['source']), Path(d['finalPath'])
assert sha(source) == d['source_sha256']
assert sha(final) == d['final_sha256']
assert d['changed_slides'] == [52] and len(d['renders']) == 52
assert all(not item['changed'] for item in d['renders'][:51])
assert read(H / 'residualCamera-122-exact-package.json')['other_members_byte_identical']
assert read(H / 'residualCamera-122-final-validation.json')['finalSha256'] == d['final_sha256']
probe = read(O / 'result.json')
assert probe['state'] == 'completed' and len(probe['checks']) == 26
assert all(item['passed'] for item in probe['checks'])
assert not probe['human_training_started'] and not probe['real_coefficients_fitted']
freeze = read(O / 'frozen-view-addendum.json')
assert freeze['passed'] and len(freeze['checks']) == 11
assert all(item['passed'] for item in freeze['checks'])
assert not freeze['new_gpu_run'] and not freeze['numerical_render_code_changed']
bridge = read(O / 'historical-n6-bridge.json')
assert bridge['passed'] and bridge['original_files_unchanged']
assert not bridge['old_candidate_promoted']
screens = ['report-ui.png', 'report-lower-ui.png', 'viewer-ui.png']
assert all((O / name).stat().st_size > 10000 for name in screens)
stamp = datetime.now(timezone.utc).isoformat()
mirror = T / 'shuolee/reports/02_有限视角重建优化路线.pptx'
assert sha(mirror) == d['source_sha256']
shutil.copy2(final, source)
shutil.copy2(final, mirror)
assert sha(source) == sha(mirror) == d['final_sha256']
d.update(canonical_replaced=True, updated=stamp, actual_slide52_reviewed=True,
         first51_render_and_package_unchanged=True)
save(receipt, d)
save(O / 'ppt-update.json', dict(updated=stamp, path=str(source), sha256=sha(source),
     slides=52, changed=[52], first51_unchanged=True, actual_slide52_reviewed=True,
     native_PowerPoint_tested=False, receipt=str(receipt)))
save(O / 'viewer-ui-review.json', dict(
    updated=stamp, report_tab='40', main_preview_tab='31',
    urls=['http://127.0.0.1:8870/body-smplx.html#robust122-output-residual',
          'http://127.0.0.1:8870/?model=latest'],
    actual_review=[
        'Report top and lower sections inspected in the browser: complete residual camera contract, synthetic tests, historical N6 check and real-quality limits visible.',
        'Main viewer reloaded: historical 119 sofa model, 204649 points and 89 cameras retained; latestDiagnostic updated.',
        'PPT slide 52 rendered and visually inspected; other 51 render hashes and unrelated package members unchanged.'
    ],
    screenshots={name: sha(O / name) for name in screens}, tab15_untouched=True,
    new_person_model=False, physical_quality_accepted=False,
    new_residual_native_v4_OpenFile_UI=False,
    full_native_residual_capacity=False, official_external_GUI=False,
    company00_tested=False))
p = read(O / 'publication.json')
p.update(updated=stamp, report_UI_verified=True, PPT_pending=False,
         UI_evidence='viewer-ui-review.json', PPT='ppt-update.json')
save(O / 'publication.json', p)
q = read(R / 'queue-state.json')
q.update(updated=stamp, phase='R2_residual_camera_publication_complete', running=None)
q['last_goal_turn'].update(publication_complete=True,
    completed='R2 output-residual implementation, 26 synthetic integration and 11 freeze checks complete; historical N6 mapping agrees. Report/main preview and PPT52 reviewed. Real calibration and portrait quality remain unaccepted.')
save(R / 'queue-state.json', q)
c = read(C / 'local-research-status.json')
c.update(updated_at=stamp, source_queue_updated=stamp, next=q['next'],
         local_GPU_task_active=False, latest_native_residual_camera_publication=str(O / 'publication.json'))
c['viewer'].update(current_122_ui_verified=True,
    current_122_ui_scope='R2 output-residual report and latestDiagnostic actually reviewed. Historical 119 sofa and 89 cameras retained. New residual v4 native OpenFile UI is not yet tested. No new person model or physical acceptance.')
save(C / 'local-research-status.json', c)
save(C / 'local-residual-camera-handoff.json', dict(
    updated=stamp, stage='R2 explicit output-residual projection/training/native-package integration',
    evidence=str(O / 'review.json'), publication=str(O / 'publication.json'),
    url='http://127.0.0.1:8870/body-smplx.html#robust122-output-residual',
    result='Versioned renderer v3 and native package v4 retain complete per-camera K/E and Brown 6+2+4 coefficients. 26 synthetic integration checks and 11 CPU freeze checks pass; trained native response replays exactly. Historical N6 projection direction and pixel-origin transport agree; its failed real validation is unchanged.',
    limits='No real distortion fit, no person training, no company environment change. Real face 13.049 px/full-frame/material gates remain failed. Full-resolution residual capacity and new v4 native OpenFile UI are pending.',
    company_action='Information only. Do not merge research modules into stable company deployment. No acknowledgement loop.',
    next=q['next']))
print(json.dumps(dict(publication_complete=True, ppt=str(source),
                     goal_active=True, physical_quality_pass=False), ensure_ascii=False))
