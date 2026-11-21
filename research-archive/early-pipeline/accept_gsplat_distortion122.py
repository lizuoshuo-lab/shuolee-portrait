from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
O = R / 'gsplat-distortion-research'
C = H.parent / 'shuolee-coordination'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert not (O / 'viewer-ui-review.json').exists()
receipt = H / 'gsplatDistortion-122-deck-receipt.json'
d = read(receipt)
source, final = Path(d['source']), Path(d['finalPath'])
assert sha(source) == d['source_sha256']
assert sha(final) == d['final_sha256']
assert d['changed_slides'] == [52] and len(d['renders']) == 52
assert all(not item['changed'] for item in d['renders'][:51])
assert read(H / 'gsplatDistortion-122-exact-package.json')['other_members_byte_identical']
assert read(H / 'gsplatDistortion-122-final-validation.json')['finalSha256'] == d['final_sha256']
probe = read(O / 'probe-result.json')
assert probe['state'] == 'completed' and len(probe['checks']) == 8
assert all(item['passed'] for item in probe['checks'])
assert not probe['human_training_started'] and not probe['estimated_calibration']
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
     receipt=str(receipt)))
save(O / 'viewer-ui-review.json', dict(
    updated=stamp, report_tab='40', main_preview_tab='31',
    urls=['http://127.0.0.1:8870/body-smplx.html#robust122-gsplat-distortion',
          'http://127.0.0.1:8870/?model=latest'],
    actual_review=[
        'Report top and lower sections inspected in the real browser: support table, synthetic controls, installed gradient limits, and physical-calibration boundaries visible.',
        'Main viewer reloaded: historical119sofa Gaussian model and89cameras retained; latestDiagnostic shows the new research conclusion.',
        'PPT slide52 rendered and inspected; other51 render hashes and unrelated package members unchanged.'
    ],
    screenshots={name: sha(O / name) for name in screens}, tab15_untouched=True,
    new_person_model=False, physical_quality_accepted=False,
    residual_distortion_in_person_pipeline=False, official_external_GUI=False,
    company00_tested=False))
p = read(O / 'publication.json')
p.update(updated=stamp, report_UI_verified=True, PPT_pending=False,
         UI_evidence='viewer-ui-review.json', PPT='ppt-update.json')
save(O / 'publication.json', p)
q = read(R / 'queue-state.json')
q.update(updated=stamp, phase='gsplat_distortion_research_publication_complete', running=None)
q['last_goal_turn'].update(publication_complete=True,
    completed='gsplat source audit and8 synthetic CUDA checks complete; actual report/main viewer andPPT52 reviewed. Existing119model retained. Residual camera integration and real portrait quality remain unaccepted.')
save(R / 'queue-state.json', q)
c = read(C / 'local-research-status.json')
c.update(updated_at=stamp, source_queue_updated=stamp, next=q['next'],
         local_GPU_task_active=False, latest_gsplat_distortion_publication=str(O / 'publication.json'))
c['viewer'].update(current_122_ui_verified=True,
    current_122_ui_scope='gsplat distortion research report andlatestDiagnostic actually reviewed. Historical119sofa/89cameras retained. No new person model orphysical acceptance.')
save(C / 'local-research-status.json', c)
save(C / 'local-gsplat-distortion-handoff.json', dict(
    updated=stamp, stage='gsplat distortion source andCUDA capability research',
    evidence=str(O / 'review.json'), publication=str(O / 'publication.json'),
    url='http://127.0.0.1:8870/body-smplx.html#robust122-gsplat-distortion',
    result='Installed1.5.3 supports Brown6+2+4 andKB4. Single synthetic CUDA probe confirms distorted rendering andGaussian gradients; localUT/ray3D branch returns noK/distortion gradients andEbackward isunsupported. Currentlocalv2 adapter rejectsBrown residuals. Officialmain source pinned forresearch only, no upgrade.',
    limits='No real-camera coefficient fitting, no persontraining, nocompanyenvironment change. Realface13.049px/fullframe/material gates unchanged.',
    company_action='Information only. Do not upgrade ormerge research modules into stable113/119 deployment. No acknowledgement loop.',
    next=q['next']))
print(json.dumps(dict(publication_complete=True, ppt=str(source),
                     goal_active=True, physical_quality_pass=False), ensure_ascii=False))
