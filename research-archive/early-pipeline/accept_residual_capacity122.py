from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
A, O = R / 'native-residual-capacity', R / 'native-residual-ui'
C = H.parent / 'shuolee-coordination'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert not (A / 'publication-ui-review.json').exists()
d = read(H / 'residualCapacity-122-deck-receipt.json')
source, final = Path(d['source']), Path(d['finalPath'])
assert sha(source) == d['source_sha256'] and sha(final) == d['final_sha256']
assert d['changed_slides'] == [52] and len(d['renders']) == 52
assert all(not x['changed'] for x in d['renders'][:51])
assert read(H / 'residualCapacity-122-exact-package.json')['other_members_byte_identical']
assert read(H / 'residualCapacity-122-final-validation.json')['finalSha256'] == d['final_sha256']
assert all(c['passed'] for c in read(A / 'result.json')['checks'])
assert all(c['passed'] for c in read(O / 'review.json')['checks'])
screens = ['report-ui.png', 'report-lower-ui.png', 'viewer-ui.png']
assert all((O / n).stat().st_size > 10000 for n in screens)
stamp = datetime.now(timezone.utc).isoformat()
mirror = T / 'shuolee/reports/02_有限视角重建优化路线.pptx'
assert sha(mirror) == d['source_sha256']
shutil.copy2(final, source)
shutil.copy2(final, mirror)
assert sha(source) == sha(mirror) == d['final_sha256']
d.update(canonical_replaced=True, updated=stamp, actual_slide52_reviewed=True, first51_render_and_package_unchanged=True)
save(H / 'residualCapacity-122-deck-receipt.json', d)
save(A / 'ppt-update.json', dict(updated=stamp, path=str(source), sha256=sha(source), slides=52, changed=[52], first51_unchanged=True, actual_slide52_reviewed=True, native_PowerPoint_tested=False))
save(A / 'publication-ui-review.json', dict(updated=stamp, report_tab='40', main_preview_tab='31', native_UI_tab='44',
    screenshots={n: sha(O / n) for n in screens},
    actual_review=['Report new section, table, actual image and limits inspected.',
                   'Main viewer reloaded:119 sofa,204649points and89cameras retained;latestDiagnostic updated.',
                   'New PPT52 visually reviewed;previous51renders andunrelated package bytes unchanged.'],
    native_UI_evidence=str(O / 'review.json'), tab15_untouched=True, physical_quality_accepted=False, company00_tested=False))
p = read(A / 'publication.json')
p.update(updated=stamp, report_UI_verified=True, PPT_pending=False, UI_evidence='publication-ui-review.json', PPT='ppt-update.json')
save(A / 'publication.json', p)
q = read(R / 'queue-state.json')
q.update(updated=stamp, phase='R3_R4_publication_complete', running=None)
q['last_goal_turn'].update(publication_complete=True, completed='Full-native residual R3 capacity and R4 actual UI complete,19+20checks; reports/main119/PPT52 reviewed. Real calibration andquality remain failed.')
save(R / 'queue-state.json', q)
c = read(C / 'local-research-status.json')
c.update(updated_at=stamp, source_queue_updated=stamp, next=q['next'], local_GPU_task_active=False, latest_native_residual_capacity_publication=str(A / 'publication.json'))
c['viewer'].update(current_122_ui_verified=True, current_122_ui_scope='R3/R4 report and actual v4 native ZIP import,2recorded-full-size cameras,RGB/alpha,downloads/rejection/cache tested. Main119sofa204649/89cameras unchanged. No orbit,arbitraryimport,personquality orcompany acceptance.')
save(C / 'local-research-status.json', c)
save(C / 'local-residual-capacity-handoff.json', dict(updated=stamp,
    stage='R3 full-native residual capacity and R4 actual native UI',
    evidence=[str(A / 'review.json'), str(O / 'review.json')], publication=str(A / 'publication.json'),
    url='http://127.0.0.1:8870/body-smplx.html#robust122-residual-capacity', native_UI_url='http://127.0.0.1:8877/',
    result='200kSH3,1520x2704+2704x1520,fresh onejointupdate.19capacitychecks and20actualUIresultchecks pass. DownloadedZIP/OpenFile retainsfullKED/exactarrays,twoactualGPUrecordedviews floatRGB/alpha maxdiff0. Only LFlauncherrepair after retainedpreCUDA failure; v5lens check separatesfloat32worldclosure, no kernel orcamera change.',
    limits='No realDfit/persontraining/18kstability/orbit/officialviewer/companyacceptance. Physicalface13.049px/full464frame/materialgatesfailed. Oldv1/v2/v3 and119retained.',
    company_action='Information only. Do not merge research adapters into stable113. No acknowledgement loop.', next=q['next']))
print(json.dumps(dict(publication_complete=True, ppt=str(source), goal_active=True, physical_quality_pass=False), ensure_ascii=False))
