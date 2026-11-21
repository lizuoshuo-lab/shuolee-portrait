from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
OUT = T / 'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/sofa-selected-sh0-pipeline124'
C = H.parent / 'shuolee-coordination'
W = H / 'remy00-delivery-20261103'
now = datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

runtime = read(W / 'launcher-runtime.json')
assert runtime['zipHttp'] == 200 and runtime['modelPoints'] == 789005
for name in ['launcher-render.png', 'openfile-sync-package.png']:
    assert (W / name).stat().st_size > 10000
delivery = read(W / 'delivery.json')
delivery.update(updated=now, state='local_runtime_and_UI_verified_company00_pending',
    runtime=runtime, actual_windows_preset_render=True,
    actual_export_zip_openfile=True, automatic_paired_cameras=174,
    ui_evidence=['launcher-render.png', 'openfile-sync-package.png'],
    company00_verified=False, company00_failure_url_known=False)
save(W / 'delivery.json', delivery)

for rel in ['delivery/publication.json', 'pipeline-publication.json']:
    p = OUT / rel
    data = read(p)
    data['updated'] = now
    data['ui_checked' if rel.startswith('delivery/') else 'actual_UI_verified'] = True
    data['company00_verified'] = False
    data['quality_accepted'] = False
    save(p, data)

entry = dict(updated=now, receipt=str(W / 'delivery.json'),
    package=delivery['package'], package_sha256=delivery['package_sha256'],
    launcher_build='2026.11.03.1', zip_http_before=404, zip_http_after=200,
    local_windows_actual_render=True, local_offline_package_import=True,
    company00_verified=False, company00_failure_url_known=False,
    guidance='Wait for complete WPS download, launch from the synced viewer folder, Open File the Remy ZIP. Do not reuse another machine loopback URL or an old desktop root.')
for p in [R / 'protocol.json', R / 'queue-state.json', R / 'evidence-ledger.json', C / 'local-research-status.json']:
    data = read(p)
    data['updated'] = now
    if 'updated_at' in data:
        data['updated_at'] = now
    if data.get('current_stage') == 'sofa_sh0_pipeline124_delivered_review_pending_final_ui':
        data['current_stage'] = 'sofa_sh0_pipeline124_delivered_local_ui_verified'
    data['remy00_delivery_20261103'] = entry
    sofa = data.get('sofa_sh0_pipeline124')
    if isinstance(sofa, dict):
        sofa['actualFinalPresetRender'] = True
        sofa['actualReportUI'] = True
        sofa['company00Verified'] = False
    save(p, data)
save(C / 'remy00-delivery-20261103.json', entry)

ui_root = OUT / 'delivery/step-018000/ui-review'
for name in ['openfile-gaussian.png', 'openfile-points.png', 'viewer-final.png', 'report-final.png']:
    assert (ui_root / name).stat().st_size > 1000
save(ui_root / 'review.json', dict(updated=now,
    model='cross-person-124-sofa-selected-sh0-pipeline124-sh0-s018000',
    actualOpenFileGaussian=True, actualOpenFilePoints=True,
    actualPresetReload=True, pairedCameras=1, points=28121,
    actualReportUI=True, qualityAccepted=False, company00Verified=False,
    limitation='Gray preview reveals fragmented shoes; source/export parity was tested on black. A native-gray check has not uniquely separated training from preview differences.'))
print(json.dumps(dict(delivery=entry, sofa_local_ui_verified=True, goal_complete=False), ensure_ascii=False))
