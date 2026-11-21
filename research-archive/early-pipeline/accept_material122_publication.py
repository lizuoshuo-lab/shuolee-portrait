from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timezone
import hashlib
import json
import shutil

H = Path(__file__).parent
T = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
I = R / 'video-update-20261027'
C = H.parent / 'shuolee-coordination'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

stamp = datetime.now(timezone.utc).isoformat()
receipt = read(H / 'material122-deck-receipt.json')
source, final = Path(receipt['source']), Path(receipt['finalPath'])
canonical = T / 'shuolee/reports/02_有限视角重建优化路线.pptx'
assert not receipt['canonical_replaced']
assert sha(source) == sha(canonical) == receipt['source_sha256']
assert sha(final) == receipt['final_sha256']
with ZipFile(source) as a, ZipFile(final) as b:
    assert a.namelist() == b.namelist()
    changed = [name for name in a.namelist() if a.read(name) != b.read(name)]
    assert set(changed) == {'ppt/slides/slide52.xml', 'ppt/notesSlides/notesSlide52.xml'}
assert len(receipt['renders']) == 52
assert all(not item['changed'] for item in receipt['renders'][:51])
for name in ['report-ui.png', 'viewer-ui.png']:
    assert (I / name).is_file()
validation = read(H / 'material122-final-validation.json')
assert validation['finalSha256'] == receipt['final_sha256']
assert validation['packageIntegrity']['status'] == 'pass'
assert validation['presentationLayout']['finding_count'] == 0
for destination in [canonical, source]:
    temporary = destination.with_suffix('.replacement.pptx')
    assert not temporary.exists()
    shutil.copy2(final, temporary)
    assert sha(temporary) == receipt['final_sha256']
    temporary.replace(destination)
receipt.update(
    accepted=stamp, canonical_replaced=True, canonical=str(canonical), delivery=str(source),
    changed_zip_members=changed, first51_members_and_render_hashes_identical=True,
    changed_slide_visually_reviewed=True, PowerPoint_or_WPS_runtime_verified=False,
    visual_review='Before/after slide52 PNGs inspected. Same historical nose material figure explicitly labelled historical. Updated title and two conclusions fit without clipping; source pixels unchanged.'
)
save(H / 'material122-deck-receipt.json', receipt)
save(I / 'ppt-update.json', {key: value for key, value in receipt.items() if key != 'renders'})
ui = {
    'updated': stamp, 'report_tab': '40', 'viewer_tab': '31',
    'report_url': 'http://127.0.0.1:8870/body-smplx.html#robust122-new-input',
    'actual_report_reviewed': True, 'latestDiagnostic_display_verified': True,
    'model': 'new-capture-119-sofa-clothed', 'points': 204649, 'cameras': 89,
    'viewer_version': '2026.10.27.5', 'quality_accepted': False,
    'historical_tab15_untouched': True, 'native_interactive_GUI_verified': False,
    'screenshots': ['video-update-20261027/report-ui.png', 'video-update-20261027/viewer-ui.png'],
    'remote00_verified': False, 'WPS_cloud_verified': False, 'training_started': False,
}
save(I / 'viewer-ui-review.json', ui)
publication = read(R / 'publication.json')
publication.update(updated=stamp, R1d_C1a_UI_verified=True,
    R1d_C1a_UI_review='video-update-20261027/viewer-ui-review.json',
    new_input_PPT_update='video-update-20261027/ppt-update.json')
save(R / 'publication.json', publication)
queue = read(R / 'queue-state.json')
queue['updated'] = stamp
queue['last_goal_turn']['remaining'] = 'Fresh new-video observations and output-K contract, independent physical subpixel camera admission and static training.'
queue['last_goal_turn']['evidence'] += ['video-update-20261027/viewer-ui-review.json', 'video-update-20261027/ppt-update.json']
save(R / 'queue-state.json', queue)
ledger = read(R / 'evidence-ledger.json')
ledger['updated'] = stamp
ledger['confirmed'].append('New-video intake/material/pixel-origin report actually reviewed in tab40; tab31 retained119 historical preview with204649points/89cameras and updated diagnostic. Same52-pagePPT onlyslide52 text/notes changed; other51 members and renders identical. No native interactive or company00 verification.')
ledger['artifacts'] += ['video-update-20261027/viewer-ui-review.json', 'video-update-20261027/ppt-update.json']
save(R / 'evidence-ledger.json', ledger)
status = read(C / 'local-research-status.json')
company = read(C / 'company00-status.json')
status['updated_at'] = stamp
status['source_queue_updated'] = stamp
status['latest_input_update']['actual_report_and_preview_UI_reviewed'] = True
status['latest_input_update']['same_PPT_slide52_updated'] = True
status['historical_121_face_metric_scope'] = 'These are faceROI-selected exploratory residuals, not certified physical face-point errors. C1a found hat and incidental background tracks; original values retained without residual filtering.'
status['company00_handoff_received'] = {
    'updated_at': stamp, 'source': 'company00-status.json', 'state': company['state'],
    'scope': 'User-pasted server evidence reports verified head source alias applied and fresh113_002 worker PID8115 started. Current liveness/head success/Gaussian training unverified. Main research did not log in, execute server commands or modify113.'
}
save(C / 'local-research-status.json', status)
print(json.dumps({'published': str(source), 'PPT_only_slide52_changed': True, 'actual_UI_verified': True, 'new_training_started': False}, ensure_ascii=False))
