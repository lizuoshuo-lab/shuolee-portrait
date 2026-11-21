from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timezone
import hashlib, json, shutil

H = Path(__file__).parent
T = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
B = R / 'video-update-20261027/background-geometry'
C = H.parent / 'shuolee-coordination'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')

stamp = datetime.now(timezone.utc).isoformat()
receipt = read(H / 'projection122-deck-receipt.json')
source, final = Path(receipt['source']), Path(receipt['finalPath'])
canonical = T / 'shuolee/reports/02_有限视角重建优化路线.pptx'
assert not receipt['canonical_replaced']
assert sha(source) == sha(canonical) == receipt['source_sha256']
assert sha(final) == receipt['final_sha256']
with ZipFile(source) as a, ZipFile(final) as b:
    assert set(a.namelist()) == set(b.namelist())
    changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
    assert set(changed) == {'ppt/slides/slide52.xml', 'ppt/notesSlides/notesSlide52.xml', 'ppt/media/n1_material122.png'}
    owners = [n for n in a.namelist() if n.startswith('ppt/slides/_rels/') and b'n1_material122.png' in a.read(n)]
    assert owners == ['ppt/slides/_rels/slide52.xml.rels'], owners
assert len(receipt['renders']) == 52 and all(not s['changed'] for s in receipt['renders'][:51])
v = read(H / 'projection122-final-validation.json')
assert v['finalSha256'] == receipt['final_sha256']
assert v['packageIntegrity']['status'] == 'pass' and v['presentationLayout']['finding_count'] == 0
assert v['firstPartyImport']['passed'] and v['firstPartyImport']['slideCount'] == 52
screens = ['report-ui.png', 'report-images-ui.png', 'viewer-ui.png']
assert all((B / n).is_file() for n in screens)
for destination in [canonical, source]:
    temp = destination.with_suffix('.replacement.pptx')
    assert not temp.exists()
    shutil.copy2(final, temp)
    assert sha(temp) == receipt['final_sha256']
    temp.replace(destination)
receipt.update(accepted=stamp, canonical_replaced=True, canonical=str(canonical), delivery=str(source), changed_zip_members=changed,
               first51_members_and_render_hashes_identical=True, changed_slide_visually_reviewed=True, PowerPoint_or_WPS_runtime_verified=False,
               visual_review='Final slide PNG viewed: two rows of six native square crops have unchanged aspect; captions fit. Comparison excludes anchor278 only for layout; full seven-view report remains available.')
save(H / 'projection122-deck-receipt.json', receipt)
save(B / 'ppt-update.json', {k: x for k, x in receipt.items() if k != 'renders'})
ui = dict(updated=stamp, report_tab='40', viewer_tab='31', report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n6',
          actual_report_reviewed=True, all_three_crop_rows_reviewed=True, latestDiagnostic_display_verified=True,
          model='new-capture-119-sofa-clothed', points=204649, cameras=89, viewer_version='2026.10.27.5', quality_accepted=False,
          historical_tab15_untouched=True, screenshots=[dict(file=f, sha256=sha(B/f)) for f in screens],
          remote00_verified=False, WPS_cloud_verified=False, training_started=False, new_PLY=False)
save(B / 'viewer-ui-review.json', ui)
p = read(B / 'publication.json')
p.update(updated=stamp, UI_verified=True, PPT_updated=True)
save(B / 'publication.json', p)
p = read(R / 'publication.json')
p.update(updated=stamp, N5_N6_UI_verified=True, N5_N6_UI_review='video-update-20261027/background-geometry/viewer-ui-review.json',
         N5_N6_PPT_update='video-update-20261027/background-geometry/ppt-update.json')
save(R / 'publication.json', p)
q = read(R / 'queue-state.json')
q.update(updated=stamp, phase='residual_projection_comparison_reviewed_and_published')
q['last_goal_turn']['remaining'] = 'Spatial/time/material camera diagnosis and all-frame training admission; new PDF matrix-role claims require source verification.'
q['last_goal_turn']['evidence'] += ['video-update-20261027/background-geometry/viewer-ui-review.json', 'video-update-20261027/background-geometry/ppt-update.json']
save(R / 'queue-state.json', q)
e = read(R / 'evidence-ledger.json')
e['updated'] = stamp
e['confirmed'].append('N5/N6 table and native crops actually reviewed in report tab40. Main viewer tab31 shows new diagnostic while retaining119 model204649points/89cameras. Same52-pagePPT last slide updated with comparison; first51 ZIP content and render hashes unchanged. No physical quality pass or00deployment.')
e['artifacts'] += q['last_goal_turn']['evidence'][-2:]
save(R / 'evidence-ledger.json', e)
s = read(C / 'local-research-status.json')
s.update(updated_at=stamp, source_queue_updated=stamp, N5_N6_UI_verified=True, same_PPT_lastpage_updated=True)
s['viewer']['current_122_ui_scope'] = 'N5/N6 report and new diagnostic visually verified; retained119 model. R1e four synthetic recorded-camera renders verified separately.'
s['renderer_readiness']['stage'] = q['phase']
save(C / 'local-research-status.json', s)
print(json.dumps(dict(PPT_published=str(source), N5_N6_UI_verified=True, first51_unchanged=True, training=False), ensure_ascii=False))
