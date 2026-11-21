"""Publish the reviewed existing deck and actual browser-text review; no experiment."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, zipfile

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
REC = T / 'shuolee/current/local-clothed-surface-123'
D = REC / 'smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis'
O = D / 'root-policy-control'
SYNC = Path('D:/00/工作/查看器')
def lp(p): return Path('\\\\?\\' + str(Path(p).resolve()))
def read(p): return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p, v): lp(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(p): return hashlib.sha256(lp(p).read_bytes()).hexdigest()

stamp = datetime.now(timezone.utc).isoformat()
receipt_path = H / 'root-policy123-reviewed-deck-receipt.json'
receipt = read(receipt_path)
candidate, canonical = Path(receipt['finalPath']), Path(receipt['source'])
assert sha(candidate) == receipt['final_sha256']
assert sha(canonical) == receipt['source_sha256'], 'Canonical changed; do not overwrite a newer deck.'
assert receipt['first51_render_hashes_identical']
assert receipt['actual_changed_slide_review']['reviewed']
with zipfile.ZipFile(candidate) as newer, zipfile.ZipFile(canonical) as older:
    assert set(newer.namelist()) == set(older.namelist())
    changed = sorted(n for n in newer.namelist() if newer.read(n) != older.read(n))
    assert changed == ['ppt/notesSlides/notesSlide52.xml', 'ppt/slides/slide52.xml'], changed
backup = H / ('prior-root-policy-deck-' + receipt['source_sha256'] + '.pptx')
if not backup.exists(): shutil.copy2(canonical, backup)
assert sha(backup) == receipt['source_sha256']
shutil.copy2(candidate, canonical)
report_mirror = T / 'shuolee/reports' / canonical.name
shutil.copy2(candidate, report_mirror)
assert sha(canonical) == sha(report_mirror) == receipt['final_sha256']
receipt.update(canonical_replaced=True, canonical_replaced_at=stamp, root_actual_slide52_image_review=True,
               root_verified_changed_zip_members=changed, canonical_sha256=sha(canonical),
               previous_deck_backup=str(backup), report_mirror=str(report_mirror))
save(receipt_path, receipt)

ui = dict(updated=stamp, tab_id='32', url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#root-policy-control',
          action='Reloaded the existing right-side browser tab after disk publication; marked deliverable.',
          actual_AX_review=True, actual_table_review=True,
          observed=dict(section='根位姿初始化对照：两种结果几乎一致', frames=[0,1,2],
                        held_face_max_px=[0.345,0.282,0.306], surface_RMSE_mm=[5.56,8.97,11.73],
                        synthetic_scope_explicit=True, numerical_convergence_not_claimed=True),
          screenshot_verified=False, screenshot_limitation='Browser screenshot API returned Cannot take screenshot with 0 width.',
          real_person_quality_passed=False, new_PLY=False, historical_model_not_replaced=True,
          tab15_untouched=True, company00_tested=False, WPS_cloud_sync_tested=False)
save(O/'viewer-ui-review.json', ui)
pub = read(O/'publication.json')
pub.update(updated=stamp, viewer_UI_pending=False, viewer_text_and_table_verified=True,
           viewer_screenshot_verified=False, viewer_ui_review=str(O/'viewer-ui-review.json'),
           deck_pending=False, PPT=str(canonical), PPT_sha256=sha(canonical),
           PPT_changed_slides=[52], PPT_first51_unchanged=True, PPT_receipt=str(receipt_path),
           root_reviewed_slide52_png=str(H/'root-policy123-reviewed-final-52.png'))
save(O/'publication.json', pub)
for dest in [R/'video10-global-initializer', T/'viewer/diagnostics/video10-global-initializer', SYNC/'diagnostics/video10-global-initializer']:
    for name in ['publication.json','viewer-ui-review.json']:
        shutil.copy2(lp(O/name), dest/'root-policy-control'/name)

p = H.parent/'shuolee-coordination/local-research-status.json'
v = read(p)
v.update(updated_at=stamp, latest_publication=str(O/'publication.json'),
         latest_viewer_ui_review=str(O/'viewer-ui-review.json'),
         current_report=str(canonical), current_report_sha256=sha(canonical),
         latest_image_count=81, latest_image_delivery_receipt=str(D/'fan-observation/publication.json'),
         viewer_text_and_table_verified=True, viewer_screenshot_verified=False,
         viewer_UI_image_acceptance=False, company_installation_unchanged=True)
save(p,v)
print(json.dumps(dict(canonical=str(canonical),sha256=sha(canonical),changed_zip_members=changed,
                     actual_browser_table_verified=True,screenshot_verified=False,new_PLY=False),ensure_ascii=False))
