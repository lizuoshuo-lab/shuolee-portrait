"""Accept the visually inspected slide52 and preserve the diagnostic UI status."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
F=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis/fan-observation'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,x):lp(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
receipt=H/'fan-observation123-deck-receipt.json';r=read(receipt)
assert not r['canonical_replaced']
assert r['changed_slides']==[52] and len(r['renders'])==52
assert all(not x['changed'] for x in r['renders'][:51])
assert read(H/'fan-observation123-exact-package.json')['other_members_byte_identical']
assert sha(r['source'])==r['source_sha256'] and sha(r['finalPath'])==r['final_sha256']
assert read(F/'generic-checks.json')['state']=='saved_output_and_native_adapter_checks_passed'
assert read(F/'viewer-ui-review.json')['actual_AX_result_and_table_verified']
shutil.copy2(r['finalPath'],r['source'])
shutil.copy2(r['finalPath'],T/'shuolee/reports'/Path(r['source']).name)
r.update(canonical_replaced=True,updated=stamp,actual_final_slide52_image_reviewed=True,
         first51_byte_and_render_checks_passed=True)
save(receipt,r)
p=F/'publication.json';v=read(p);v.update(updated=stamp,deck_pending=False,deck=r['source'],deck_sha256=r['final_sha256'],
    deck_receipt=str(receipt),only_slide52_changed=True,first51_byte_and_render_checks_passed=True,
    viewer_text_and_table_verified=True,viewer_screenshot_verified=False,viewer_UI_pending=True,
    viewer_UI_limitation='IAB screenshot returned0width. Native source figures were viewed; interface pixels remain unverified.')
save(p,v)
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    for name in ['assessment.json','publication.json','viewer-ui-review.json']:
        shutil.copy2(lp(F/name),dest/'fan-observation'/name)
p=H.parent/'shuolee-coordination/local-research-status.json';v=read(p);v.update(updated_at=stamp,
    current_report=r['source'],current_report_sha256=r['final_sha256'],latest_publication=str(F/'publication.json'),
    latest_viewer_ui_review=str(F/'viewer-ui-review.json'),viewer_text_and_table_verified=True,
    viewer_screenshot_verified=False,active_goal_remains=True)
save(p,v)
print(json.dumps(dict(canonical_sha256=sha(r['source']),changed_slides=[52],record_images=81,goal_complete=False)))
