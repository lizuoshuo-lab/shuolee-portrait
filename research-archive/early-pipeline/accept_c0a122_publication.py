from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timezone
import hashlib, json, shutil

B=Path(__file__).parent
T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'; O=R/'camera-observability'
C=B.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
r=read(B/'c0a122-deck-receipt.json');source=Path(r['source']);final=Path(r['finalPath'])
canonical=T/'shuolee/reports/02_有限视角重建优化路线.pptx'
assert sha(source)==sha(canonical)==r['source_sha256']
assert sha(final)==r['final_sha256']
with ZipFile(source) as a,ZipFile(final) as b:
    assert a.namelist()==b.namelist()
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
assert len(r['renders'])==52 and all(not x['changed'] for x in r['renders'][:51])
for dest in [canonical,source]:
    assert sha(dest)==r['source_sha256']
    tmp=dest.with_suffix('.replacement.pptx');assert not tmp.exists()
    shutil.copy2(final,tmp);assert sha(tmp)==r['final_sha256'];tmp.replace(dest)
r.update(accepted=stamp,canonical_replaced=True,canonical=str(canonical),delivery=str(source),
         changed_zip_members=changed,first51_members_and_render_hashes_identical=True,
         changed_slide_visually_reviewed=True,
         visual_review='Actual before/after PNG inspected. Existing image/frame is explicitly historical; title, caption, two text lines and footer fit. No image changes or text burned into source image.',
         validation_first_failure='Finalizer subprocess missed RUNTIME_NODE_MODULES. Corrected runtime environment; no source PPT replaced until integrity/layout/font/import/render checks passed.',
         PowerPoint_or_WPS_runtime_verified=False)
save(B/'c0a122-deck-receipt.json',r)
save(O/'ppt-update.json',{k:v for k,v in r.items() if k!='renders'})
for name in ['report-ui.png','viewer-ui.png']: assert (O/name).is_file()
ui=dict(updated=stamp,report_url='http://127.0.0.1:8870/body-smplx.html#robust122-c0a',report_tab='40',viewer_tab='31',
    actual_report_content_and_table_viewed=True,latestDiagnostic_display_verified=True,
    preserved_model='new-capture-119-sofa-clothed',preserved_points=204649,preserved_cameras=89,viewer_version='2026.10.27.5',
    historical_tab15_untouched=True,model_changed=False,new_native_interactive_GUI=False,
    screenshot=['camera-observability/report-ui.png','camera-observability/viewer-ui.png'],
    remote00_verified=False,WPS_cloud_verified=False,training_started=False)
save(O/'viewer-ui-review.json',ui)
receipt=read(R/'publication.json');receipt.update(updated=stamp,C0a_browser_UI_verified=True,C0a_UI_review='camera-observability/viewer-ui-review.json',C0a_PPT_update='camera-observability/ppt-update.json')
save(R/'publication.json',receipt)
ledger=read(R/'evidence-ledger.json');ledger['updated']=stamp
for item in ['camera-observability/viewer-ui-review.json','camera-observability/ppt-update.json','native-package-loader-plan.json']:
    if item not in ledger['artifacts']:ledger['artifacts'].append(item)
ledger['confirmed'].append('C0a report and existing preview actually refreshed;119model/89cameras preserved. Canonical52pagePPT slide52 text/notes updated; first51 package members and render hashes identical.')
save(R/'evidence-ledger.json',ledger)
state=read(C/'local-research-status.json');state['updated_at']=stamp
state['camera_observability'].update(viewer_UI_verified=True,ppt_last_slide_updated=True)
state['company00_handoff_received']={'updated_at':stamp,'source':'company00-status.json','state':read(C/'company00-status.json')['state'],
    'scope':'00 user logs locate113 failure at head after prepare; Gaussian training not reached in that launcher. No local research change or server restart.'}
save(C/'local-research-status.json',state)
print(json.dumps({'published':str(source),'first51_unchanged':True,'actual_UI_checked':True,'training_started':False},ensure_ascii=False))
