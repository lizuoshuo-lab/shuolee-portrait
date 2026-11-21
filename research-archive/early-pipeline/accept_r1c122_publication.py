from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timezone
import hashlib,json,shutil
H=Path(__file__).parent
T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-package-loader';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
r=read(H/'r1c122-deck-receipt.json');source=Path(r['source']);final=Path(r['finalPath'])
canonical=T/'shuolee/reports/02_有限视角重建优化路线.pptx'
assert sha(source)==sha(canonical)==r['source_sha256'] and sha(final)==r['final_sha256']
with ZipFile(source) as a,ZipFile(final) as b:
    assert a.namelist()==b.namelist()
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
assert len(r['renders'])==52 and all(not x['changed'] for x in r['renders'][:51])
for destination in [canonical,source]:
    tmp=destination.with_suffix('.replacement.pptx');assert not tmp.exists()
    shutil.copy2(final,tmp);assert sha(tmp)==r['final_sha256'];tmp.replace(destination)
r.update(accepted=stamp,canonical_replaced=True,canonical=str(canonical),delivery=str(source),
         changed_zip_members=changed,first51_members_and_render_hashes_identical=True,
         changed_slide_visually_reviewed=True,PowerPoint_or_WPS_runtime_verified=False,
         visual_review='Before/after slide52 PNGs inspected. Same historical material image/crop and fonts. Updated title and two lines fit without clipping. No source image editing.')
save(H/'r1c122-deck-receipt.json',r);save(O/'ppt-update.json',{k:v for k,v in r.items() if k!='renders'})
for name in ['report-ui.png','viewer-ui.png']:assert (O/name).is_file()
ui={'updated':stamp,'report_tab':'40','viewer_tab':'31','actual_report_reviewed':True,
    'report_url':'http://127.0.0.1:8870/body-smplx.html#robust122-r1c','latestDiagnostic_display_verified':True,
    'model':'new-capture-119-sofa-clothed','points':204649,'cameras':89,'quality_accepted':False,
    'viewer_version':'2026.10.27.5','historical_tab15_untouched':True,'new_native_interactive_GUI':False,
    'screenshot':['native-package-loader/report-ui.png','native-package-loader/viewer-ui.png'],
    'remote00_verified':False,'WPS_cloud_verified':False,'training_started':False}
save(O/'viewer-ui-review.json',ui)
receipt=read(R/'publication.json');receipt.update(updated=stamp,R1c_browser_UI_verified=True,R1c_UI_review='native-package-loader/viewer-ui-review.json',R1c_PPT_update='native-package-loader/ppt-update.json');save(R/'publication.json',receipt)
progress=read(O/'progress.json');progress.update(stage='completed_reviewed_published',updated=stamp,pid=None);save(O/'progress.json',progress)
ledger=read(R/'evidence-ledger.json');ledger['updated']=stamp
ledger['confirmed'].append('R1c report and existing119 preview actually refreshed,204649points/89cameras retained. Same52-pagePPT onlyslide52 text/notes updated; first51 package members and render hashes identical.')
ledger['artifacts']+=['native-package-loader/viewer-ui-review.json','native-package-loader/ppt-update.json'];save(R/'evidence-ledger.json',ledger)
state=read(C/'local-research-status.json');state['updated_at']=stamp
state['renderer_readiness'].update(native_package_UI_report_published=True,actual_native_interactive_UI=False,native_package_PPT_updated=True)
save(C/'local-research-status.json',state)
print(json.dumps({'published':str(source),'first51_unchanged':True,'UI_reviewed':True,'native_interactive_GUI':False,'training_started':False},ensure_ascii=False))
