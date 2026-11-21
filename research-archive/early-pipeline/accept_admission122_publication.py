from pathlib import Path
from zipfile import ZipFile
from datetime import datetime,timezone
import hashlib,json,shutil

H=Path(__file__).parent;T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';K=R/'video-update-20261027/output-intrinsics-audit';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'admission122-deck-receipt.json')
source,final=Path(receipt['source']),Path(receipt['finalPath']);canonical=T/'shuolee/reports/02_有限视角重建优化路线.pptx'
assert not receipt['canonical_replaced']
assert sha(source)==sha(canonical)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
with ZipFile(source) as a,ZipFile(final) as b:
    assert set(a.namelist())==set(b.namelist())
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
assert len(receipt['renders'])==52 and all(not s['changed'] for s in receipt['renders'][:51])
validation=read(H/'admission122-final-validation.json')
assert validation['finalSha256']==receipt['final_sha256'] and validation['packageIntegrity']['status']=='pass' and validation['presentationLayout']['finding_count']==0
assert all((K/n).is_file() for n in ['report-ui.png','viewer-ui.png'])
for destination in [canonical,source]:
    temp=destination.with_suffix('.replacement.pptx');assert not temp.exists()
    shutil.copy2(final,temp);assert sha(temp)==receipt['final_sha256'];temp.replace(destination)
receipt.update(accepted=stamp,canonical_replaced=True,canonical=str(canonical),delivery=str(source),changed_zip_members=changed,
    first51_members_and_render_hashes_identical=True,changed_slide_visually_reviewed=True,PowerPoint_or_WPS_runtime_verified=False,
    visual_review='Both PNGs viewed. Text fits and same six-column native crop stays byte-identical with unchanged geometry; title/conclusions/notes alone updated.')
save(H/'admission122-deck-receipt.json',receipt);save(K/'ppt-update.json',{k:v for k,v in receipt.items() if k!='renders'})
ui=dict(updated=stamp,report_tab='40',viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n4',
    actual_report_reviewed=True,latestDiagnostic_display_verified=True,model='new-capture-119-sofa-clothed',points=204649,cameras=89,
    viewer_version='2026.10.27.5',quality_accepted=False,historical_tab15_untouched=True,
    screenshots=[dict(file=f,sha256=sha(K/f)) for f in ['report-ui.png','viewer-ui.png']],
    remote00_verified=False,WPS_cloud_verified=False,training_started=False,new_PLY=False)
save(K/'viewer-ui-review.json',ui)
p=read(R/'publication.json');p.update(updated=stamp,N3_N4_UI_verified=True,N3_N4_UI_review='video-update-20261027/output-intrinsics-audit/viewer-ui-review.json',N3_N4_PPT_update='video-update-20261027/output-intrinsics-audit/ppt-update.json');save(R/'publication.json',p)
q=read(R/'queue-state.json');q['updated']=stamp
q['last_goal_turn']['remaining']='Finish actual R1e native UI validation; physical outputK/material/camera evidence and fresh training remain unaccepted'
q['last_goal_turn']['evidence']+=['video-update-20261027/output-intrinsics-audit/viewer-ui-review.json','video-update-20261027/output-intrinsics-audit/ppt-update.json'];save(R/'queue-state.json',q)
ledger=read(R/'evidence-ledger.json');ledger['updated']=stamp
ledger['confirmed'].append('N3/N4 report actually refreshed and viewed in tab40. Mainviewer tab31 shows current diagnostic and retained119 model204649points/89cameras. Same52pagePPT slide52 text/notes only updated; other51 pages and allimagebytes unchanged. No new person training/model or00verification.')
ledger['artifacts']+=q['last_goal_turn']['evidence'][-2:];save(R/'evidence-ledger.json',ledger)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,N3_N4_UI_verified=True,same_PPT_lastpage_updated=True);save(C/'local-research-status.json',s)
print(json.dumps(dict(PPT_published=str(source),N3_N4_UI_verified=True,first51_unchanged=True,training=False),ensure_ascii=False))
