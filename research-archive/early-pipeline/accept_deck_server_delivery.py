from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile
import hashlib, json, shutil

B=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
canonical=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx')
delivery=Path('D:/00/报告/02_有限视角重建优化路线.pptx')
r=json.loads((B/'server-delivery-deck-receipt.json').read_text(encoding='utf-8'))
final=Path(r['finalPath'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(canonical)==sha(delivery)==r['source_sha256']
assert sha(final)==r['final_sha256']
with ZipFile(canonical) as a,ZipFile(final) as b:
    assert a.namelist()==b.namelist()
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
    assert b.testzip() is None
assert len(r['renders'])==52 and r['changed_slides']==[52]
previous=json.loads((B/'human117-deck-receipt.json').read_text(encoding='utf-8'))
assert all(r['renders'][i]['render_sha256']==previous['renders'][i]['render_sha256'] for i in range(51))
backup=B/'server-delivery-source.pptx'
if backup.exists():assert sha(backup)==r['source_sha256']
else:shutil.copy2(canonical,backup)
for destination in [canonical,delivery]:
    temporary=destination.with_suffix('.replacement.pptx')
    assert not temporary.exists()
    shutil.copy2(final,temporary)
    assert sha(temporary)==r['final_sha256']
    temporary.replace(destination)
    assert sha(destination)==r['final_sha256']
r.update(accepted=datetime.now(timezone.utc).isoformat(),canonical_replaced=True,delivery=str(delivery),canonical=str(canonical),
    changed_zip_members=changed,first51_render_hashes_identical=True,changed_slide_visually_reviewed=True,
    image_unchanged=True,powerpoint_or_wps_execution_verified=False,private_source_backup=str(backup),
    review='Direct before/after PNG inspection: title, source images, body text and footer fit without clipping; only last slide text/notes changed.')
(B/'server-delivery-deck-receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
log=Path('D:/00/工作日志/2026-10-26/报告修订');log.mkdir(parents=True,exist_ok=True)
(log/'优化路线_009末页核验.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'accepted':str(delivery),'sha256':r['final_sha256'],'first51_unchanged':True},ensure_ascii=False))
