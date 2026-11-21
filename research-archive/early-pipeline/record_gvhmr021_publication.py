from pathlib import Path
from datetime import datetime
import hashlib
import json
import urllib.request

BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
BUILD=Path(__file__).resolve().parent
PHASE=BASE/'current/smplx-gvhmr-articulation-fresh-021'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2),'utf-8')
receipt=read(BUILD/'gvhmr021-deck-receipt.json')
canonical=Path(receipt['source'])
assert receipt['canonical_replaced'] and len(receipt['unchanged_rendered_pixels'])==54
assert sha(canonical)==receipt['final_sha256']
assert (BUILD/'gvhmr021-final-renders/55.png').is_file()
assert (BUILD/'gvhmr021-page-proof.png').is_file()
cleanup=read(PHASE/'publication-cleanup.json')
assert cleanup['state']=='completed'
urls=['body-smplx.html']+['body-fitting-method/gvhmr-fitting/'+p.name for p in (BASE.parent/'viewer/body-fitting-method/gvhmr-fitting').iterdir() if p.is_file()]
for relative in urls:
    with urllib.request.urlopen('http://127.0.0.1:8870/'+relative,timeout=10) as r:
        assert r.status==200
now=datetime.now().astimezone().isoformat(timespec='seconds')
publication=dict(created=now,page='http://127.0.0.1:8870/body-smplx.html#gvhmr-fitting-021',
    page_file=str(BASE.parent/'viewer/body-smplx.html'),
    browser_screenshot=str(BUILD/'gvhmr021-page-proof.png'),
    canonical_presentation=str(canonical),presentation_sha256=sha(canonical),
    changed_slides=[55],preserved_slides=list(range(1,55)),
    validation_receipt=str(BUILD/'gvhmr021-final-validation.json'),
    rendering_boundary='All55 slides rendered; first54 XML/media/notes and rendered pixels unchanged; slide55 visually reviewed. Native PowerPoint not executed.',
    cleanup={'manifest':str(PHASE/'publication-cleanup.json'),'files':cleanup['total_files'],'bytes':cleanup['total_bytes']},
    verified_http_200=urls,body_geometry_accepted=False,new_body_fit_updates=2800,
    new_gaussian_updates=0,remote_deployed=False,viewer_model_replaced=False)
save(PHASE/'publication.json',publication)
status_path=BASE/'work-status.json'
status=read(status_path)
status['updated']=now
status['gvhmr_articulation_021']['publication']=publication
status['latest_report']=dict(updated=now,canonical_path=str(canonical),final_sha256=sha(canonical),
    final_bytes=canonical.stat().st_size,slides=55,changed_slides=[55],all55_rendered=True,
    first54_pixel_identical=True,slide55_visually_reviewed=True,nativePowerPointVerified=False,
    validated_output=str(canonical))
status['latest_local_execution']['state']='reviewed_and_published_partial_not_accepted'
status['latest_local_execution']['running']=False
save(status_path,status)
print(json.dumps({'canonical_sha256':sha(canonical),'http_assets_verified':len(urls),'duplicate_files_removed':cleanup['total_files'],'body_fit_updates':2800,'new_gaussian_updates':0,'accepted':False},ensure_ascii=False))
