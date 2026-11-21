from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,re

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
A=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/observation-contract'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
p=H/'smplify-observation123-deck-receipt.json';r=read(p)
assert not r['canonical_replaced'] and r['changed_slides']==[52]
assert all(not x['changed'] for x in r['renders'][:51])
assert sha(r['source'])==r['source_sha256'] and sha(r['finalPath'])==r['final_sha256']
assert read(H/'smplify-observation123-exact-package.json')['other_members_byte_identical']
shutil.copy2(r['finalPath'],r['source'])
shutil.copy2(r['finalPath'],T/'shuolee/reports'/Path(r['source']).name)
r.update(canonical_replaced=True,updated=stamp,actual_before_after_slide52_reviewed=True,figure_aspect_ratio_preserved=True)
save(p,r)
ui={'updated':stamp,'tab_id':'32','url':'http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#smplify-observation-contract',
    'actual_existing_tab_reloaded':True,'AX_new_section_verified':True,'AX_record_path_and43_images_verified':True,
    'source_evidence_images_actually_reviewed':True,'screenshot_verified':False,
    'screenshot_limitation':'Current IAB screenshot returned Cannot take screenshot with 0 width. No alternate capture path was used.',
    'marked_deliverable':True,'tab15_untouched':True,'new_PLY':False,
    'company00_remote_verified':False,'WPS_cloud_sync_verified':False}
save(A/'viewer-ui-review.json',ui)
p=A/'publication.json';q=read(p)
q.update(updated=stamp,actual_UI_review_pending=False,UI_AX_verified=True,UI_screenshot_verified=False,
         deck=r['source'],deck_sha256=r['final_sha256'],only_slide52_changed=True,
         first51_byte_and_render_checks_passed=True)
save(p,q)
destinations=[R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]
for d in destinations:
    target=d/'smplifyx-reference-control/observation-contract'
    for name in ['viewer-ui-review.json','publication.json']:
        shutil.copy2(A/name,target/name)
    page=(d/'index.html').read_text(encoding='utf-8')
    section=re.search(r'<section id="smplify-observation-contract">.*?</section>',page,flags=re.S).group()
    imgs=re.findall(r'<img[^>]+src="([^"]+)"',section)
    assert len(imgs)==9
    assert all((d/image).is_file() for image in imgs)
policy=H.parent/'shuolee-coordination/image-record-policy.json';v=read(policy)
v.update(updated=stamp,latest_verified_dataset='005',latest_image_count=43,
         latest_delivery_receipt=str(A/'record-delivery.json'),new_work_log_image_writes=False)
save(policy,v)
print(json.dumps({'record_images':43,'record_path':'D:/00/记录/005','image_assets_resolve_in_all3_viewers':True,
                  'only_slide52_updated':True,'gaussian_steps':0,'goal_active':True},ensure_ascii=False))
