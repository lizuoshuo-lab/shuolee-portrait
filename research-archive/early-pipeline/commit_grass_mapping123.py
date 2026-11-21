from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=T/'shuolee/current/local-clothed-surface-123';A=REC/'grass-exploratory/provenance-audit'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();p=H/'grass-mapping123-aspect-deck-receipt.json';r=read(p)
assert not r['canonical_replaced'] and r['changed_slides']==[52] and all(not z['changed'] for z in r['renders'][:51])
assert sha(Path(r['source']))==r['source_sha256'] and sha(Path(r['finalPath']))==r['final_sha256']
assert read(H/'grass-mapping123-aspect-exact-package.json')['other_members_byte_identical']
shutil.copy2(r['finalPath'],r['source']);shutil.copy2(r['finalPath'],T/'shuolee/reports'/Path(r['source']).name)
r.update(canonical_replaced=True,updated=stamp,actual_last_page_visual_review=True,figure_aspect_ratio_preserved=True)
save(p,r)
p=A/'publication.json';q=read(p);q.update(updated=stamp,deck=r['source'],deck_sha256=r['final_sha256'],deck_only_slide52_changed=True,first51_render_hashes_identical=True);save(p,q)
for d in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(p,d/'grass-exploratory/provenance-audit/publication.json')
for folder in [REC,R]:
    p=folder/'evidence-ledger.json';x=read(p);x['updated']=stamp
    if folder==REC:
        x['entries'].append(dict(id='grass_smplify_semantics_shared_position',updated=stamp,evidence=str(A/'mapping-audit/result.json'),single_component_control=str(A/'mapping-audit/mean-ablation-result.json'),
            finding='Not a fullSMPLify-X fit. Officialhand/foot/faceindices andcoordinatechains agree; directbodyjoints differfromCOCOregression. Singlecommonworldpelvis replacement changes15.34to78.77px; missingjaw/expression andrestrictedhandfit confirmed.',
            physical_root_cause_unique=False,gaussian_steps=0))
    else:
        for f in ['mapping-audit/result.json','mapping-audit/mean-ablation-result.json','mapping-audit/visual-review.json','viewer-ui-review.json']:
            if str(A/f) not in x['artifacts']:x['artifacts'].append(str(A/f))
    save(p,x)
print(json.dumps(dict(deck=r['source'],only_slide52_changed=True,UI_updated=True,gaussian_steps=0,goal_complete=False),ensure_ascii=False))
