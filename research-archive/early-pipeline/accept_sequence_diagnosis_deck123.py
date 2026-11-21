from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();rp=H/'sequence-diagnosis123-fixed-deck-receipt.json';r=read(rp)
assert not r['canonical_replaced']
assert sha(r['source'])==r['source_sha256'] and sha(r['finalPath'])==r['final_sha256']
assert len(r['renders'])==52 and all(not x['changed'] for x in r['renders'][:51])
assert read(H/'sequence-diagnosis123-fixed-exact-package.json')['other_members_byte_identical']
shutil.copy2(r['finalPath'],r['source']);shutil.copy2(r['finalPath'],T/'shuolee/reports'/Path(r['source']).name)
r.update(canonical_replaced=True,updated=stamp,actual_final_slide52_image_reviewed=True);save(rp,r)
p=D/'publication.json';q=read(p);q.update(updated=stamp,deck_sha256=r['final_sha256'],deck_receipt=str(rp),
    final_fix_status_in_deck=True,viewer_post_fix_text_verified=True);save(p,q)
p=D/'viewer-ui-review.json';q=read(p);q.update(updated=stamp,actual_post_fix_AX_state_verified=True);save(p,q)
for dest in [T/'shuolee/current/robust-portrait-122/video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    for name in ['publication.json','viewer-ui-review.json']:shutil.copy2(D/name,dest/'shared-sequence-diagnosis'/name)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p);q.update(updated_at=stamp,current_report_sha256=r['final_sha256']);save(p,q)
print(json.dumps(dict(canonical_sha256=sha(r['source']),changed_slides=[52],goal_complete=False)))
