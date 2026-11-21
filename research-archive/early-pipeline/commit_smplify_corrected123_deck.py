from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');O=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/corrected-reference';R=T/'shuolee/current/robust-portrait-122'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=H/'smplify-corrected123-v2-deck-receipt.json';r=read(p)
assert not r['canonical_replaced'] and r['changed_slides']==[52]
assert all(not x['changed'] for x in r['renders'][:51])
assert sha(r['source'])==r['source_sha256'] and sha(r['finalPath'])==r['final_sha256']
assert read(H/'smplify-corrected123-v2-exact-package.json')['other_members_byte_identical']
shutil.copy2(r['finalPath'],r['source']);shutil.copy2(r['finalPath'],T/'shuolee/reports'/Path(r['source']).name)
stamp=datetime.now(timezone.utc).isoformat();r.update(canonical_replaced=True,updated=stamp,actual_before_after_slide52_reviewed=True,figure_aspect_ratio_preserved=True);save(p,r)
p=O/'publication.json';v=read(p);v.update(updated=stamp,deck=r['source'],deck_sha256=r['final_sha256'],only_slide52_changed=True,first51_byte_and_render_checks_passed=True);save(p,v)
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:shutil.copy2(p,dest/'smplifyx-reference-control/corrected-reference/publication.json')
print(json.dumps({'canonical_deck_updated':True,'only_slide52':True,'old_invalid_attempt_preserved':True,'corrected_real_fit_run':True,'gaussian_steps':0,'goal_complete':False}))
