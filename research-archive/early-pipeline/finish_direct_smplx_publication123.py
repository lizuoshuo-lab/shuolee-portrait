from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';OUT=REC/'direct-smplx-bindings'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
receipt_path=H/'direct-smplx123-v2-deck-receipt.json';d=read(receipt_path)
source=Path(d['source']);candidate=Path(d['finalPath'])
assert not d['canonical_replaced'] and sha(source)==d['source_sha256'] and sha(candidate)==d['final_sha256']
assert d['changed_slides']==[52] and len(d['renders'])==52
backup=H/'direct-smplx123-previous-canonical.pptx';assert not backup.exists();shutil.copy2(source,backup)
shutil.copy2(candidate,source)
mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';assert mirror.exists();shutil.copy2(candidate,mirror)
assert sha(source)==sha(mirror)==d['final_sha256']
d.update(canonical_replaced=True,canonical_copy=stamp,actual_last_slide_visual_review=True,
         aspect_ratio_corrected_before_promotion=True,previous_canonical_backup=str(backup));save(receipt_path,d)
p=OUT/'publication.json';q=read(p)
dom=(OUT/'browser-report.txt').read_text(encoding='utf-8')
expanded=(OUT/'browser-bindings-expanded.txt').read_text(encoding='utf-8')
assert 'direct-2d-3d' in dom and '0.04482' in dom and 'nose4' in expanded and 'right_eye1' in expanded
q.update(updated=stamp,diagnostic_DOM_verified=True,actual_binding_table_expanded=True,
    screenshot_verified=False,screenshot_limit='Previously documented zero-width browser screenshot failure; no retry or bypass.',
    canonical_PPT_updated=True,canonical_PPT=str(source),canonical_PPT_sha256=d['final_sha256'],
    PPT_first51_bytes_and_renders_preserved=True,PPT_last_slide_actual_reviewed=True)
save(p,q)
for name in ['visual-review.json','result.json','input-contract-result.json','bindings.json','candidate-manifest.json','plan.json','publication.json']:
    dest=Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐/direct-smplx-bindings')
    shutil.copy2(OUT/name,dest/name)
p=REC/'next-action.json';q=read(p);q.update(updated=stamp,diagnostic_DOM_verified=True,actual_UI_verified=False,
    UI_scope='Report contents and expanded bindings only; no new real cloud loaded.',
    publication=str(OUT/'publication.json'));save(p,q)
print(json.dumps(dict(canonical_PPT=str(source),first51_preserved=True,report_DOM_verified=True,
    synthetic_fit_only=True,real_subject_quality_accepted=False),ensure_ascii=False))
