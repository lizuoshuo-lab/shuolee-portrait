from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
OUT=REC/'field-preflight/native-material/topology-review/background-full-review'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();rp=H/'field-background123-deck-receipt.json';d=read(rp)
dom=(OUT/'browser-report.txt').read_text(encoding='utf-8')
images=(OUT/'browser-images-expanded.txt').read_text(encoding='utf-8')
assert '102个原生观测' in dom and '109602' in dom and 'no_reviewed_static_background_FIT_tracks' in dom
assert images.count('image 背景轨迹')==10
source=Path(d['source']);candidate=Path(d['finalPath'])
assert not d['canonical_replaced'] and sha(source)==d['source_sha256'] and sha(candidate)==d['final_sha256']
assert d['changed_slides']==[52] and len(d['renders'])==52
backup=H/'field-background123-previous-canonical.pptx';assert not backup.exists()
shutil.copy2(source,backup);shutil.copy2(candidate,source)
mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';assert mirror.exists();shutil.copy2(candidate,mirror)
assert sha(source)==sha(mirror)==d['final_sha256']
d.update(canonical_replaced=True,updated=stamp,actual_last_slide_reviewed=True,previous_canonical_backup=str(backup),first51_renders_and_package_preserved=True);save(rp,d)
p=OUT/'publication.json';q=read(p);q.update(updated=stamp,diagnostic_DOM_verified=True,actual_images_disclosure_expanded=True,
    screenshot_verified=False,screenshot_limit='Known zero-width browser screenshot failure; no retry or bypass. All10 original evidence sheets and final changed slide inspected locally.',
    canonical_PPT_updated=True,canonical_PPT=str(source),canonical_PPT_sha256=d['final_sha256'],PPT_first51_bytes_and_renders_preserved=True,PPT_last_slide_actual_reviewed=True)
save(p,q)
p=REC/'evidence-ledger.json';q=read(p);entry=next(e for e in q['entries'] if e.get('id')=='field_background_complete_observation_review');entry.update(report_DOM_verified=True,PPT_updated=True,actual_training_or_model_UI_verified=False);save(p,q)
p=REC/'next-action.json';q=read(p);q.update(updated=stamp,diagnostic_DOM_verified=True,actual_UI_verified=False,UI_scope='Existing diagnostic text, full10-row review table and expanded10 evidence sheets; no new fitted mesh or Gaussian cloud.',publication=str(OUT/'publication.json'),publication_next=None);save(p,q)
for dest in [T/'shuolee/current/robust-portrait-122/video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(OUT/'publication.json',dest/'field-background-full/publication.json')
print(json.dumps(dict(canonical_PPT=str(source),PPT_first51_unchanged=True,reviewed_observations=102,actual_report_DOM_checked=True,real_SMPLX_fit_executed=False,training_started=False,quality_accepted=False),ensure_ascii=False))
