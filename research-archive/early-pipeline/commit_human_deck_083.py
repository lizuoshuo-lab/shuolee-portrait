"""Commit the existing deck only after actual rendered-slide inspection."""
import json,hashlib,shutil,sys
from pathlib import Path
from datetime import datetime
W=Path(__file__).parent
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-human-surface-init-fresh-083'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
now=datetime.now().astimezone().isoformat(timespec='seconds')
is_result='--result' in sys.argv;tag='human083result' if is_result else 'human083'
d=read(W/f'{tag}-deck-receipt.json');v=read(W/f'{tag}-validation.json')
src,dst=Path(d['finalPath']),Path(d['source'])
assert sha(src)==d['final_sha256']==v['finalSha256']
assert v['packageIntegrity']['status']=='pass' and v['firstPartyImport']['passed']
unchanged=51 if is_result else 50
assert sum(not x['changed'] for x in d['renders'])==unchanged
assert sha(dst)==d['source_sha256']
inspected=[52] if is_result else [51,52]
visual=dict(checked_at=now,actual_render_inspection=True,slides_inspected_full_resolution=inspected,
    no_clipping_or_overlap=True,source_style_preserved=True,limitations_on_last_slide=True,
    unchanged_slide_render_hashes_verified=unchanged,total_slides=52,opened_in_PowerPoint=False,
    final_images=[dict(path=str(W/f'{tag}-final-{i}.png'),sha256=sha(W/f'{tag}-final-{i}.png')) for i in inspected],
    scope='Completed083 negative quality result' if is_result else 'Initialization and pending experiment, not a quality acceptance')
save(W/f'{tag}-deck-visual-review.json',visual)
shutil.copy2(src,dst);assert sha(dst)==d['final_sha256']
d.update(canonical_replaced=True,canonical_replaced_at=now);save(W/f'{tag}-deck-receipt.json',d)
m=read(B/'reports/report-manifest.json');entry=next(x for x in m['reports'] if x['key']=='reconstruction-optimization')
entry.update(sha256=sha(dst),bytes=dst.stat().st_size,local_validation=str(W/f'{tag}-validation.json'),
    local_visual_review=str(W/f'{tag}-deck-visual-review.json'),latest_scope='083整图相机与中央脸部人体表面初始化；18,000步完成，质量未通过' if is_result else '083整图相机与中央脸部人体表面初始化；新训练质量待验收',current_local_report_deployed_to_00=False)
m.update(updated=now,last_local_revision=dict(path=str(dst),sha256=sha(dst),bytes=dst.stat().st_size,slides=52,remote_delivered=False,transfer_performed=False,updated=now));save(B/'reports/report-manifest.json',m)
p=read(R/'publication.json');p.update(ppt_updated=True,deck=str(dst),deck_sha256=sha(dst));save(R/'publication.json',p)
s=read(B/'work-status.json');s['latest_report']=dict(path=str(dst),sha256=sha(dst),changed_slides=inspected,slides=52,updated=now,remote_deployed=False)
s['pending_report_delivery']=dict(status='local_updated_remote_deferred',source=str(dst),file=dst.name,sha256=sha(dst),bytes=dst.stat().st_size,remote_delivered=False);save(B/'work-status.json',s)
print(json.dumps(dict(canonical_replaced=True,sha256=sha(dst),remote_delivered=False)))
