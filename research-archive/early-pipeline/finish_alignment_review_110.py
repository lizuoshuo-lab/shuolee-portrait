from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
T=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/material-support-gradient-110';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
r=read(T/'human110-review-deck-receipt.json');v=read(T/'human110-review-validation.json')
source,final=Path(r['source']),Path(r['finalPath'])
assert not r['canonical_replaced'] and sha(source)==r['source_sha256']
assert sha(final)==r['final_sha256']==v['finalSha256']
assert v['packageIntegrity']['status']=='pass' and v['presentationLayout']['exitCode']==0 and v['firstPartyImport']['passed']
assert r['changed_slides']==[52] and len(r['renders'])==52
assert all(not x['changed'] for x in r['renders'][:51]) and r['renders'][51]['changed']
assert (O/'published-alignment-review.png').is_file()
assert not read(O/'decision.json')['training_admitted']
model=V/'models/whole-body-shared-world/shuolee_098_s18000_f0100.ply'
assert sha(model)=='37ec0b641f0f04672798dd7b66c832d217a0be4ef90de6d886f9f87a552178f9'
stage=source.with_suffix('.staged.pptx');assert not stage.exists()
shutil.copy2(final,stage);assert sha(stage)==r['final_sha256'];stage.replace(source)
now=datetime.now().astimezone().isoformat()
r.update(canonical_replaced=True,canonical_sha256=sha(source),canonical_path=str(source));save(T/'human110-review-deck-receipt.json',r)
shutil.copy2(T/'human110-review-final-52.png',O/'slide52.png')
shutil.copy2(T/'human110-review-validation.json',O/'presentation-validation.json')
save(O/'presentation-review.json',dict(updated=now,canonical=str(source),sha256=sha(source),slides=52,
    changed_slides=[52],first51_render_hashes_unchanged=True,changed_slide_visually_inspected=True,
    actual_powerpoint_application_opened=False,independent_review_incorporated=True))
p=read(O/'publication.json');p.update(updated=now,ui_verified=True,screenshot=str(O/'published-alignment-review.png'),
    presentation_pending=False,presentation_sha256=sha(source),independent_review_complete=True,
    actual_gaussian='098',latest_diagnostic='110',actual_model_sha256=sha(model))
save(O/'publication.json',p)
s=read(B/'work-status.json');assert s['latest_diagnostic']['round']=='110'
s['latest_report'].update(sha256=sha(source),updated=now);s['updated']=now;save(B/'work-status.json',s)
h=read(O/'handoff.json');h.update(updated=now,presentation_sha256=sha(source),publication_complete=True,
    next_training_allowed=False,goal_complete=False);save(O/'handoff.json',h)
files=[T/n for n in ['check_support_target_bias_110.py','publish_alignment_review_110.py','update_deck_alignment_review_110.mjs',Path(__file__).name]]
save(O/'review-finalization-sources.json',dict(created=now,files=[dict(path=str(p),sha256=sha(p)) for p in files]))
print(json.dumps(dict(canonical=str(source),sha256=sha(source),report=p['report'],changed_slides=[52],
    actual_gaussian='098',training_started=False,quality_accepted=False),ensure_ascii=False))
