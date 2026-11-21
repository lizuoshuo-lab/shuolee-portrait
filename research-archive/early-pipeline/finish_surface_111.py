from pathlib import Path
from datetime import datetime
import json,hashlib,shutil,urllib.request
T=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/surface-point-alignment-111';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
r=read(T/'human111-deck-receipt.json');v=read(T/'human111-validation.json')
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
with urllib.request.urlopen('http://127.0.0.1:8870/latest-result.json',timeout=10) as q:
    live=json.load(q)
assert live['round']=='098' and live['latestDiagnostic']=='111' and live['sha256']==sha(model)
assert live['qualityAccepted'] is False and live['report']=='body-smplx.html#surface111-result'
stage=source.with_suffix('.staged.pptx');assert not stage.exists()
shutil.copy2(final,stage);assert sha(stage)==r['final_sha256'];stage.replace(source)
now=datetime.now().astimezone().isoformat()
r.update(canonical_replaced=True,canonical_sha256=sha(source),canonical_path=str(source));save(T/'human111-deck-receipt.json',r)
shutil.copy2(T/'human111-final-52.png',O/'slide52.png')
shutil.copy2(T/'human111-validation.json',O/'presentation-validation.json')
save(O/'presentation-review.json',dict(updated=now,canonical=str(source),sha256=sha(source),slides=52,
    changed_slides=[52],first51_render_hashes_unchanged=True,changed_slide_visually_inspected=True,
    actual_powerpoint_application_opened=False))
p=read(O/'publication.json');p.update(updated=now,ui_verified=True,screenshot=str(O/'published-alignment-review.png'),
    html_sha256=sha(V/'body-smplx.html'),presentation_pending=False,presentation_sha256=sha(source),
    actual_gaussian='098',latest_diagnostic='111',actual_model_sha256=sha(model),latest_manifest_http_verified=True,
    viewer_state='User changed prior latest tab to historical070; retained that selection. Report tab updated and visibly verified.',
    remote00_deployed=False)
save(O/'publication.json',p)
s=read(B/'work-status.json');assert s['latest_diagnostic']['round']=='111'
s['latest_report'].update(sha256=sha(source),updated=now);s['updated']=now;save(B/'work-status.json',s)
h=read(O/'handoff.json');h.update(updated=now,presentation_sha256=sha(source),publication_complete=True,
    presentation_pending=False,next_training_allowed=False,goal_complete=False,
    native_backend_result='Existing local gsplat 1.5.3 center-depth features fail exact slanted surface depth control. Do not wire as surface position.',
    next_step_is_concrete='Inspect official hbb1/diff-surfel-rasterization forward/backward intersection depth and filtered fallback; implement an isolated pixel-consistent sparse observation candidate, validate before training.',
    user_manual_historical_viewer_preserved=True);save(O/'handoff.json',h)
files=[T/n for n in ['probe_surface_backend_111.py','preflight_2dgs_111.py','plot_surface_evidence_111.py','publish_surface_111.py','update_deck_111.mjs',Path(__file__).name]]
save(O/'finalization-sources.json',dict(created=now,files=[dict(path=str(x),sha256=sha(x)) for x in files]))
print(json.dumps(dict(canonical=str(source),sha256=sha(source),report=p['report'],changed_slides=[52],
    actual_gaussian='098',training_started=False,quality_accepted=False,goal_complete=False),ensure_ascii=False))
