from pathlib import Path
from datetime import datetime
import json,hashlib,shutil,urllib.request
W=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/surface-native-alignment-112';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)

r=read(W/'human112-readable-deck-receipt.json');v=read(W/'human112-readable-validation.json')
source,final=Path(r['source']),Path(r['finalPath'])
assert not r['canonical_replaced'] and sha(source)==r['source_sha256']
assert sha(final)==r['final_sha256']==v['finalSha256']
assert v['packageIntegrity']['status']=='pass' and v['presentationLayout']['exitCode']==0 and v['firstPartyImport']['passed']
assert r['changed_slides']==[52] and len(r['renders'])==52 and all(not x['changed'] for x in r['renders'][:51])
assert (O/'published-alignment-review.png').exists() and (O/'published-initialization-review.png').exists()
assert read(O/'official-fixed-result.json')['passed'] and read(O/'native-alignment-result.json')['passed']
assert read(O/'initial-normal-fixed-covariance-result.json')['covariance_rgb_preserved']
model=V/'models/whole-body-shared-world/shuolee_098_s18000_f0100.ply'
assert sha(model)=='37ec0b641f0f04672798dd7b66c832d217a0be4ef90de6d886f9f87a552178f9'
with urllib.request.urlopen('http://127.0.0.1:8870/latest-result.json',timeout=10) as response: live=json.load(response)
assert live['round']=='098' and live['latestDiagnostic']=='112' and live['sha256']==sha(model)
assert not live['qualityAccepted'] and live['report']=='body-smplx.html#surface112-result'
for asset in ['112-result.json','112-native-depth.json','112-alignment-controls.json','112-initialization.json','112-initial-normal.png']:
    with urllib.request.urlopen('http://127.0.0.1:8870/body-shared-current/'+asset,timeout=10) as response:
        assert response.status==200 and len(response.read())>0
stage=source.with_suffix('.staged.pptx');assert not stage.exists()
shutil.copy2(final,stage);assert sha(stage)==r['final_sha256'];stage.replace(source)
now=datetime.now().astimezone().isoformat()
r.update(canonical_replaced=True,canonical_sha256=sha(source),canonical_path=str(source));save(W/'human112-readable-deck-receipt.json',r)
shutil.copy2(W/'human112-readable-final-52.png',O/'slide52.png')
shutil.copy2(W/'human112-readable-validation.json',O/'presentation-validation.json')
save(O/'presentation-review.json',dict(updated=now,canonical=str(source),sha256=sha(source),slides=52,
    changed_slides=[52],first51_render_hashes_unchanged=True,changed_slide_visually_inspected=True,
    readable_revision='Expanded only slide52evidence image within vacant content region; typography and other objects retained',
    actual_powerpoint_application_opened=False))
p=read(O/'publication.json');p.update(updated=now,ui_verified=True,screenshot=str(O/'published-alignment-review.png'),
    image_screenshot=str(O/'published-initialization-review.png'),html_sha256=sha(V/'body-smplx.html'),
    presentation_pending=False,presentation_sha256=sha(source),actual_gaussian='098',latest_diagnostic='112',
    actual_model_sha256=sha(model),latest_manifest_http_verified=True,remote00_deployed=False,
    viewer_state='Existing report tab updated and verified; user-selected historical070 viewer retained')
save(O/'publication.json',p)
state=read(B/'work-status.json');assert state['latest_diagnostic']['round']=='112'
state['latest_report'].update(sha256=sha(source),updated=now);state['updated']=now;save(B/'work-status.json',state)
h=read(O/'handoff.json');h.update(updated=now,presentation_sha256=sha(source),publication_complete=True,
    presentation_pending=False,goal_complete=False,next_training_allowed=False,
    next_step_is_concrete='Implement fresh silhouette-surface normals/confidence and native3DGS sparse contribution bridge; retain same registered cameras and robust shared material-point uncertainty. Do not reuse learned098initialization or silently switch to2DGS.',
    current_goal_turn='progress',user_manual_historical_viewer_preserved=True)
save(O/'handoff.json',h)
files=[W/n for n in ['build_official_surface_112.py','fix_official_surface_112.py','publish_surface_112.py','update_deck_112.mjs',Path(__file__).name]]
save(O/'finalization-sources.json',dict(created=now,files=[dict(path=str(x),sha256=sha(x)) for x in files]))
print(json.dumps(dict(canonical=str(source),sha256=sha(source),report=p['report'],changed_slides=[52],
    actual_gaussian='098',new_training=False,quality_accepted=False,goal_complete=False),ensure_ascii=False))
