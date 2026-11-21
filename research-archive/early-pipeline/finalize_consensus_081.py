from pathlib import Path
from datetime import datetime
import hashlib, json, shutil

BUILD=Path(__file__).resolve().parent
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/static-observation-consensus-081';V=B.parent/'viewer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now().astimezone().isoformat(timespec='seconds')
assert not (O/'completion.json').exists()
receipt=read(BUILD/'consensus081-deck-receipt.json')
candidate=Path(receipt['finalPath']);canonical=Path(receipt['source'])
assert sha(candidate)==receipt['final_sha256']
assert sha(canonical)==receipt['source_sha256'],'Canonical report changed since import'
validation=read(BUILD/'consensus081-validation.json')
assert validation['finalSha256']==sha(candidate)
assert validation['packageIntegrity']['status']=='pass'
assert validation['presentationLayout']['finding_count']==0
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
browser=read(O/'browser-check.json')
assert len(browser['page']['images'])==4 and all(x['loaded'] for x in browser['page']['images'])
assert browser['model']['report']=='body-smplx.html#camera-residual'
assert browser['model']['model'].startswith('079 ') and browser['model']['following']
assert '质量尚未通过验收' in browser['model']['status']
ply=V/'models/whole-body-static-surface/shuolee_079_s18000_f0100.ply'
cam=B/'current/full-scene-localization-074/cameras.json'
assert sha(ply)=='9463d848c81c688732b14e1d87199e37cefb5532bc82a580c8e227c26a14d6bd'
assert sha(cam)=='5fe1cf0afff5dc856958873895fbb7d6c626b656557226dcd1909cb14ba2eefe'
latest=read(V/'latest-result.json');assert latest['round']=='079' and latest['diagnosticRound']=='081'
assert sha(V/'model-whole-body-static-surface.json')==latest['revision']
review=dict(checked=now,presentation_sha256=sha(candidate),inspected_slides=[52],
    render_path=str(BUILD/'consensus081-final-52.png'),render_sha256=sha(BUILD/'consensus081-final-52.png'),
    visual_findings=[],text_clipping_observed=False,images_preserved=True,
    other_51_slides_render_hash_unchanged=True,native_powerpoint_verified=False,native_wps_verified=False)
save(BUILD/'consensus081-deck-visual-review.json',review)
shutil.copyfile(candidate,canonical);assert sha(canonical)==receipt['final_sha256']
receipt.update(canonical_replaced=True,canonical_replaced_at=now);save(BUILD/'consensus081-deck-receipt.json',receipt)
mp=B/'reports/report-manifest.json';manifest=read(mp)
report=next(x for x in manifest['reports'] if x['key']=='reconstruction-optimization')
report.update(sha256=sha(canonical),bytes=canonical.stat().st_size,slides=52,
    local_validation=str(BUILD/'consensus081-validation.json'),
    local_visual_review=str(BUILD/'consensus081-deck-visual-review.json'),
    latest_scope='081静态观测覆盖与联合外参核验；人体表面初始化机制检查',
    current_local_report_deployed_to_00=False)
manifest['updated']=now
manifest['last_local_revision']=dict(path=str(canonical),sha256=sha(canonical),bytes=canonical.stat().st_size,
    slides=52,remote_delivered=False,transfer_performed=False,updated=now);save(mp,manifest)
shots=[O/'page-verified.png',O/'page-result-verified.png']
ui=dict(verified=now,page='http://127.0.0.1:8870/body-smplx.html#camera-residual',
    actual_browser_check=browser,current_model_unchanged=True,current_model_sha256=sha(ply),
    screenshots=[dict(path=str(p),sha256=sha(p)) for p in shots],pc00_tested_this_round=False)
save(O/'final-ui-verification.json',ui)
v=read(O/'joint-bundle/visual-manifest.json');v.update(visually_reviewed=True,checked=now,
    finding='Same unwarped original crop; joint BA does not eliminate nose and mouth projection mismatch. No new Gaussian rendering.');save(O/'joint-bundle/visual-manifest.json',v)
publication=read(O/'publication.json');publication.update(browser_verified=True,
    browser_verification=str(O/'final-ui-verification.json'),ppt_updated=True,ppt_path=str(canonical),ppt_sha256=sha(canonical),verified=now);save(O/'publication.json',publication)
completion=dict(completed=now,diagnostic_round='081',workflow_completed=True,quality_accepted=False,
    root_cause_established=False,physical_motion_proven=False,static_filter_accepted=False,joint_camera_accepted=False,
    source_camera_updates=0,gaussian_optimizer_updates=0,latest_gaussian_round='079',latest_gaussian_sha256=sha(ply),
    source_camera_sha256=sha(cam),browser_verified=True,ppt_updated=True,pc00_deployed=False,
    goal_complete=False,goal_turn_classification='progress',consecutive_blocked_turns=0,
    evidence=['protocol.json','result.json','joint-bundle/protocol.json','joint-bundle/result.json',
              'decision.json','source-route-review.json','final-ui-verification.json','publication.json'],
    next='Audit and freshly fit one shared human/head surface under existing full-image cameras before changing Gaussian initialization. No new training yet.')
save(O/'completion.json',completion)
sp=B/'work-status.json';status=read(sp);assert status['latest_diagnostic']['round']=='081'
status['updated']=now;status['latest_diagnostic'].update(completion=str(O/'completion.json'),browser_verified=True,ppt_updated=True)
status['local_training_running']=False;save(sp,status)
for p in [BUILD/'update_deck_081.mjs',Path(__file__)]:shutil.copyfile(p,O/'source'/p.name)
save(O/'source-manifest.json',dict(created=now,files=[dict(path=p.name,sha256=sha(p)) for p in sorted((O/'source').iterdir()) if p.is_file()]))
print(json.dumps(dict(completed=True,gaussian_round='079',diagnostic_round='081',ppt_sha256=sha(canonical),
    browser_verified=True,quality_accepted=False,goal_complete=False),ensure_ascii=False))
