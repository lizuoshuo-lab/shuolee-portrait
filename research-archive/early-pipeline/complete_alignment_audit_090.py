from pathlib import Path
import json, hashlib, shutil, datetime
W=Path(__file__).parent
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/joint-calibration-model-audit-090'; V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8'); q.replace(p)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
latest=read(V/'latest-result.json'); meta=read(V/'model-whole-body-static-human-init.json')
assert latest['round']=='083' and latest['latestDiagnostic']==meta['latestDiagnostic']=='090'
assert sha(V/'models/whole-body-static-human-init/shuolee_083_s18000_f0100.ply')==latest['sha256']=='31845b1665f27c67ff2d2afa00cae488f3c7f8f31f364dcf25b902296263fb4f'
assert sha(B/'current/full-scene-localization-074/cameras.json')=='5fe1cf0afff5dc856958873895fbb7d6c626b656557226dcd1909cb14ba2eefe'
d=read(W/'human090-deck-receipt.json'); validation=read(W/'human090-validation.json'); src=Path(d['finalPath']); dst=Path(d['source'])
assert sha(src)==d['final_sha256']==validation['finalSha256'] and sha(dst)==d['source_sha256']
assert validation['packageIntegrity']['status']=='pass' and validation['firstPartyImport']['passed']
assert len(d['renders'])==52 and sum(not x['changed'] for x in d['renders'])==51
save(W/'human090-deck-visual-review.json',dict(updated=now,actually_inspected=True,changed_slide=52,all51_unchanged_render_hashes_verified=True,no_new_images_or_layouts=True,opened_in_PowerPoint=False,final_slide_sha256=sha(W/'human090-final-52.png')))
shutil.copy2(src,dst); assert sha(dst)==d['final_sha256']; d.update(canonical_replaced=True,updated=now); save(W/'human090-deck-receipt.json',d)
m=read(B/'reports/report-manifest.json'); entry=next(x for x in m['reports'] if x['key']=='reconstruction-optimization')
entry.update(sha256=sha(dst),bytes=dst.stat().st_size,local_validation=str(W/'human090-validation.json'),local_visual_review=str(W/'human090-deck-visual-review.json'),latest_scope='088至090：图像观测、逐帧焦距及共享相机模型均未给出可采用修正；继续原生训练投影核验。',current_local_report_deployed_to_00=False)
m.update(updated=now,last_local_revision=dict(path=str(dst),sha256=sha(dst),bytes=dst.stat().st_size,slides=52,remote_delivered=False,transfer_performed=False,updated=now)); save(B/'reports/report-manifest.json',m)
save(O/'ui-verification.json',dict(updated=now,actual_browser_verified=True,loaded_images=2,no_horizontal_overflow=True,current_gaussian_round='083',latest_diagnostic='090',new_file_import_test=False,existing083_file_bytes_unchanged=True,viewer_follow_latest=True,actual_report_href=meta['report'],actual_status=meta['status'],screenshots=[dict(path=str(O/n),sha256=sha(O/n)) for n in ['page-verified.png','viewer-current.png']]))
pub=read(O/'publication.json'); pub.update(actual_browser_verified=True,deck_updated=True,deck=str(dst),deck_sha256=sha(dst)); save(O/'publication.json',pub)
s=read(B/'work-status.json'); s['updated']=now; s['latest_diagnostic']['ui_verified']=True
s['latest_report']=dict(path=str(dst),sha256=sha(dst),changed_slides=[52],slides=52,updated=now,remote_deployed=False)
s['pending_report_delivery']=dict(status='local_updated_remote_deferred',source=str(dst),file=dst.name,sha256=sha(dst),bytes=dst.stat().st_size,remote_delivered=False)
save(B/'work-status.json',s)
decision=read(O/'decision.json'); decision['workflow_complete']=True; save(O/'decision.json',decision); shutil.copy2(O/'decision.json',V/'body-static-surface-current/decision.json')
save(O/'completion.json',dict(updated=now,round='090',workflow_complete=True,goal_complete=False,quality_accepted=False,root_cause_established=False,goal_turn_classification='progress',external_blocker=False,repeated_blocking_condition_count=0,candidate_adopted=False,gaussian_updates=0,camera_updates=0,new_gaussian_ply=False,current_gaussian_round='083',actual_browser_verified=True,deck=str(dst),deck_sha256=sha(dst),next_bounded_work=s['next_bounded_work'],remote_deployed=False))
for n in ['run_calibration_model_090.py','check_calibration_models_090.py','run_evaluate_calibration_090.sh','publish_alignment_audit_090.py','update_deck_090.mjs','complete_alignment_audit_090.py']:
    shutil.copy2(W/n,O/n)
print(json.dumps(dict(workflow_complete=True,goal_complete=False,diagnostic='090',deck_sha256=sha(dst))))
