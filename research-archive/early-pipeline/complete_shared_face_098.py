from pathlib import Path
import json,hashlib,datetime,shutil

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-shared-face-fresh-098';V=B.parent/'viewer'
W=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)

now=datetime.datetime.now().astimezone().isoformat()
deck=read(W/'human098-deck-receipt.json');canonical=Path(deck['source'])
assert deck['canonical_replaced'] and sha(canonical)==deck['final_sha256']
assert len(deck['renders'])==52 and deck['changed_slides']==[52]
assert all((R/name).is_file() for name in ['published-report.png','latest-viewer.png','standalone-gaussian.png','standalone-points.png'])
latest=read(V/'latest-result.json');assert latest['round']=='098' and not latest['qualityAccepted']
deck['canonical_path']=str(canonical);deck['canonical_sha256']=sha(canonical)
deck['temporary_duplicates_removed']=[str(W/'human098-candidate.pptx'),deck['finalPath']]
assert not any(Path(p).exists() for p in deck['temporary_duplicates_removed'])
save(W/'human098-deck-receipt.json',deck)
shutil.copy2(W/'human098-deck-receipt.json',R/'presentation-review.json')
shutil.copy2(W/'human098-validation.json',R/'presentation-validation.json')
proof=read(R/'publication.json');proof.update(updated=now,ui_report_verified=True,latest_viewer_verified=True,
    screenshots={name:dict(path=str(R/name),sha256=sha(R/name)) for name in ['published-report.png','latest-viewer.png']},
    report_model_link_verified='http://127.0.0.1:8870/?model=whole-body-shared-world',
    presentation=dict(path=str(canonical),sha256=sha(canonical),slides=52,changed_slides=[52],
                      import_and_render_verified=True,native_powerpoint_verified=False),
    quality_accepted=False,pc00_deployed=False)
save(R/'publication.json',proof)
e=read(R/'evaluation/result.json');e['actual_visual_review_pending']=False
e['visual_decision']=str(R/'decision.json');save(R/'evaluation/result.json',e)
shutil.copy2(R/'evaluation/result.json',V/'body-shared-current/evaluation.json')
state=read(B/'work-status.json')
state['previous_latest_diagnostic']=state.get('latest_diagnostic')
state['latest_diagnostic']=dict(round='098',path=str(R),status='full_training_reviewed_not_quality_accepted',
    camera_updates=0,shared_world_points=50,shared_point_definition_fully_satisfied=False,
    numerical_shared_target_reduction=True,visible_ghosting_fixed=False,
    report=proof['report'],ui_verified=True)
state['latest_local_report_revision']=dict(path=str(canonical),sha256=sha(canonical),slides=52,
    remote_delivered=False,local_import_and_visual_review_passed=True)
state['current_report_revision']=dict(path=str(canonical),sha256=sha(canonical),changed_slides=[52],
    first51_render_hashes_unchanged=True,validated=True,native_powerpoint_verified=False)
state['latest_user_clarification']='所有可见相机的同名面部特征，共同约束唯一全局三维点；不能仅优化两两深度或点到平面误差。'
state['local_training_running']=False;state['goal_accepted']=False;state['updated']=now
save(B/'work-status.json',state)
sources=R/'analysis-sources';manifest=read(sources/'manifest.json')
for src in [Path(__file__),W/'update_deck_098.mjs']:
    shutil.copy2(src,sources/src.name)
    manifest['files'].append(dict(source=str(src),copy=src.name,sha256=sha(src)))
save(sources/'manifest.json',manifest)
print(json.dumps(dict(round='098',report_verified=True,viewer_verified=True,ppt=str(canonical),quality_accepted=False,goal_complete=False)))
