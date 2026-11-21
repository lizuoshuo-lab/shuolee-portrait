from pathlib import Path
import json,hashlib,datetime,shutil
B=Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/barycentric-arap-face-086';V=B.parent/'viewer';W=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');q.replace(p)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
e=read(O/'evaluation.json');fit=read(O/'surface-fit-result.json');check=read(O/'adapter-check.json');p=read(O/'publication.json')
assert check['all_passed'] and fit['finished_prescribed_steps'] and not e['all_gates_passed']
assert fit['flipped_triangles']==0 and e['held_track_ratio']<.8 and e['held_nosemouth_ratio']>1
latest=read(V/'latest-result.json');meta=read(V/'model-whole-body-static-human-init.json')
assert latest['round']=='083' and latest['latestDiagnostic']==meta['latestDiagnostic']=='086'
assert latest['report']==meta['report']=='body-smplx.html#static-surface-fit'
assert sha(V/'models/whole-body-static-human-init/shuolee_083_s18000_f0100.ply')==latest['sha256']=='31845b1665f27c67ff2d2afa00cae488f3c7f8f31f364dcf25b902296263fb4f'
assert sha(B/'current/full-scene-localization-074/cameras.json')=='5fe1cf0afff5dc856958873895fbb7d6c626b656557226dcd1909cb14ba2eefe'
assert sha(B/'reports/02_有限视角重建优化路线.pptx')==p['deck_sha256']=='20a893ec720cd6655c4bd38f8f824eaff2172a76df59c09a83b990f864368453'
save(O/'ui-verification.json',dict(updated=now,actual_browser_verified=True,loaded_images=2,current_gaussian_round='083',latest_diagnostic='086',
    new_file_import_test=False,existing083_file_bytes_unchanged=True,viewer_follow_latest=True,actual_report_href=meta['report'],
    actual_status='083静态高斯：鼻口重影仍未解决。最新086几何检查未通过，未替换点云。',
    screenshots=[dict(path=str(O/n),sha256=sha(O/n)) for n in ['page-verified.png','viewer-current.png']]))
p['actual_browser_verified']=True;save(O/'publication.json',p)
s=read(B/'work-status.json');s['updated']=now;s['latest_diagnostic']['ui_verified']=True;save(B/'work-status.json',s)
d=read(O/'decision.json');d['workflow_complete']=True;save(O/'decision.json',d);shutil.copy2(O/'decision.json',V/'body-static-surface-current/decision.json')
save(O/'source-references.json',dict(updated=now,sources=[
    dict(url='https://github.com/isl-org/Open3D/blob/v0.18.0/cpp/open3d/geometry/TriangleMeshDeformation.cpp',use='Spokes ARAP local rotations and global Laplacian equations; code read and native nonrigid comparison performed'),
    dict(url='https://github.com/isl-org/Open3D/blob/v0.18.0/cpp/open3d/geometry/TriangleMesh.cpp',use='ComputeEdgeWeightsCot averages opposite-angle cotangents and clamps negative result to0'),
    dict(local='flame_model/flame.py',use='Official full landmark face indices and barycentric embedding verified against native forward; no licensed model published')]))
save(O/'completion.json',dict(updated=now,round='086',workflow_complete=True,goal_turn_classification='progress',previous_turn_classification='progress',
    goal_complete=False,quality_accepted=False,root_cause_established=False,external_blocker=False,repeated_blocking_condition_count=0,
    fresh_neutral_fit=True,actual_surface_fit=True,surface_fit_steps=50,candidate_adopted=False,
    arap_adapter_checks_passed=2,held_track_error_before=10.204929052982767,held_track_error_after=6.1599159711479885,
    held_nosemouth_error_before=7.137710828257409,held_nosemouth_error_after=8.49400614602202,
    new_gaussian_updates=0,camera_updates=0,new_gaussian_ply=False,current_gaussian_round='083',
    named_landmark_audit_complete=True,actual_browser_verified=True,website=p['report_url'],
    deck=p['deck'],deck_sha256=p['deck_sha256'],next_bounded_work=s['next_bounded_work'],remote_deployed=False))
for name in ['run_barycentric_arap_086.sh','review_barycentric_arap_086.sh','audit_anatomical_086.sh','review_handle_sources_086.sh','complete_barycentric_arap_086.py']:
    shutil.copy2(W/name,O/name)
print(json.dumps(dict(workflow_completed=True,goal_complete=False,new_ply=False,model_round='083',diagnostic_round='086',quality_accepted=False)))
