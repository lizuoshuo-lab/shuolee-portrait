import pathlib,json,hashlib,datetime,shutil
B=pathlib.Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');O=B/'current/observed-static-face-surface-085';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');q.replace(p)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
p=read(O/'publication.json');e=read(O/'evaluation.json');check=read(O/'arap-interface-check.json')
assert check['all_passed'] and not e['all_gates_passed']
meta=read(V/'model-whole-body-static-human-init.json');latest=read(V/'latest-result.json')
assert meta['report']==latest['report']=='body-smplx.html#static-surface-fit'
assert meta['gaussianCount']==latest['points']==300220 and latest['round']=='083'
assert sha(V/'models/whole-body-static-human-init/shuolee_083_s18000_f0100.ply')==latest['sha256']=='31845b1665f27c67ff2d2afa00cae488f3c7f8f31f364dcf25b902296263fb4f'
assert sha(B/'reports/02_有限视角重建优化路线.pptx')==p['deck_sha256']=='3a440f657d3500b9a9298c0771c0647b01525ce7297e31292099b3085a2e85ff'
save(O/'ui-verification.json',dict(updated=now,actual_browser_verified=True,loaded_images=1,current_gaussian_round='083',new_file_import_test=False,
    existing083_file_bytes_unchanged=True,viewer_follow_latest=True,actual_report_href=meta['report'],
    screenshots=[dict(path=str(O/n),sha256=sha(O/n)) for n in ['page-verified.png','viewer-current.png']]))
p['actual_browser_verified']=True;save(O/'publication.json',p)
s=read(B/'work-status.json');s['updated']=now;s['latest_diagnostic']['ui_verified']=True
s['next_bounded_work']['interface_preflight']=dict(path=str(O/'arap-interface-check.json'),open3d_version='0.18.0',identity_passed=True,translation_passed=True,actual_subject_fit=False)
s['next_bounded_work']['do_not_repeat']+=' Official Open3D0.18.0 neutral identity and uniform translation checks already passed. Remaining preflight is image-track-to-handle mapping and the one fresh subject-fit protocol.'
save(B/'work-status.json',s)
save(O/'completion.json',dict(updated=now,round='085',workflow_complete=True,goal_turn_classification='progress',previous_turn_classification='progress',
    goal_complete=False,root_cause_established=False,quality_accepted=False,external_blocker=False,repeated_blocking_condition_count=0,
    new_gaussian_updates=0,camera_updates=0,new_gaussian_ply=False,current_gaussian_round='083',surface_fit_steps=4,surface_candidate_rejected=True,
    fresh_neutral_fit=True,admitted_tracks=26,fit_tracks=17,held_tracks=9,held_transfer_observations=118,
    arap_interface_checks_passed=2,arap_subject_fit_performed=False,actual_browser_verified=True,website=p['report_url'],
    deck=p['deck'],deck_sha256=p['deck_sha256'],next_bounded_work=s['next_bounded_work'],remote_deployed=False))
shutil.copy2(__file__,O/pathlib.Path(__file__).name)
print(json.dumps(dict(workflow_completed=True,goal_complete=False,new_ply=False,model_round='083',next='ARAP mapping and fresh static subject-fit protocol')))
