from pathlib import Path
import datetime,hashlib,json,shutil
W=Path('C:/Users/Administrator/Documents/New project');B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/shared-point-bundle-convergence-101';build=W/'shuolee-slide-update'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
r=read(build/'human101-deck-receipt.json');v=read(build/'human101-validation.json')
canonical=Path(r['source']);final=Path(r['finalPath']);now=datetime.datetime.now().astimezone().isoformat()
assert sha(canonical)==r['source_sha256'] and sha(final)==r['final_sha256']==v['finalSha256']
assert r['changed_slides']==[52] and all(not q['changed'] for q in r['renders'][:51])
assert v['packageIntegrity']['status']=='pass' and v['presentationLayout']['finding_count']==0
shutil.copy2(final,canonical);assert sha(canonical)==r['final_sha256']
r.update(canonical_replaced=True,canonical_sha256=sha(canonical),actual_slide52_visual_review=True,first51_render_hashes_unchanged=True,
    reviewed_at=now,native_powerpoint_tested=False)
save(build/'human101-deck-receipt.json',r)
shutil.copy2(build/'human101-final-52.png',O/'presentation-slide52.png')
shutil.copy2(build/'human101-validation.json',O/'presentation-validation.json')
save(O/'presentation-review.json',r)
p=read(O/'publication.json');assert (O/'published-report.png').stat().st_size>10000
p.update(ui_verified=True,verified_at=now,report_tab=2,latest_viewer_tab=15,
    actual_latest_status='101 shared points convergence with residuals;098 remains310961points',
    screenshot=str(O/'published-report.png'),presentation=str(canonical),presentation_sha256=sha(canonical))
save(O/'publication.json',p)
state=read(B/'work-status.json');state.update(updated=now,local_training_running=False)
state['latest_diagnostic']['report_url']=p['report']
state['latest_diagnostic']['ui_verified']=True
state['latest_diagnostic']['presentation_updated']=True
state['next_bounded_work'].update(
    first='Design a finite3Dsupport check around each unique Xk. Use all actual native contributor weights; compare stable normal/slanted support and separated-layer counterexamples before deciding any loss. Existing point-plane and center-depth substitutes are both insufficient.',
    evidence='100nativeRGB/oldgradientparity passed but slantedbias P90/targetsigma1.984>1: rejected, no newGaussian.101explicit Ceres tolerances converge41iterations; solution nearlyidentical099, residualfloor remains.',
    freeze='Do not change074/075cameras or098Gaussians based solely on101solver convergence. Retain no-dynamic, no-glasses-specialist, fresh200k/18k, all175registered cameras, foregroundRGB, wholebody/novelview/standalonePLY gates.',
    no_running_experiment=True)
save(B/'work-status.json',state)
save(O/'handoff.json',dict(created=now,completed=['100finite center-depth numerical candidate rejected before training','101shared-point identity proved and default stopping configuration diagnosed','101fresh3arm jointBA solved with same100iteration budget','existingwebsite and canonicalPPT updated and actually reviewed'],
    no_running_experiment=True,latest_gaussian_round='098',camera_source_unchanged=True,quality_accepted=False,goal_complete=False,
    important={'tolerance_fix_is_not_accuracy_gain':True,'same_numeric_input_as099':True,'pyceres_isolated':'/opt/gs-slam-lab/shuolee-bundle-audit/pyceres-2.3'},next=state['next_bounded_work']))
print(json.dumps(dict(canonical=str(canonical),sha256=sha(canonical),report=p['report'],new_gaussian=False)))
