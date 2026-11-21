"""Close this evidence-publication cycle, leaving the reconstruction goal active."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
from audit_grayT123_camera_support import REC,ROOT,read,save,sha
H=Path(__file__).parent;T=REC.parents[2];C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
stamp=datetime.now(timezone.utc).isoformat()
con=read(REC/'camera-support-conclusion.json');ppt=read(H/'surface123-camera-support-deck-receipt.json')
assert ppt['canonical_replaced'] and sha(Path(ppt['canonical_path']))==ppt['final_sha256']
assert not (ROOT/'gaussian-run').exists()
latest=read(T/'viewer/latest-result.json');sync=read(Path('D:/00/工作/查看器/latest-result.json'))
# Local and synced viewers deliberately use different progress entry filenames.
assert {k:v for k,v in latest.items() if k not in ['latestDiagnostic','report','updated']}=={k:v for k,v in sync.items() if k not in ['latestDiagnostic','report','updated']}
assert {k:v for k,v in latest['latestDiagnostic'].items() if k!='reportUrl'}=={k:v for k,v in sync['latestDiagnostic'].items() if k!='reportUrl'}
assert latest['latestDiagnostic']['reportUrl']=='body-smplx.html#camera-support'
assert sync['latestDiagnostic']['reportUrl']=='progress.html#camera-support'
ui=dict(updated=stamp,actual_UI_review=True,progress_tab='40',viewer_tab='31',
    progress_screenshot=str(REC/'grayT-camera-support-ui.png'),main_viewer_screenshot=str(REC/'grayT-support-main-viewer-ui.png'),
    observed='Progress page shows all9point-position crops and new evidence; mainviewer shows WhiteT309523Gaussians93cameras with the new diagnostic entry.',
    new_Gaussian_model_published=False,quality_accepted=False,company00_verified=False,
    diagnostic=latest['latestDiagnostic'])
save(REC/'grayT-camera-support-ui-review.json',ui)
for p in [REC/'grayT-camera-support-ui.png',REC/'grayT-support-main-viewer-ui.png',REC/'grayT-camera-support-ui-review.json']:
    shutil.copy2(p,L/p.name)
next_plan=dict(created=stamp,name='Frozen grayT material identity and static consistency follow-up',status='planned_not_executed',
    hypothesis='Some genuine face material observations exist before final camera selection. Their static consistency may differ from RTMW semantic landmarks or from background; current evidence does not identify the unique cause.',
    stage1='Use the same saved25FIT and3HELD whole-face-region components. Finish physical point/occlusion review without reading residuals for selection; all flags and failures remain in the evidence.',
    stage2='For the original HELD components use already saved alternate-anchor/check 3D projections. Do not recompute cameras or relabel FIT as independent. Compare actual identity-supported face checks with separately reviewed static-scene checks; report coverage and uncertainty.',
    optional_point_only_diagnostic='Only after a separate explicit plan, fit diagnostic shared3D points for the original FIT material candidates with K/E frozen. This would not be a camera candidate, training initializer or independent validation.',
    budget=dict(no_feature_extraction=True,no_new_BA=True,no_new_K=True,no_training=True,CPU_seconds=120,full_review_max_components=28),
    stopping_rules=['If identity or staticity cannot be established, report this lack of evidence; do not delete high errors or claim <1px.',
                    'If saved HELD material already disproves static consistency, preserve that result before any new model intervention.',
                    'Do not restart failed round123 supervisor or resume any grayT training.'],
    sources=dict(attrition=str(REC/'grayT-track-attrition/result.json'),support=str(REC/'grayT-camera-support-audit/result.json')))
save(REC/'grayT-material-followup-plan.json',next_plan)
handoff=dict(updated=stamp,goal_active=True,training_running=False,worker_pid=None,current_publication_worker=None,
    conclusion=con,actual_UI=ui,same52slide_PPT=dict(path=ppt['canonical_path'],sha256=ppt['final_sha256'],changed_slides=[52],actual_visual_review=True),
    next_plan=str(REC/'grayT-material-followup-plan.json'),
    no_unique_physical_cause_claim=True,old_results_preserved=True,company00_used=False)
save(REC/'camera-support-handoff.json',handoff);save(L/'camera-support-handoff.json',handoff)
save(C/'local-surface123-delivery.json',handoff)
save(C/'local-surface123-status.json',dict(updated=stamp,state='alignment_diagnosis_active_old_queue_stopped',training_running=False,
    summary=con['summary'],latest_handoff=str(REC/'camera-support-handoff.json'),next_plan=handoff['next_plan'],company_server_used=False,physical_quality_accepted=False))
s=read(C/'local-research-status.json');s.update(updated_at=stamp,current_local_camera_support_diagnosis=con,local_GPU_task_active=False,training_running=False,next_plan=handoff['next_plan']);save(C/'local-research-status.json',s)
n=read(REC/'next-action.json');n.update(updated=stamp,next_plan=handoff['next_plan'],actual_UI_verified=True,
    next=[con['next'],'Read grayT-material-followup-plan.json before any further computation; it is planned, not executed.',
          'Original52pagePPT now updated only slide52, canonicalSHA='+ppt['final_sha256']+'. Preserve all previousresults; no camera/fit/training change.']);save(REC/'next-action.json',n)
ledger=read(REC/'evidence-ledger.json');kind='grayT_camera_support_published_UI_and_same_PPT'
assert not any(e.get('kind')==kind for e in ledger['entries'])
ledger['entries'].append(dict(kind=kind,date=stamp,evidence=handoff));ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
print(json.dumps(dict(PPT_SHA256=ppt['final_sha256'],UI_verified=True,next_plan=handoff['next_plan'],training_running=False,goal_active=True),ensure_ascii=False))
