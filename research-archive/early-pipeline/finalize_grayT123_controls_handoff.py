"""Finalize one reviewed publication cycle; retain the active reconstruction goal."""
from pathlib import Path
from datetime import datetime,timezone
import shutil,json
from audit_grayT123_camera_support import REC,ROOT,read,save,sha

H=Path(__file__).parent;T=REC.parents[2];C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
stamp=datetime.now(timezone.utc).isoformat();con=read(REC/'alignment-control-conclusion.json');ppt=read(H/'surface123-controls-deck-receipt.json')
assert ppt['canonical_replaced'] and sha(Path(ppt['canonical_path']))==ppt['final_sha256']
assert not (ROOT/'gaussian-run').exists()
geom=read(REC/'grayT-skin-localization-review/geometry-result.json')
assert all(sha(Path(a['path']))==a['sha256'] for a in geom['sources'])
latest=read(T/'viewer/latest-result.json');sync=read(Path('D:/00/工作/查看器/latest-result.json'))
assert latest['modelId']==sync['modelId']=='local-surface-123-whiteT-s18000'
assert latest['sha256']==sync['sha256']=='df4b686f0e667f6cef08a4f24d6d1688d9b5efde74e8d53e06d20b2ab5702841'
assert latest['latestDiagnostic']['reportUrl']=='body-smplx.html#alignment-controls'
assert sync['latestDiagnostic']['reportUrl']=='progress.html#alignment-controls'
screens=['grayT-controls-summary-ui.png','grayT-controls-gallery-ui.png','grayT-controls-near-ui.png','grayT-controls-main-viewer-ui.png']
for name in screens:assert (REC/name).exists();shutil.copy2(REC/name,L/name)
ui=dict(updated=stamp,actual_UI_review=True,progress_tab='40',main_viewer_tab='31',historical_tab15_untouched=True,
    actual_display='Progress page shows41photometric localizations,4same-frame projections,6near-scene component sheets and the new residual table. Main viewer retains WhiteT309523Gaussians/93cameras and displays the new diagnostic entry. One Playwright role locator failed; fresh native AXbutton selection expanded the existingdetails, verified by screenshot.',
    screenshot_paths=[str(REC/name) for name in screens],new_model_published=False,
    model_unchanged=latest['modelId'],quality_accepted=False,company00_verified=False)
save(REC/'grayT-controls-ui-review.json',ui);save(L/'grayT-controls-ui-review.json',ui)
handoff=dict(updated=stamp,goal_mode='active_not_timer',goal_complete=False,training_running=False,
    current_subject='grayT',current_worker_pid=None,conclusion=con,actual_UI=ui,
    same52slide_PPT=dict(path=ppt['canonical_path'],sha256=ppt['final_sha256'],changed_slides=[52],actual_last_page_visual_review=True,other51_render_hashes_identical=True),
    completed_no_repeat=['grayT-near-scene-support-review','grayT-skin-localization-review','grayT-material-identity-review','grayT-material-static-check','grayT-skin-texture-probe'],
    next_plan=str(REC/'alignment-admission-implementation-plan.json'),
    constraints='No old123supervisor or grayTGaussian restart. Registered near-scene audit and single photometric localization pluspointcomparison complete. No cameras or oldgeometry changed. Nextplan implements separate finealignment admission in future localqueue only; no oldsupervisor restart.',
    no_unique_physical_cause_claim=True,company00_used=False)
save(REC/'alignment-controls-handoff.json',handoff);save(L/'alignment-controls-handoff.json',handoff);save(C/'local-surface123-delivery.json',handoff)
save(C/'local-surface123-status.json',dict(updated=stamp,state='alignment_controls_published',training_running=False,
    summary=con['summary'],latest_handoff=str(REC/'alignment-controls-handoff.json'),next_plan=handoff['next_plan'],company_server_used=False,physical_quality_accepted=False))
n=read(REC/'next-action.json');n.update(updated=stamp,latest_handoff=str(REC/'alignment-controls-handoff.json'),actual_UI_verified=True,
    next=['Implement the registered alignment-admission-implementation-plan.json. Do not repeat completed material, near-scene or photometric localization controls. No training is admitted.',
        'The same52pagePPT now has canonical SHA='+ppt['final_sha256']+'; edit only52if new substantive evidence.','GrayTcamera/SMPL-X original files and both old18ktrained models remain unchanged. No grayT Gaussian training exists.'])
save(REC/'next-action.json',n)
state=read(C/'local-research-status.json');state.update(updated_at=stamp,current_local_alignment_control_diagnosis=con,
    local_GPU_task_active=False,training_running=False,next_plan=handoff['next_plan'],latest_handoff=str(REC/'alignment-controls-handoff.json'));save(C/'local-research-status.json',state)
ledger=read(REC/'evidence-ledger.json');kind='grayT_alignment_controls_UI_and_same_PPT'
assert not any(e.get('kind')==kind for e in ledger['entries'])
ledger['entries'].append(dict(kind=kind,date=stamp,evidence=handoff));ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
print(json.dumps(dict(goal_active=True,training_running=False,UI_verified=True,PPT_sha256=ppt['final_sha256'],next_plan=handoff['next_plan']),ensure_ascii=False))
