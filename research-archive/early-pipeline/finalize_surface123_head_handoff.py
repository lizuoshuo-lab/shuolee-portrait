"""Save actual local UI/PPT evidence and next bounded action; no goal-completion claim."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
from audit_grayT123_camera_support import REC,ROOT,read,save,sha
H=Path(__file__).parent;T=REC.parents[2];V=T/'viewer';S=Path('D:/00/工作/查看器')
L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123');C=H.parent/'shuolee-coordination'
stamp=datetime.now(timezone.utc).isoformat();con=read(REC/'head-rigid-conclusion.json')
receipt=read(H/'surface123-head-deck-receipt.json');assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
assert sha(Path(receipt['canonical_path']))==receipt['final_sha256']
assert sha(Path(receipt['mirror_path']))==receipt['final_sha256']
assert not (ROOT/'gaussian-run').exists()
screens=['head-rigid-summary-ui.png','head-rigid-gallery-ui.png','head-rigid-main-viewer-ui.png']
for name in screens:assert (REC/name).stat().st_size>10000;shutil.copy2(REC/name,L/name)
latest=read(V/'latest-result.json');assert latest['modelId']=='local-surface-123-whiteT-s18000' and not latest['qualityAccepted']
assert latest['latestDiagnostic']['reportUrl']=='body-smplx.html#head-rigid'
assert read(S/'latest-result.json')['modelId']==latest['modelId']
ply=V/'models/local-surface-123-whiteT-s18000/shuolee_123_whiteT_s18000.ply'
assert sha(ply)=='df4b686f0e667f6cef08a4f24d6d1688d9b5efde74e8d53e06d20b2ab5702841'
ui=dict(updated=stamp,actual_browser_review=True,progress_tab='40',main_viewer_tab='31',historical_tab15_untouched=True,
    main_model=latest['modelId'],points=309523,cameras=93,model_file_sha256=sha(ply),
    latest_diagnostic_visible=True,native_nose_gallery_opened_and_images_visible=True,
    screenshots=[dict(path=str(REC/name),sha256=sha(REC/name)) for name in screens],
    no_new_model_or_camera_published=True,new_Gaussian_points_UI_acceptance_not_applicable=True,
    company00_not_operated=True,cloud_sync_not_verified=True,quality_accepted=False)
save(REC/'head-rigid-ui-review.json',ui)
handoff=dict(updated=stamp,classification='substantive_progress_not_goal_completion',summary=con['summary'],
    engineering=str(REC/'alignment-admission-conclusion.json'),diagnostic=str(REC/'grayT-head-rigid-review/result.json'),
    UI_review=str(REC/'head-rigid-ui-review.json'),report='http://127.0.0.1:8870/body-smplx.html#head-rigid',
    canonical_presentation=receipt['canonical_path'],presentation_sha256=receipt['final_sha256'],only_slide52_changed=True,
    next_plan=str(REC/'grayT-proxy-visibility-material-plan.json'),no_GPU_or_training_process_found=True,
    local_training_running=False,quality_accepted=False,goal_active=True,
    do_not_repeat=['admission tests','near-scene and photometric controls','head-rigid fit','old123queue'])
save(REC/'head-rigid-handoff.json',handoff)
n=read(REC/'next-action.json');n.update(updated=stamp,actual_UI_verified=True,latest_handoff=str(REC/'head-rigid-handoff.json'),
    current_worker_pid_wsl=None,publication_worker_pid_windows=None,completed_this_turn=[
        'Fail-closed alignment guard:27contract cases+4entry boundaries,6actual WSL denials.',
        '8native head/torso material-support crops reviewed.',
        'One registered shared/perframe rigid proxy diagnostic with independent skin check, failed1px.',
        'Same viewer/latestDiagnostic/log and slide52 published, actual UI inspected.']);save(REC/'next-action.json',n)
publication=read(REC/'publication.json');publication.update(updated=stamp,UI_checked=True,report=handoff['report'],
    actual_UI_review=str(REC/'head-rigid-ui-review.json'),training_started=False);save(REC/'publication.json',publication)
q=read(REC.parent/'robust-portrait-122/queue-state.json');q.update(updated=stamp,running=None,
    last_goal_turn=dict(classification='progress',current=con['summary'],quality_accepted=False),latest_handoff=str(REC/'head-rigid-handoff.json'));save(REC.parent/'robust-portrait-122/queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_local_head_rigid_diagnostic=con,
    local_GPU_task_active=False,training_running=False,latest_handoff=handoff,next_plan=handoff['next_plan']);save(C/'local-research-status.json',c)
save(C/'local-surface123-status.json',handoff)
for p in [REC/'head-rigid-handoff.json',REC/'head-rigid-ui-review.json',REC/'publication.json',H/'surface123-head-deck-receipt.json']:
    shutil.copy2(p,L/p.name)
print(json.dumps(dict(report=handoff['report'],UI_verified=True,PPT_sha256=receipt['final_sha256'],training_started=False,goal_active=True),ensure_ascii=False))
