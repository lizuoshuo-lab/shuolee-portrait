from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
A=REC/'grass-exploratory/provenance-audit';M=A/'mapping-audit';OUT=REC/'smplifyx-reference-control'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();assert not (OUT/'plan.json').exists()
prior=read(M/'plan.json')
save(OUT/'plan.json',dict(registered=stamp,state='registered_not_executed',purpose='Determine whether official SMPL-X joint semantics and SMPLify-X staged fitting can align individual original images before any shared-world initialization.',
    basis=str(M/'result.json'),causal_observation=str(M/'mean-ablation-result.json'),
    authorization='User asks to investigate the mapping and points out SMPLify body/face/hand ability, within the active generic reconstruction goal.',
    views=[0,31,62],source_frames=[0,155,310],selection='Already preregistered mapping-review views; not chosen by measured fit error.',
    input_hashes=prior['inputs'],camera_policy='Use recorded complete integer-center K perimage without optimization. NoSfM/E change. Camera at identity in each separate diagnostic frame; person root is fitted in that camera frame.',
    mapping='OfficialSMPL-X direct joints selected by semantic names into detectorCOCO17 order; officialfeet6/hands42/staticface51. Noartificial neck/pelvis observations, no nearestvertex binding.',
    stages='Root placement then bodyshape/pose then hands then jaw/expression/visibleface, based on pinned officialSMPLify-X. Keep finite predeclared budgets and identical configuration across views.',
    implementation_boundary='Use existing licensedSMPL-X andVPoserV02_05; adapt API, fullK and detector conventions explicitly. This is a local SMPLify-X-style reference unless full official execution is actually established. Never copy defaultfocal5000 or defaultimagecenter over suppliedK.',
    prior_policy='Fresh individual savednetwork outputs allowed as independent perimageinitializers. Nofailedsharedfit, oldoptimizedgeometry, Gaussians or optimizer state.',
    validation_before_real_fit=['Compare directjointindex/name contract against pinnedofficialsource','Check all optimized parameters participate and receivefinitegradients, includinghandPCA,jaw,expression andpose latent','KnownSMPL-X oneviewprojection recovery sanitycheck with the exactcamera andoptimizer adapter; notrepeatoldA0c unchanged','Inspect fixednativeface/hand/bodycrops and register visible/occluded/unknown before real fitting; detectorconfidence isnotvisibilitytruth'],
    budget=dict(reference_implementation=1,real_views=3,optimizer_evaluations_max_per_view=300,gpu_seconds_max_per_view=180,large_downloads=0,gaussian_steps=0),
    stop=['Nonfinite or negative-depth collapse','Failedsemantics/projection/gradient check','Budgetended: savefailure without extra schedules or tuning','Unobserved face/hand cannot be scored as reconstructed accurately'],
    report=['Sameunwarpedimage overlay andface/handcrops, allvalid residual distributions andpredeclaredvisibility','Do not equate lowfittingerror with independentphysicalaccuracy','State exactadaptations comparedtoofficialsource; no fullreproduction claim without actualrun'],
    no_repeated_legacy_trial=True,no_dynamic_Gaussians=True,no_joint_world_location_average=True,no_new_camera_candidate=True,
    next_if_single_view_good='Diagnose compatibility of fitted camera-frame bodies with static sharedworld and backgroundcameras separately. Do notcombine3differentbodies into atrainingpointcloud.',
    goal_complete=False))
for folder in [REC,R]:
    p=folder/'queue-state.json';q=read(p);q.update(updated=stamp,current_stage='grass_smplify_mapping_and_shared_position_audit_complete',worker_pid=None,worker_command=None,training_running=False,gaussian_steps=0,next_plan=str(OUT/'plan.json'));save(p,q)
p=REC/'next-action.json';q=read(p);q.update(updated=stamp,current_stage='grass_smplify_mapping_and_shared_position_audit_complete',next_plan=str(OUT/'plan.json'),reference_fit_authorized=True,reference_fit_started=False);save(p,q)
p=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/local-research-status.json');q=read(p);q.update(updated_at=stamp,current_stage='grass_smplify_mapping_and_shared_position_audit_complete',next_plan=str(OUT/'plan.json'),latest_visual_review=str(M/'visual-review.json'));save(p,q)
save(A/'viewer-ui-review.json',dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#grass-smplify-mapping',actual_UI_loaded=True,reloaded=True,tab='32',browser='iab',historical_tab15_untouched=True,
    evidence_dom_sha256=sha(A/'viewer-dom.txt'),new_heading_and_metrics_seen=True,three_figure_elements_seen=True,source_images_opened_separately=True,rendered_screenshot_verified=False,
    screenshot_limitation='Existing zero-width screenshot failure retained, no alternative capture used.',new_point_cloud_published=False,company00_UI_verified=False,cloud_sync_verified=False))
p=A/'publication.json';q=read(p);q.update(actual_UI_review_pending=False,actual_UI_review=str(A/'viewer-ui-review.json'),next_plan=str(OUT/'plan.json'));save(p,q)
for d in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    target=d/'grass-exploratory/provenance-audit'
    for f in ['viewer-ui-review.json','publication.json']:save(target/f,read(A/f))
print(json.dumps(dict(plan=str(OUT/'plan.json'),state='registered_not_executed',UI_verified=True),ensure_ascii=False))
