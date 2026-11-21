from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
R=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\raw-domain-121')
def save(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=json.loads((R/'protocol.json').read_text());now=datetime.now(timezone.utc).isoformat()
supp=dict(registered=now,trigger='Measured native feature support: grass frame224 face/body0 tracks, portrait frame266 face4 with0heldout; near-duplicate cross-split families35/79/95/13.',
    hypothesis='Global feature budget is occupied by background texture, leaving insufficient independent face/body material support. Native ROI feature allocation can address support, not necessarily geometry or motion.',
    main_variable='Add native crop SIFT allocation for face and body to unchanged whole-frame features; no pixel scale/K/frame/pair/ransac threshold change.',
    validation_hygiene='Quarantine cross-split near-duplicates; never relabel heldout observations as fit. This is a validation correction, not evidence of geometric improvement.',
    preserved=['24 source frames per video','native pixels','fixed conditionalK','mutual ratio0.75','existing3viewcycle pairs','original source hashes'],
    max_additional_feature_runs=1,max_seconds_per_person=900,cpu_only=True,existing_official_models_only=True,
    detector_purpose='YOLO and RTMW define approximate ROI only. Descriptor pixels are native unresized crops; add integer crop offset. No landmark index used as physical identity.',
    budgets=dict(body_SIFT=1000,face_SIFT=500,contrastThreshold=.02),
    selection='Per-frame1px suppression before matching; existing split protected by3px overlap exclusion. Mixed split components quarantined.',
    stop_conditions=['ambiguous person ROI','no added face/body material support','physical patch mismatch','insufficient independently held face/body','time budget'],
    worker=str(H/'supplement_native_roi121.py'),sha256=hashlib.sha256((H/'supplement_native_roi121.py').read_bytes()).hexdigest())
save('roi-supplement-plan.json',supp);p['native_ROI_supplement']=supp;p['updated']=now;save('protocol.json',p)
save('implementation-repairs.json',dict(repairs=[dict(script='probe_native_calibrated121.py',failure='Progress timestamp passed function instead of now() causing JSON TypeError before any subject result was saved.',repair='Called now() after process exit; reran identical seed/config. No threshold or data change; first result then completed.')]))
save('queue-state.json',dict(updated=now,state='native_roi_supplement_preregistered',training_started=False,gpu_job=None,automation_id='118',next='Run one native ROI supplement, audit physical identity/leakage/regions, then global calibrated camera reconstruction. No duplicate old118/119 training.'))
print('SUPPLEMENT_PREREGISTERED')
