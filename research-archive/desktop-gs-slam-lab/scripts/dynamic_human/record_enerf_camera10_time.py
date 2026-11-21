from pathlib import Path
import json,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-camera10-time-050'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
summary=read(OUT/'paired-comparison/summary.json')
assert read(OUT/'independent-state-audit.json')['status'].startswith('paired6000updates_exact')
review=dict(status='six_whole_person_and17_predeclared_native_detail_panels_actually_reviewed',
    whole_person_cameras=['00','06','12','01','08','17'],detail_regions=read(OUT/'fixed-detail-regions.json')['regions'],
    findings={
      '00':'Printed shirt slightly cleaner; left hanging arm remains elongated and streaked, wrist and airborne shoe do not recover; foreground score worsens.',
      '06':'Blue-shirt folds slightly cleaner, crossed forearms and hands remain distorted, wrong thin structure beside occluded shoulder remains, toy still blurred.',
      '12':'Printed shirt, blue forearm and held-toy outline partly improve; hanging arm retains wall-colored floaters and stretched surface; airborne shoe remains smeared.',
      'training':'01 still streaked and overall score worsens;08/17 back-shirt surfaces relatively clear but fingers/occlusion remain unresolved.'},
    decision='Do not adopt camera10-only retiming as a clarity fix. Preserve as mixed local observation control. No clear-human acceptance.',
    manipulation='Native integer-coordinate crops and white label padding only; no resampling, enhancement or generated pixels.',goal_complete=False)
save(OUT/'visual-review.json',review)
save(OUT/'final-summary.json',dict(status='paired_camera10_time_control_mixed_view_quality_limb_clarity_unaccepted',
    updates_each=6000,baseline_points=143073,candidate_points=141868,
    heldout_foreground_delta_db={r['camera']:r['foreground_delta_db'] for r in summary['records'] if r['split']=='heldout'},
    all_metrics='paired-comparison/summary.json',input_isolation='Only camera10 RGB and corresponding same-rule mask changes750to749.',
    fixed='Exact initial parameters and6000camera/background draws; other14observations and allcamera matrices/referencephotos unchanged.',
    independent_state_audit='independent-state-audit.json',visual_review='visual-review.json',
    conclusion=review['decision'],goal_complete=False))
mp=LAB/'manifests/dynamic-human-work.json';m=read(mp)
m['outdoor_camera10_time_control']='outputs/dynamic-human-enerf-camera10-time-050/final-summary.json'
m['status']='single_camera_time_control_mixed_clarity_unaccepted'
m['current_runs']=[]
entry='050 two6000update models exact paired initial/sampling checked;48nativePNGs24panels6wholeperson17fixed-detail comparisons verified/reviewed. Camera10 shift mixed, limbs still smeared.'
if entry not in m['completed']:m['completed'].append(entry)
m['next_actions']=['Fit globally consistent relative-index hypothesis from049training-only fixed tracks, then validate at760 without using it in fitting.',
 'Resolve multi-view geometry and moving-limb contradictions before temporal/full-scene integration.']
m['goal_turn_classification']='Progress:050 trained, evaluated and visually reviewed; no external blocker, clear moving-human goal unmet.'
m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();save(mp,m)
shutil.copy2(__file__,OUT/'source-snapshot'/Path(__file__).name)
print('050 recorded; no clarity acceptance')
