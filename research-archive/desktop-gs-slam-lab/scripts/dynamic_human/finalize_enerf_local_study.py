"""Archive the completed bounded study without accepting the long-term goal."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil

LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
pipeline=read(out/'pipeline-state.json');assert pipeline['status']=='all_training_complete_evaluation_pending'
for h in pipeline['history']:assert h['exit_code']==0 and h['marker_exists']
image_audit=read(out/'independent-image-audit.json');assert image_audit['images']==225
state_audit=read(out/'independent-state-audit.json');assert state_audit['per_model_total_update_budget']==20400
actor=read(out/'actor-interface/render-equivalence.json');assert all(v['maximum_channel_difference']==0 for v in actor['views'])
statistics=read(out/'paired-statistics.json')
observations=[]
for c in ['00','06','12']:
    for t in [34,46,58]:
        for p in [1,2]:observations.append(f'comparison/camera{c}-frame{t:03d}-person{p}.png')
for t in [49,50,51]:
    for p in [1,2]:observations.append(f'comparison/camera00-frame{t:03d}-person{p}.png')
observations += [f'evaluation/camera{c}-frame058-offset-index.png' for c in ['00','06','12']]
observations += ['comparison/first-fit-training-camera01-person2.png']
for name in observations:assert (out/name).is_file()
visual=dict(status='actual_images_reviewed_clarity_not_accepted',viewed_images=observations,main_samples='All18upper-body panels at34,46,58 for3heldout appearance cameras;6adjacent panels at49,50,51 for worst-delta camera00;3full final renders;1first-fit training-only panel',findings=['Static multi-time representation mixes hand and held-object positions; dynamic models recover main posture, clothing and lower-body silhouette','Local index adjustment restores some shirt lettering and reduces some hand/toy holes; no uniform improvement over all views','At58, panda/hand/sleeve surfaces remain broken and blurred, especially oblique camera06; white-shirt hand and watermelon-like object retain jagged edges','Camera06 full render retains isolated dark fragment below/left of people and sleeve-side fragment; full frames not cleaned','Worst quantitative delta atcamera00/frame50 retained and compared with49/51; no perceptual acceptance from mean scores'],videos='3videos decoded fully; no claim that every video frame was individually visually inspected or playback flicker accepted',clothing_detail_accepted=False,moving_hand_detail_accepted=False,whole_scene_accepted=False,synthetic_enhancement=False)
save(out/'visual-review.json',visual)
summary=dict(status='bounded_continuous_control_complete_clarity_not_accepted',completed_utc=datetime.now(timezone.utc).isoformat(),models=statistics['models'],pairwise=[{k:v for k,v in c.items() if k!='values'} for c in statistics['comparisons']],total_updates_each=20400,images=225,person_regions=450,temporal_transitions=216,video_frames_each=25,static_points=state_audit['static_points'],dynamic_points={r['condition']:r['points'] for r in state_audit['dynamic']},visual_review='visual-review.json',input_audit='input-isolation-audit.json',image_audit='independent-image-audit.json',state_audit='independent-state-audit.json',temporal_metric=image_audit['temporal_summary'],actor_interface='actor-interface/scene-contract.json',actor_render_reconstruction='actor-interface/render-equivalence.json',scope=statistics['limits'],decision='Retain local-time model as research candidate; no global replacement or clarity acceptance. No original datasets changed.',next=['Hold current observations fixed and isolate motion-stage scale/opacity/surface support from time correspondence','Use training-only full-canvas foreground consistency to diagnose fragments before any cleanup; do not use heldout masks to delete points','Validate unseen times/larger motion and other performers under separately frozen protocols','Then test shared-coordinate static-background/deformable-person composition and OmniRe import; current node export is not that integration'],goal_complete=False)
save(out/'final-summary.json',summary)
archive=out/'source-snapshot';archive.mkdir(exist_ok=True);sources=[]
names=['prepare_enerf_local_time_study.py','freeze_enerf_local_appearance.py','run_enerf_local_study.py','train_enerf_first_frame.py','train_enerf_motion.py','train_enerf_multitime_static.py','evaluate_enerf_local_study.py','compose_enerf_local_study.py','compose_enerf_local_failures.py','audit_enerf_local_inputs.py','audit_enerf_local_calibration.py','audit_enerf_local_results.py','audit_enerf_local_states.py','export_enerf_actor_contract.py','audit_enerf_actor_render.py','summarize_enerf_local_study.py','enerf_native_loader.py','prepare_enerf_geometry.py','audit_enerf_camera_time.py','solve_enerf_time_graph.py','finalize_enerf_local_study.py']
for name in names:
    src=LAB/'scripts/dynamic_human'/name;data=src.read_bytes();target=archive/name
    if target.exists():assert target.read_bytes()==data
    else:shutil.copy2(src,target)
    sources.append(dict(file=name,sha256=hashlib.sha256(data).hexdigest()))
save(archive/'receipt.json',dict(status='actual_used_source_versions_preserved',sources=sources))
manifest=LAB/'manifests/dynamic-human-work.json';work=read(manifest)
work.update(status='local_continuous_control_verified_surface_motion_refinement_pending',updated=datetime.now(timezone.utc).isoformat(),current_runs=[],external_input_blocker=None,remaining_quality_issue='Continuous native foreground retains hand/toy/sleeve breakup; local retiming gives mean+0.356dB but21/75views regress. Static-background joint scene not yet integrated.',outdoor_local_continuous_control='outputs/dynamic-human-enerf-local-time-029/final-summary.json',outdoor_actor_interface='outputs/dynamic-human-enerf-local-time-029/actor-interface/scene-contract.json',next_actions=summary['next'],pending=summary['next'],previous_goal_turn_classification='progress: actual3models20400updates each,225native renders450regions216transitions3videos independently verified;24heldout panels and3full renders reviewed; lossless actor-node interface passed actual render reconstruction. Clarity remains unaccepted, no external blocker.')
new='Local029: explicit calibration/appearance split;3matched20400update models,225renders450regions216transitions3decoded videos; local offset+0.356dB54/75improved;24heldout panels reviewed; lossless150543point25state actor nodes and3render matches; clarity not accepted'
if new not in work['completed']:work['completed'].append(new)
save(manifest,work)
print(json.dumps(dict(status=summary['status'],reviewed_images=len(observations),source_snapshots=len(sources),current_runs=work['current_runs']),indent=2))
