from pathlib import Path
import json,shutil,numpy as np
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-heldout-cameras-068'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
pair=read(R/'paired-comparison/summary.json');tm=read(R/'temporal-comparison/summary.json');protocol=read(R/'review-protocol.json')
assert read(R/'independent-image-audit.json')['images']==54
assert len(tm['transitions'])==15 and len(tm['videos'])==3
save(R/'observation-visual-review.json',dict(status='9even_time_pose_overlays_actually_viewed',images=[f'camera{c}-frame{t:06d}-observations.png' for c in ['00','06','12'] for t in [744,750,756]],
 findings={'00':'Most visible body/feet plausible; facial points hallucinated on back-facing light person. Face excluded from clockfit.',
 '06':'Severe occlusion: dark-person head often predicted on light-person shoulder/back; hidden arm/wrist predictions can switch actors. Face excluded; body outliers remain.',
 '12':'Overlapping forearms/wrists can be ambiguous; no prediction treated as ground truth.'},
 qualification='Compatibility gate uses only even-time observations against frozen training geometry. Visual review documents limits, does not tune offsets on odd RGB.'))
full=[f"camera{r['camera']}-frame{r['frame']:06d}-full-person.png" for r in protocol['full_person']]
detail=[r['panel'] for r in pair['details']]
assert len(full)==9 and len(detail)==18
findings=dict(camera00='Toy silhouette and clothing graphic improve. Hand/finger edges and shoes remain smoothed; camera00frame753 has minimal foreground gain, not regression.',
 camera06='Toy detail improves but severe overlap retains streaked/partially blended appearance around black shirt, toy and hands. Some projected dark forearm ROIs contain the occluding light shirt instead of the target arm.',
 camera12='Toy markings and silhouette are clearer across all three fixed full views. Fingers remain fused and shoes smooth; camera12frame755 dark forearm region regresses.',
 timing='Camera00/06/12frame751 clock panels actually viewed; fitted-time render usually aligns movement better, but clock fitting is conditional and camera06 late odd frames do not improve.')
save(R/'visual-review.json',dict(status='9fixed_full18details_minimumgain_and3clock_panels_actually_viewed',full_person=full,details=detail,
 additional_minimum_gain='camera00-frame000753-full-person.png',clock_panels=[f'camera{c}-frame000751-clock-comparison.png' for c in ['00','06','12']],
 findings=findings,video_qualification='All3videos mechanically decoded; visual review only enumerated still images, not full playback.',
 clarity_accepted=False,goal_complete=False))
regions={n:dict(mean_delta_db=float(np.mean([r['delta_db'] for r in pair['details'] if r['region']==n])),improved=sum(r['delta_db']>0 for r in pair['details'] if r['region']==n),count=9) for n in ['dark_right_forearm','light_left_forearm']}
save(R/'final-summary.json',dict(status='3heldout_appearance_cameras54native_renders_clockfit_and_visual_review_complete',
 cameras=['00','06','12'],calibration_even_images=21,evaluation_odd_images=18,appearance_updates=0,body_updates=0,graph_updates=0,
 clock_offsets_frames=read(R/'timing-summary.json')['offsets_frames'],foreground=pair['validation_foreground'],per_camera=pair['per_camera'],
 graph_clock_mean_delta_db=pair['clock_mean_delta_db'],fixed_projected_regions=regions,temporal_difference_residual=tm['mean_temporal_residual'],visual_findings=findings,
 qualification='Cameras excluded from064-067 appearance/pose/initialization, but used in earlier development. Newcamera clockfit uses21even-time RTMW predictions;18odd-time images only appearance evaluation. Neither fresh final benchmark nor ground-truth synchronization. ROI labels identify projected geometry, not semantic visibility masks. Matte and keypoints are predictions.',
 next='Diagnose per-actor alpha/color/depth contributions before choosing finer hand or independent-object motion.',goal_complete=False))
shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(dict(regions=regions,foreground=pair['validation_foreground'],temporal=tm['mean_temporal_residual'])))
