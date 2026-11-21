"""Persist actual N12d material review before any geometric residual is computed."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
G=I/'clip03-grid-coverage'; O=G/'array-cache-repair'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (O/'visual-review.json').exists()
v=read(O/'visual-index.json'); c=read(O/'context-index.json'); result=read(O/'result.json')
assert len(v['rows'])==83 and len(v['pages'])==14 and len(c['rows'])==16
assert all(sha(O/n)==h for n,h in c['frozen_hashes'].items())
contexts={r['row'] for r in c['rows']}
labels={
 0:('pole_step_depth_boundary_uncertain','Pole base overlaps stone and foliage; persistent material identity is not established.'),
 1:('foliage','Center is on leaves near the nominal pole.'),
 2:('hydrant_outline_uncertain','Selected support approaches hydrant rim/outline against foliage.'),
 3:('hydrant_outline_uncertain','Full context identifies the same hydrant, but selected center is at its left outer rim against vegetation; not a uniquely fixed surface point.'),
 4:('foliage','Center is foliage beneath the hydrant.'),
 5:('hydrant_valve_coarse','Same valve detail and neighboring hydrant order; coarse material candidate, fine point uncertainty unmeasured.'),
 6:('foliage','Center is foliage near the hydrant.'),
 7:('foliage','Center is foliage beside the hydrant.'),
 8:('foliage','Center is foliage above the steps.'),
 9:('stone_riser_coarse','Same stone riser context as N12c row11; coarse persistent surface candidate, exact point uncertainty unmeasured.'),
 10:('foliage','Selected center is foliage in front of the nominal stairs.'),
 11:('step_lip_vegetation_boundary_uncertain','Native context places the selected center at the top step lip/vegetation boundary; not certified as a single surface point.'),
 12:('step_ground_gap_uncertain','Selected center/support reaches the dark gap and vegetation between stair base and ground; mixed depth cannot be resolved from these crops.'),
 13:('foliage','Center is foliage above the steps.'),
 20:('window_reflection_or_repeated_texture_uncertain','Selected center lies in repeated glazing with reflected/interior texture; context identifies the building, not one fixed material point.'),
 21:('window_reflection_or_repeated_texture_uncertain','Window-band context is stable but central glazing texture and repeated bays prevent independent surface certification.'),
 22:('building_corner_coarse','Same protruding floor-band corner at the outer facade, anchored by the adjacent red wall and floor ordering. Coarse fixed edge candidate; exact localization unmeasured.'),
 23:('window_reflection_or_repeated_texture_uncertain','Central signal is within glass panes, not an independently identified opaque structural mark.'),
 24:('window_reflection_or_repeated_texture_uncertain','Heldout crop center is repeated glass/reflected texture; do not admit merely to satisfy the nonground holdout count.'),
 25:('foreground_fence_vegetation_facade_overlap_uncertain','Center combines foreground fence/vegetation and distant facade; same building does not establish same depth/material.'),
 26:('fence_road_depth_junction_uncertain','Fence post/wire overlaps the white road stripe behind; center/support is a depth junction, not certified as a surface point.')}
for n in range(14,20): labels[n]=('roof_structure_coarse','Native256px context anchors the same roof structural bay by the solid red end wall and column ordering in all five views. No visible bay switch; coarse fixed-material candidate, not subpixel ground truth.')
for n in range(27,51): labels[n]=('ground_texture_coarse','Native aggregate/paint/crack pattern remains plausible in the same ground neighborhood. Ground is nearly planar and exact independent subpixel identity is unmeasured.')
for n in range(51,67):
 label='face_surface_coarse' if n in {52,55,56,57,59,60} else ('hair_coarse' if n==58 else 'white_clothing_in_head_roi')
 labels[n]=(label,'All available native crops inspected. Head ROI membership is not a face landmark or static-material guarantee. Blink/expression and independent pixel uncertainty remain to be tested.')
for n in range(67,83):
 label='shoe' if n in {68,69,72,73,74,77} else ('hand_clothing_boundary' if n in {78,81} else ('sock' if n==79 else 'shorts'))
 labels[n]=(label,'All available native crops inspected. Plausible person-material track; not used to fit background cameras or certify person stationarity.')
rows=[]
for t in v['rows']:
 n=t['row']; label,reason=labels[n]
 nonground=n in {5,9,14,15,16,17,18,19,22}; ground=27<=n<=50
 rows.append(dict(row=n,track=t['track'],family=t['family'],split=t['split'],frames=[x['frame'] for x in t['observations']],
   reviewed_material=label,reason=reason,coarse_static_candidate=nonground or ground,coarse_static_nonground_candidate=nonground,
   context_reviewed=n in contexts,subpixel_physical_identity_certified=False,camera_fit_admitted=False))
coarse={s:sum(x['coarse_static_candidate'] and x['split']==s for x in rows) for s in ['fit','heldout']}
nonground={s:sum(x['coarse_static_nonground_candidate'] and x['split']==s for x in rows) for s in ['fit','heldout']}
assert coarse==dict(fit=22,heldout=11) and nonground==dict(fit=7,heldout=2)
frames=v['columns']; shared=[]
for a,b in zip(frames[:-1],frames[1:]):
 shared.append(dict(frames=[a,b],fit=sum(x['coarse_static_candidate'] and x['split']=='fit' and a in x['frames'] and b in x['frames'] for x in rows)))
face={s:sum(x['reviewed_material']=='face_surface_coarse' and x['split']==s for x in rows) for s in ['fit','heldout']}
review=dict(stage='N12d',updated=datetime.now(timezone.utc).isoformat(),state='selection_comparison_review_complete_no_geometry',
 actual_review=dict(native128px_rows=list(range(83)),pages=v['pages'],native256px_context_rows=sorted(contexts),all_available_observations_reviewed=True,annotations_in_images=False),
 counts=Counter(x['reviewed_material'] for x in rows),rows=rows,coarse_static_candidates=coarse,coarse_static_nonground_candidates=nonground,actual_face_candidates=face,
 inherited_geometry_gate=dict(required_total=dict(fit=20,heldout=8),required_nonground=dict(fit=6,heldout=3),passed=False,retrospectively_relaxed=False,
   reason='The 83 reviewed rows provide22FIT/11heldout coarse static families, but only7FIT/2heldout nonground families, below6/3. No new camera solve is admitted. Repeated glazing and mixed-depth edges are not promoted to pass the gate.',
   adjacent_shared_reviewed_fit=shared,spatial_and_parallax_admission='Not evaluated as a pass because nonground holdout support already fails; camera observability remains unknown.'),
 comparison=dict(baseline='N12b',only_selection_rule_changed=True,all_7_raw_detector_identities_bit_exact=True,all_7_original_8000_descriptor_arrays_bit_exact=True,
   person_selected_ids_unchanged=True,cycles=[5004,4329],tracks=[2740,2519],mixed_old_fit_heldout_families_quarantined=3,
   frame232_roof=dict(selected=[4,136],cycles=[1,49]),frame232_facade_search=dict(selected=[11,280],cycles=[5,52]),
   physical_scope='More roof material support is visible under the registered sampling scheme. Different track membership/family quarantine is a consequence of allocation. Counts are not independent physical points, and increased coverage does not establish camera accuracy or explain the whole ghosting error.'),
 implementation_repair=dict(original_failure_preserved=str(G/'failure.json'),record=str(G/'implementation-repair.json'),change='Cache NPZ descriptor arrays once instead of repeatedly decompressing them inside equality loop; no detector/descriptor/matcher/threshold changes.',original_active_seconds=84.687,repaired_seconds=result['elapsed_seconds'],combined_under_180_seconds=True),
 limitations=['Only7 of464 frames; no all-video registration.', 'No geometric residuals computed; material labels were assigned without residual selection.',
  'Reviewed representative sample is finite. Unreviewed families remain unclassified, not rejected as physically false.',
  'Native128px and256px crops preserve input pixels but cannot establish independent <1px measurement uncertainty.',
  'Ground-dominated static support and sparse distant roof tracks do not independently constrain all camera freedoms.',
  'First YAML4000x3000 conditionalK unchanged; exact exporter K and rounding still unverified.'],
 next='No second grid setting or threshold sweep. Preserve completed N12d. Before further camera solving, evaluate whether the two credible heldout roof families and ground permit an independent physical-support/uncertainty study; any additional material evidence must be preregistered and cannot move heldout families into FIT. Do not use uncertain glass/occlusion points or old optimized cameras to bypass support failure. Continue independent renderer/pipeline engineering when physical evidence is insufficient.',
 source_hashes={n:sha(O/n) for n in ['result.json','observations.json','families.json','visual-index.json','context-index.json','allocation.json']},
 K_changed=False,new_cameras=0,training_started=False,quality_accepted=False)
save(O/'visual-review.json',review)
save(O/'progress.json',dict(state=review['state'],pid=None,updated=review['updated'],next=review['next']))
print(json.dumps(dict(coarse=coarse,nonground=nonground,face=face,adjacent=shared,geometry_admitted=False),ensure_ascii=False))
