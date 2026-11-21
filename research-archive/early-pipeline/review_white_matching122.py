"""Persist actual N12b material review. No solver, residual selection or new detections."""
from pathlib import Path
from datetime import datetime, timezone
import json,hashlib,collections

I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
O=I/'clip03-native-matching'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (O/'visual-review.json').exists()
idx=read(O/'visual-index.json'); ctx=read(O/'context-index.json'); result=read(O/'result.json')
assert len(idx['rows'])==104 and len(idx['pages'])==18 and len(ctx['rows'])==16
assert {n:sha(O/n) for n in result['hashes']}==result['hashes']==ctx['frozen_hashes']
ground=set([24,25]+list(range(30,48))+list(range(63,72)))
woody={7,8,9,10,53}
mixed={26,28,60,61,62}
banner={27,29}
face={72,73,77,79,80,84};hair={78}
records=[]
for r in idx['rows']:
    n=r['row']; label=''; reason=''; confirmed=False
    if n<72:
        if n==12:
            label='rigid_building_roof_detail';confirmed=True
            reason='Native256px context resolves the same upper red building/aperture detail across232/248/264/280. Coarse physical identity plausible; not a measured subpixel reference.'
        elif n in ground:
            label='ground_texture_candidate'
            reason='Static-looking ground aggregate/paint/crack context is consistent at coarse scale. Fine repetitive aggregate has no independently measured pixel identity. Keep as ground candidates, not subpixel truth.'
        elif n in woody:
            label='woody_branch_or_trunk_contour_uncertain'
            reason='Recognizable tree structure, but the selected feature is a contour/junction with possible foliage overlap. Material-surface identity and static status are not independently certified.'
        elif n in mixed:
            label='fence_vegetation_or_building_depth_overlap_uncertain'
            reason='Context contains real rigid infrastructure, but crop center/support mixes fence, vegetation or repetitive facade; cannot assign one verified physical surface/depth from the reviewed observations.'
        elif n in banner:
            label='fabric_banner_texture'
            reason='Lettering appears trackable but lies on hanging fabric. Do not use it as independently static camera support.'
        else:
            label='foliage_candidate'
            reason='Selected center is in foliage/leaf or thin branch detail. Nearby hydrant, pole or steps visible in context do not make this feature a rigid-object point.'
    elif n<88:
        if n in face:
            label='actual_face_candidate'
            reason='Face material/feature region is visible; headROI label alone was insufficient. Eyelid changes atframe200 warn about nonrigidity/visibility. No fixedcamera reprojection or pixel uncertainty measured.'
        elif n in hair:
            label='hairline_candidate';reason='Selected headROI is hair/hairline rather than eyes/nose/mouth; possible hair and silhouette dependence.'
        else:
            label='headROI_actually_white_shirt'
            reason='128px native crops show collar, shoulder or chest clothing. This is a real observation, but cannot be counted as facial support.'
    else:
        if n in {88,90,92,96,97,103}:label='printed_shorts_candidate'
        elif n==98:label='sock_repetitive_rib_candidate'
        elif n==100:label='hand_clothing_boundary_candidate'
        else:label='shoe_lace_or_panel_candidate'
        reason='Person material only; never camera-fit background. Pattern identity is plausible at coarse scale, but nonrigid motion and subpixel localization remain unmeasured.'
    records.append(dict(row=n,family=r['family'],track=r['track'],split=r['split'],spatial_bin=r['spatial_bin'],frames=[o['frame'] for o in r['observations']],
      category_from_detector=r['category'],reviewed_material=label,reason=reason,
      reviewed_static_nonground_coarse_candidate=confirmed,subpixel_physical_identity_certified=False,camera_fit_admitted=False,
      context_reviewed=n in {v['row'] for v in ctx['rows']}))
counts=collections.Counter(r['reviewed_material'] for r in records)
counts_by_split={s:dict(collections.Counter(r['reviewed_material'] for r in records if r['split']==s)) for s in ['fit','heldout']}
nonground={s:sum(r['reviewed_static_nonground_coarse_candidate'] and r['split']==s for r in records) for s in ['fit','heldout']}
review=dict(stage='N12b',updated=datetime.now(timezone.utc).isoformat(),state='reviewed_material_support_gate_not_met',
 actual_review=dict(native128px_rows=list(range(104)),pages=idx['pages'],native256px_context_rows=[r['row'] for r in ctx['rows']],all_observations_displayed=True,annotations_in_images=False),
 counts=counts,counts_by_split=counts_by_split,rows=records,
 gate=dict(coarse_camera_passed=False,confirmed_nonground_coarse=nonground,required_nonground=dict(fit=6,heldout=3),
   grounds_if_all_counted={s:sum(r['row'] in ground and r['split']==s for r in records) for s in ['fit','heldout']},
   failure='Even if every reviewed ground candidate were accepted, the pre-registered6FIT/3heldout rigidnon-ground support gate is unmet (1FIT/0heldout identifiable coarse building detail). Do not run fixedK geometry on this reviewed selection.',
   first_pair_shared_static_gate_evaluated=False,reason_not_evaluated='Necessary nonground gate already fails; no camera solver or residual exists.'),
 findings=[
  'The72selected background families are not72certified static physical points. Spatial quotas plus longest-track review remain dominated by foliage and ground; this identifies a material-support selection problem, not a measured intrinsic error.',
  '16headROI families contain6actualface,1hairline,9clothing rows. Facial support bysplit is4FIT/2heldout, with no independent subpixel uncertainty bound. Do not report16facial tracks.',
  'Eyelid appearance changes are visible, especially frame200, independently of any estimated projection. This proves some non-static image appearance; it does not quantify globalhead/bodymotion or explain all ghosting.',
  'All underlying2740tracks/1584families and1292camera-candidate families remain intact. Only72backgroundfamilies were visually reviewed; unreviewed families are not classified as wrong or unavailable.',
  'FirstK4000basis is unchanged. No new numerical reprojection error was measured, noK/pose estimated, no464-frame registration or training. Old hat failures remain unresolved.',
  'Next useful variable is semantic selection of existing observations on visible rigid structures, independent of residuals, not another detector/K/BA sweep. Audit whether the pool actually contains those materials before registering a new bounded geometry attempt.'
 ],
 source_hashes={n:sha(O/n) for n in ['observations.json','families.json','visual-index.json','context-index.json','result.json']},
 no_residual_based_selection=True,fit_heldout_unchanged=True,features_rerun=False,K_changed=False,new_cameras=0,training_started=False,quality_accepted=False)
save(O/'visual-review.json',review)
save(O/'progress.json',dict(state=review['state'],pid=None,updated=review['updated'],next='Read-only semantic allocation audit of frozen existing observations; do not rerun N12b extraction or relax this gate.'))
print(json.dumps(dict(counts=counts,nonground=nonground,camera=False,training=False),ensure_ascii=False))
