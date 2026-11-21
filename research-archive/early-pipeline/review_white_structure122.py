"""Actual N12c native-context review; retained observations, no residual or refitting."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, collections
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
A=I/'clip03-structure-allocation'; W=I/'clip03-native-matching'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (A/'visual-review.json').exists()
v=read(A/'visual-index.json'); c=read(A/'context-index.json'); r=read(A/'result.json')
assert len(v['rows'])==20 and len(v['pages'])==4 and len(c['rows'])==16
assert {n:sha(A/n) for n in r['hashes']}==r['hashes']
assert {n:sha(A/n) for n in c['frozen_hashes']}==c['frozen_hashes']
rows=[]
for t in v['rows']:
    n=t['row']; coarse=False
    if n==3:
        label='hydrant_valve_coarse_material_candidate';coarse=True
        reason='Center is on the hydrant valve detail. The neighboring hydrants and tree supports keep the same order in all three observed views; no confirmed object switch here. Fine surface/contour localization remains unmeasured.'
    elif n==11:
        label='stair_riser_coarse_material_candidate';coarse=True
        reason='Center lies on the textured stone riser, with the same steps and background tree supports in all three observations. Coarse identity is plausible; repetitive stone texture and exact subpixel location remain unmeasured.'
    elif n in {2,5}:
        label='hydrant_outline_or_rim_with_foliage_uncertain'
        reason='Hydrant context is stable but the selected center/support approaches the rim or outer contour against vegetation. It is not independently confirmed as a persistent point on one surface. Do not infer rigid point identity merely from the object in the crop.'
    elif n in {0,9}:
        label='pole_step_or_vegetation_depth_junction_uncertain'
        reason='Center/support lies near the pole boundary or its base, with steps and vegetation at other depths. A visible pole does not establish a material point. Preserve as ambiguous before any geometry.'
    elif n in {17,18,19}:
        label='facade_vegetation_fence_overlap_uncertain'
        reason='Large context contains distant repetitive windows and a foreground fence/vegetation layer. The selected center/support cannot be certified as a unique facade surface point from these crops.'
    else:
        label='foliage_inside_structure_search_region'
        reason='Native crop center is in leaves/branches beside the nominal object. Search-region inclusion is not static-object or material certification.'
    rows.append(dict(row=n,track=t['track'],family=t['family'],split=t['split'],frames=[o['frame'] for o in t['observations']],
      search_label=t['semantic_category'],reviewed_material=label,reason=reason,coarse_static_nonground_candidate=coarse,
      subpixel_physical_identity_certified=False,camera_fit_admitted=False,context_reviewed=n in {x['row'] for x in c['rows']}))
counts=collections.Counter(x['reviewed_material'] for x in rows)
coarse={s:sum(x['coarse_static_nonground_candidate'] and x['split']==s for x in rows) for s in ['fit','heldout']}
assert coarse==dict(fit=2,heldout=0)
review=dict(stage='N12c',updated=datetime.now(timezone.utc).isoformat(),state='readonly_allocation_and_material_review_complete_no_geometry',
 actual_review=dict(native128px_rows=list(range(20)),pages=v['pages'],native256px_context_rows=[x['row'] for x in c['rows']],all_observations_reviewed=True,annotations_in_images=False),
 counts=counts,rows=rows,new_coarse_nonground_candidates=coarse,combined_with_N12b=dict(fit=3,heldout=0),
 inherited_geometry_gate=dict(required_nonground=dict(fit=6,heldout=3),passed=False,retrospectively_relaxed=False,
   reason='Even counting the two newly plausible coarse material candidates, combined reviewed non-ground support3FIT/0heldout still fails6/3. No camera fit is run.'),
 selection_evidence=[
   dict(frame=184,region=0,material_search='upper lamp fixture',detected=232,eligible=230,selected=5,paired=0,cycled=0),
   dict(frame=184,region=1,material_search='lower lamp fixture',detected=163,eligible=163,selected=1,paired=0,cycled=0),
   dict(frame=232,region=17,material_search='distant facade including overlapping vegetation',detected=6432,eligible=6417,selected=11,paired=10,cycled=5),
   dict(frame=280,region=20,material_search='distant facade',detected=4584,eligible=4280,selected_background=0,cycled_background=0)
 ],
 confirmed_implementation='N12b selects top SIFT response separately in three large horizontal background bands, with no per-cell or per-material quota. It detects many points in built-structure search regions but can allocate nearly all descriptor slots to other high-response material. This is a verified coverage failure mechanism, not a measured causal share of ghosting.',
 limitations=[
  'Detector counts include multiple SIFT scales/orientations and non-rigid material inside coarse search polygons; they are not counts of independent physical points.',
  'Region counts overlap and must not be summed as unique counts. A lower selected count does not prove the discarded points would match correctly.',
  '43 unreviewed candidate families touched the search regions;20 were selected by the preregistered cap,23 remain unreviewed and are not labeled wrong.',
  'The seven frames are a subset of464. No new camera, residual metric, independent pixel uncertainty or training result exists.',
  'FirstK4000basis, family split and source observations are unchanged. No second undistortion or switch to4096/fovK.',
 ],
 next='Design one controlled selection-only comparison using the same native images and detector parameters, a fixed spatial/structure coverage allocation, and full detector metadata. Baseline remains the frozen N12b result. First preserve and verify exact detector identity; do not run threshold/model/BA sweeps or reuse old optimized geometry. Register the finite plan before execution. No new extraction has been authorized by this review alone.',
 source_hashes={n:sha(A/n) for n in ['result.json','allocation.json','visual-index.json','context-index.json']},
 original_observations_sha256=sha(W/'observations.json'),original_families_sha256=sha(W/'families.json'),
 K_changed=False,features_recomputed=False,matches_recomputed=False,new_cameras=0,training_started=False,quality_accepted=False)
save(A/'visual-review.json',review)
save(A/'progress.json',dict(state=review['state'],pid=None,updated=review['updated'],next=review['next']))
print(json.dumps(dict(counts=counts,new_coarse=coarse,combined=review['combined_with_N12b'],geometry=False),ensure_ascii=False))
