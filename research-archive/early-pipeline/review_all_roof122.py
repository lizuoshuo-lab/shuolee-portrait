"""Record N12f native-pixel review before geometric fitting or residual inspection."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from itertools import combinations
import json, hashlib
import numpy as np
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
O=I/'clip03-grid-coverage/array-cache-repair'; D=I/'clip03-roof-review'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (D/'visual-review.json').exists()
p=read(I/'clip03-roof-review-plan.json'); v=read(D/'visual-index.json'); old=read(O/'visual-review.json'); ov=read(O/'visual-index.json')
for name,h in p['input_hashes'].items(): assert sha(Path(name))==h
assert len(v['rows'])==31 and len(v['pages'])==6 and len(v['context_pages'])==8
uncertain={
 18:'The building and floor order agree, but the selected center/support lies at glazing and the projecting white floor band. The precise opaque material point is ambiguous at this resolution.',
 21:'The selected support lies along the outer red column adjacent to glazing/sky. The same building edge is visible, but a unique fixed surface point along the edge is not established.',
 25:'The crop is dominated by the outer vertical building outline against sky and a narrow white side face. Aperture/depth-boundary ambiguity remains; the same building alone is insufficient.'}
rows=[]; accepted={}
for t in v['rows']:
 n=t['row']; good=n not in uncertain
 reason=uncertain.get(n,'The red end wall, vent, beam/column order and floor-band corners anchor the same structural neighborhood across every available view. No visible bay/floor switch. Coarse fixed-architecture candidate only; localization and mixed-depth support remain uncalibrated.')
 rows.append(dict(row=n,family=t['family'],track=t['track'],split=t['split'],frames=[o['frame'] for o in t['observations']],label='roof_structure_coarse' if good else 'structural_edge_material_ambiguous',reason=reason,coarse_static_candidate=good,subpixel_identity_certified=False))
 if good: accepted[t['family']]=dict(**t,review_source='N12f',material='roof_structure_coarse',nonground=True)
# Retain earlier image judgments; no new residual-driven relabelling.
for oldrow in old['rows']:
 if not oldrow['coarse_static_candidate']: continue
 t=ov['rows'][oldrow['row']]
 if t['semantic_category']=='distant_roof':
  assert t['family'] in accepted
  continue
 assert t['family'] not in accepted
 accepted[t['family']]=dict(**t,review_source='N12d',material=oldrow['reviewed_material'],nonground=oldrow['coarse_static_nonground_candidate'])
tracks=sorted(accepted.values(),key=lambda t:t['family']); frames=v['columns']
counts={s:sum(t['split']==s for t in tracks) for s in ['fit','heldout']}
ng={s:sum(t['split']==s and t['nonground'] for t in tracks) for s in ['fit','heldout']}
coverage=[]
for f in frames:
 obs=[(t,next(o for o in t['observations'] if o['frame']==f)) for t in tracks if any(o['frame']==f for o in t['observations'])]
 bins=sorted({(min(2,int(o['uv'][0]*3/1520)),min(2,int(o['uv'][1]*3/2704))) for t,o in obs if t['split']=='fit'})
 coverage.append(dict(frame=f,fit=sum(t['split']=='fit' for t,o in obs),heldout=sum(t['split']=='heldout' for t,o in obs),nonground_fit=sum(t['split']=='fit' and t['nonground'] for t,o in obs),nonground_heldout=sum(t['split']=='heldout' and t['nonground'] for t,o in obs),fit_bins_3x3=bins))
pairs=[]
for a,b in combinations(frames,2):
 common=[t for t in tracks if t['split']=='fit' and {a,b}<=set(o['frame'] for o in t['observations'])]
 pairs.append(dict(frames=[a,b],fit=len(common),nonground_fit=sum(t['nonground'] for t in common)))
total_gate=counts['fit']>=20 and counts['heldout']>=8 and ng['fit']>=6 and ng['heldout']>=3
frame_gate=all(x['fit']>=8 and len(x['fit_bins_3x3'])>=3 for x in coverage)
save(D/'reviewed-support.json',dict(tracks=tracks,frames=frames,all_observations_unchanged=True,old_whole_family_split_preserved=True,fit_coordinates='Original frozen N12d SIFT only. N12e photometric measurements are diagnostic and not substituted.',candidate_camera=None))
review=dict(stage='N12f',updated=datetime.now(timezone.utc).isoformat(),state='exhaustive_roof_material_review_complete',actual_review=dict(native128_rows=list(range(31)),native256_context_rows=list(range(31)),pages=v['pages'],context_pages=v['context_pages'],all_available_observations_reviewed=True,annotations_in_images=False),rows=rows,roof_counts=dict(Counter(t['split'] for t in v['rows'] if t['row'] not in uncertain)),ambiguous_rows=sorted(uncertain),coarse_combined=counts,coarse_nonground=ng,per_frame=coverage,pair_support=pairs,gate=dict(counts_passed=total_gate,per_frame_8fit_3bins_passed=frame_gate,initial_pair_12fit_available=any(x['fit']>=12 for x in pairs),requires_registered_fresh_geometry=True,thresholds_unchanged=True,training=False),limitations=['All roof candidates are concentrated on one distant building; many observations are only in frames232..280. Counts are not independent depth coverage.', 'Seven frames are a diagnostic subset of464, not registered full video.', 'Near-duplicate families remain defined by the frozen8px whole-family rule; visual structural correspondence does not certify <1px ground truth.', 'Three ambiguous outlines/glazing supports were retained in the evidence and excluded before any geometric residual.', 'N12d two nonground heldout count described its reviewed sample, not all detected roof candidates. This exhaustive audit corrects review coverage without new extraction or split changes.'],source_hashes={n:sha(D/n) for n in ['result.json','visual-index.json','reviewed-support.json']},original_data_hashes=p['input_hashes'],new_features=False,new_matches=False,K_changed=False,new_cameras=0,training_started=False,quality_accepted=False)
save(D/'visual-review.json',review)
print(json.dumps(dict(counts=counts,nonground=ng,per_frame=coverage,adjacent=[x for x in pairs if x['frames'][1]-x['frames'][0]==16],gate=review['gate']),ensure_ascii=False))
