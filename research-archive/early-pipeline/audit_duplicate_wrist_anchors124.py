"""All-frame, read-only audit of duplicated SMPL-X wrists and native2D labels."""
import os
from collections import Counter
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read,save,sha,now
import official_smplifyx_entry123 as entry
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster
import audit_sofa_knee_objective124 as a

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
OUT=a.ROOT/'duplicate-wrist-anchor-audit'
assert not OUT.exists()
SOURCES={name:a.ROOT/(name+'-input') for name in ('winterhat','sofa','indoor')}
SOURCES['grass']=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
frozen={str(p):sha(p) for src in SOURCES.values() for p in (src/'data/inputs.json',src/'data/observations.npz')}
frozen[str(a.OFFICIAL/'smplifyx/utils.py')]=sha(a.OFFICIAL/'smplifyx/utils.py')
OUT.mkdir()
save(OUT/'plan.json',dict(updated=now(),hypothesis='The native body wrist and hand-root observations can disagree while the official joint mapper uses the exact same3D skeletal anchor for both. Quantify all frozen frames, without any fitting or confidence/coordinate changes.',
    budget=dict(cpu_only=True,subjects=4,optimizer_steps=0,new_observations=0),
    interpretation='For two2D targets separated by d pixels and one projected3D anchor, at least one residual is >=d/2. This is a geometric contradiction for a simultaneous1px target when d>=2, not proof which detector is physically correct.',
    sources={str(Path(__file__)):sha(__file__)},inputs=frozen))
utils,_,_,_=entry.load_official(a.OFFICIAL)
mapping=np.asarray(utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25'))
duplicates={int(k):np.where(mapping==k)[0].tolist() for k,v in Counter(mapping.tolist()).items() if v>1}
assert duplicates=={20:[7,25],21:[4,46]},duplicates
result={}
for subject,src in SOURCES.items():
    rows=read(src/'data/inputs.json')['records']
    with np.load(src/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    assert list(obs['frame_ids'].astype(str))==[r['frame'] for r in rows]
    conf,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    targets=np.zeros((len(rows),118,3),np.float32)
    dst,source=np.r_[entry.COCO,19:25,25:67,67:118],np.r_[0:23,91:133,40:91]
    targets[:,dst,:2]=obs['keypoints'][:,source];targets[:,dst,2]=conf[:,source]
    records=[]
    for i,row in enumerate(rows):
        t,_=mask_outside_raster(targets[i:i+1],row['wh'])
        for side,first,second in [('left',7,25),('right',4,46)]:
            x,y=t[0,first],t[0,second];distance=float(np.linalg.norm(x[:2]-y[:2]))
            records.append(dict(frame=row['frame'],side=side,body_xy=x[:2].tolist(),hand_root_xy=y[:2].tolist(),
                confidence=[float(x[2]),float(y[2])],both_active=bool(x[2]>0 and y[2]>0),
                distance_px=distance,max_residual_lower_bound_px=distance/2))
    active=[r for r in records if r['both_active']]
    values=np.array([r['distance_px'] for r in active])
    result[subject]=dict(frames=len(rows),active_pairs=len(active),pairs_incompatible_with_both_strictly_below1px=int((values>=2).sum()),
        distance_median_px=float(np.median(values)),distance_p90_px=float(np.quantile(values,.9)),
        distance_max_px=float(values.max()),records=records)
for path,digest in frozen.items():assert sha(path)==digest,path
save(OUT/'result.json',dict(updated=now(),subjects=result,duplicate_mapper_indices=duplicates,
    native_observations_unchanged=True,confidence_values_unchanged=True,new_fitter_calls=0,source_hashes_verified=True,
    quality_accepted=False,gaussian_admitted=False,
    limits=['Detector confidence is a relative heuristic, not calibrated physical accuracy.',
            'No conclusion that either body or hand-root label is the correct physical wrist location.',
            'This does not establish the unique cause of knee drift or all portrait ghosting.',
            'Do not pick frame-dependent winners, move2D labels, or relax the gate to make a fit pass.']))
print({s:{k:v for k,v in x.items() if k!='records'} for s,x in result.items()},flush=True)
