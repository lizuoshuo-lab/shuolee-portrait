"""Correct report-region semantics without changing any observation or solve."""
from pathlib import Path
import hashlib,json
from datetime import datetime
import numpy as np
from PIL import Image
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/material-support-gradient-110';P=B/'current/background-architecture-109'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def stats(x):
    x=np.asarray(x)
    return dict(n=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9))) if len(x) else dict(n=0)
rows=read(B/'current/person-static-shared-face-fresh-098/data/training-inputs.json')['records']
bounds=[]
for row in rows:
    p=Path(row['mask'].replace('/mnt/c/','C:/'));m=np.asarray(Image.open(p).convert('L'))>127
    yy=np.flatnonzero(m.any(1));bounds.append((int(yy[0]),int(yy[-1])))
def region(group,frame,y):
    if group in (1,2):return {1:'face_box_lower',2:'face_box_upper'}[group]
    lo,hi=bounds[frame];v=(y-lo)/max(1,hi-lo)
    return 'head_height_outside_face_box' if v<.24 else 'upper_body_height' if v<.62 else 'lower_body_height'
LABELS=dict(face_box_lower='面部框下半部',face_box_upper='面部框上半部',
            head_height_outside_face_box='面框外头部高度带',upper_body_height='肩臂与躯干高度带',lower_body_height='腿鞋高度带')
records=read(O/'records.json');groupids=dict(face_lower=1,face_upper=2,torso=3,legs_shoes=4)
new=[]
for r in records:
    q=dict(r);q['reported_region']=region(groupids[r['group']],r['raw']-1,r['observed_uv'][1]);new.append(q)
save(O/'records-regions.json',new)
gradient={}
for k in LABELS:
    rr=[r for r in new if r['reported_region']==k and not r['empty']]
    gradient[k]=dict(queries=len(rr),center_opposition_count=sum(r['center_descent_opposes_fixed_assignment'] for r in rr),
        metrics={name:stats([r[name] for r in rr]) for name in ['xyz_gradient_cosine','rasterweight_gradient_over_direct','center_target_rms_world','within_pixel_center_scatter_world']})
with np.load(B/'current/shared-point-bundle-convergence-101/input.npz') as z:
    obs=z['observations'];off=z['offsets'];tid=z['original_track_ids'];held=z['held'];groups=z['groups']
lookup={int(t):(int(lo),int(hi)) for t,lo,hi in zip(tid,off[:-1],off[1:])}
details=read(P/'evaluation-tracks.json');summary={}
for arm in ['fixed_cameras','background_bundle']:
    values={k:[] for k in LABELS};counts={k:set() for k in LABELS}
    for r in details[arm]['records']:
        if all(g in [5,6] for g in r['held_group']):continue
        lo,hi=lookup[r['id']];oo=obs[lo:hi];mask=held[lo:hi];gg=groups[lo:hi][mask]
        assert np.array_equal(oo[mask,0],r['frames']) and np.array_equal(gg,r['held_group'])
        for ob,g,err in zip(oo[mask],gg,r['error']):
            if g==0:continue
            name=region(int(g),int(ob[0]),ob[2]);values[name].append(err);counts[name].add(r['id'])
    summary[arm]={k:dict(**stats(v),tracks=len(counts[k])) for k,v in values.items()}
errata=dict(created=datetime.now().astimezone().isoformat(),
    source='audit_camera_residual_080.py:regions_for_observations; reporting labels only, applied after image-only tracking.',
    finding='Legacygroup3 namedtorso means all foreground above62percent personheight outside the cropped facebox. It includes hair and neck. Label is broader than torso.',
    confirmed_example=dict(raw=136,track_id=9609,legacy_group='torso',observed_location='hair_top_in_source_photo'),
    interpretation='A reporting-label defect. Not proof of a camera-coordinate mapping defect; labels were not used as semantic part targets in Gaussian098.',
    correction='Preserve original arrays and legacy metrics. Re-report held observations by original foreground-height bands0-.24/.24-.62/.62-1, retaining facebox subdivisions. Height bands are not anatomical segmentation.',
    labels=LABELS,gradient_region_summary=gradient,camera109_height_reanalysis=summary,
    source_model_changed=False,cameras_changed=False,queries_added_or_removed=False,
    original109_candidate_still_rejected=True)
save(O/'region-reporting-correction.json',errata)
print(json.dumps(dict(gradient_regions={k:v['queries'] for k,v in gradient.items()},camera109=summary),ensure_ascii=False))
