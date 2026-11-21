"""Finish JSON metadata from retained outputs after numpy-int serialization failure."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from review_full_sequence123 import read,sha,relocate

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle/local-face-input-control'
F=O.parent/'full94-shared-shape'
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

plan=read(O/'plan.json');failure=read(O/'observation-failure.json')
assert 'int64 is not JSON serializable' in failure['error']
assert not (O/'observation-result.json').exists()
assert all(sha(relocate(p))==h for p,h in plan['source_hashes'].items())
with np.load(O/'observations.npz',allow_pickle=False) as z:
    ids=list(z['frame_ids'].astype(str));fan=z['fan68_native'];old=z['rtmw51_native'];hm=z['heatmaps']
assert ids==plan['frames'] and fan.shape==(3,68,2) and old.shape==(3,51,2) and hm.shape==(3,1,68,64,64)
assert all(np.isfinite(x).all() for x in [fan,old,hm])
rows={r['frame']:r for r in read(F/'data/training-inputs.json')['records']}
records=[];images=[]
for i,frame in enumerate(ids):
    row=rows[frame];assert sha(relocate(row['image']))==row['source_sha256']
    box=np.array(row['face_box_xyxy']);center=(box[:2]+box[2:])/2.;center[1]-=(box[3]-box[1])*.12
    scale=float((box[2]-box[0]+box[3]-box[1])/195.)
    d=np.linalg.norm(fan[i,17:]-old[i],axis=1)
    scores=hm[i,0].reshape(68,-1).max(1)
    records.append(dict(frame_id=frame,source_frame=row['source_frame_index'],source_sha256=row['source_sha256'],
        face_box=box.tolist(),center=center.tolist(),scale=scale,network_input_wh=[256,256],input_crop_extent_native=200.*scale,
        face51_displacement_px=dict(median=float(np.median(d)),maximum=float(d.max())),
        scores_min=float(scores.min()),scores_max=float(scores.max()),inference_seconds=None,
        physical_visibility='not_established',quality_accepted=False))
    paths=[O/f'{frame}-face.png',O/f'{frame}-points.png'];assert all(p.is_file() for p in paths)
    images.append(dict(frame_id=frame,context=str(paths[0]),eight_points=str(paths[1]),context_sha256=sha(paths[0]),points_sha256=sha(paths[1])))
result=dict(updated=datetime.now(timezone.utc).isoformat(),state='complete_metadata_recovered_from_existing_outputs',
    records=records,images=images,actual_visual_review=False,observations_sha256=sha(O/'observations.npz'),plan_sha256=sha(O/'plan.json'),
    recovered_only_metadata=True,recovery_gpu_calls=0,inference_repeated=False,
    original_failure_retained=str(O/'observation-failure.json'),original_source_unchanged=True,
    exact_inference_timing_not_recovered=True,fit_updates=0,gaussian_steps=0,new_downloads=0)
save(O/'observation-result.json',result);save(O/'observation-progress.json',dict(state='complete_metadata_recovered',pid=None))
print(json.dumps(dict(state=result['state'],frames=ids,rows=records),ensure_ascii=False))
