from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-head-align-20261111')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime
from shuolee_portrait.head_alignment import fit_head_alignment, HEAD_COLUMNS
base=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official')
assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
proof=dict(schema='head-alignment-cpu-archive-proof-v1',state='passed',
           source_sha256=runtime.sha(PROJECT/'src/shuolee_portrait/head_alignment.py'),subjects={})
MUTABLE={'vertices_camera','vertices_world','joints_camera','joints_projected_native','body_pose_axis_angle'}
for s in ('016','006'):
 root=Path('/opt/gs-slam-lab')/('shuolee-smpl-delivery-round19-'+s)/'run'
 admission=runtime.read(root/'gaussian-run/face-supplement/admission.json')
 selection=runtime.read(root/'fit/selection.json')
 held=set(selection['retained_holdouts'])
 excluded={Path(row[key]) for row in runtime.read(root/'data/inputs.json')['records'] if row['frame'] in held
           for key in ('image','mask')}
 read=Path.read_bytes
 def observed(p):
  assert p not in excluded, p
  return read(p)
 with patch.object(Path,'read_bytes',observed):
  b,c,a=fit_head_alignment(root,assets=assets,official_root=base,auxiliary_frames=admission['chosen_frame_ids'])
 tracked={}
 for mode,d in (('fitted',b),('joint-candidates',c)):
  with np.load(root/f'fit/{mode}.npz',allow_pickle=False) as old:
   assert set(d)==set(old.files)
   assert all(d[k].dtype==old[k].dtype and d[k].shape==old[k].shape for k in old.files)
   assert all(np.array_equal(d[k],old[k]) for k in old.files if k not in MUTABLE)
   outside=np.setdiff1d(np.arange(63),HEAD_COLUMNS)
   assert np.array_equal(d['body_pose_axis_angle'][:,outside],old['body_pose_axis_angle'][:,outside])
   names=old['frame_ids'].tolist()
   unchanged=[i for i,f in enumerate(names) if f not in a['changed_frames']]
   assert all(np.array_equal(d[k][unchanged],old[k][unchanged]) for k in MUTABLE if k in old.files)
   tracked[mode]=[f for i,f in enumerate(names) if not np.array_equal(d['body_pose_axis_angle'][i],old['body_pose_axis_angle'][i])]
 assert set(tracked['joint-candidates'])==set(a['changed_frames'])
 assert len(a['fitted_frames'])==28 and len(a['retained_holdouts'])==4
 ear=[(q['before'][r]['rms_px'],q['after'][r]['rms_px']) for q in a['frames']
      for r in q['before'] if r.endswith('_ear')]
 proof['subjects'][s]=dict(audit=a,actual_changed=tracked,eligible_ear_observations=len(ear),
       mean_ear_before_px=float(np.mean([x for x,_ in ear])),
       mean_ear_after_px=float(np.mean([y for _,y in ear])),
       trial_failure_counts=dict(Counter(f for q in a['frames'] for t in q['trials'] for f in t['failures'])),
       archive_dtype_shapes_unchanged=True,fixed_arrays_unchanged=True,heldout_source_never_opened=True)
p=Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round20/input-audit/head-alignment-archive-cpu-proof.json')
runtime.save(p,proof)
print(json.dumps(dict(path=str(p),sha256=runtime.sha(p),subjects={s: {k:v for k,v in d.items()
      if k in ('actual_changed','eligible_ear_observations','mean_ear_before_px','mean_ear_after_px')}
      for s,d in proof['subjects'].items()}),indent=2))
