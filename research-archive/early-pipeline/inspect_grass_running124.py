import json
from pathlib import Path
import numpy as np
r=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
o=r/'cross-subject-sh0-124/grass-temporal'
b=r/'smplifyx-reference-control/official-first-cycle/full94-shared-shape'
p=json.loads((o/'temporal-progress.json').read_text())
records=[]
for item in p['records']:
 n=item['frame']; f=o/'official-fit/frames'/n/'winner-camera-space.npz'
 if not f.exists():continue
 with np.load(f) as z,np.load(b/'frames'/n/'winner-camera-space.npz') as old:
  records.append(dict(frame=n,camera_equal=np.array_equal(z['camera_translation'],old['camera_translation']),K_equal=np.array_equal(z['K_integer_centers'],old['K_integer_centers']),shape_equal=np.array_equal(z['betas'],old['betas']),world_arrays='vertices_world' in z,finite=bool(np.isfinite(z['vertices_camera']).all())))
print(json.dumps(dict(committed=len(records),all_interface_parity=all(x['camera_equal'] and x['K_equal'] and x['shape_equal'] and x['finite'] for x in records),frames=[x['frame'] for x in records]),indent=2))