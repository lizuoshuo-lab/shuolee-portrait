"""Fresh training-frame726 visual hull and shared canonical material bindings."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2,torch
from scipy.spatial import cKDTree
from train_enerf_camera10_time import visual_hull,tensor
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-second-interval-079';O=R/'initialization-expanded-y';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'initial.npz').exists();assert (R/'motion-audit/audit.json').exists();torch.set_num_threads(4)
ref=np.load(R/'head/reference.npz');v=ref['vertices'].reshape(-1,3)
save(O/'protocol.json',dict(status='fixed_before_hull_creation',reference_frame=726,
 inputs='Only15training cameras at originalrawframe726, originalnativeRGB/matte; no oddtime or00/06/12input read.',
 geometry='Same prior14-of15mask agreement,7x7dilation,0.012world-unit voxel shell anddepth-awaretrainingcolor averaging. Bounds from newfittedSMPL min/max expanded0.75worldunits, plus0.5 alongpositiveY.',
 bounds_correction='First attempt preserved ininitialization/:28occupiedvoxels touchedpositiveYface. Expandonlythatface0.5; do not clip or silentlydrop thosevoxels.',
 approximation='Images have frozen clock offsets; initial raw-frame visual hull approximates canonical physical726. Both later appearance branches share identical initialization and must optimize it.',
 bindings='Nearest final079head-refined canonical SMPL vertex; approximate for garments/toys/hair. No individual finger or object articulation introduced.',goal_complete=False))
hi=v.max(0).astype(np.float64);hi[1]+=.5
save(O/'geometry-audit.json',dict(point_quantiles=[v.min(0).tolist(),np.median(v,0).tolist(),hi.tolist()],frame=726,source='New fitted canonicalSMPL bounds; positiveYexpanded0.5afterrecordedboundaryfailure; valuesareexplicitboundsanchorsnotstatisticalquantiles'))
obs=[];inputs=[]
for c in read(R/'protocol.json')['training_cameras']:
 path=R/'native-input'/f'camera{c}-frame000726.npz';z=np.load(path);m=z['matte'];inputs.append(dict(camera=c,frame=726,sha256=sha(path)))
 obs.append(dict(id=c,rgb=tensor(z['rgb']/255),matte=tensor(m),valid=tensor(z['valid']),K=tensor(z['K']),w2c=tensor(z['w2c']),dilated=tensor(cv2.dilate(m,np.ones((7,7),np.uint8)))))
init=visual_hull(O,obs);dist,ids=cKDTree(v).query(init['points'],workers=4)
np.savez_compressed(R/'initial.npz',points=init['points'],colors=init['colors'],anchor=ids.astype(np.int64),anchor_distance=dist)
save(O/'binding-audit.json',dict(status='newtrainingonly_visualhull_and_canonical726_bindings_ready',inputs=inputs,points=len(ids),person_counts=[int((ids//6890==i).sum()) for i in [0,1]],
 distance_quantiles=np.quantile(dist,[0,.5,.9,.99,1]).tolist(),initial_sha256=sha(R/'initial.npz'),reference_sha256=sha(R/'head/reference.npz'),heldout_inputs_read=False,goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name)
print('NEW_INTERVAL_INITIALIZATION',len(ids),flush=True)
