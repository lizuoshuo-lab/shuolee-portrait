"""CPU-only projection/epipolar fixture, no learned model or subject optimization."""
import datetime as dt
import json
import numpy as np
from scipy.spatial.transform import Rotation
from run_roma116_diagnostic import R,epipolar_distance
from evaluate_pixsfm116_candidate import project,point_fit

k=np.array([1459.3513748328846,762.0972081111069,1351.114577699115,.0011426040608948861])
e=np.repeat(np.eye(4)[None],3,axis=0)
e[1,:3,:3]=Rotation.from_euler('y',6,degrees=True).as_matrix()
e[2,:3,:3]=Rotation.from_euler('y',-11,degrees=True).as_matrix()
centers=np.array([[0,0,0],[.3,.04,0],[-.5,.1,.03]])
e[:,:3,3]=-np.einsum('nij,nj->ni',e[:,:3,:3],centers)
points=np.array([[.1,.2,3.2],[-.4,-.5,4.4],[.6,-.6,2.8]])
obs=np.array([project(x,e,k)[0] for x in points])
epi=epipolar_distance(obs[:,0],obs[:,2],e[0],e[2],k)
perturbed=obs[:,2].copy();perturbed[:,1]+=10
bad=epipolar_distance(obs[:,0],perturbed,e[0],e[2],k)
fitted=[];held=[]
for x,p in zip(points,obs):
    y,projection,z,ok=point_fit(np.c_[np.arange(3),p],np.array([False,True,False]),e,k)
    assert ok and (z>0).all();fitted.append(np.linalg.norm(x-y));held.append(np.linalg.norm(projection[1]-p[1]))
assert epi.max()<1e-8 and bad.min()>5
assert max(fitted)<1e-8 and max(held)<1e-8
result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),scope='CPU synthetic distinct camera poses, known SIMPLE_RADIAL, no subject model inference',
    exact_epipolar_max_undistorted_px=float(epi.max()),wrong_match_epipolar_min_undistorted_px=float(bad.min()),
    reconstructed_xyz_max_error=float(max(fitted)),held_projection_max_raw_px=float(max(held)),passed=True,
    physical_074_calibration_verified=False,roma_neural_inference_run=False)
(R/'geometry-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
