"""Analytic counterexample: shared point-to-plane distance is not point agreement."""
from pathlib import Path
import json
import numpy as np

R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098')
x=np.array([0.,0.,3.]);ray=np.array([.02,0.,1.])
normals=np.array([[0.,0.,1.],[1.,0.,.02]])
normals/=np.linalg.norm(normals,axis=1,keepdims=True)
offsets=-(normals@x)
hits=ray[None]*(-offsets/(normals@ray))[:,None]
plane_error=normals@x+offsets
point_distance=np.linalg.norm(hits-x,axis=1)
assert np.max(np.abs(plane_error))<1e-12 and np.linalg.norm(hits[0]-hits[1])>1.5
result=dict(shared_point=x.tolist(),same_camera_ray=ray.tolist(),plane_normals=normals.tolist(),
            plane_offsets=offsets.tolist(),shared_point_plane_error=plane_error.tolist(),
            ray_plane_intersections=hits.tolist(),intersection_distance_to_target=point_distance.tolist(),
            intersection_separation=float(np.linalg.norm(hits[0]-hits[1])),
            conclusion='Two distinct planes can both contain the same shared 3D landmark while their intersections with an observed ray disagree. The 098 plane objective is therefore necessary at best, not a sufficient implementation of point-to-point convergence.',
            caveat='Synthetic algebraic counterexample proving a nullspace. It does not by itself establish which planes or camera errors caused any observed human artifact.',
            training_updates=0,camera_updates=0)
(R/'shared-plane-nullspace.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
