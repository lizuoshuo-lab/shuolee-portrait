from pathlib import Path
import json, hashlib, os
import numpy as np
import cv2
B=Path(('C:' if os.name=='nt' else '/mnt/c')+'/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/full-scene-localization-074'
R=B/'current/person-static-shared-face-fresh-098'
O=B/'current/shared-point-ray-consistency-099';O.mkdir(exist_ok=True)
cam=json.loads((S/'cameras.json').read_text(encoding='utf-8'))
g=json.loads((S/'camera-gauge-comparison.json').read_text(encoding='utf-8'))
face=np.load(R/'shared-targets/shared-face-points.npz')
Er=np.array([r['w2c'] for r in cam['records']]); E=face['w2c']; K=face['K'][0]
aligned=np.load(S/'camera-comparison.npz')['new_aligned']
assert np.array_equal(E,aligned)
scale=g['scale'];Q=np.array(g['rotation']);t=np.array(g['translation'])
# Raw world Xr -> aligned world Xa = scale * Q @ Xr + t.
Ra=Er[:,:3,:3]@Q.T
ta=Er[:,:3,3]*scale-np.einsum('fij,j->fi',Ra,t)
assert np.max(abs(E[:,:3,:3]-Ra))<1e-10
assert np.max(abs(E[:,:3,3]-ta))<1e-10
f,cx,cy,k=cam['intrinsics'];Kr=np.array([[f,0,cx-.5],[0,f,cy-.5],[0,0,1.]])
Ka=K.copy();Ka[:2,2]-=.5
X=face['xyz'][face['point_available']]
Xr=(X-t)@Q/scale
cr=np.einsum('fij,pj->fpi',Er[:,:3,:3],Xr)+Er[:,None,:3,3]
xy=cr[...,:2]/cr[...,2:]
raw=xy*(1+k*(xy*xy).sum(-1,keepdims=True))*f+np.array([cx-.5,cy-.5])
rect=cv2.undistortPoints(raw.reshape(-1,1,2),Kr,np.array([k,0,0,0,0.]),P=Ka).reshape(raw.shape)
ca=np.einsum('fij,pj->fpi',E[:,:3,:3],X)+E[:,None,:3,3]
uv=ca[...,:2]/ca[...,2:]*f+K[:2,2]-.5
err=float(np.max(abs(rect-uv)));assert err<1e-8
out=dict(raw_and_training_w2c_identical=False,reason='A single global similarity changes the world gauge; undistortion changes the principal point. Both preserve the same viewing rays.',
 global_similarity_scale=scale,global_similarity_rotation=Q.tolist(),global_similarity_translation=t.tolist(),
 raw_intrinsics=cam['intrinsics'],training_intrinsics=K.tolist(),
 raw_to_training_projection_max_px=err,training_equals_published_aligned_cameras=True,
 source_updates=0,physical_calibration_accuracy_proven=False)
(O/'camera-coordinate-mapping.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
