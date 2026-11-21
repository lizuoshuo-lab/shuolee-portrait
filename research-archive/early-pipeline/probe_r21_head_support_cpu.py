from pathlib import Path
import json,sys,hashlib
from collections import Counter
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/mnt/d/Projects/Shuolee-smpl-head-support-20261111/src')
from shuolee_portrait import runtime as rt
asset=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
with np.load(asset,allow_pickle=False) as a:
 dominant=a['weights'].argmax(1)
 lmk=np.unique(np.r_[a['lmk_faces_idx'].flatten(),a['dynamic_lmk_faces_idx'].flatten()])
 protected=np.isin(dominant,[12,15,22,23,24])
def Q(a): return np.round(np.percentile(a,[50,90,95,99,100]),6).tolist() if len(a) else []
def normalize(v): return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
result={}
for subject in ('016','006'):
 root=Path('/opt/gs-slam-lab')/('shuolee-smpl-delivery-round20-'+subject)/'run'
 out=root/'gaussian-run/step-010000'
 p=PlyData.read(str(out/'bound-local.ply'))['vertex'].data
 world=PlyData.read(str(out/'world.ply'))['vertex'].data
 npz=np.load(out/'reference-mesh.npz')
 verts,faces=npz['vertices_world'],npz['faces']
 mask=protected.copy(); mask[faces[lmk].flatten()]=True
 head=mask[faces].all(1)
 print(subject,p.dtype.names,'head_faces',head.sum())
 ids=p['binding_0'].astype(int)
 face=verts[faces]
 edge=face[:,1]-face[:,0]
 n=normalize(np.cross(edge,face[:,2]-face[:,0]))
 x=normalize(edge)
 z=normalize(np.cross(x,n))
 basis=np.stack((x,n,z),-1)
 scale=(np.linalg.norm(edge,axis=1)+np.abs(((face[:,2]-face[:,0])*z).sum(-1)))*.5
 local_xyz=np.stack([p[v] for v in 'xyz'],1)
 wxyz=face.mean(1)[ids]+np.einsum('pij,pj->pi',basis[ids],local_xyz)*scale[ids,None]
 assert np.max(np.abs(wxyz-np.stack([world[v] for v in 'xyz'],1)))<2e-6
 logs=np.stack([p[f'scale_{i}'] for i in range(3)],1)
 q=np.stack([p[f'rot_{i}'] for i in (1,2,3,0)],1)
 sigma=np.exp(logs)*scale[ids,None]
 rotation=Rotation.from_quat(q).as_matrix()
 sigma_n=np.linalg.norm(rotation[:,1]*sigma,axis=1)
 sigma_major=sigma.max(1)
 opacity=(1+np.exp(-p['opacity']))**-1
 signed=local_xyz[:,1]*scale[ids]
 outward=signed+3*sigma_n
 head_active=head[ids] & (opacity>=.10)
 face_active=head_active & np.isin(dominant[faces[ids]], [15,22,23,24]).all(1)
 data=dict(points=len(p),head_faces=int(head.sum()),head_points=int(head[ids].sum()),
  active_head=int(head_active.sum()),active_cranium=int(face_active.sum()),bound_ply_sha256=rt.sha(out/'bound-local.ply'),
  world_ply_sha256=rt.sha(out/'world.ply'),mesh_sha256=rt.sha(out/'reference-mesh.npz'))
 for name,sel in [('head',head_active),('cranium',face_active)]:
  data[name]=dict(sigma_major_m=Q(sigma_major[sel]),sigma_normal_m=Q(sigma_n[sel]),signed_center_m=Q(signed[sel]),
     outward_3sigma_m=Q(outward[sel]),opacity=Q(opacity[sel]),axis_ratio=Q((sigma_major/np.maximum(sigma.min(1),1e-8))[sel]),
     native_outward_masks={str(mm):int((sel & (outward>mm*.001)).sum()) for mm in (5,10,15,20,25,30)},
     anisotropic_normal_over5mm=int((sel & (sigma_n>.005)).sum()),
     long_axis_with_outward_support=int((sel & (sigma_major>.008)&(outward>.015)).sum()))
 print(json.dumps(data,indent=2))
 result[subject]=data
path=Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round21/input-audit/head-ellipsoid-baseline-cpu.json')
rt.save(path,dict(schema='shuolee-head-ellipsoid-baseline/1',subjects=result,smplx_asset_sha256=rt.sha(asset),
 quantiles=[.5,.9,.95,.99,1.],outward_metric='signed bound triangle normal offset plus 3 normal sigmas',
 interpretation='Reference pose geometric proxy; hair or clothing may be genuinely outside bare mesh.',
 source_script_sha256=rt.sha(__file__)))
print(path,rt.sha(path))
