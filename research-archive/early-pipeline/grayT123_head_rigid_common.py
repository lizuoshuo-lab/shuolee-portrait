"""Fixed-data, diagnostic-only rigid head projection. No production mutations."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native,stats

OUT=REC/'grayT-head-rigid-review'
PLAN=REC/'grayT-head-rigid-diagnostic-plan.json'
MODEL=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')

def transform(p,center,radius):
    R=Rotation.from_rotvec(p[:3]).as_matrix();H=np.eye(4)
    H[:3,:3]=R;H[:3,3]=center-R@center+radius*p[3:]
    return H

def project(X,E,K,H=None):
    X=np.atleast_2d(X);E=np.asarray(E);E=E[None] if E.ndim==2 else E
    if H is None:H=np.eye(4)
    H=np.asarray(H)
    if H.ndim==2:
        q=X@H[:3,:3].T+H[:3,3];cam=np.einsum('nij,kj->nki',E[:,:3,:3],q)+E[:,:3,3,None].transpose(0,2,1)
    else:
        q=np.einsum('nij,kj->nki',H[:,:3,:3],X)+H[:,:3,3,None].transpose(0,2,1)
        cam=np.einsum('nij,nkj->nki',E[:,:3,:3],q)+E[:,:3,3,None].transpose(0,2,1)
    px=cam@K.T
    return px[...,:2]/px[...,2:],cam[...,2]

def jacobian(fun,p):
    cols=[]
    for j in range(len(p)):
        e=np.zeros_like(p);e[j]=1e-6
        cols.append((fun(p+e)-fun(p-e))/(2*e[j]))
    return np.stack(cols,axis=1)

def load():
    p=read(PLAN);files=[PLAN]+[Path(v) for v in p['sources'].values()]+[MODEL]
    hashes=[sha(f) for f in files]
    with np.load(ROOT/'fit/fitted.npz') as z:f={k:z[k] for k in ['frame_ids','vertices_world','faces','projected_face51']}
    with np.load(MODEL) as z:m={k:z[k] for k in ['lmk_faces_idx','lmk_bary_coords']}
    o=np.load(ROOT/'data/observations.npz');w=np.load(ROOT/'fit/observation-weights.npz');z=np.load(ROOT/'camera/solver-output.npz')
    rows=read(ROOT/'data/training-inputs.json')['records'];frames=np.array(p['frames']);ii=frames-1
    assert list(f['frame_ids'])==list(o['frame_ids'])==[r['frame'] for r in rows]
    assert o['joint_names'][50:59].tolist()==[f'face-{i}' for i in range(27,36)]
    assert np.max(np.abs(f['vertices_world']-f['vertices_world'][0]))==0
    tri=f['vertices_world'][0].astype(float)[f['faces']]
    face=(tri[m['lmk_faces_idx']]*m['lmk_bary_coords'][:,:,None]).sum(1)
    X=face[10:19];K=z['K'];E=z['w2c'][ii];uv=o['keypoints'][ii,50:59].astype(float)
    weight=(w['face'][ii,10:19]*w['face_surface_visibility'][ii,10:19]).astype(float)
    for i in ii:
        assert np.max(np.abs(np.array(rows[i]['K_integer_centers'])-K))==0
        assert np.max(np.abs(np.array(rows[i]['w2c'])-z['w2c'][i]))==0
    pred,_=project(X,E,K);delta=float(np.max(np.abs(pred-f['projected_face51'][ii,10:19])))
    assert delta<.001
    center=X.mean(0);radius=float(np.sqrt(np.mean(np.sum((X-center)**2,axis=1))))
    assert np.isfinite(uv).all() and np.isfinite(weight).all() and (weight>=0).all()
    return dict(plan=p,files=files,hashes=hashes,rows=rows,frames=frames,ii=ii,X=X,K=K,E=E,uv=uv,weight=weight,
        center=center,radius=radius,projection_delta=delta)
