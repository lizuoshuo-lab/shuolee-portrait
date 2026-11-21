from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
from plyfile import PlyData
import torch
torch.set_num_threads(4)
sys.path.insert(0,'/mnt/d/Projects/Shuolee-smpl-head-support-20261111/src')
from shuolee_portrait.head_support import CONFIG, _geometry_gate, _proposal, _rotations, _support
from shuolee_portrait import runtime as rt
from utils.graphics_utils import compute_face_orientation

assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
with np.load(assets,allow_pickle=False) as m:
    protected=np.isin(m['weights'].argmax(1),[12,15,22,23,24])
    landmarks=np.unique(np.r_[m['lmk_faces_idx'].flatten(),m['dynamic_lmk_faces_idx'].flatten()])
def array(p,keys):return torch.from_numpy(np.stack([p[k] for k in keys],1).copy())

for s in ('016','006'):
    root=Path('/opt/gs-slam-lab')/('shuolee-smpl-delivery-round20-'+s)/'run/gaussian-run/step-010000'
    p=PlyData.read(str(root/'bound-local.ply'))['vertex'].data
    mesh=np.load(root/'reference-mesh.npz')
    faces=mesh['faces'].astype(np.int64)
    mask=protected.copy()
    mask[faces[landmarks].flatten()]=True
    head=torch.from_numpy(mask[faces].all(1))
    matrix,scale=compute_face_orientation(torch.from_numpy(mesh['vertices_world']),
                                          torch.from_numpy(faces),return_scale=True)
    binding=torch.from_numpy(p['binding_0'].astype(np.int64))
    xyz=array(p,'xyz')
    scaling=array(p,[f'scale_{i}' for i in range(3)])
    q=array(p,[f'rot_{i}' for i in range(4)])
    alpha=torch.from_numpy(p['opacity'].copy()).sigmoid()[:,None]
    dc=array(p,[f'f_dc_{i}' for i in range(3)])[:,None]
    rot=_rotations(q)
    geo=_support(xyz,scaling.exp(),rot,scale[binding,0])
    c=CONFIG
    allowed=head[binding]&(alpha[:,0]>=c['active_opacity'])&(geo['sigma_major_m']>=c['min_major_sigma_m'])
    allowed &= (geo['outward_3sigma_m']>=c['min_outward_support_m'])&(geo['sigma_normal_m']>=c['min_normal_sigma_m'])
    ids=allowed.nonzero(as_tuple=True)[0]
    score=geo['outward_3sigma_m']*geo['sigma_normal_m']*alpha[:,0]
    ids=ids[score[ids].argsort(descending=True)[:c['shortlist_limit']]]
    model=NS(_xyz=xyz,_scaling=scaling,binding=binding,get_opacity=alpha,_features_dc=dc)
    world=NS(reference_scale=scale[:,0],log_cap=torch.full((len(scale),),torch.inf),
             limits=NS(cap=torch.full_like(scaling,torch.inf)))
    centers,logs,opacities,colors,upper,center_cap,cap=_proposal(model,world,ids,rot[ids],c)
    assert centers.shape==(len(ids),3,3) and opacities.shape==(len(ids),3,1)
    assert torch.all(centers[:,:,1]<=xyz[ids,None,1]) and torch.all(logs<=upper[:,None])
    assert torch.equal(colors[:,0],dc[ids])
    assert torch.allclose((centers[:,1]+centers[:,2])/2,centers[:,0])
    gate=_geometry_gate(model,world,ids,rot[ids],NS(centers=centers,log_sigma=logs),c)
    assert gate['accepted']
    print(s,int(allowed.count_nonzero()),len(ids),min(gate['outward_reduction_m']),max(gate['outward_reduction_m']))
