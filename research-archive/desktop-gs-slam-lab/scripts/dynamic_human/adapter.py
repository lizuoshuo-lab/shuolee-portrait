"""Explicit gsplat/SciPy adapter for the pinned Dynamic3DGaussians source.

The original optimization and regularization code stays unmodified. This is a
local compatibility port, not a bitwise reproduction of the original rasterizer.
"""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree
from gsplat import rasterization

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'repos/Dynamic3DGaussians'
DATA=LAB/'data/dynamic-human/dynamic3dgaussians/data'
OUT=LAB/'outputs/dynamic-human-reconstruction-003'

class Renderer:
    def __init__(self,raster_settings):self.camera=raster_settings
    def __call__(self,means3D,colors_precomp,rotations,opacities,scales,means2D,**kwargs):
        c=self.camera
        image,alpha,info=rasterization(means3D,rotations,scales,opacities.squeeze(-1),colors_precomp,c.w2c[None],c.K[None],c.width,c.height,near_plane=c.near,far_plane=c.far,packed=False,rasterize_mode='classic',backgrounds=torch.zeros((1,3),device='cuda'))
        if info['means2d'].requires_grad:
            # gsplat returns derivatives in pixels; the original density-control
            # threshold is applied to normalized-device-coordinate derivatives.
            factor=means3D.new_tensor([c.width/2,c.height/2])
            def keep_screen_gradient(gradient):
                means2D.grad=torch.nn.functional.pad(gradient[0]*factor,(0,1)).detach()
                return gradient
            info['means2d'].register_hook(keep_screen_gradient)
        radii=info['radii'][0]
        if radii.ndim==2:radii=radii.max(-1).values
        return image[0].permute(2,0,1),radii,alpha[0,...,0]

def setup_camera(w,h,k,w2c,near=.01,far=100):
    return SimpleNamespace(width=w,height=h,K=torch.tensor(k,dtype=torch.float32,device='cuda'),w2c=torch.tensor(w2c,dtype=torch.float32,device='cuda'),near=near,far=far)

def knn(points,k):
    distances,indices=cKDTree(np.asarray(points,np.float64)).query(points,k=k+1,workers=-1)
    return distances[:,1:]**2,indices[:,1:]

def load_official():
    raster=ModuleType('diff_gaussian_rasterization');raster.GaussianRasterizer=Renderer;raster.GaussianRasterizationSettings=SimpleNamespace
    sys.modules['diff_gaussian_rasterization']=raster
    sys.modules.setdefault('open3d',ModuleType('open3d'))
    sys.path.insert(0,str(ROOT))
    import helpers
    helpers.setup_camera=setup_camera;helpers.o3d_knn=knn
    import train
    return train,helpers

def dataset(t,metadata,sequence='basketball',camera_indices=None):
    from PIL import Image
    result=[]
    for c in range(len(metadata['fn'][t])) if camera_indices is None else camera_indices:
        fn=metadata['fn'][t][c];image=np.asarray(Image.open(DATA/sequence/'ims'/fn).convert('RGB')).copy()
        maskfile=DATA/sequence/'seg'/fn.replace('.jpg','.png')
        seg=np.asarray(Image.open(maskfile)).astype(np.float32) if maskfile.exists() else None
        item=dict(cam=setup_camera(metadata['w'],metadata['h'],metadata['k'][t][c],metadata['w2c'][t][c],near=1,far=100),im=torch.from_numpy(image).to('cuda',torch.float32).permute(2,0,1)/255,id=c,filename=fn)
        if seg is not None:
            assert seg.max()<=1 and seg.min()>=0
            tensor=torch.from_numpy(seg).cuda();item['seg']=torch.stack((tensor,torch.zeros_like(tensor),1-tensor))
        result.append(item)
    return result

def initialize(train,metadata):
    # Only path resolution differs from the author's initialize_params.
    original=np.load
    def load(path,*args,**kwargs):
        if str(path).replace('\\','/')=='./data/basketball/init_pt_cld.npz':path=DATA/'basketball/init_pt_cld.npz'
        return original(path,*args,**kwargs)
    np.load=load
    try:return train.initialize_params('basketball',metadata)
    finally:np.load=original
