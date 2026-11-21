"""Native silhouette supervision for shared mesh offsets and bound Gaussian center offsets."""
from pathlib import Path

import cv2
import numpy as np
import torch


class SurfaceContours:
    def __init__(self, directory, frames, actor, mode):
        self.mode=mode
        self.crops={}
        self.actor=actor
        self.tri=actor.mesh.faces.int().contiguous()
        if mode in ('mesh','both'):
            import nvdiffrast.torch as dr
            self.dr=dr
            self.ctx=dr.RasterizeCudaContext(device=self.tri.device)
        self.checked=set()
        for f in frames:
            K,_,R,C=f.rasterizer.calibration
            with np.load(directory/Path(f.name).with_suffix('.npz'),allow_pickle=False) as data:
                x,y=map(int,data['origin'])
                h,w=data['alpha'].shape
                edge=data['edge']!=0
                if not edge.any():
                    continue
                distance=cv2.distanceTransform((~edge).astype(np.uint8),cv2.DIST_L2,5)
                band=(distance<=max(6,round(min(h,w)*.055)))&data['usable']
                target=data['alpha'][band].astype(np.float32)/255
            gpu=lambda v: torch.as_tensor(v,device=self.tri.device,dtype=torch.float32)
            self.crops[f.name]=(x,y,w,h,gpu(K),gpu(R),gpu(C),
                torch.as_tensor(band,device=self.tri.device),gpu(target))
        if not self.crops:
            raise ValueError('No usable exterior contours for the training images')

    def mesh_image(self,frame):
        x,y,w,h,K,R,C,_,_=self.crops[frame.name]
        W,H=(w+7)//8*8,(h+7)//8*8
        vertices=self.actor.mesh.vertices_world(frame)
        camera=(vertices-C)@R
        pixel=camera@K.T
        z=camera[:,2]
        clip=torch.stack((2*(pixel[:,0]+(.5-x)*z)/W-z, 2*(pixel[:,1]+(.5-y)*z)/H-z,
                          z-.01,z),dim=1)[None].contiguous()
        raster,_=self.dr.rasterize(self.ctx,clip,self.tri,(H,W))
        hard=(raster[...,3:4]>0).to(clip)
        return self.dr.antialias(hard,raster,clip,self.tri)[0,:h,:w,0]

    def backward(self,frame,gaussian_alpha):
        if frame.name not in self.crops:
            return
        x,y,w,h,_,_,_,band,target=self.crops[frame.name]
        for key in ('mesh','centers'):
            if self.mode not in (key,'both'):
                continue
            if key=='mesh':
                predicted=self.mesh_image(frame)
                parameter=self.actor.mesh.offset
            else:
                predicted=gaussian_alpha[y:y+h,x:x+w]
                parameter=self.actor.head_offset
            # Targeted backward prevents opacity and covariance from absorbing the new silhouette term.
            loss=.15*(predicted[band]-target).abs().mean()
            grad,=torch.autograd.grad(loss,parameter,retain_graph=True)
            if not torch.isfinite(grad).all():
                raise RuntimeError(f'Nonfinite {key} contour gradient')
            if parameter.grad is None:
                parameter.grad=grad.detach()
            else:
                parameter.grad.add_(grad.detach())
            if key not in self.checked and grad.abs().max()>0:
                print(f'Native {key} contour gradient on {frame.name}: {float(grad.abs().max()):.7g}',flush=True)
                self.checked.add(key)
