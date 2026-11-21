"""Articulated hand surface Gaussians with shared color and bounded local offsets.

Local research implementation: MANO supplies topology/continuous initialization,
not ground truth. New samples composite in the same depth ordering as existing GS.
"""
import numpy as np
import torch
from hamer.models.mano_wrapper import MANO
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from enerf_residual_graph import rotation_matrix
from fit_hamer_consensus import basis

class ManoGaussianSurface(torch.nn.Module):
    def __init__(self,parameters,clock_path,scene_scale,color,refine_pose,device='cuda'):
        super().__init__();self.scene_scale=float(scene_scale);self.refine_pose=bool(refine_pose)
        self.mano=MANO(model_path='/mnt/d/GS-SLAM-Data/models/mano_v1_2/numpy',is_rhand=True,use_pca=False,flat_hand_mean=True).to(device)
        self.mano.requires_grad_(False)
        z=np.load(parameters)
        for k in ['root_base','pose','translation','betas','log_scale']:self.register_buffer('base_'+k,torch.as_tensor(z[k],dtype=torch.float32,device=device))
        clock=np.load(clock_path);self.lookup={(str(c),int(f)):i for i,(c,f) in enumerate(zip(clock['camera'],clock['frame']))}
        self.register_buffer('time_basis',torch.tensor(basis(clock['capture_time']),dtype=torch.float32,device=device))
        self.register_buffer('faces',torch.tensor(self.mano.faces,dtype=torch.long,device=device));n=len(self.faces)
        self.pose_delta=torch.nn.Parameter(torch.zeros_like(self.base_pose),requires_grad=refine_pose)
        self.translation_delta=torch.nn.Parameter(torch.zeros_like(self.base_translation),requires_grad=refine_pose)
        self.offset_raw=torch.nn.Parameter(torch.zeros((n,3),device=device))
        self.log_scale_delta=torch.nn.Parameter(torch.zeros((n,3),device=device))
        self.opacity_logits=torch.nn.Parameter(torch.full((n,),float(np.log(.02/.98)),device=device))
        colors=torch.as_tensor(color,dtype=torch.float32,device=device).clamp(.02,.98).expand(n,3).clone()
        self.color_logits=torch.nn.Parameter(torch.logit(colors))

    def geometry(self,camera,frame):
        B=self.time_basis[self.lookup[camera,frame]][None]
        pose=self.base_pose+.25*self.pose_delta.tanh()
        tr=self.base_translation+.01*self.scene_scale*self.translation_delta.tanh()
        rv=(B@pose.reshape(len(B[0]),-1)).reshape(1,16,3);rot=rotation_matrix(rv)
        out=self.mano(global_orient=torch.eye(3,device=B.device)[None,None],hand_pose=rot[:,1:],betas=self.base_betas[None],pose2rot=False)
        R=rot[:,0]@self.base_root_base;vertices=((out.vertices-out.joints[:,:1])*self.base_log_scale.exp())@R.transpose(-1,-2)+(B@tr)[:,None]
        tri=vertices[0,self.faces];e=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
        x=torch.nn.functional.normalize(e,dim=-1);normal=torch.nn.functional.normalize(torch.cross(e,e2,dim=-1),dim=-1);y=torch.cross(normal,x,dim=-1);F=torch.stack([x,y,normal],-1)
        means=tri.mean(1)+(F@(.005*self.scene_scale*self.offset_raw.tanh())[...,None])[...,0]
        edge_rms=torch.sqrt((e.square().sum(-1)+e2.square().sum(-1)+(tri[:,2]-tri[:,1]).square().sum(-1))/3).clamp_min(1e-5)
        initial=torch.stack([.55*edge_rms,.55*edge_rms,.0825*edge_rms],-1)
        scales=initial*torch.exp(np.log(3)*self.log_scale_delta.tanh())
        cov=(F*scales.square()[:,None,:])@F.transpose(-1,-2)
        return means,cov,self.opacity_logits.sigmoid(),self.color_logits.sigmoid(),vertices[0]

    def regularization(self):
        return (.0003*self.offset_raw.tanh().square().mean()
                +.00001*self.log_scale_delta.tanh().square().mean()
                +.001*(.25*self.pose_delta.tanh()).square().mean()
                +.001*self.translation_delta.tanh().square().mean())

def render_hand_surface(p,old_model,hand,camera,frame,W,K,width,height,degree=2,disable_new=False,emit_surface_weight=False):
    means,cov,linear,meta=old_model.geometry(p,camera,frame);i=meta[0]
    center=-W[:3,:3].T@W[:3,3]
    directions=(old_model.base.inverse[i,p['anchor']]@torch.linalg.solve(linear,(means-center)[...,None]))[...,0]
    directions=torch.nn.functional.normalize(directions,dim=-1)
    color=(spherical_harmonics(degree,directions,torch.cat([p['sh0'],p['shN']],1))+.5).clamp_min(0)
    alpha=p['opacities'].sigmoid()
    if not disable_new:
        hm,hc,ha,hcolor,_=hand.geometry(camera,frame)
        means=torch.cat([means,hm]);cov=torch.cat([cov,hc]);alpha=torch.cat([alpha,ha]);color=torch.cat([color,hcolor])
    if emit_surface_weight:
        tag=torch.zeros((len(color),1),dtype=color.dtype,device=color.device)
        if not disable_new:tag[len(p['means']):]=1
        color=torch.cat([color,tag],-1)
    return rasterization(means,None,None,alpha,color,W[None],K[None],width,height,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
