"""Continuous, camera-independent MANO hypothesis from uncertain HaMeR 2D candidates.

Local research diagnostic, not ground-truth supervision or Gaussian reconstruction.
All image/camera/time data stay frozen. No monocular HaMeR depth is imported.
"""
from pathlib import Path
import argparse, json, hashlib, time, shutil
import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from hamer.models.mano_wrapper import MANO
from enerf_residual_graph import rotation_matrix

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
O=L/'outputs/dynamic-human-hand-consensus-097'
P=L/'outputs/dynamic-human-hand-prior-096'
S=L/'outputs/dynamic-human-enerf-second-interval-079'
MODEL='/mnt/d/GS-SLAM-Data/models/mano_v1_2/numpy'
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
KNOTS=np.array([718]*4+[720,723,726,729,732]+[735]*4,dtype=float)
NCONTROL=len(KNOTS)-4
def basis(t):return BSpline.design_matrix(np.asarray(t),KNOTS,3,extrapolate=False).toarray()
def project(X,K,W):
    q=np.einsum('nij,nkj->nki',W[:,:3,:3],X)+W[:,None,:3,3]
    v=np.einsum('nij,nkj->nki',K,q)
    return v[:,:,:2]/v[:,:,2:],q[:,:,2]
def summarize(a):
    a=np.asarray(a)
    return dict(count=int(a.size),median_px=float(np.median(a)),p90_px=float(np.quantile(a,.9)),mean_px=float(a.mean()))

def inputs():
    rows=read(P/'input-manifest.json')['records']
    z=np.load(S/'heldout/capture-affines-estimated.npz')
    times={(str(c),int(f)):float(t) for c,f,t in zip(z['camera'],z['frame'],z['capture_time'])}
    aa=[]
    for r in rows:
        p=P/'predictions'/f"camera{r['camera']}-frame{r['frame']:06d}.npz"
        with np.load(p) as a:aa.append({k:a[k] for k in a.files})
    assert len(rows)==105 and not set(r['camera'] for r in rows)&{'00','06','12'}
    xy=np.array([a['source_keypoints_2d'][1] for a in aa],np.float64)
    spread=np.array([np.max(np.linalg.norm(a['source_keypoints_2d']-a['source_keypoints_2d'][1:2],axis=-1),axis=0) for a in aa])
    weight=4/(4+spread**2)
    return rows,aa,times,xy,weight,spread

def free_init(xy,K,W,t,w):
    """Robust continuous linear triangulation; each joint has a cubic world path."""
    u=(t-726)/6;B=u[:,None]**np.arange(4)
    h=np.concatenate([xy,np.ones((*xy.shape[:2],1))],-1)
    x=np.einsum('nij,nkj->nki',np.linalg.inv(K),h);x=x[:,:,:2]/x[:,:,2:]
    coeff=[]
    for j in range(21):
        eq=W[:,:2,:]-x[:,j,:,None]*W[:,2:3,:]
        A=np.einsum('nak,nl->nalk',eq[:,:,:3],B).reshape(len(t)*2,12)
        y=-eq[:,:,3].ravel();ww=w[:,j].copy()
        for iteration in range(12):
            wt=np.repeat(np.sqrt(ww),2)
            q=np.linalg.lstsq(A*wt[:,None],y*wt,rcond=None)[0].reshape(4,3)
            X=B@q;uv,depth=project(X[:,None],K,W)
            err=np.linalg.norm(uv[:,0]-xy[:,j],axis=-1)
            ww=w[:,j]/np.sqrt(1+(err/4)**2)
        coeff.append(q)
    return np.array(coeff)

def align_rotation(a,b):
    # Both point sets are wrist-centered, column rotation R maps a into b.
    U,_,Vt=np.linalg.svd(a.T@b)
    return Vt.T@np.diag([1,1,np.linalg.det(Vt.T@U.T)])@U.T

class SharedHand(torch.nn.Module):
    def __init__(self,init,device='cuda'):
        super().__init__()
        self.mano=MANO(model_path=MODEL,is_rhand=True,use_pca=False,flat_hand_mean=True).to(device)
        self.register_buffer('root_base',torch.as_tensor(init['root_base'],dtype=torch.float32,device=device))
        for key in ['pose','translation','betas','log_scale']:
            self.register_parameter(key,torch.nn.Parameter(torch.as_tensor(init[key],dtype=torch.float32,device=device)))
    def forward(self,B):
        n=len(B);p=(B@self.pose.reshape(NCONTROL,-1)).reshape(n,16,3)
        R=rotation_matrix(p);I=torch.eye(3,device=B.device).expand(n,1,3,3)
        out=self.mano(global_orient=I,hand_pose=R[:,1:],betas=self.betas[None].expand(n,-1),pose2rot=False)
        root=R[:,0]@self.root_base;T=B@self.translation;s=self.log_scale.exp()
        J=(out.joints-out.joints[:,:1])*s
        V=(out.vertices-out.joints[:,:1])*s
        return J@root.transpose(-1,-2)+T[:,None],V@root.transpose(-1,-2)+T[:,None],R,root

def initialization(rows,aa,times,xy,weight,fit):
    K=np.array([a['K'] for a in aa]);W=np.array([a['w2c'] for a in aa])
    t=np.array([times[r['camera'],r['frame']] for r in rows]);B=basis(t)
    coeff=free_init(xy[fit],K[fit],W[fit],t[fit],weight[fit])
    # Shape and local rotations use only the fitting subset, never the diagnostic views.
    betas=np.median(np.array([a['mano_betas'][1] for a in aa])[fit],axis=0)
    rv=Rotation.from_matrix(np.array([a['mano_hand_pose'][1] for a in aa])[fit].reshape(-1,3,3)).as_rotvec().reshape(sum(fit),45)
    bf=B[fit];wf=np.sqrt(np.median(weight[fit],axis=1))
    D=np.diff(np.eye(NCONTROL),n=2,axis=0)
    cp=np.linalg.lstsq(np.vstack([bf*wf[:,None],.25*D]),np.vstack([rv*wf[:,None],np.zeros((len(D),45))]),rcond=None)[0]
    m=MANO(model_path=MODEL,is_rhand=True,use_pca=False,flat_hand_mean=True)
    sample_t=np.linspace(t[fit].min(),t[fit].max(),41);bs=basis(sample_t)
    with torch.no_grad():
        local=m(global_orient=torch.eye(3)[None,None].expand(len(bs),1,3,3),hand_pose=rotation_matrix(torch.from_numpy((bs@cp).reshape(-1,15,3)).float()),betas=torch.from_numpy(betas).float()[None].expand(len(bs),-1),pose2rot=False).joints.numpy()
    local-=local[:,:1].copy()
    free=np.einsum('nk,jkd->njd',((sample_t-726)/6)[:,None]**np.arange(4),coeff)
    free-=free[:,:1].copy();palm=[5,9,13,17]
    scale=np.median(np.linalg.norm(free[:,palm],axis=-1)/np.linalg.norm(local[:,palm],axis=-1))
    assert .1<scale<10,scale
    rots=np.array([align_rotation(a[palm],b[palm]) for a,b in zip(local,free)])
    base=rots[np.argmin(abs(sample_t-726))]
    rvroot=Rotation.from_matrix(rots@base.T).as_rotvec()
    rootcp=np.linalg.lstsq(np.vstack([bs,.1*D]),np.vstack([rvroot,np.zeros((len(D),3))]),rcond=None)[0]
    rootxyz=((t-726)/6)[:,None]**np.arange(4)@coeff[0]
    trans=np.linalg.lstsq(B,rootxyz,rcond=None)[0]
    init=dict(root_base=base,pose=np.concatenate([rootcp[:,None],cp.reshape(NCONTROL,15,3)],1),translation=trans,betas=betas,log_scale=np.log(scale))
    return init,coeff,K,W,t,B

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['excluded_views','all_views'],required=True);parser.add_argument('--steps',type=int,default=1200);args=parser.parse_args()
    torch.set_num_threads(6);torch.manual_seed(42);O.mkdir(exist_ok=True);out=O/args.variant;out.mkdir(exist_ok=True)
    assert not (out/'summary.json').exists()
    rows,aa,times,xy,weight,spread=inputs()
    fit=np.array([r['camera'] not in ['04','09','13'] if args.variant=='excluded_views' else True for r in rows])
    init,free,K,W,t,B=initialization(rows,aa,times,xy,weight,fit)
    np.savez(out/'initialization.npz',**init,free_coeff=free,t=t,B=B,fit=fit,xy=xy,weight=weight,crop_spread_px=spread,K=K,w2c=W)
    protocol=dict(status='fixed_before_optimization',variant=args.variant,steps=args.steps,fit_photos=int(fit.sum()),diagnostic_photos=int((~fit).sum()),diagnostic_cameras=['04','09','13'] if args.variant=='excluded_views' else [],knots=KNOTS.tolist(),parameters='one shared shape(10) and scale;9 cubic B-spline controls x16 rotation vectors and3 wrist coordinates; no camera-dependent deformation',inputs='central-crop HaMeR predicted21joints; weight=4/(4+max_crop_displacement_px^2); Cauchy scale4px. Predictions are NOT GT; crop variation does not establish visibility.',frozen='native RGB, K,w2c,estimated capturetimes, body and allGaussianparameters',initial_scale=float(np.exp(init['log_scale'])),initialization_uses_excluded_views=False,monocular_depth_used=False,scope='qualify soft articulated hypothesis only; no skin textures or Gaussian reconstruction',goal_complete=False)
    save(out/'protocol.json',protocol)
    model=SharedHand(init);tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
    bt=tensor(B);kt=tensor(K);wt=tensor(W);yt=tensor(xy);ww=tensor(weight);fi=torch.from_numpy(fit).cuda()
    pose0=model.pose.detach().clone();beta0=model.betas.detach().clone();scale0=model.log_scale.detach().clone()
    # Dense trajectory derivatives are evaluated in frame units, shared by all cameras.
    grid=np.linspace(719,734,61);b1=tensor(BSpline(KNOTS,np.eye(NCONTROL),3).derivative(1)(grid));b2=tensor(BSpline(KNOTS,np.eye(NCONTROL),3).derivative(2)(grid))
    optimizer=torch.optim.Adam([{'params':[model.pose],'lr':.008},{'params':[model.translation],'lr':.001},{'params':[model.betas],'lr':.008},{'params':[model.log_scale],'lr':.001}])
    started=time.monotonic();history=[];torch.cuda.reset_peak_memory_stats()
    def evaluate():
        J,V,R,root=model(bt)
        cam=J@wt[:,:3,:3].transpose(-1,-2)+wt[:,None,:3,3]
        proj=cam@kt.transpose(-1,-2);uv=proj[:,:,:2]/proj[:,:,2:].clamp_min(.1)
        return J,V,R,root,uv,cam[:,:,2]
    with torch.no_grad():initial_uv=evaluate()[4].cpu().numpy()
    for step in range(1,args.steps+1):
        optimizer.zero_grad(set_to_none=True);J,V,R,root,uv,depth=evaluate();err2=(uv-yt).square().sum(-1)
        reproj=(ww[fi]*16*torch.log1p(err2[fi]/16)).sum()/ww[fi].sum()
        cp=model.pose.reshape(NCONTROL,-1)
        reg_acc_rot=2*(b2@cp).square().mean();reg_vel_rot=.05*(b1@cp).square().mean()
        reg_acc_pos=2000*(b2@model.translation).square().mean()
        reg_pose=.3*(model.pose[:,1:]-pose0[:,1:]).square().mean()
        reg_shape=.02*model.betas.square().mean()+.02*(model.betas-beta0).square().mean()
        reg_scale=2*(model.log_scale-scale0).square()
        reg_depth=100*torch.relu(.2-depth).square().mean()
        loss=reproj+reg_acc_rot+reg_vel_rot+reg_acc_pos+reg_pose+reg_shape+reg_scale+reg_depth
        assert torch.isfinite(loss),step;loss.backward()
        if step==1:
            for name in ['pose','translation','betas','log_scale']:assert torch.isfinite(getattr(model,name).grad).all() and getattr(model,name).grad.abs().max()>0,name
        torch.nn.utils.clip_grad_norm_([model.pose,model.translation,model.betas,model.log_scale],100)
        optimizer.step()
        factor=.2+.8*(1-step/args.steps)
        for group,lr in zip(optimizer.param_groups,[.008,.001,.008,.001]):group['lr']=lr*factor
        if step==1 or step%100==0:
            e=torch.sqrt(err2).detach().cpu().numpy();record=dict(step=step,loss=float(loss),robust_reprojection=float(reproj),fit_median_px=float(np.median(e[fit])),diagnostic_median_px=float(np.median(e[~fit])) if (~fit).any() else None,scale=float(model.log_scale.exp()),elapsed_seconds=time.monotonic()-started)
            history.append(record);save(out/'progress.json',record);print(args.variant,json.dumps(record),flush=True)
    with torch.no_grad():J,V,R,root,uv,depth=[x.detach().cpu().numpy() for x in evaluate()]
    state={key:getattr(model,key).detach().cpu().numpy() for key in ['root_base','pose','translation','betas','log_scale']}
    np.savez(out/'parameters.npz',**state,knots=KNOTS)
    np.savez(out/'predictions.npz',camera=np.array([r['camera'] for r in rows]),frame=np.array([r['frame'] for r in rows]),capture_time=t,fit=fit,joints_world=J,vertices_world=V,rotations=R,root_rotations=root,projected_xy=uv,depth=depth,initial_projected_xy=initial_uv,faces=model.mano.faces)
    error=np.linalg.norm(uv-xy,axis=-1);err0=np.linalg.norm(initial_uv-xy,axis=-1)
    summary=dict(status='continuous_articulated_fit_complete_visual_qualification_pending',variant=args.variant,fit_photos=int(fit.sum()),diagnostic_photos=int((~fit).sum()),steps=args.steps,seconds=time.monotonic()-started,max_GPU_GB=torch.cuda.max_memory_allocated()/1024**3,initial_scale=float(np.exp(init['log_scale'])),final_scale=float(np.exp(state['log_scale'])),fit_initial=summarize(err0[fit]),fit_final=summarize(error[fit]),diagnostic_initial=summarize(err0[~fit]) if (~fit).any() else None,diagnostic_final=summarize(error[~fit]) if (~fit).any() else None,per_camera={c:summarize(error[np.array([r['camera']==c for r in rows])]) for c in sorted(set(r['camera'] for r in rows))},minimum_camera_depth=float(depth.min()),parameters_sha256=sha(out/'parameters.npz'),no_Gaussian_updates=True,scope='Residuals to neural predictions, not observed finger accuracy. Native source-image qualification still required.',goal_complete=False)
    save(out/'summary.json',summary);save(out/'history.json',history);shutil.copy2(__file__,out/Path(__file__).name)
    print('COMPLETE',args.variant,json.dumps(summary),flush=True)
if __name__=='__main__':main()
