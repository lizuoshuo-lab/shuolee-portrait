"""Train-only ENeRF native-resolution foreground 3DGS diagnostic.

Uses installed gsplat 1.3 DefaultStrategy, not a reproduction of 4K4D.
No author point cloud, heldout pixels, sharpening or image-generation input.
"""
from pathlib import Path
import argparse,json,time,os,random,sys,hashlib
import numpy as np
import cv2,torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from PIL import Image
from gsplat.rendering import rasterization_2dgs
from gsplat.strategy import DefaultStrategy
import torch.nn.functional as F

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def ss(a,b):
    # Gaussian-window SSIM, RGB NCHW, data range one.
    v=torch.arange(11,device=a.device,dtype=a.dtype)-5;v=torch.exp(-v*v/(2*1.5**2));v/=v.sum();w=(v[:,None]*v[None,:]).expand(3,1,11,11)
    mu=F.conv2d(a,w,groups=3);mv=F.conv2d(b,w,groups=3)
    va=F.conv2d(a*a,w,groups=3)-mu*mu;vb=F.conv2d(b*b,w,groups=3)-mv*mv;cov=F.conv2d(a*b,w,groups=3)-mu*mv
    return (((2*mu*mv+.01**2)*(2*cov+.03**2))/((mu*mu+mv*mv+.01**2)*(va+vb+.03**2))).mean()
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')

def visual_hull(out,obs):
    f=out/'train-visual-hull-init.npz'
    if f.exists():return np.load(f)
    q=np.array(json.loads((out/'geometry-audit.json').read_text())['point_quantiles'])
    lo=q[0]-.75;hi=q[2]+.75;step=.012
    axes=[np.arange(l,h,step,dtype=np.float32) for l,h in zip(lo,hi)]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3);parts=[]
    for start in range(0,len(xyz),150000):
        x=tensor(xyz[start:start+150000]);votes=torch.zeros(len(x),device='cuda')
        for o in obs:
            c=x@o['w2c'][:3,:3].T+o['w2c'][:3,3];p=c@o['K'].T;uv=(p[:,:2]/p[:,2:]).round().long();h,w=o['matte'].shape
            ok=(c[:,2]>.1)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<w)&(uv[:,1]<h)
            take=ok.nonzero().ravel();votes[take]+=(o['dilated'][uv[take,1],uv[take,0]]>.5)
        parts.append((votes>=len(obs)-1).cpu().numpy())
    occ=np.concatenate(parts).reshape(tuple(len(a) for a in axes))
    boundary=sum(int(np.take(occ,i,axis=k).sum()) for k in range(3) for i in (0,-1))
    if boundary:
        save(out/'visual-hull-bounds-failure.json',dict(bounds=[lo.tolist(),hi.tolist()],boundary_voxels=boundary,faces=[[int(np.take(occ,i,axis=k).sum()) for i in (0,-1)] for k in range(3)]))
        raise RuntimeError(f'Visual hull touches initialization bounds: {boundary}')
    shell=occ&~binary_erosion(occ,iterations=1);pts=xyz[shell.reshape(-1)];assert 1000<len(pts)<300000
    x=tensor(pts);cs=torch.zeros_like(x);ws=torch.zeros(len(x),device='cuda')
    for o in obs:
        c=x@o['w2c'][:3,:3].T+o['w2c'][:3,3];p=c@o['K'].T;uv=(p[:,:2]/p[:,2:]).round().long();h,w=o['matte'].shape
        ok=(c[:,2]>.1)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<w)&(uv[:,1]<h);take=ok.nonzero().ravel()
        flat=uv[take,1]*w+uv[take,0];zmin=torch.full((h*w,),float('inf'),device='cuda');zmin.scatter_reduce_(0,flat,c[take,2],reduce='amin')
        front=c[take,2]<(zmin[flat]+.025);take=take[front];alpha=o['matte'][uv[take,1],uv[take,0]]
        cs[take]+=o['rgb'][uv[take,1],uv[take,0]]*alpha[:,None];ws[take]+=alpha
    col=(cs/ws.clamp_min(1e-5)[:,None]).cpu().numpy();missing=ws.cpu().numpy()<.1
    if missing.any():col[missing]=col[~missing][cKDTree(pts[~missing]).query(pts[missing])[1]]
    np.savez_compressed(f,points=pts,colors=col,voxel_size=step)
    save(out/'initialization.json',dict(method='Training-view foreground visual-hull shell, 14/15 silhouette agreement with three-pixel dilation',points=len(pts),voxel_size_world_units=step,bounds=[lo.tolist(),hi.tolist()],bounds_touched=boundary,training_cameras=[o['id'] for o in obs],frame=json.loads((out/'geometry-audit.json').read_text())['frame'],author_geometry_used=False,heldout_rgb_or_masks_used=False))
    print('HULL',len(pts),flush=True);return np.load(f)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=6000);ap.add_argument('--run',default='first-frame');ap.add_argument('--surface-reg',action='store_true');a=ap.parse_args();out=Path(a.out);run=out/a.run;run.mkdir(exist_ok=True)
    if (run/'completion.json').exists():raise FileExistsError('Completed run')
    torch.set_num_threads(8);torch.manual_seed(73);np.random.seed(73);random.seed(73)
    obs=[]
    for f in sorted(out.glob('train-camera-*.npz')):
        d=np.load(f);m=d['matte'];y,x=np.where(m>.1);box=[max(0,int(x.min())-32),max(0,int(y.min())-32),min(m.shape[1],int(x.max())+33),min(m.shape[0],int(y.max())+33)]
        obs.append(dict(id=f.stem[-2:],rgb=tensor(d['rgb']/255),matte=tensor(m),valid=tensor(d['valid']),K=tensor(d['K']),w2c=tensor(d['w2c']),box=box,dilated=tensor(cv2.dilate(m,np.ones((7,7),np.uint8))),surface_valid=tensor(cv2.erode(m,np.ones((5,5),np.uint8))*d['valid'])))
    assert len(obs)==15
    init=visual_hull(out,obs);pts=init['points'];col=init['colors'];N=len(pts);C0=.28209479177387814
    params=torch.nn.ParameterDict(dict(means=torch.nn.Parameter(tensor(pts)),scales=torch.nn.Parameter(tensor(np.full((N,3),np.log(.009)))),quats=torch.nn.Parameter(tensor(np.load(out/'surface-orientation.npz')['quaternions_wxyz'])),opacities=torch.nn.Parameter(tensor(np.full(N,-2.1972246))),sh0=torch.nn.Parameter(tensor((col-.5)/C0)[:,None,:]),shN=torch.nn.Parameter(torch.zeros(N,8,3,device='cuda'))))
    scene_scale=float(np.linalg.norm(pts.max(0)-pts.min(0))*.5)
    rates=dict(means=.00016*scene_scale,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.0025/20)
    optim={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
    strategy=DefaultStrategy(refine_start_iter=400,refine_stop_iter=min(4000,a.steps-1000),reset_every=2000,refine_every=100,verbose=True,key_for_gradient='gradient_2dgs')
    strategy.check_sanity(params,optim);state=strategy.initialize_state(scene_scale)
    spec=dict(method='gsplat1.5.3 2DGaussian surface control',surface_regularization=a.surface_reg,frame=json.loads((out/'geometry-audit.json').read_text())['frame'],native_resolution=[1920,1080],resize=False,crop='Training matte bounding rectangle plus 32 px margin; K principal point translated consistently',steps=a.steps,seed=73,train_cameras=[o['id'] for o in obs],heldout_cameras=['00','06','12'],spherical_harmonics_max_degree=2,lr=rates,loss='0.8 RGB L1 + 0.2 DSSIM + 0.1 matte L1, random uniform background per iteration',renderer='classic, near .1 far 20, eps2d .3',model_scope='Two foreground people and their held objects, no reconstructed static background or temporal model yet',model_selection='Fixed final iteration, no heldout selection',point_limit=500000)
    if (run/'protocol.json').exists():assert json.loads((run/'protocol.json').read_text())==spec
    else:save(run/'protocol.json',spec)
    # Precompute independent observation/background schedules so a change in
    # densification RNG consumption cannot change the paired observations.
    view_rng=random.Random(73)
    view_schedule=[view_rng.randrange(len(obs)) for _ in range(a.steps)]
    bg_schedule=torch.rand(a.steps,3,generator=torch.Generator(device='cpu').manual_seed(9073))
    np.savez_compressed(run/'sampling-schedule.npz',camera=np.array([obs[j]['id'] for j in view_schedule]),background=bg_schedule.numpy())
    initial={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in params.items()}
    save(run/'initial-and-sampling-audit.json',dict(initial_parameters=initial,initial_gaussians=len(params['means']),
        target_schedule_sha256=hashlib.sha256(np.array(view_schedule,dtype=np.int32).tobytes()).hexdigest(),
        background_schedule_sha256=hashlib.sha256(bg_schedule.numpy().tobytes()).hexdigest(),
        qualification='Same initial tensors and fixed target/background draws. Densification can legitimately diverge after changed input.',goal_complete=False))
    bg_schedule=bg_schedule.cuda()
    start=0;history=[];elapsed=0
    if (run/'resume.pt').exists():
        cp=torch.load(run/'resume.pt',weights_only=False,map_location='cpu');params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.cuda()) for k,v in cp['params'].items()});optim={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
        for k,v in optim.items():v.load_state_dict(cp['optim'][k])
        state={k:v.cuda() if torch.is_tensor(v) else v for k,v in cp['strategy'].items()};start=cp['step'];random.setstate(cp['random']);np.random.set_state(cp['numpy']);torch.set_rng_state(cp['torch']);torch.cuda.set_rng_state(cp['cuda']);history=cp['history'];elapsed=cp['elapsed']
    began=time.time()-elapsed
    def checkpoint(step):
        cp=dict(params={k:v.detach().cpu() for k,v in params.items()},optim={k:v.state_dict() for k,v in optim.items()},strategy=state,step=step,random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),history=history,elapsed=time.time()-began)
        torch.save(cp,run/'resume.partial');os.replace(run/'resume.partial',run/'resume.pt')
    def render(o,degree,box=None):
        K=o['K'].clone();h,w=o['matte'].shape
        if box is not None:
            x0,y0,x1,y1=box;w=x1-x0;h=y1-y0;K[0,2]-=x0;K[1,2]-=y0
        r,alpha,normals,depthnormals,distortion,median,info=rasterization_2dgs(params['means'],params['quats'],params['scales'].exp(),params['opacities'].sigmoid(),torch.cat([params['sh0'],params['shN']],1),o['w2c'][None],K[None],w,h,sh_degree=degree,packed=False,near_plane=.1,far_plane=20,eps2d=.3,render_mode='RGB+ED',distloss=True)
        if depthnormals.ndim==3:depthnormals=depthnormals[None]
        info['surface_normals']=normals;info['depth_normals']=depthnormals;info['distortion']=distortion
        return r[...,:3],alpha,info
    for i in range(start,a.steps):
        o=obs[view_schedule[i]];x0,y0,x1,y1=o['box'];bg=bg_schedule[i];degree=min(2,i//1000)
        rgb,alpha,info=render(o,degree,o['box']);target_alpha=o['matte'][y0:y1,x0:x1,None];valid=o['valid'][y0:y1,x0:x1,None]
        target=o['rgb'][y0:y1,x0:x1]*target_alpha+bg*(1-target_alpha);pred=rgb[0]+bg*(1-alpha[0]);pred=pred*valid;target=target*valid
        l1=(pred-target).abs().mean();loss=.8*l1+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-target_alpha).abs()*valid).mean()
        if a.surface_reg:
            support=o['surface_valid'][y0:y1,x0:x1];denom=support.sum().clamp_min(1)
            normal_error=1-(info['surface_normals'][0]*(info['depth_normals'][0]*alpha[0].detach())).sum(-1)
            nl=(normal_error*support).sum()/denom
            dl=(info['distortion'][0,...,0]*support).sum()/denom
            if i>=600:loss=loss+.01*dl
            if i>=1400:loss=loss+.05*nl
            if i==1400:
                reggrad=torch.autograd.grad(.05*nl+.01*dl,params['means'],retain_graph=True)[0]
                assert torch.isfinite(reggrad).all() and reggrad.abs().sum()>0
                save(run/'surface-regularizer-gradient-check.json',dict(update_index=i,normal_weight=.05,distortion_weight=.01,means_gradient_l1=float(reggrad.abs().sum()),status='surface_loss_has_finite_nonzero_geometry_gradient'))
        else:nl=loss.new_zeros(());dl=loss.new_zeros(())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        strategy.step_pre_backward(params,optim,state,i,info);loss.backward()
        if i==start:assert params['means'].grad is not None and torch.isfinite(params['means'].grad).all() and params['means'].grad.abs().sum()>0
        for op in optim.values():op.step();op.zero_grad(set_to_none=True)
        strategy.step_post_backward(params,optim,state,i,info,packed=False)
        optim['means'].param_groups[0]['lr']=rates['means']*(.01**((i+1)/a.steps))
        if len(params['means'])>spec['point_limit']:checkpoint(i+1);raise RuntimeError('Point budget exceeded')
        if (i+1)%100==0 or i==0:
            torch.cuda.synchronize();row=dict(step=i+1,loss=float(loss),normal_term=float(nl),distortion_term=float(dl),points=len(params['means']),elapsed_seconds=time.time()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(run/'progress.json',row);print(json.dumps(row),flush=True)
        if (i+1)%1000==0 or i+1==a.steps:
            checkpoint(i+1)
            with torch.no_grad():
                r,al,_=render(obs[0],degree);pic=(r[0]+.5*(1-al[0])).clamp(0,1).cpu().numpy();Image.fromarray((pic*255+.5).astype('uint8')).save(run/f'train-01-step-{i+1:05d}.png')
    save(run/'completion.json',dict(status='first_frame_trained_heldout_and_clarity_review_pending',step=a.steps,points=len(params['means']),elapsed_seconds=time.time()-began,max_gpu_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':main()
