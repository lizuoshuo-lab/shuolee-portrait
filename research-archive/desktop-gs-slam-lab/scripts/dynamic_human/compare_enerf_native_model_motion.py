"""Frozen Gaussian motion versus native patch correspondences; no model updates."""
from pathlib import Path
import json, hashlib, time, shutil
import cv2, numpy as np, torch
import enerf_graph_renderer as module
from enerf_graph_renderer import GraphMotion, ResidualGraphRenderer
from enerf_residual_graph import transform_gaussians

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
R=L/'outputs/dynamic-human-enerf-native-motion-tracks-075'
I=L/'outputs/dynamic-human-enerf-articulated-control-064'
read=lambda p:json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
assert not (R/'model-comparison-completion.json').exists()
sources={
    'baseline067':dict(root=L/'outputs/dynamic-human-enerf-residual-graph-067',motion=L/'outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz'),
    'correctRGB074':dict(root=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074',motion=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/head/capture-affines.npz')}
# Resolve the documented067 motion path from its immutable training protocol.
baseline=sources['baseline067']
expected=read(baseline['root']/'residual_graph/protocol.json')['motion_sha256']
if not baseline['motion'].exists() or sha(baseline['motion'])!=expected:
    matches=[p for p in (L/'outputs').glob('dynamic-human-enerf-*066/**/capture-affines.npz') if sha(p)==expected]
    assert len(matches)==1, matches
    baseline['motion']=matches[0]
for s in sources.values():
    s['checkpoint']=s['root']/'residual_graph/resume.pt'
    assert sha(s['motion'])==read(s['root']/'residual_graph/protocol.json')['motion_sha256']
protocol=dict(status='fixed_before_model_motion_measurement',
    models={k:dict(checkpoint_sha256=sha(s['checkpoint']),motion_sha256=sha(s['motion'])) for k,s in sources.items()},
    formula='f(p)=sum_i T_i(p)*alpha_i(p)*(pi(mu_i(target))-pi(mu_i(source)))/sum_i T_i(p)*alpha_i(p). Visibility/order/covariance are from source750.',
    interpretation='Visibility-weighted projected Gaussian-center motion; not exact optical flow inside a rotating or deforming finite Gaussian footprint. No camera-to-camera correspondence inferred.',
    primary_comparison='Intersect accepted image tracks with source alpha>=0.95, source assigned actor contribution>=0.9 and positive target depth contribution>=0.99 in both models. Report all accepted tracks separately; no endpoint-error filtering.',
    variance='Weighted center-motion second moments reported; primary result does not select by residual or motion variance.',
    model_updates=0,goal_complete=False)
save(R/'model-motion-protocol.json',protocol)
torch.set_num_threads(8);cv2.setNumThreads(4)
original=module.rasterization;capture={}
def capture_raster(*args,**kwargs):
    capture['args']=args;capture['kwargs']=kwargs
    return original(*args,**kwargs)
module.rasterization=capture_raster

def sample(a,xy):
    z=cv2.remap(a.astype(np.float32),xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return z[:,0]

def geometry(p,model,c,t,cov):
    i=model.base.lookup[c,t];ids=p['anchor'];A=model.base.A[i,ids];b=model.base.b[i,ids]
    means=(A@p['means'][...,None])[...,0]+b
    C=A@cov@A.transpose(-1,-2)
    nodes=(model.base.A[i,model.graph.node_anchor]@model.graph.node_points[...,None])[...,0]+model.base.b[i,model.graph.node_anchor]
    rr,dd=model.graph((model.times[i]-750)/6)
    return transform_gaussians(means,C,nodes,rr,dd,p['node_indices'],p['node_weights']/p['node_weights'].sum(-1,keepdim=True))[:2]

began=time.monotonic();checks=[]
with torch.no_grad():
    for label,s in sources.items():
        out=R/label;out.mkdir(exist_ok=True)
        cp=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
        p={k:v.cuda() for k,v in cp['params'].items()}
        graph=GraphMotion(s['root']/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);graph.eval()
        model=ResidualGraphRenderer(s['motion'],graph);del cp
        cov,_=module.quat_scale_to_covar_preci(p['quats'],p['scales'].exp(),compute_covar=True,compute_preci=False,triu=False)
        actor=(p['anchor']//6890);onehot=torch.nn.functional.one_hot(actor,2).float()
        for source in read(R/'preparation.json')['records']:
            c=source['camera'];tracks=np.load(R/'tracks'/f'camera{c}.npz');xy=tracks['source_xy'];frames=tracks['target_frame']
            z=np.load(I/'native-input'/f'camera{c}-frame000750.npz');W=tensor(z['w2c']);K=tensor(z['K'])
            rgb,alpha,_=model.render(p,c,750,W,K,1920,1080,2)
            args=capture['args'];kw=capture['kwargs'];means=args[0];colors=args[4]
            computed,C=geometry(p,model,c,750,cov)
            assert torch.max(abs(computed-means))<1e-7 and torch.max(abs(C-kw['covars']))<1e-7
            def project(X):
                camera=X@W[:3,:3].T+W[:3,3];h=camera@K.T
                return h[:,:2]/h[:,2:3],camera[:,2]
            uv0,_=project(means)
            # Check compositing and sampled-channel convention against actual RGB,
            # a zero-flow identity and a known constant projected displacement.
            smoke_features=torch.cat([colors,torch.ones_like(onehot)*torch.tensor([2.,-3.],device='cuda'),onehot],1)
            smoke,smoke_a,_=original(*args[:4],smoke_features,*args[5:],**kw)
            smoke_errors=dict(rgb=float((smoke[...,:3]-rgb).abs().max()),alpha=float((smoke_a-alpha).abs().max()),
                constant=float((smoke[...,3:5]-alpha*torch.tensor([2.,-3.],device='cuda')).abs().max()),
                actor_sum=float((smoke[...,5:7].sum(-1)-alpha[...,0]).abs().max()),
                reference_identity=float((project(computed)[0]-uv0).abs().max()))
            assert max(smoke_errors.values())<1e-4,smoke_errors
            afull=alpha[0,...,0].cpu().numpy();a=sample(afull,xy)
            actor_mass=sample(smoke[0,...,5:7].cpu().numpy(),xy)
            base_rgb=sample(rgb[0].cpu().numpy(),xy)
            sums=[];second=[];valids=[];errors=[];epe=[]
            for t in frames:
                target,_=geometry(p,model,c,int(t),cov);uv1,depth=project(target)
                displacement=uv1-uv0;positive=(depth>.1).float()
                features=torch.cat([displacement,displacement.square(),positive[:,None]],-1)
                value,al,_=original(*args[:4],features,*args[5:],**kw)
                ae=float((al-alpha).abs().max());assert ae<1e-5
                sampled=sample(value[0].cpu().numpy(),xy)
                sums.append(sampled[:,:2]);second.append(sampled[:,2:4]);valids.append(sampled[:,4]);errors.append(ae)
            sums=np.stack(sums);second=np.stack(second);valids=np.stack(valids)
            flow=sums/np.maximum(a[None,:,None],1e-8)
            variance=np.maximum(0,second/np.maximum(a[None,:,None],1e-8)-flow**2)
            np.savez_compressed(out/f'camera{c}.npz',source_xy=xy,target_frame=frames,
                flow_numerator=sums,second_moment_numerator=second,positive_depth_mass=valids,
                alpha=a,actor_mass=actor_mass,source_premult_rgb=base_rgb,flow=flow,variance=variance,
                predicted_target_xy=xy[None]+flow)
            checks.append(dict(model=label,camera=c,source_geometry_exact=True,smoke=smoke_errors,max_target_alpha_difference=max(errors),output_sha256=sha(out/f'camera{c}.npz')))
            print('MODEL_PATCH_MOTION',label,c,len(xy),flush=True)
        assert sha(s['checkpoint'])==protocol['models'][label]['checkpoint_sha256']
        del p,model,graph,cov;capture.clear();torch.cuda.empty_cache()
save(R/'model-comparison-completion.json',dict(status='both_frozen_models78native_training_motion_contexts_each_complete',checks=checks,
    seconds=time.monotonic()-began,model_updates=0,goal_complete=False))
shutil.copy2(__file__,R/Path(__file__).name)
