"""Native training-only 3DGS background from audited known-camera triangulation."""
from pathlib import Path
import argparse,json,time,random,os,hashlib
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
from PIL import Image
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from enerf_background_data import load_background,HELDOUT
from audit_enerf_input import digest,read_cameras
from train_enerf_first_frame import ss,tensor,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--init',required=True);ap.add_argument('--parent',default=None);a=ap.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=True);init_path=Path(a.init)
    assert not (out/'completion.json').exists(), 'Completed run is immutable'
    torch.set_num_threads(8);cv2.setNumThreads(8);random.seed(75);np.random.seed(75);torch.manual_seed(75)
    train=sorted(set(read_cameras(root/'optimized'))-set(HELDOUT)); observations=load_background(root,train)
    sources=[dict(camera=o['camera'],path=o['source'],sha256=o['sha256']) for o in observations]
    for o in observations:
        np.savez_compressed(out/f'train-camera-{o["camera"]}.npz',rgb=o['rgb'],valid=o['valid'],K=o['K'],w2c=o['w2c'])
        for k in ['rgb','valid','K','w2c']:o[k]=tensor(o[k]/255 if k=='rgb' else o[k])
    initial=np.load(init_path);pts=initial['points'];col=initial['colors'];N=len(pts);assert N==6480
    distance=cKDTree(pts).query(pts,k=4)[0][:,1:]
    standard_deviation=np.clip(np.sqrt((distance**2).mean(1)),.001,.1)
    scene_scale=float(np.linalg.norm(pts.max(0)-pts.min(0))*.5)
    rates=dict(means=.00016*scene_scale,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.0025/20)
    protocol=dict(status='fixed_before_background_optimization',dataset='actor5_6',steps=6000,seed=75,
        train_cameras=train,heldout_cameras=HELDOUT,sources=sources,initial_points=N,initial_sha256=digest(init_path),
        calibration_sha256={n:digest(root/'optimized'/n) for n in ['intri.yml','extri.yml']},
        native_resolution=[1920,1080],resize=False,source='Author released empty-scene bkgd references; fixed optimized OpenCV cameras',
        initialization='Audited032 training-only triangulated points; initial isotropic std RMS3nearest distance clipped[.001,.1] released units, opacity.1, identity wxyz rotation, median source color',
        rates=rates,mean_lr_schedule='Exponential to0.01initial over6000steps',sh_degree='0then1at1000then2at2000',
        loss='0.8 valid-normalized RGB L1 +0.2(1-SSIM) on valid-zeroed RGB +0.01 valid-normalized opacity deficit; random uniform RGB background',
        density=dict(strategy='gsplat1.3 DefaultStrategy',start=400,stop=4000,every=100,reset_every=2000,point_limit=500000),
        renderer='classic, eps2d0.3, near0.1 far20, background compositing after rasterization',
        model_selection='Fixed6000 final, no evaluation images for stopping or model selection',
        evaluation='All3heldout background references and fixed train01,08,17, full/native crops, valid image PSNR and alpha coverage, fixed top0:800 and bottom900:1080image bands (not semantic labels)',
        scope='Static background only; no human model or joint dynamic-scene completion; physical world scale unverified',
        source_sha256={n:digest(Path(__file__).parent/n) for n in ['train_enerf_background.py','enerf_background_data.py','train_enerf_first_frame.py']})
    if a.parent:
        parent=Path(a.parent);cp_parent=torch.load(parent,map_location='cpu',weights_only=False)
        assert 0<cp_parent['step']<6000
        protocol['recovery']=dict(parent_sha256=digest(parent),parent_step=cp_parent['step'],parent_points=len(cp_parent['params']['means']),reason='Parent halted after exceeding engineering density budget. Continue existing optimizer/state to6000 with fixed point count; no subsequent growth, prune or opacity reset. No heldout images inspected for this decision.')
        protocol['density']=dict(strategy='Fixed existing support after parent density budget stop',point_count=protocol['recovery']['parent_points'])
        if not (out/'resume.pt').exists():torch.save(cp_parent,out/'resume.pt')
        print('RECOVERY',json.dumps(protocol['recovery']),flush=True)
        del cp_parent
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:save(out/'protocol.json',protocol)
    params=torch.nn.ParameterDict(dict(means=torch.nn.Parameter(tensor(pts)),scales=torch.nn.Parameter(tensor(np.log(standard_deviation)[:,None].repeat(3,axis=1))),quats=torch.nn.Parameter(tensor(np.tile([1,0,0,0],(N,1)))),opacities=torch.nn.Parameter(tensor(np.full(N,np.log(.1/.9)))),sh0=torch.nn.Parameter(tensor((col-.5)/.28209479177387814)[:,None,:]),shN=torch.nn.Parameter(torch.zeros(N,8,3,device='cuda'))))
    optim={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
    strategy=DefaultStrategy(refine_start_iter=400,refine_stop_iter=4000,reset_every=2000,refine_every=100,verbose=False)
    strategy.check_sanity(params,optim);state=strategy.initialize_state(scene_scale);start=0;history=[];elapsed=0
    if (out/'resume.pt').exists():
        cp=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.cuda()) for k,v in cp['params'].items()})
        optim={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
        for k,v in optim.items():v.load_state_dict(cp['optim'][k])
        state={k:v.cuda() if torch.is_tensor(v) else v for k,v in cp['strategy'].items()};start=cp['step'];history=cp['history'];elapsed=cp['elapsed']
        random.setstate(cp['random']);np.random.set_state(cp['numpy']);torch.set_rng_state(cp['torch']);torch.cuda.set_rng_state(cp['cuda'])
    began=time.time()-elapsed
    def render(o,degree):
        return rasterization(params['means'],params['quats'],params['scales'].exp(),params['opacities'].sigmoid(),torch.cat([params['sh0'],params['shN']],1),o['w2c'][None],o['K'][None],1920,1080,sh_degree=degree,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
    def checkpoint(step):
        cp=dict(params={k:v.detach().cpu() for k,v in params.items()},optim={k:v.state_dict() for k,v in optim.items()},strategy=state,step=step,history=history,elapsed=time.time()-began,random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
        torch.save(cp,out/'resume.partial');os.replace(out/'resume.partial',out/'resume.pt')
    for i in range(start,6000):
        o=observations[random.randrange(len(observations))];degree=min(2,i//1000);bg=torch.rand(3,device='cuda')
        rgb,alpha,info=render(o,degree);pred=rgb[0]+bg*(1-alpha[0]);target=o['rgb'];valid=o['valid'][...,None]
        l1=((pred-target).abs()*valid).sum()/(valid.sum()*3)
        loss=.8*l1+.2*(1-ss((pred*valid).permute(2,0,1)[None],(target*valid).permute(2,0,1)[None]))+.01*((1-alpha[0])*valid).sum()/valid.sum()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        strategy.step_pre_backward(params,optim,state,i,info);loss.backward()
        if i==start:assert params['means'].grad is not None and torch.isfinite(params['means'].grad).all() and params['means'].grad.abs().sum()>0
        for op in optim.values():op.step();op.zero_grad(set_to_none=True)
        if not a.parent:strategy.step_post_backward(params,optim,state,i,info,packed=False)
        optim['means'].param_groups[0]['lr']=rates['means']*(.01**((i+1)/6000))
        if not a.parent and len(params['means'])>500000:checkpoint(i+1);raise RuntimeError('Point budget exceeded')
        if (i+1)%100==0 or i==0:
            torch.cuda.synchronize();row=dict(step=i+1,loss=float(loss),points=len(params['means']),elapsed_seconds=time.time()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(out/'progress.json',row);print(json.dumps(row),flush=True)
        if (i+1)%1000==0:
            checkpoint(i+1)
            with torch.no_grad():
                r,al,_=render(observations[0],degree);picture=(r[0]+.5*(1-al[0])).clamp(0,1).cpu().numpy();Image.fromarray(np.rint(picture*255).astype('uint8')).save(out/f'train01-step{i+1:05d}.png')
    for k,v in params.items():assert torch.isfinite(v).all(),k
    save(out/'completion.json',dict(status='background_trained_evaluation_pending',step=6000,points=len(params['means']),elapsed_seconds=time.time()-began,max_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=digest(out/'resume.pt')))

if __name__=='__main__':main()
