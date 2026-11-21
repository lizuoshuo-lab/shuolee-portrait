"""Matched-update static foreground baseline over all local motion times."""
from pathlib import Path
import argparse,json,time,random,hashlib,os
import numpy as np,torch
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from enerf_native_loader import EnerfNativeDataset
from train_enerf_first_frame import ss,tensor,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--root',required=True);ap.add_argument('--steps',type=int,default=14400);a=ap.parse_args();out=Path(a.out);run=out/'static-multitime';run.mkdir(exist_ok=True);assert not (run/'completion.json').exists()
    torch.set_num_threads(8);torch.manual_seed(75);np.random.seed(75);random.seed(75);ds=EnerfNativeDataset(a.manifest,'train',a.root);observations=[]
    for i,row in enumerate(ds.rows):
        o=ds[i];m=o['foreground_matte'];ys,xs=np.where(m>.1);x0=max(0,int(xs.min())-32);y0=max(0,int(ys.min())-32);x1=min(m.shape[1],int(xs.max())+33);y1=min(m.shape[0],int(ys.max())+33);K=o['K'].copy();K[0,2]-=x0;K[1,2]-=y0
        observations.append(dict(camera=o['camera'],common_frame=o['common_frame'],source_frame=o['frame'],rgb=np.ascontiguousarray(o['rgb'][y0:y1,x0:x1]),matte=np.ascontiguousarray(m[y0:y1,x0:x1]),valid=np.ascontiguousarray(o['valid_pixels'][y0:y1,x0:x1]),K=K,w2c=o['w2c']))
        if len(observations)%75==0:print('STATIC_INPUTS',len(observations),flush=True)
    assert len(observations)==375
    initial_path=out/'first-frame/resume.pt';initial=torch.load(initial_path,weights_only=False,map_location='cpu');assert initial['step']==6000
    initial_spec=json.loads((out/'first-frame/protocol.json').read_text());rates=initial_spec['lr'];params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.cuda()) for k,v in initial['params'].items()});del initial
    init=np.load(out/'train-visual-hull-init.npz')['points'];scene_scale=float(np.linalg.norm(init.max(0)-init.min(0))*.5)
    optim={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()};strategy=DefaultStrategy(refine_start_iter=400,refine_stop_iter=min(10000,a.steps-1000),reset_every=4000,refine_every=200,verbose=False);state=strategy.initialize_state(scene_scale);strategy.check_sanity(params,optim)
    protocol=dict(method='Static multi-time3DGS foreground, SAME locally corrected observations as motion branch',initial_checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),seed=75,first_frame_shared_updates=6000,additional_updates=a.steps,total_updates=6000+a.steps,observations=375,common_frames=sorted({o['common_frame'] for o in observations}),params_optimized=list(params),one_shared_state_all_times=True,loss='0.8L1+0.2DSSIM+0.1alpha, random background, same native crop+32px',lr=rates,sampling='Uniform over375time-camera observations',density='Same gsplat DefaultStrategy class, refine400..10000every200, reset4000, max500000. Static permits scale/opacity/density updates; motion freezes them after first frame.',model_selection='Fixed final budget; no test images or scores used for stopping')
    if (run/'protocol.json').exists():assert json.loads((run/'protocol.json').read_text())==protocol
    else:save(run/'protocol.json',protocol)
    start=0;elapsed=0;history=[]
    if (run/'resume.pt').exists():
        cp=torch.load(run/'resume.pt',weights_only=False,map_location='cpu');params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.cuda()) for k,v in cp['params'].items()});optim={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
        for k,v in optim.items():v.load_state_dict(cp['optim'][k])
        state={k:v.cuda() if torch.is_tensor(v) else v for k,v in cp['strategy'].items()};start=cp['step'];elapsed=cp['elapsed'];history=cp['history'];random.setstate(cp['random']);np.random.set_state(cp['numpy']);torch.set_rng_state(cp['torch']);torch.cuda.set_rng_state(cp['cuda'])
    began=time.time()-elapsed
    def checkpoint(step):
        torch.save(dict(params={k:v.detach().cpu() for k,v in params.items()},optim={k:v.state_dict() for k,v in optim.items()},strategy=state,step=step,elapsed=time.time()-began,history=history,random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),run/'resume.partial');os.replace(run/'resume.partial',run/'resume.pt')
    for i in range(start,a.steps):
        o=random.choice(observations);rgb=tensor(o['rgb']/255);matte=tensor(o['matte'])[...,None];valid=tensor(o['valid'])[...,None];h,w=o['matte'].shape;bg=torch.rand(3,device='cuda')
        image,alpha,info=rasterization(params['means'],params['quats'],params['scales'].exp(),params['opacities'].sigmoid(),torch.cat([params['sh0'],params['shN']],1),tensor(o['w2c'])[None],tensor(o['K'])[None],w,h,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
        pred=(image[0]+bg*(1-alpha[0]))*valid;target=(rgb*matte+bg*(1-matte))*valid;loss=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-matte).abs()*valid).mean();assert torch.isfinite(loss)
        strategy.step_pre_backward(params,optim,state,i,info);loss.backward()
        for op in optim.values():op.step();op.zero_grad(set_to_none=True)
        strategy.step_post_backward(params,optim,state,i,info,packed=False);optim['means'].param_groups[0]['lr']=rates['means']*(.01**((i+1)/a.steps))
        if len(params['means'])>500000:checkpoint(i+1);raise RuntimeError('Static point budget exceeded')
        if (i+1)%200==0 or i==0:
            row=dict(step=i+1,loss=float(loss),points=len(params['means']),elapsed_seconds=time.time()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(run/'progress.json',row);print(json.dumps(row),flush=True)
        if (i+1)%1000==0 or i+1==a.steps:checkpoint(i+1)
    torch.save({k:v.detach().cpu() for k,v in params.items()},run/'model.pt');save(run/'completion.json',dict(status='static_multitime_training_complete_evaluation_pending',additional_steps=a.steps,total_steps=6000+a.steps,points=len(params['means']),elapsed_seconds=time.time()-began,max_gpu_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
