"""Native ENeRF motion pilot with persistent Gaussians and local rigidity.

Local gsplat/SH adaptation of Dynamic3DGaussians temporal losses. Not the
official 4K4D, EasyVolcap or OmniRe training pipeline. Train-view pixels only.
"""
from pathlib import Path
import argparse,json,random,time,os,hashlib
import numpy as np,torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from PIL import Image
from gsplat import rasterization
from enerf_native_loader import EnerfNativeDataset
from train_enerf_first_frame import tensor,ss,save
from adapter import load_official

def cpus(p):return {k:v.detach().cpu() for k,v in p.items()}
def atomic(path,x):torch.save(x,path.with_suffix('.partial'));os.replace(path.with_suffix('.partial'),path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--root',required=True);ap.add_argument('--iterations',type=int,default=600)
    ap.add_argument('--initial-checkpoint');ap.add_argument('--support',choices=['frozen','scales','opacities','both'],default='frozen');ap.add_argument('--max-common-frame',type=int)
    a=ap.parse_args();out=Path(a.out);run=out/'motion';run.mkdir(exist_ok=True,parents=True)
    if (run/'completion.json').exists():raise FileExistsError('Completed run')
    torch.set_num_threads(8);torch.manual_seed(74);np.random.seed(74);random.seed(74)
    tr,h=load_official();ds=EnerfNativeDataset(a.manifest,'train',a.root);spec=json.loads(Path(a.manifest).read_text());frames=sorted(set(r.get('common_frame',r['frame']) for r in ds.rows));assert len(frames)>1
    if a.max_common_frame is not None:frames=[t for t in frames if t<=a.max_common_frame];assert len(frames)>1
    initial_path=Path(a.initial_checkpoint) if a.initial_checkpoint else out/'first-frame/resume.pt'
    initial=torch.load(initial_path,weights_only=False,map_location='cpu');assert initial['step']==6000
    p={k:v.cuda() for k,v in initial['params'].items()};names=['means','quats','sh0','shN'];points=p['means'].cpu().numpy();dist,indices=cKDTree(points).query(points,k=21,workers=8)
    idx=torch.tensor(indices[:,1:],device='cuda');d0=tensor(dist[:,1:]);weight=torch.exp(-2000*d0*d0);N=len(points)
    support_names=[]
    if a.support in ['scales','both']:support_names.append('scales')
    if a.support in ['opacities','both']:support_names.append('opacities')
    names+=support_names;initial_support={k:p[k].detach().clone() for k in ['scales','opacities']}
    protocol=dict(method='Local native-resolution persistent Gaussian motion adaptation; Dynamic3DGaussians rigid/rotation/isometry losses, gsplat SH renderer',source_commit='7dbbd4dec404308524ff402756bdb8143a2589b0',frames=frames,heldout_times=spec['heldout_frames'],train_cameras=sorted(set(r['camera'] for r in ds.rows)),heldout_cameras=spec['heldout_cameras'],iterations_each=a.iterations,initial_checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),params_optimized=names,frozen=[k for k in ['scales','opacities'] if k not in support_names]+['camera calibration','Gaussian identity/count'],neighbors=20,neighbor_samples_per_step=20000,neighbor_weight='exp(-2000 distance_squared) in released world coordinates, metric scale not independently established',loss_weights=dict(rgb_l1=.8,dssim=.2,alpha_l1=.1,local_rigidity=4,neighbor_rotation=4,isometry=2,sh_color_continuity=.01),adaptations=['SH degree2 instead of RGB-only','Native RGB/matte with valid rectification support','Foreground only; no floor/background loss','Uniform sampled Gaussian centers for temporal regularizers','Clone previous colors and geometry; variable-gap velocity initialization','Adam reset per time, final fixed budget; no heldout model selection'],target='Movement and novel-view clarity of two foreground people and held objects, not a complete large scene',no_heldout_geometry_or_images=True)
    if support_names:
        protocol['support_control']=dict(variant=a.support,trainable=support_names,lr=dict(scales=.0005,opacities=.002),temporal_prior='0.01 mean squared change of log-scale and/or sigmoid-opacity from fixed previous time',global_bounds='log-scale within initial +/-ln2; opacity logits within initial +/-2',point_count='fixed, no splitting or pruning',source='Local controlled extension, not an author-provided Dynamic3DGaussians variant')
    if any('common_frame' in r for r in ds.rows):
        protocol['time_mapping']=spec['time_mapping'];protocol['split_scope']=spec['split_scope'];protocol['source_manifest_sha256']=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest();protocol['calibration_visible_cameras']=spec.get('calibration_visible_cameras',[])
    if (run/'protocol.json').exists():assert json.loads((run/'protocol.json').read_text())==protocol
    else:save(run/'protocol.json',protocol)
    prev=None;prev_t=None;cur_t=frames[0];history=[];elapsed=0
    if (run/'resume.pt').exists():
        cp=torch.load(run/'resume.pt',weights_only=False,map_location='cpu');p={k:v.cuda() for k,v in cp['params'].items()};prev={k:v.cuda() for k,v in cp['previous'].items()};cur_t=cp['frame'];prev_t=cp['previous_frame'];history=cp['history'];elapsed=cp['elapsed'];random.setstate(cp['random']);np.random.set_state(cp['numpy']);torch.set_rng_state(cp['torch']);torch.cuda.set_rng_state(cp['cuda'])
    else:atomic(run/f'frame-{frames[0]:03d}.pt',cpus(p))
    began=time.time()-elapsed
    for t in frames[1:]:
        if t<=cur_t:continue
        observations=[]
        for i,r in enumerate(ds.rows):
            if r.get('common_frame',r['frame'])!=t:continue
            o=ds[i];m=o['foreground_matte'];y,x=np.where(m>.1);x0=max(0,int(x.min())-32);y0=max(0,int(y.min())-32);x1=min(m.shape[1],int(x.max())+33);y1=min(m.shape[0],int(y.max())+33);K=o['K'].copy();K[0,2]-=x0;K[1,2]-=y0
            observations.append(dict(camera=o['camera'],rgb=tensor(o['rgb'][y0:y1,x0:x1]/255),matte=tensor(m[y0:y1,x0:x1,None]),valid=tensor(o['valid_pixels'][y0:y1,x0:x1,None]),K=tensor(K),w2c=tensor(o['w2c'])))
        anchor={k:v.detach().clone() for k,v in p.items()};anchor_q=F.normalize(anchor['quats'],dim=-1);inv=anchor_q.clone();inv[:,1:]*=-1
        p={k:torch.nn.Parameter(v.detach().clone(),requires_grad=k in names) for k,v in p.items()}
        with torch.no_grad():
            if prev is not None:
                factor=(t-cur_t)/(cur_t-prev_t);p['means'].add_((anchor['means']-prev['means'])*factor)
        assert anchor['sh0'].data_ptr()!=p['sh0'].data_ptr()
        rates=dict(means=.0004,quats=.001,sh0=.001,shN=.00005,scales=.0005,opacities=.002)
        optim=torch.optim.Adam([dict(params=[p[k]],lr=rates[k]) for k in names],eps=1e-15)
        for it in range(a.iterations):
            o=random.choice(observations);height,width=o['matte'].shape[:2];bg=torch.rand(3,device='cuda')
            image,alpha,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),o['w2c'][None],o['K'][None],width,height,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
            pred=(image[0]+bg*(1-alpha[0]))*o['valid'];target=(o['rgb']*o['matte']+bg*(1-o['matte']))*o['valid']
            im=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean()
            sample=torch.randint(N,(min(20000,N),),device='cuda');nb=idx[sample];w=weight[sample]
            rel=h.quat_mult(F.normalize(p['quats'],dim=-1),inv);rot=tr.build_rotation(rel[sample]);offset=p['means'][nb]-p['means'][sample,None];old=anchor['means'][nb]-anchor['means'][sample,None]
            rigid=h.weighted_l2_loss_v2((rot.transpose(1,2)[:,None]@offset[...,None]).squeeze(-1),old,w)
            consistency=h.weighted_l2_loss_v2(rel[nb],rel[sample,None],w);iso=h.weighted_l2_loss_v1(offset.square().sum(-1).add(1e-20).sqrt(),d0[sample],w)
            color=(p['sh0']-anchor['sh0']).abs().mean()+(p['shN']-anchor['shN']).abs().mean();loss=im+4*rigid+4*consistency+2*iso+.01*color
            if 'scales' in support_names:loss=loss+.01*(p['scales']-anchor['scales']).square().mean()
            if 'opacities' in support_names:loss=loss+.01*(p['opacities'].sigmoid()-anchor['opacities'].sigmoid()).square().mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss frame{t} step{it}')
            loss.backward();optim.step();optim.zero_grad(set_to_none=True)
            if support_names:
                with torch.no_grad():
                    if 'scales' in support_names:p['scales'].clamp_(min=initial_support['scales']-np.log(2),max=initial_support['scales']+np.log(2))
                    if 'opacities' in support_names:p['opacities'].clamp_(min=initial_support['opacities']-2,max=initial_support['opacities']+2)
            if it in [0,a.iterations-1]:print('MOTION',t,it+1,float(loss),float(im),flush=True)
        for k in ('scales','opacities'):
            if k not in support_names:assert torch.equal(p[k],anchor[k])
        assert len(p['means'])==N
        history.append(dict(frame=t,updates=a.iterations,loss=float(loss),image_loss=float(im),elapsed_seconds=time.time()-began,max_gpu_bytes=torch.cuda.max_memory_allocated()))
        if support_names:
            with torch.no_grad():
                history[-1]['support_change']={k:dict(mean_abs_from_previous=float((p[k]-anchor[k]).abs().mean()),max_abs_from_initial=float((p[k]-initial_support[k]).abs().max())) for k in support_names}
        atomic(run/f'frame-{t:03d}.pt',cpus(p));atomic(run/'resume.pt',dict(params=cpus(p),previous=cpus(anchor),frame=t,previous_frame=cur_t,history=history,elapsed=time.time()-began,random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
        prev=anchor;prev_t=cur_t;cur_t=t;save(run/'progress.json',history[-1]);print('FRAME_COMPLETE',t,round(time.time()-began,1),flush=True)
        del observations,optim
    save(run/'completion.json',dict(status='motion_training_complete_heldout_temporal_review_pending',frames=frames,updates=(len(frames)-1)*a.iterations,points=N,elapsed_seconds=time.time()-began,max_gpu_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':main()
