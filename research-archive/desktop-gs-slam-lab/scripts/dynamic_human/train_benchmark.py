"""Checkpointed, equal-update-count static/dynamic human reconstruction pilot."""
from pathlib import Path
import argparse,random,json,time,hashlib,os
import numpy as np
import torch
from PIL import Image
from adapter import LAB,OUT,DATA,load_official,dataset,initialize

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def cpu_params(params):return {k:v.detach().cpu() for k,v in params.items()}
def save_state(path,params):
    tmp=path.with_suffix('.partial');torch.save(cpu_params(params),tmp);os.replace(tmp,path)

def checkpoint(path,params,variables,optimizer,t,i,todo):
    state=dict(params=cpu_params(params),variables={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in variables.items() if k!='means2D'},optimizer=optimizer.state_dict(),next_frame=t,next_iteration=i,todo=[b['id'] for b in todo],python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    tmp=path.with_suffix('.partial');torch.save(state,tmp);os.replace(tmp,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','dynamic','static'],default='smoke');parser.add_argument('--frames',type=int,default=31);parser.add_argument('--initial-iters',type=int,default=10000);parser.add_argument('--step-iters',type=int,default=2000);args=parser.parse_args()
    torch.manual_seed(73);np.random.seed(73);random.seed(73);torch.set_num_threads(8)
    tr,h=load_official();md=json.loads((DATA/'basketball/train_meta.json').read_text());params,variables=initialize(tr,md);optimizer=tr.initialize_optimizer(params,variables)
    folder=OUT/('smoke' if args.mode=='smoke' else args.mode);folder.mkdir(exist_ok=True)
    if args.mode!='smoke' and (folder/'completion.json').exists():raise RuntimeError('Completed model exists; no overwrite')
    if args.mode=='smoke':
        batch=dataset(0,md,camera_indices=[0])[0];times=[];losses=[];start=time.perf_counter()
        for i in range(20):
            tick=time.perf_counter();loss,variables=tr.get_loss(params,batch,variables,True);loss.backward()
            assert variables['means2D'].grad is not None and torch.isfinite(variables['means2D'].grad).all()
            assert torch.isfinite(loss) and torch.isfinite(params['means3D'].grad).all()
            optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();losses.append(float(loss.detach()));times.append(time.perf_counter()-tick)
        with torch.no_grad():rgb,_,_=tr.Renderer(batch['cam'])(**h.params2rendervar(params))
        Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8)).save(folder/'train-view-after20.png')
        save(folder/'checks.json',dict(status='passed',loss_start=losses[0],loss_end=losses[-1],median_step_seconds=float(np.median(times[5:])),points=len(params['means3D']),max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),screen_gradient_nonzero=bool(torch.count_nonzero(variables['means2D'].grad)),scope='one training camera, 20 optimization updates; not a reconstruction-quality result'))
        print(json.dumps(json.loads((folder/'checks.json').read_text()),indent=2));return
    spec=dict(mode=args.mode,sequence='basketball',frames=list(range(args.frames)),initial_iterations=args.initial_iters,per_dynamic_frame_iterations=args.step_iters,total_gradient_updates=args.initial_iters+(args.frames-1)*args.step_iters,seed=73,source_commit='7dbbd4dec404308524ff402756bdb8143a2589b0',rasterizer='local gsplat compatibility port, classic rasterization, pixel-to-NDC projected-gradient conversion',knn='SciPy cKDTree replaces Open3D KNN',resolution=[md['w'],md['h']],train_camera_ids=md['cam_id'][0],test_camera_ids=json.loads((DATA/'basketball/test_meta.json').read_text())['cam_id'][0],evaluation='test RGB never used in loss, camera corrections not fitted to test views; authors supplied initialization point cloud retained, its generating camera subset is not documented in release',max_points=1200000)
    if (folder/'protocol.json').exists():assert json.loads((folder/'protocol.json').read_text())==spec
    else:save(folder/'protocol.json',spec)
    cache={};history=[];start=time.perf_counter();resume_frame=0;resume_iteration=0;resume_todo=[]
    if (folder/'resume.pt').exists():
        saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
        params={k:torch.nn.Parameter(v.cuda().requires_grad_(True)) for k,v in saved['params'].items()}
        variables={k:v.cuda() if torch.is_tensor(v) else v for k,v in saved['variables'].items()}
        optimizer=tr.initialize_optimizer(params,variables);optimizer.load_state_dict(saved['optimizer'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        resume_frame=saved['next_frame'];resume_iteration=saved['next_iteration'];resume_todo=saved['todo'];history=json.loads((folder/'training-log.json').read_text())
        start-=history[-1]['elapsed_seconds'] if history else 0
        print('RESUME',args.mode,resume_frame,resume_iteration,flush=True)
    def getframe(t):
        if t not in cache:
            # Keep at most two complete timesteps on GPU.
            if len(cache)>=2:cache.pop(next(iter(cache)))
            cache[t]=dataset(t,md)
        return cache[t]
    if args.mode=='static':
        steps=[(0,spec['total_gradient_updates'])]
    else:steps=[(t,args.initial_iters if t==0 else args.step_iters) for t in range(args.frames)]
    for t,iterations in steps:
        if t<resume_frame:continue
        first_iteration=resume_iteration if t==resume_frame else 0
        if args.mode=='dynamic' and t>0 and first_iteration==0:params,variables=tr.initialize_per_timestep(params,variables,optimizer)
        todo=[getframe(t)[c] for c in resume_todo] if t==resume_frame and first_iteration>0 else [];frame_start=time.perf_counter()
        for i in range(first_iteration,iterations):
            if args.mode=='static':
                chosen_t=random.randrange(args.frames);chosen_c=random.randrange(len(md['fn'][0]));batch=dataset(chosen_t,md,camera_indices=[chosen_c])[0]
            else:batch=tr.get_batch(todo,getframe(t))
            loss,variables=tr.get_loss(params,batch,variables,t==0 or args.mode=='static');loss.backward()
            with torch.no_grad():
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at frame {t}, iteration {i}')
                if t==0:params,variables=tr.densify(params,variables,optimizer,i)
                if len(params['means3D'])>spec['max_points']:raise RuntimeError('Predeclared memory/point budget exceeded')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                if i%250==0 or i==iterations-1:
                    row=dict(mode=args.mode,frame=t,iteration=i+1,loss=float(loss),points=len(params['means3D']),elapsed_seconds=time.perf_counter()-start,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated());history.append(row);save(folder/'training-log.json',history);save(OUT/'progress.json',row);print('TRAIN',args.mode,t,i+1,'loss',round(float(loss),5),'points',len(params['means3D']),flush=True)
                if t==0 and (i+1)%2000==0:
                    save_state(folder/f'initial-{i+1:05d}.pt',params)
                    checkpoint(folder/'resume.pt',params,variables,optimizer,t,i+1,todo)
        save_state(folder/f'frame-{t:04d}.pt',params)
        if args.mode=='dynamic' and t==0:variables=tr.initialize_post_first_timestep(params,variables,optimizer)
        checkpoint(folder/'resume.pt',params,variables,optimizer,t+1,0,[])
        print('FRAME COMPLETE',args.mode,t,'seconds',round(time.perf_counter()-frame_start,1),flush=True)
    save(folder/'completion.json',dict(status='trained_evaluation_pending',mode=args.mode,wall_seconds=time.perf_counter()-start,gradient_updates=spec['total_gradient_updates'],frames=args.frames,protocol_sha256=digest(folder/'protocol.json'),final_points=len(params['means3D']),max_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':main()
