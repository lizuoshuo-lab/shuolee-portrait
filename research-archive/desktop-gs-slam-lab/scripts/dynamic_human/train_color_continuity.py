"""Matched forks from one first-frame model: aliased versus frozen color reference."""
from pathlib import Path
import argparse,json,random,time,hashlib
import numpy as np
import torch
from adapter import LAB,DATA,OUT as BASE,load_official,dataset
from train_benchmark import checkpoint,save_state,save

OUT=LAB/'outputs/dynamic-human-color-continuity-004'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['aliased','frozen'],required=True);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);folder=OUT/args.mode;folder.mkdir(exist_ok=True)
    assert not (folder/'completion.json').exists(),'Do not overwrite a completed diagnostic'
    torch.set_num_threads(8);torch.manual_seed(73);np.random.seed(73);random.seed(73)
    tr,h=load_official();md=json.loads((DATA/'basketball/train_meta.json').read_text());source=BASE/'dynamic/frame-0000.pt'
    protocol=dict(mode=args.mode,frames=list(range(31)),source_first_frame_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed=73,iterations_per_following_frame=2000,variant_updates=60000,shared_first_frame_training_updates=10000,optimizer='both forks start with identical zero Adam moments; a zero-gradient initialization step creates state without changing parameters',single_intervention='freeze previous-frame color with detach().clone()' if args.mode=='frozen' else 'retain pinned-source detach() alias',heldout_role='diagnostic follow-up on the previously inspected four cameras; not a fresh independent test',test_tuning='no per-camera color fit or image enhancement; no hyperparameter selection in this contrast')
    if (folder/'protocol.json').exists():assert json.loads((folder/'protocol.json').read_text())==protocol
    else:save(folder/'protocol.json',protocol)
    params={k:torch.nn.Parameter(v.cuda()) for k,v in torch.load(source,map_location='cpu',weights_only=True).items()}
    centers=np.linalg.inv(md['w2c'][0])[:,:3,3];radius=1.1*np.max(np.linalg.norm(centers-np.mean(centers,0)[None],axis=-1))
    variables=dict(scene_radius=radius,max_2D_radius=torch.zeros(len(params['means3D']),device='cuda'))
    optimizer=tr.initialize_optimizer(params,variables)
    for value in params.values():value.grad=torch.zeros_like(value)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    variables=tr.initialize_post_first_timestep(params,variables,optimizer)
    save_state(folder/'frame-0000.pt',params);history=[];start=time.perf_counter();start_frame=1
    if (folder/'resume.pt').exists():
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
        params={k:torch.nn.Parameter(v.cuda()) for k,v in state['params'].items()};variables={k:v.cuda() if torch.is_tensor(v) else v for k,v in state['variables'].items()}
        optimizer=tr.initialize_optimizer(params,variables);optimizer.load_state_dict(state['optimizer']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        start_frame=state['next_frame'];assert state['next_iteration']==0;history=json.loads((folder/'training-log.json').read_text());start-=history[-1]['elapsed_seconds'] if history else 0
    for t in range(start_frame,31):
        params,variables=tr.initialize_per_timestep(params,variables,optimizer)
        if args.mode=='frozen':variables['prev_col']=params['rgb_colors'].detach().clone()
        assert (variables['prev_col'].data_ptr()!=params['rgb_colors'].data_ptr())==(args.mode=='frozen')
        actual_previous=params['rgb_colors'].detach().clone();views=dataset(t,md);todo=[]
        for i in range(2000):
            batch=tr.get_batch(todo,views);loss,variables=tr.get_loss(params,batch,variables,False);loss.backward()
            with torch.no_grad():
                assert torch.isfinite(loss)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                if i%500==0 or i==1999:
                    c=params['rgb_colors'];row=dict(mode=args.mode,frame=t,iteration=i+1,loss=float(loss),actual_previous_color_l1=float(h.l1_loss_v2(c,actual_previous)),implemented_color_regularizer=float(h.l1_loss_v2(c,variables['prev_col'])),rgb_min=float(c.min()),rgb_max=float(c.max()),rgb_channels_outside_01_fraction=float(((c<0)|(c>1)).float().mean()),elapsed_seconds=time.perf_counter()-start)
                    history.append(row);save(folder/'training-log.json',history);print('COLOR TRAIN',args.mode,t,i+1,round(row['loss'],6),'color reference loss',round(row['implemented_color_regularizer'],6),flush=True)
        save_state(folder/f'frame-{t:04d}.pt',params);checkpoint(folder/'resume.pt',params,variables,optimizer,t+1,0,[])
        del views,actual_previous
    save(folder/'completion.json',dict(status='trained_diagnostic_evaluation_pending',frames=31,variant_updates=60000,shared_first_frame_training_updates=10000,wall_seconds=time.perf_counter()-start,final_points=len(params['means3D']),protocol_sha256=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()))

if __name__=='__main__':main()
