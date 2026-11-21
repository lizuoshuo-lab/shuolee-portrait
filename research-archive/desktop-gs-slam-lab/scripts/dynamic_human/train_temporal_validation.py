"""Continue both frozen-protocol color branches on previously unused timesteps."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import time
import numpy as np
import torch
from adapter import LAB, DATA, load_official, dataset
from train_benchmark import checkpoint, save_state, save

OUT = LAB/'outputs/dynamic-human-temporal-validation-006'
SOURCE = LAB/'outputs/dynamic-human-color-continuity-004'


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode', choices=['aliased','frozen'], required=True);args=p.parse_args()
    protocol=json.loads((OUT/'evaluation-protocol.json').read_text())
    for item in protocol['implementation_files']:
        assert hashlib.sha256((LAB/item['path']).read_bytes()).hexdigest()==item['sha256']
    deadline=time.monotonic()+3*3600
    while not (SOURCE/args.mode/'completion.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Prior branch has not completed')
        time.sleep(10)
    folder=OUT/args.mode;folder.mkdir(exist_ok=True)
    assert not (folder/'completion.json').exists()
    origin=SOURCE/args.mode/'resume.pt'
    receipt=dict(source_resume_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
        first_frame=31,last_frame=60,mode=args.mode,updates=60000,
        optimizer='Continue same Adam state and all RNG states; do not reset at frame 31',
        evaluation_protocol_sha256=hashlib.sha256((OUT/'evaluation-protocol.json').read_bytes()).hexdigest())
    if (folder/'protocol.json').exists():assert json.loads((folder/'protocol.json').read_text())==receipt
    else:save(folder/'protocol.json',receipt)
    statefile=folder/'resume.pt' if (folder/'resume.pt').exists() else origin
    state=torch.load(statefile,map_location='cpu',weights_only=False)
    assert 31<=state['next_frame']<=60 and state['next_iteration']==0
    torch.set_num_threads(8);tr,h=load_official()
    params={k:torch.nn.Parameter(v.cuda()) for k,v in state['params'].items()}
    variables={k:v.cuda() if torch.is_tensor(v) else v for k,v in state['variables'].items()}
    optimizer=tr.initialize_optimizer(params,variables);optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    start_frame=state['next_frame'];del state
    assert all(g['lr']==0 for g in optimizer.param_groups if g['name'] in ['logit_opacities','log_scales','cam_m','cam_c'])
    md=json.loads((DATA/'basketball/train_meta.json').read_text())
    history=json.loads((folder/'training-log.json').read_text()) if (folder/'training-log.json').exists() else []
    started=time.perf_counter()-(history[-1]['elapsed_seconds'] if history else 0)
    for t in range(start_frame,61):
        params,variables=tr.initialize_per_timestep(params,variables,optimizer)
        if args.mode=='frozen':variables['prev_col']=params['rgb_colors'].detach().clone()
        assert (variables['prev_col'].data_ptr()!=params['rgb_colors'].data_ptr())==(args.mode=='frozen')
        previous=params['rgb_colors'].detach().clone();views=dataset(t,md);todo=[]
        for i in range(2000):
            batch=tr.get_batch(todo,views);loss,variables=tr.get_loss(params,batch,variables,False);loss.backward()
            with torch.no_grad():
                assert torch.isfinite(loss)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                if i%500==0 or i==1999:
                    row=dict(frame=t,iteration=i+1,loss=float(loss),
                        implemented_color_regularizer=float(h.l1_loss_v2(params['rgb_colors'],variables['prev_col'])),
                        actual_previous_color_l1=float(h.l1_loss_v2(params['rgb_colors'],previous)),
                        elapsed_seconds=time.perf_counter()-started)
                    history.append(row);save(folder/'training-log.json',history)
                    print('VALIDATION TRAIN',args.mode,t,i+1,round(row['loss'],6),flush=True)
        save_state(folder/f'frame-{t:04d}.pt',params)
        checkpoint(folder/'resume.pt',params,variables,optimizer,t+1,0,[])
        del views,previous
    save(folder/'completion.json',dict(status='trained_validation_evaluation_pending',first_frame=31,last_frame=60,
        frames=30,updates=60000,final_points=len(params['means3D']),wall_seconds=time.perf_counter()-started))


if __name__=='__main__':main()
