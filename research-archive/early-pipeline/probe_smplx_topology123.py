"""Fixed-state official loss/gradient timing; no fit or optimizer step."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import re
import sys
import time

import numpy as np
import official_smplifyx_entry123 as entry


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return entry.sha(p)


def log_stages(root):
    path=root.parent/'run-full94.log'
    text=path.read_text(encoding='utf-8',errors='replace')
    start=0;rows=[]
    for match in re.finditer(r'\{"frame_id": "frame_\d+"[^\n]*\}',text):
        row=json.loads(match.group());part=text[start:match.start()];start=match.end()
        row['stages']=[dict(stage=int(i),seconds=float(t)) for i,t in re.findall(r'Stage (\d+) done after ([\d.]+) seconds',part)]
        row['camera_init_seconds']=[float(t) for t in re.findall(r'Camera initialization done after ([\d.]+)',part)]
        report=read(root/'frames'/row['frame_id']/'result.json')
        assert len(row['stages'])==5*report['orientation_count'] and len(row['camera_init_seconds'])==1
        assert row['elapsed_seconds']==report['elapsed_seconds']
        rows.append(row)
    assert len(rows)==94 and len(set(r['frame_id'] for r in rows))==94
    summary=[]
    for stage in range(5):
        values=[s['seconds'] for row in rows for s in row['stages'] if s['stage']==stage]
        summary.append(dict(stage=stage,count=len(values),sum_seconds=sum(values),median_seconds=float(np.median(values))))
    return dict(source_sha256=sha(path),frames=rows,stage_summary=summary,
        final_two_stage_fraction=sum(v['sum_seconds'] for v in summary[3:])/sum(v['sum_seconds'] for v in summary),
        caveat='Upstream stages use wall clock; frame totals use monotonic clock. Their mismatch was already observed; do not add these into an E2E benchmark.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);args=p.parse_args()
    plan_path=Path(args.plan);plan=read(plan_path);root=Path(plan['root']);official=Path(plan['official_root'])
    out=root/'reviews'/plan.get('output_directory','official-topology-cost');out.mkdir(exist_ok=False)
    started=time.monotonic()
    for path,digest in plan['source_hashes'].items(): assert sha(path)==digest,path
    entry.adapted_entry_source(official)  # Pinned upstream source verification only.
    save(out/'original-stage-timing.json',log_stages(root))
    project=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
    assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
    sys.path.insert(0,str(project/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(assets,out)
    import torch
    import smplx
    from torch.profiler import profile,ProfilerActivity
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    utils,prior,fitting,optimizer=entry.load_official(official)
    loader=entry.module(project/'src/avatar_core/shuolee_vposer.py','_topology_cost_vposer')
    loader.MODEL=assets/'vposer/V02_05';loader.CODE=project/'third_party/human_body_prior'
    with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        vposer,_,_=loader.load_vposer('cuda')
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        body=smplx.SMPLX(str(assets/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,
            num_expression_coeffs=10,use_pca=True,num_pca_comps=12,flat_hand_mean=False,
            use_face_contour=False,create_body_pose=False,create_transl=False,
            joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cuda()
        body.betas.requires_grad_(False)
        parents=body.parents.detach().cpu().clone()
        assert parents.ndim==1 and len(parents)==55 and int(parents[0])==-1
        class VposerAPI(torch.nn.Module):
            def decode(self,x,output_type='aa'):
                assert output_type=='aa'
                return vposer.decode(x)['pose_body'].reshape(x.shape[0],-1)
        class Camera(torch.nn.Module):
            def __init__(self,K,t):
                super().__init__();self.register_buffer('K',torch.as_tensor(K,dtype=torch.float32,device='cuda'))
                self.translation=torch.nn.Parameter(torch.as_tensor(t,dtype=torch.float32,device='cuda').clone())
            def forward(self,x):
                q=(x+self.translation[:,None])@self.K.T
                return q[...,:2]/q[...,2:]
        cfg=entry.load_official_config(official)
        cfg.update(interpenetration=False,visualize=False,save_meshes=False,use_cuda=True,model_type='smplx')
        inputs=read(root/'data/training-inputs.json')['records']
        with np.load(root/'fit/fitted.npz',allow_pickle=False) as z: targets=z['observed_native_keypoints'].copy()
        results=[];profiles=[]
        for frame_id in plan['frames']:
            slot=next(i for i,r in enumerate(inputs) if r['frame']==frame_id)
            with np.load(root/'frames'/frame_id/'winner-camera-space.npz',allow_pickle=False) as z: saved=dict(z)
            body.reset_params(**{name:saved[name] for name,_ in body.named_parameters()})
            latent=torch.nn.Parameter(torch.as_tensor(saved['pose_embedding'],device='cuda').clone())
            cam=Camera(saved['K_integer_centers'],saved['camera_translation'])
            gt=torch.as_tensor(targets[slot:slot+1,:,:2],device='cuda');conf=torch.as_tensor(targets[slot:slot+1,:,2],device='cuda')
            joint_weights=torch.ones((1,118),device='cuda');joint_weights[:,[1,9,12]]=0
            joint_weights[:,25:67]=cfg['hand_joints_weights'][-1];joint_weights[:,67:]=cfg['face_joints_weights'][-1]
            l2=lambda:prior.L2Prior().cuda()
            loss=fitting.create_loss(loss_type='smplify',rho=cfg['rho'],joint_weights=joint_weights,
                use_joints_conf=True,use_face=True,use_hands=True,interpenetration=False,
                body_pose_prior=l2(),shape_prior=l2(),expr_prior=l2(),jaw_prior=l2(),
                left_hand_prior=l2(),right_hand_prior=l2(),angle_prior=prior.SMPLifyAnglePrior().cuda()).cuda()
            loss.reset_loss_weights(dict(data_weight=1000/inputs[slot]['wh'][1],
                body_pose_weight=cfg['body_pose_prior_weights'][-1],shape_weight=cfg['shape_weights'][-1],
                bending_prior_weight=3.17*cfg['body_pose_prior_weights'][-1],
                hand_prior_weight=cfg['hand_pose_prior_weights'][-1],expr_prior_weight=cfg['expr_weights'][-1],
                jaw_prior_weight=[float(v) for v in cfg['jaw_pose_prior_weights'][-1].split(',')]))
            params=[p for p in body.parameters() if p.requires_grad]+[latent]
            opt,create_graph=optimizer.create_optimizer(params,**cfg)
            monitor=fitting.FittingMonitor(visualize=False,model_type='smplx',maxiters=cfg['maxiters'])
            monitor.steps=0
            closure=monitor.create_fitting_closure(opt,body,cam,gt,loss,conf,joint_weights,
                return_verts=True,return_full_pose=True,use_vposer=True,vposer=VposerAPI(),pose_embedding=latent,create_graph=create_graph)
            reference=None;timings=[];before=[p.detach().clone() for p in params]+[cam.translation.detach().clone()]
            for mode in ['cuda_parent_indices','cpu_parent_indices']:
                body.parents=parents.to('cuda' if mode.startswith('cuda') else 'cpu')
                with torch.no_grad():
                    state=body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
                    reprojection=cam(state.joints)
                    replay_error=float(abs(reprojection-torch.as_tensor(saved['joints_projected_native'],device='cuda')).max())
                    vertices=state.vertices.detach().clone();joints=state.joints.detach().clone()
                assert replay_error<.001
                seconds=[]
                for rep in range(2+plan['timed_closures']):
                    if time.monotonic()-started>plan['max_seconds']:raise RuntimeError('Probe time budget exceeded.')
                    cam.translation.grad=None
                    torch.cuda.synchronize();start=time.monotonic();value=closure();torch.cuda.synchronize()
                    if rep>=2:seconds.append(time.monotonic()-start)
                    assert torch.isfinite(value)
                grads=[None if p.grad is None else p.grad.detach().clone() for p in params]+[cam.translation.grad.detach().clone()]
                assert all(g is None or torch.isfinite(g).all() for g in grads)
                state=dict(loss=value.detach().clone(),vertices=vertices,joints=joints,grads=grads)
                if reference is None:reference=state
                parity=dict(loss=float(abs(reference['loss']-state['loss'])),
                    vertices_max=float(abs(reference['vertices']-vertices).max()),joints_max=float(abs(reference['joints']-joints).max()),
                    gradients_max=max(float(abs(a-b).max()) for a,b in zip(reference['grads'],grads) if a is not None and b is not None),
                    grad_none_pattern_equal=[g is None for g in reference['grads']]==[g is None for g in grads])
                assert parity['loss']==0 and parity['vertices_max']==0 and parity['joints_max']==0 and parity['gradients_max']==0 and parity['grad_none_pattern_equal']
                timings.append(dict(mode=mode,wall_seconds=seconds,median_seconds=float(np.median(seconds)),saved_projection_max_px=replay_error,parity=parity))
                if frame_id==plan['frames'][0]:
                    cam.translation.grad=None
                    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False) as prof:
                        closure();torch.cuda.synchronize()
                    events=[]
                    for event in prof.key_averages():
                        events.append(dict(name=event.key,count=event.count,self_cpu_us=event.self_cpu_time_total,
                            cpu_total_us=event.cpu_time_total,self_device_us=getattr(event,'self_device_time_total',0.)))
                    profiles.append(dict(mode=mode,events=events))
                del grads,state,vertices,joints
            assert all(torch.equal(a,b) for a,b in zip(before,params+[cam.translation]))
            assert len(opt.state)==0
            results.append(dict(frame_id=frame_id,parameters_unchanged=True,optimizer_state_entries=0,timings=timings))
        for path,digest in plan['source_hashes'].items():assert sha(path)==digest,path
        save(out/'profiler.json',profiles)
        result=dict(state='completed',plan_sha256=sha(plan_path),elapsed_seconds=time.monotonic()-started,
            frames=results,optimizer_updates=0,upstream_source_modified=False,production_runtime_changed=False,
            parent_values=parents.tolist(),only_control_variable='device of invariant integer55-node parent topology',
            limitation='Loss/gradient replay at three saved states, not a full optimization or E2E benchmark. No candidate pose or training result produced.')
        save(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
