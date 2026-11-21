"""One diagnostic replay of saved33; original solver, loss and stop tests intact."""
import argparse
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import trace_smplifyx_stage19_123 as trace
import audit_sofa_temporal_objective124 as audit
from smplify_temporal124 import install_loss_factory
from smplify_observation_validity124 import mask_outside_raster
from smplify_score_contract123 import relative_confidence

entry=trace.entry
read,save,sha,now=trace.read,trace.save,trace.sha,trace.now
OUT=audit.RUN/'stage33-diagnostic-replay'

def target():
    row=read(audit.SRC/'data/inputs.json')['records'][32]
    with np.load(audit.SRC/'data/observations.npz') as z:obs=dict(z)
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    x=np.zeros((1,118,3),np.float32)
    dst=np.r_[entry.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    source=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    x[0,dst,:2]=obs['keypoints'][32,source];x[0,dst,2]=confidence[32,source]
    return row,mask_outside_raster(x,row['wh'])[0]

def verify():
    p=read(OUT/'plan.json')
    for path,digest in p['frozen'].items():assert sha(path)==digest,path
    return p

def prepare():
    assert not OUT.exists() and read(audit.RUN/'temporal-progress.json')['pid'] is None
    row,targets=target()
    files=[Path(__file__),Path(trace.__file__),Path(audit.__file__),Path(entry.__file__),Path(trace.probe.__file__),
        Path(__file__).with_name('smplify_temporal124.py'),Path(__file__).with_name('run_official_sequence123_topology.py'),
        audit.SRC/'data/inputs.json',audit.SRC/'data/observations.npz',Path(row['image']),audit.RUN/'plan.json']
    files += [audit.RUN/f'official-fit/frames/frame_{i:06d}/{n}' for i in (32,33) for n in ('winner-camera-space.npz','result.json')]
    files += [audit.OFFICIAL/p for p in entry.PINNED]
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),frame='frame_000033',purpose='Observe the exact original five-stage trajectory including temporal loss; distinguish early prior reset from final stage selection and stopping.',
        diagnostic_only=True,fitter_calls=1,max_seconds=300,no_new_training_candidate=True,no_pose_substitution=True,
        initial_state='Replay recorded same-run32 body/root and original zero face/hands solely as a diagnostic, not a new model initializer.',
        instrumentation='Reuse existing original monitor/step observer. Recognize TemporalLoss via isinstance instead of the literal class name. No solver/loss/gradient/stopping changes.',
        temporal_weights_unchanged=True,original_stages_unchanged=True,
        parity_tolerances=dict(vertices_camera=2e-6,joints_projected_native=.002,pose_embedding=2e-6,global_orient=2e-6,camera_translation=1e-7,betas=0),
        interpretation_gate='Interpret the replay as the saved failure only after endpoint parity. Preserve and stop on mismatch; do not loosen tolerance or rerun.',
        frozen={str(p):sha(p) for p in files}))
    print('SOFA_SINGLE_REPLAY_REGISTERED',flush=True)

def worker():
    p=verify();assert not (OUT/'progress.json').exists()
    save(OUT/'progress.json',dict(state='diagnostic_replay',pid=os.getpid(),updated=now()))
    originals=entry.adapted_entry_source,entry.load_official_config,entry.load_official
    start=time.monotonic()
    try:
        import torch,smplx
        from smplx.lbs import batch_rodrigues
        torch.set_num_threads(4)
        row,targets=target();K=np.array(row['K_integer_centers'])
        previous=trace.load_npz(audit.RUN/'official-fit/frames/frame_000032/winner-camera-space.npz')
        expected=trace.load_npz(audit.RUN/'official-fit/frames/frame_000033/winner-camera-space.npz')
        utils,_,_,_=originals[2](audit.OFFICIAL)
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        loader=entry.module(audit.PROJECT/'src/avatar_core/shuolee_vposer.py','_sofa_trace124_vposer')
        loader.MODEL,loader.CODE=audit.ASSETS/'vposer/V02_05',audit.PROJECT/'third_party/human_body_prior'
        vposer,_,_=loader.load_vposer('cuda')
        body=smplx.SMPLX(str(audit.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
            joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).to('cuda')
        trace.probe.cache_static_topology(body)
        stages,states=[],[];receipt={}
        neighbour=np.zeros((1,165),np.float32);neighbour[:,:3]=previous['global_orient'];neighbour[:,3:66]=previous['body_pose_axis_angle']
        source=inspect.getsource(trace.install_monitor_trace)
        old="type(loss).__name__ == 'SMPLifyLoss'"
        assert source.count(old)==1
        scope=dict(trace.install_monitor_trace.__globals__)
        exec(compile(source.replace(old,'isinstance(loss, fitting.SMPLifyLoss)'),'<trace-temporal-subclass-recognition-only>','exec'),scope)
        observer=scope['install_monitor_trace']
        entry._root_seed_original_source=originals[0]
        def config(root):
            c=originals[1](root);assert c['maxiters']==30
            c.update(_seed_pose_embedding=previous['pose_embedding'],_seed_global_orient=previous['global_orient'],_seed_root_enabled=True)
            return c
        def load(root):
            values=originals[2](root)
            dt=row['time_seconds']-read(audit.SRC/'data/inputs.json')['records'][31]['time_seconds']
            install_loss_factory(values[2],neighbour,dt,read(audit.RUN/'plan.json')['nominal_dt_seconds'],receipt)
            observer(values[2],stages,states)
            return values
        entry.adapted_entry_source,entry.load_official_config,entry.load_official=trace.probe.adapted_source,config,load
        fit_start=time.monotonic()
        arrays,report=entry.fit_one(official_root=audit.OFFICIAL,image=trace.sequence.checked_image(row),keypoints=targets,
            K_integer_centers=K,body_model=body,vposer2=vposer,result_path=OUT/'official-result.pkl',shared_betas=previous['betas'],freeze_betas=True,device='cuda',diff_path=OUT/'adapter.diff')
        fit_seconds=time.monotonic()-fit_start
        parity={k:float(np.max(np.abs(arrays[k]-expected[k]))) for k in p['parity_tolerances']}
        save(OUT/'endpoint-parity.json',dict(differences=parity,tolerances=p['parity_tolerances']))
        trace.sequence.save_npz(OUT/'traced-winner.npz',**arrays)
        assert all(parity[k]<=limit for k,limit in p['parity_tolerances'].items()),parity
        assert report['orientation_count']==1 and len(stages)==6
        trace.OUT=OUT
        evaluations=trace.analyse_states(body,stages,states,torch.as_tensor(targets,device='cuda'),arrays)
        rotation=[]
        with torch.no_grad():
            ref=batch_rodrigues(torch.as_tensor(previous['body_pose_axis_angle'].reshape(-1,3),device='cuda'))
            for i,state in enumerate(states):
                latent=torch.as_tensor(state['latent'],device='cuda')
                aa=vposer.decode(latent,output_type='aa').reshape(-1,3)
                rot=batch_rodrigues(aa);tr=(ref.transpose(1,2)@rot).diagonal(dim1=-2,dim2=-1).sum(-1)
                angles=torch.rad2deg(torch.acos(((tr-1)/2).clamp(-1,1))).cpu().numpy()
                rotation.append(dict(state=i,kind=state['kind'],stage=state['stage'],label=state['label'],right_hip_from32_deg=float(angles[1]),right_knee_from32_deg=float(angles[4])))
        serial=[{k:({n:v.tolist() for n,v in value.items()} if k=='weights' else value) for k,value in s.items() if not k.startswith('_')} for s in stages]
        verify()
        save(OUT/'result.json',dict(updated=now(),state='trajectory_complete_pending_review',fit_seconds=fit_seconds,elapsed_seconds=time.monotonic()-start,
            endpoint_parity_passed=True,endpoint_parity=parity,stages=serial,evaluations=evaluations,rotation_from_neighbour=rotation,
            original_report=report,actual_visual_review=False,no_new_training_candidate=True,quality_accepted=False,gaussian_admitted=False))
        save(OUT/'progress.json',dict(state='completed',pid=None,updated=now()))
        print('SOFA_REPLAY_ENDPOINT_PARITY_PASSED',parity,flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',dict(updated=now(),error=repr(exc)))
        save(OUT/'progress.json',dict(state='failed',pid=None,updated=now()))
        raise
    finally:entry.adapted_entry_source,entry.load_official_config,entry.load_official=originals

def run():
    import fcntl
    p=verify();assert not (OUT/'progress.json').exists()
    lease=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (OUT/'replay.log').open('x') as log:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=p['max_seconds'])
    print('SOFA_DIAGNOSTIC_REPLAY_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','worker']);args=ap.parse_args();dict(prepare=prepare,run=run,worker=worker)[args.mode]()
