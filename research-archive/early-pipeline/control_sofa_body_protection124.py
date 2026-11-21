"""One preregistered causal check; no full sequence or Gaussian admission."""
import argparse
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import trace_sofa_knee_stop124 as base
from smplify_face_hand_partition124 import adapt_source, ALLOWED

a, trace, entry = base.audit, base.trace, base.entry
read,save,sha,now = base.read,base.save,base.sha,base.now
OUT = a.RUN / 'body-protected-final-stage-control'


def verify():
    p=read(OUT/'plan.json')
    for path,digest in p['frozen'].items(): assert sha(path)==digest,path
    return p


def prepare():
    assert not OUT.exists()
    assert read(base.OUT/'conclusion.json')['actual_visual_review']
    assert read(a.RUN/'quality-stop-decision.json')['pid'] is None
    row,_=base.target()
    files=[Path(__file__),Path(base.__file__),Path(inspect.getsourcefile(adapt_source)),Path(trace.__file__),
           Path(entry.__file__),Path(trace.probe.__file__),Path(inspect.getsourcefile(base.stage_config)),
           Path(inspect.getsourcefile(base.install_loss_factory)),
           base.OUT/'trajectory.npz',base.OUT/'result.json',base.OUT/'conclusion.json',
           a.RUN/'plan.json',a.SRC/'data/inputs.json',a.SRC/'data/observations.npz',Path(row['image'])]
    files += [a.frame(a.RUN,i) for i in (16,17)]
    files += [a.OFFICIAL/name for name in entry.PINNED]
    entry._root_seed_original_source=entry.adapted_entry_source
    source,_=trace.probe.adapted_source(a.OFFICIAL)
    adapt_source(source)
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),frame='frame_000016',diagnostic_only=True,
        hypothesis='Protect the stage3 body solution from stage4 face/hand coupling by changing only the final optimizer parameter list. Determine whether face and hand fitting can be retained while preserving the repaired knee.',
        final_stage_trainable=list(ALLOWED),body_stage_prefix_unchanged=True,
        preserved=['original loss and all weights','original LBFGS and monitor','stage3/4 schedule','same17 initializer for matched diagnostic only','fixed K and translation policy','frozen shared beta','same native observations and temporal target'],
        budget=dict(fitter_calls=1,frames=1,max_seconds=300),
        prefix_parity_tolerance=2e-6,
        comparative_gate='Require exact stage3 prefix parity, unchanged body/root/beta/translation through final stage, no worse confidence-weighted face or hand pixel means than the saved full-parameter stage4 endpoint, and actual native body/face/hands/feet review. Detector values are not physical truth.',
        risks=['Frozen neck/head and global pose may prevent the face from aligning.','Existing body stage3 or shared shape/camera errors remain.'],
        stopping='One frame only. Preserve and reject on prefix mismatch or face/hand regression. No parameter or threshold scan, pose substitution, full sequence or Gaussian training.',
        frozen={str(p):sha(p) for p in files}))
    print('BODY_PROTECTION_CONTROL_REGISTERED',flush=True)


def worker():
    p=verify();assert not (OUT/'progress.json').exists()
    save(OUT/'progress.json',dict(updated=now(),state='single_frame_control',pid=os.getpid()))
    originals=entry.adapted_entry_source,entry.load_official_config,entry.load_official
    start=time.monotonic()
    try:
        import torch,smplx
        torch.set_num_threads(4)
        row,targets=base.target()
        previous=a.load_npz(a.frame(a.RUN,17));baseline=a.load_npz(a.frame(a.RUN,16))
        oldtrace=a.load_npz(base.OUT/'trajectory.npz')
        oldresult=read(base.OUT/'result.json')
        utils,_,_,_=originals[2](a.OFFICIAL)
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        loader=entry.module(a.PROJECT/'src/avatar_core/shuolee_vposer.py','_body_protected_vposer124')
        loader.MODEL,loader.CODE=a.ASSETS/'vposer/V02_05',a.PROJECT/'third_party/human_body_prior'
        vposer,_,_=loader.load_vposer('cuda')
        body=smplx.SMPLX(str(a.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
            create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).to('cuda')
        trace.probe.cache_static_topology(body)
        stages,states,receipt=[],[],{}
        neighbour=np.zeros((1,165),np.float32)
        neighbour[:,:3],neighbour[:,3:66]=previous['global_orient'],previous['body_pose_axis_angle']
        source=inspect.getsource(trace.install_monitor_trace)
        hook="type(loss).__name__ == 'SMPLifyLoss'";assert source.count(hook)==1
        scope=dict(trace.install_monitor_trace.__globals__)
        exec(compile(source.replace(hook,'isinstance(loss, fitting.SMPLifyLoss)'),'<temporal-subclass-observer>','exec'),scope)
        observer=scope['install_monitor_trace'];entry._root_seed_original_source=originals[0]

        def adapted(root):
            source,diff=trace.probe.adapted_source(root)
            changed,delta=adapt_source(source)
            return changed,diff+'\n'+delta

        def config(root):
            c=base.stage_config(originals[1](root),False);assert c['maxiters']==30
            c.update(_seed_pose_embedding=previous['pose_embedding'],_seed_global_orient=previous['global_orient'],
                     _seed_root_enabled=True,_freeze_body_in_final_stage=True)
            return c

        def load(root):
            values=originals[2](root)
            dt=abs(row['time_seconds']-read(a.SRC/'data/inputs.json')['records'][16]['time_seconds'])
            base.install_loss_factory(values[2],neighbour,dt,read(a.RUN/'plan.json')['nominal_dt_seconds'],receipt)
            observer(values[2],stages,states)
            return values

        entry.adapted_entry_source,entry.load_official_config,entry.load_official=adapted,config,load
        arrays,report=entry.fit_one(official_root=a.OFFICIAL,image=trace.sequence.checked_image(row),keypoints=targets,
            K_integer_centers=np.array(row['K_integer_centers']),body_model=body,vposer2=vposer,result_path=OUT/'official-result.pkl',
            shared_betas=previous['betas'],freeze_betas=True,device='cuda',diff_path=OUT/'adapter.diff')
        serial=[{k:({n:v.tolist() for n,v in value.items()} if k=='weights' else value) for k,value in s.items() if not k.startswith('_')} for s in stages]
        save(OUT/'monitor-stages.json',dict(updated=now(),stages=serial))
        trace.sequence.save_npz(OUT/'winner-camera-space.npz',**arrays)
        assert len(stages)==3 and report['orientation_count']==1
        body_stages=[s for s in stages if s['kind']=='body']
        oldbody=[s for s in oldresult['stages'] if s['kind']=='body']
        first=states[body_stages[0]['after_state']];last=states[body_stages[1]['after_state']]
        oldidx=oldbody[0]['after_state']
        prefix={name:float(np.max(np.abs(value-oldtrace[f'state_{oldidx:03d}_parameter_{name}']))) for name,value in first['parameters'].items()}
        prefix.update(latent=float(np.max(np.abs(first['latent']-oldtrace[f'state_{oldidx:03d}_pose_embedding']))),
                      translation=float(np.max(np.abs(first['translation']-oldtrace[f'state_{oldidx:03d}_camera_translation']))))
        assert all(x<=p['prefix_parity_tolerance'] for x in prefix.values()),prefix
        unchanged=dict(body_latent=np.array_equal(first['latent'],last['latent']),
                       root=np.array_equal(first['parameters']['global_orient'],last['parameters']['global_orient']),
                       beta=np.array_equal(first['parameters']['betas'],last['parameters']['betas']),
                       translation=np.array_equal(first['translation'],last['translation']))
        assert all(unchanged.values()),unchanged
        ctx=body_stages[-1]['_context']
        selected=ctx['optimizer'].param_groups[0]['params']
        actual_names=[name for name,v in body.named_parameters() if any(v is x for x in selected)]
        assert set(actual_names)==set(ALLOWED) and len(selected)==6
        trace.OUT=OUT
        evaluations=trace.analyse_states(body,stages,states,torch.as_tensor(targets,device='cuda'),arrays)

        def metrics(projected):
            distance=np.linalg.norm(projected[0]-targets[0,:,:2],axis=1)
            def weighted(lo,hi):
                w=targets[0,lo:hi,2];return float((distance[lo:hi]*w).sum()/max(float(w.sum()),1e-8))
            return dict(knee=float(distance[13]),face=weighted(67,118),hands=weighted(25,67),feet=weighted(19,25))

        oldmetrics,newmetrics=metrics(baseline['joints_projected_native']),metrics(arrays['joints_projected_native'])
        face_hands_protected=all(newmetrics[k]<=oldmetrics[k] for k in ('face','hands'))
        verify()
        save(OUT/'result.json',dict(updated=now(),state='complete_pending_visual_review',elapsed_seconds=time.monotonic()-start,
            prefix_parity=prefix,final_body_parameters_unchanged=unchanged,actual_final_parameters=actual_names,
            metrics=dict(saved_full_parameter_endpoint=oldmetrics,body_protected_endpoint=newmetrics),
            face_hand_comparative_gate_passed=face_hands_protected,stages=serial,evaluations=evaluations,
            original_report=report,diagnostic_only=True,actual_visual_review=False,quality_accepted=False,
            gaussian_admitted=False,full_sequence_admitted=False))
        save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None))
        print('BODY_PROTECTION_COMPLETE',oldmetrics,newmetrics,face_hands_protected,flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',dict(updated=now(),error=repr(exc)))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None));raise
    finally:entry.adapted_entry_source,entry.load_official_config,entry.load_official=originals


def run():
    import fcntl
    p=verify();assert not (OUT/'progress.json').exists()
    lease=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (OUT/'fit.log').open('x') as log:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=p['budget']['max_seconds'])
    print('BODY_PROTECTION_CONTROL_FINISHED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker'])
    args=parser.parse_args();dict(prepare=prepare,run=run,worker=worker)[args.mode]()
