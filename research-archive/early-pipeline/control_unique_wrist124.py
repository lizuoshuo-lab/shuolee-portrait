"""One input-only GPU control, using the original all-parameter SMPLify-X loop."""
import argparse
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import trace_sofa_knee_stop124 as base
import prepare_unique_wrist_contract124 as inputs

a,trace,entry = base.audit,base.trace,base.entry
read,save,sha,now = base.read,base.save,base.sha,base.now
OUT = inputs.OUT/'sofa16-control'
PROTECTED = [2,3,4,5,6,7,10,11,14]


def verify():
    p=read(OUT/'plan.json')
    for path,digest in p['frozen'].items():assert sha(path)==digest,path
    return p


def metrics(projected, targets):
    distance=np.linalg.norm(projected[0]-targets[0,:,:2],axis=1)
    weight=targets[0,:,2].copy();weight[[1,9,12]]=0
    def weighted(ids):
        w=weight[ids];return float((distance[ids]*w).sum()/max(float(w.sum()),1e-8))
    return dict(knee=float(distance[13]),body=weighted(np.arange(19)),
        face=weighted(np.arange(67,118)),hands=weighted(np.arange(25,67)),
        fingers=weighted(np.r_[26:46,47:67]),feet=weighted(np.arange(19,25)),
        all_joint_distance_px=distance.tolist(),original_confidence=targets[0,:,2].tolist())


def prepare():
    assert not OUT.exists()
    pre=read(inputs.OUT/'preflight.json');assert pre['fitter_calls']==0 and all(pre['checks'].values())
    assert read(base.OUT/'conclusion.json')['actual_visual_review']
    assert read(a.RUN/'quality-stop-decision.json')['pid'] is None
    frozen=read(inputs.OUT/'plan.json')['frozen'].copy()
    for path,digest in frozen.items():assert sha(path)==digest,path
    row,_=base.target()
    paths=[Path(__file__),Path(base.__file__),Path(trace.__file__),Path(trace.probe.__file__),
           Path(a.__file__),Path(inspect.getsourcefile(base.stage_config)),
           Path(inspect.getsourcefile(base.install_loss_factory)),Path(trace.sequence.__file__),
           Path(inspect.getsourcefile(trace.probe.cache_static_topology)),
           inputs.OUT/'plan.json',inputs.OUT/'preflight.json',
           base.OUT/'trajectory.npz',base.OUT/'result.json',base.OUT/'conclusion.json',
           a.RUN/'plan.json',a.frame(a.RUN,16),a.frame(a.RUN,17),Path(row['image']),Path(row['mask'])]
    paths += [a.OFFICIAL/name for name in entry.PINNED]
    frozen.update({str(p):sha(p) for p in paths})
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),frame='frame_000016',diagnostic_only=True,
        hypothesis='Determine whether duplicated wrist-root observations contribute to the stage4 knee drift. Change only their ownership, with the original all-parameter optimizer and all40 finger observations retained.',
        intervention=read(inputs.OUT/'plan.json')['rule'],
        fixed=['Same recorded17 seed for matched diagnosis only, not a new sequence initialization.','Same native K, camera initialization, shared betas, original stages3/4, maxiters30, LBFGS, priors and temporal weights30.','No parameter freezing or added local joints.'],
        budget=dict(fitter_calls=1,frames=1,max_seconds=300),
        comparative_gate=dict(knee='strictly improve versus saved original full-parameter endpoint',
            body_face_hands_feet='original-confidence weighted means may not regress beyond0.002px numerical tolerance',
            protected_body_ids=PROTECTED,protected_body_tolerance_px=1.,
            all_original_observations_scored=True,excluded_duplicate_still_in_review=True,
            actual_native_body_face_hands_feet_review_required=True),
        risks=['Body wrist primary is not guaranteed physically correct.','Removing root targets may degrade hand pose.','Other wrong observations and monocular depth ambiguity remain.'],
        stop='One fit only. No weight/threshold/source-priority sweep, automatic full sequence, pose stitching or Gaussian admission. Preserve and report a failed comparison.',
        frozen=frozen))
    print('UNIQUE_WRIST_MATCHED_CONTROL_REGISTERED',flush=True)


def worker():
    p=verify();assert not (OUT/'progress.json').exists()
    save(OUT/'progress.json',dict(updated=now(),state='single_input_control',pid=os.getpid()))
    originals=entry.adapted_entry_source,entry.load_official_config,entry.load_official
    started=time.monotonic()
    try:
        import torch,smplx
        torch.set_num_threads(4)
        row,targets=base.target()
        previous=a.load_npz(a.frame(a.RUN,17));baseline=a.load_npz(a.frame(a.RUN,16))
        oldtrace=a.load_npz(base.OUT/'trajectory.npz');oldresult=read(base.OUT/'result.json')
        utils,_,_,_=originals[2](a.OFFICIAL)
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        fit_targets,counts=inputs.unique_wrist_observations(targets,mapping)
        assert counts['suppressed']==2
        changed=np.argwhere(fit_targets!=targets).tolist();assert changed==[[0,25,2],[0,46,2]],changed
        save(OUT/'input-delta.json',dict(updated=now(),changed_indices=changed,counts=counts,
            original_root_confidence=targets[0,[25,46],2].tolist(),all_coordinates_unchanged=True,
            all40_finger_observations_unchanged=True,original_targets_retained_in_result=True))
        loader=entry.module(a.PROJECT/'src/avatar_core/shuolee_vposer.py','_unique_wrist_vposer124')
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
        old="type(loss).__name__ == 'SMPLifyLoss'";assert source.count(old)==1
        scope=dict(trace.install_monitor_trace.__globals__)
        exec(compile(source.replace(old,'isinstance(loss, fitting.SMPLifyLoss)'),'<original-temporal-observer>','exec'),scope)
        observer=scope['install_monitor_trace'];entry._root_seed_original_source=originals[0]
        def config(root):
            c=base.stage_config(originals[1](root),False);assert c['maxiters']==30
            c.update(_seed_pose_embedding=previous['pose_embedding'],_seed_global_orient=previous['global_orient'],_seed_root_enabled=True)
            return c
        def load(root):
            values=originals[2](root)
            dt=abs(row['time_seconds']-read(a.SRC/'data/inputs.json')['records'][16]['time_seconds'])
            base.install_loss_factory(values[2],neighbour,dt,read(a.RUN/'plan.json')['nominal_dt_seconds'],receipt)
            observer(values[2],stages,states)
            return values
        entry.adapted_entry_source,entry.load_official_config,entry.load_official=trace.probe.adapted_source,config,load
        arrays,report=entry.fit_one(official_root=a.OFFICIAL,image=trace.sequence.checked_image(row),keypoints=fit_targets,
            K_integer_centers=np.array(row['K_integer_centers']),body_model=body,vposer2=vposer,result_path=OUT/'official-result.pkl',
            shared_betas=previous['betas'],freeze_betas=True,device='cuda',diff_path=OUT/'adapter.diff')
        fit_seconds=time.monotonic()-started
        arrays.update(original_native_observations=targets,fitting_native_observations=fit_targets)
        trace.sequence.save_npz(OUT/'winner-camera-space.npz',**arrays)
        serial=[{k:({n:v.tolist() for n,v in value.items()} if k=='weights' else value) for k,value in s.items() if not k.startswith('_')} for s in stages]
        save(OUT/'monitor-stages.json',dict(updated=now(),stages=serial))
        assert len(stages)==3 and report['orientation_count']==1
        checks={f:np.array_equal(arrays[f],baseline[f]) for f in ('K_integer_centers','camera_translation','betas','faces')}
        # Camera-init input indices do not include either duplicate hand root.
        assert stages[0]['kind']=='camera'
        first=states[stages[0]['after_state']];oldidx=oldresult['stages'][0]['after_state']
        camera_prefix={n:float(np.max(np.abs(v-oldtrace[f'state_{oldidx:03d}_parameter_{n}']))) for n,v in first['parameters'].items()}
        camera_prefix['latent']=float(np.max(np.abs(first['latent']-oldtrace[f'state_{oldidx:03d}_pose_embedding'])))
        camera_prefix['translation']=float(np.max(np.abs(first['translation']-oldtrace[f'state_{oldidx:03d}_camera_translation'])))
        checks['exact_camera_stage_prefix']=all(v==0 for v in camera_prefix.values())
        for current,oldstage in zip(serial,oldresult['stages']):
            assert current['weights']==oldstage['weights']
        for stage in stages:
            if stage['kind']!='body':continue
            c=stage['_context'];parameters=c['optimizer'].param_groups[0]['params']
            actual={name for name,v in body.named_parameters() if any(v is item for item in parameters)}
            expected={'global_orient','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression'}
            assert actual==expected and any(c['pose_embedding'] is item for item in parameters),(actual,len(parameters))
        checks['original_all_parameter_stages']=True
        assert all(checks.values()),checks
        trace.OUT=OUT
        evaluation=trace.analyse_states(body,stages,states,torch.as_tensor(fit_targets,device='cuda'),arrays)
        oldmetrics=metrics(baseline['joints_projected_native'],targets)
        newmetrics=metrics(arrays['joints_projected_native'],targets)
        gates=dict(knee_improved=newmetrics['knee']<oldmetrics['knee'])
        for k in ('body','face','hands','feet'):gates[k+'_protected']=newmetrics[k]<=oldmetrics[k]+.002
        gates['protected_body_points']=all(newmetrics['all_joint_distance_px'][j]<=oldmetrics['all_joint_distance_px'][j]+1. for j in PROTECTED)
        verify()
        save(OUT/'result.json',dict(updated=now(),state='complete_pending_visual_review',fit_seconds=fit_seconds,
            elapsed_seconds=time.monotonic()-started,interface_checks=checks,camera_prefix=camera_prefix,
            metrics=dict(saved_full_parameter_endpoint=oldmetrics,unique_wrist_endpoint=newmetrics),
            gates=gates,comparative_gate_passed=all(gates.values()),stages=serial,evaluations=evaluation,
            original_report=report,diagnostic_only=True,actual_visual_review=False,quality_accepted=False,
            full_sequence_admitted=False,gaussian_admitted=False))
        save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None))
        print('UNIQUE_WRIST_CONTROL_COMPLETE',{k:{n:v for n,v in m.items() if not isinstance(v,list)} for k,m in [('old',oldmetrics),('new',newmetrics)]},gates,flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',dict(updated=now(),error=repr(exc)))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None));raise
    finally:
        entry.adapted_entry_source,entry.load_official_config,entry.load_official=originals


def run():
    import fcntl
    p=verify();assert not (OUT/'progress.json').exists()
    lease=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (OUT/'fit.log').open('x') as log:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,
                       timeout=p['budget']['max_seconds'],check=True)
    print('UNIQUE_WRIST_CONTROL_FINISHED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker'])
    args=parser.parse_args();dict(prepare=prepare,run=run,worker=worker)[args.mode]()
