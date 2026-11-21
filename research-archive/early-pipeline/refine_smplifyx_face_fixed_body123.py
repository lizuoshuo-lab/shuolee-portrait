"""Uniform face-only final-stage refinement using the original SMPLify-X APIs.

No new sequence solver, pose decoder or camera fit. Every frame starts from its
saved complete state; only expression/jaw/eyes are optimizer parameters.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import replay_demo_limb_losses123 as b
import official_smplifyx_entry123 as entry
from review_smplifyx_orientations123 import metrics,load
from export_demo_body_snapshot123 import process

BODY=b.ROOT.parent
SOURCE=BODY/'body-weight-guarded-sequence'
OUT=BODY/'face-fixed-body-refinement'
PLAN=BODY/'face-fixed-body-refinement-plan.json'
HANDOFF=BODY/'face-fixed-body-refinement-gpu-handoff.json'
ACTIVE=('expression','jaw_pose','leye_pose','reye_pose')
read,save,sha=b.read,b.save,b.sha
def now():return datetime.now(timezone.utc).isoformat()

def prepare():
    assert not PLAN.exists() and not OUT.exists()
    close=read(SOURCE/'review/closeout.json');assert close['state']=='rejected_face_protection'
    causal=read(SOURCE/'face-coupling-audit/result.json');assert causal['frames']==59 and causal['body_forwards']==196
    assert read(SOURCE/'face-coupling-audit/visual-review.json')['actual_visual_review']
    entry.adapted_entry_source(b.OFFICIAL)
    cfg=entry.load_official_config(b.OFFICIAL);assert cfg['maxiters']==30 and cfg['optim_type']=='lbfgsls'
    assert cfg['face_joints_weights'][-1]==2 and cfg['hand_joints_weights'][-1]==2
    source=(b.OFFICIAL/'smplifyx/fit_single_frame.py').read_text()
    assert 'body_params = list(body_model.parameters())' in source
    assert 'final_params.append(pose_embedding)' in source
    assert 'final_params.append(camera.translation)' not in source
    frames=[f'frame_{i:06d}' for i in range(1,60)]
    inputs=[b.ROOT/'fit/fitted.npz',b.ROOT/'data/training-inputs.json',b.ASSETS/'smplx/SMPLX_NEUTRAL.npz',
        SOURCE/'selection-audit.json',SOURCE/'review/closeout.json',SOURCE/'face-coupling-audit/result.json',SOURCE/'face-coupling-audit/visual-review.json']
    for frame in frames:inputs += [root/'frames'/frame/'winner-camera-space.npz' for root in (b.ROOT,SOURCE)]
    sources=[Path(__file__),Path(entry.__file__),Path(b.__file__),Path(metrics.__code__.co_filename),Path(process.__code__.co_filename)]
    sources += [b.OFFICIAL/p for p in entry.PINNED]
    import smplx.body_models,smplx.lbs
    sources += [Path(smplx.body_models.__file__),Path(smplx.lbs.__file__)]
    save(PLAN,dict(updated=now(),state='prepared',frames=frames,frame_count=59,
        hypothesis='A local face refinement after the body-data pass may recover facial detail without letting the whole body move to trade off regions.',
        intervention='Append one original final-stage solve to every saved selected frame, with expression/jaw/eyes active and all pose/shape/camera/hand parameters frozen. This is a local stage-policy adaptation, not a new optimizer or an untouched official pipeline.',
        active_parameters=list(ACTIVE),fixed=['K','camera_translation','betas','global_orient','pose_embedding','body_pose_axis_angle','left_hand_pose','right_hand_pose','all observations/confidences'],
        stage_index=4,maxiters=30,optimizer='original lbfgsls strong_Wolfe',configuration=cfg,
        body_observation_amplitude={'frame_000033':1.,'other_frames':2.},
        body_weight_scope='Keep the previously registered body-data policy; selected independent fallback states are merely initial states. Same final-stage rule for all59; no per-frame weight tuning.',
        fixed_body_decode='Reuse the saved decoded pose of the frozen VPoser latent. Require full vertex/projection parity first; no VPoser gradient is needed for a frozen variable.',
        model_forward_budget='59 original refinement calls, each original monitor maxiters30; no additional retries',
        runtime_budget_seconds=600,expected_seconds=180,
        protections={'body17_median_delta_max_px':1.,'feet6_median_delta_max_px':1.,'face51_median_delta_max_px':.5,
            'elbow_frames':list(range(18,24)),'elbow_remaining_ratio_max':.5},
        limits={'initial_vertex_parity_m':2e-6,'initial_projection_parity_px':.002,'frozen_limb_vertex_delta_m':2e-6},
        scope=['No face parameter splice or frame selection; all59 outputs retained even if a gate fails.',
            'Head/neck remain frozen; this cannot fix a wrong head pose, inaccurate camera or missing physical correspondences.',
            'The observation residual is not independent material ground truth.',
            'Review all59 face images and frozen-limb invariants before any downstream admission.',
            'Fresh-video runtime must include earlier fitting; this extra stage alone is not a15minute pipeline.'],
        source_hashes={str(p.resolve()):sha(p) for p in sources},input_hashes={str(p):sha(p) for p in inputs},
        original_parameter_audit={'original_body_stage':['trainable body-model parameters','VPoser latent'],
          'camera_translation':'Optimized in the original initialization, not included in final body optimizer.',
          'new_stage':list(ACTIVE),'direct_pose_or_sequence_solver_added':False},
        no_automatic_followup=True,quality_accepted=False,goal_complete=False))
    print('PREPARED_FACE_ONLY_ALL59',flush=True)

def verify():
    p=read(PLAN)
    for f,h in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(f)==h,f
    return p

def worker():
    p=verify();assert not OUT.exists();OUT.mkdir()
    pid=int(os.environ['SHUOLEE_CONTROL_BASELINE_PID']);identity=os.environ['SHUOLEE_CONTROL_BASELINE_IDENTITY']
    assert process(pid)[1]=='T' and process(pid)[2]==identity
    save(OUT/'progress.json',dict(state='initializing',pid=os.getpid(),frames_completed=0))
    started=time.monotonic();rows_result=[]
    try:
        import torch,smplx
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        utils,prior,fitting,factory=entry.load_official(b.OFFICIAL);cfg=p['configuration']
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        body=smplx.SMPLX(str(b.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
            joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cuda().eval()
        body.parents=body.parents.cpu()
        for name,param in body.named_parameters():param.requires_grad_(name in ACTIVE)
        parameters=[param for name,param in body.named_parameters() if param.requires_grad]
        assert [name for name,param in body.named_parameters() if param.requires_grad]==[name for name,param in body.named_parameters() if name in ACTIVE]
        assert len(parameters)==4
        vertex_labels=body.lbs_weights.detach().cpu().numpy().argmax(1)
        limb_vertices=np.isin(vertex_labels,[1,2,4,5,7,8,10,11,13,14,16,17,18,19,20,21]+list(range(25,55)))
        aggregate=load(b.ROOT/'fit/fitted.npz');ids=list(aggregate['frame_ids'].astype(str))
        records={r['frame']:r for r in read(b.ROOT/'data/training-inputs.json')['records']}
        l2=lambda:prior.L2Prior().cuda()
        loss=fitting.create_loss(loss_type='smplify',rho=cfg['rho'],body_pose_prior=l2(),shape_prior=l2(),expr_prior=l2(),
            angle_prior=prior.SMPLifyAnglePrior().cuda(),jaw_prior=l2(),left_hand_prior=l2(),right_hand_prior=l2(),
            use_joints_conf=True,use_face=True,use_hands=True,interpenetration=False,dtype=torch.float32).cuda()
        for frame in p['frames']:
            assert time.monotonic()-started < p['runtime_budget_seconds'] and process(pid)[1]=='T'
            start=time.monotonic();saved=load(SOURCE/'frames'/frame/'winner-camera-space.npz');baseline=load(b.ROOT/'frames'/frame/'winner-camera-space.npz')
            body.reset_params(**{name:saved[name] for name,_ in body.named_parameters()})
            fixed_pose=torch.tensor(saved['body_pose_axis_angle'],device='cuda');latent=torch.tensor(saved['pose_embedding'],device='cuda')
            K=torch.tensor(saved['K_integer_centers'],dtype=torch.float32,device='cuda');translation=torch.tensor(saved['camera_translation'],device='cuda')
            class Camera:
                def __call__(self,points):
                    q=(points+translation[:,None])@K.T;return q[...,:2]/q[...,2:]
            class FrozenPose:
                def decode(self,z,output_type='aa'):
                    assert output_type=='aa' and z is latent;return fixed_pose
            camera=Camera();decoder=FrozenPose()
            target=torch.tensor(aggregate['observed_native_keypoints'][ids.index(frame):ids.index(frame)+1],device='cuda')
            weights=torch.full((1,118),1. if frame=='frame_000033' else 2.,device='cuda')
            weights[:,[1,9,12]]=0;weights[:,25:67]=cfg['hand_joints_weights'][-1];weights[:,67:]=cfg['face_joints_weights'][-1]
            final=dict(data_weight=1000/records[frame]['wh'][1],body_pose_weight=cfg['body_pose_prior_weights'][-1],
                shape_weight=cfg['shape_weights'][-1],expr_prior_weight=cfg['expr_weights'][-1],bending_prior_weight=3.17*cfg['body_pose_prior_weights'][-1],
                hand_prior_weight=cfg['hand_pose_prior_weights'][-1],jaw_prior_weight=[float(x) for x in cfg['jaw_pose_prior_weights'][-1].split(',')])
            loss.reset_loss_weights(final)
            @torch.no_grad()
            def evaluate():
                result=body(body_pose=fixed_pose,return_verts=True,return_full_pose=True)
                actual={k:v.copy() for k,v in saved.items()}
                for name,param in body.named_parameters():actual[name]=param.detach().cpu().numpy().copy()
                actual['vertices_camera']=(result.vertices+translation[:,None]).cpu().numpy()
                actual['joints_camera']=(result.joints+translation[:,None]).cpu().numpy()
                actual['joints_projected_native']=camera(result.joints).cpu().numpy()
                E=saved['w2c'];actual['vertices_world']=((actual['vertices_camera'][0]-E[:3,3])@E[:3,:3])[None]
                objective=loss(result,camera=camera,gt_joints=target[...,:2],joints_conf=target[...,2],joint_weights=weights,
                    body_model_faces=body.faces_tensor.view(-1),pose_embedding=latent,use_vposer=True)
                return actual,float(objective)
            before,obj_before=evaluate()
            parity={k:float(np.max(np.abs(before[k]-saved[k]))) for k in ('vertices_camera','joints_projected_native')}
            assert parity['vertices_camera']<p['limits']['initial_vertex_parity_m'] and parity['joints_projected_native']<p['limits']['initial_projection_parity_px'],parity
            assert np.max(np.abs(before['vertices_world']-saved['vertices_world']))<2e-6,'World contract mismatch'
            optimizer,create_graph=factory.create_optimizer(parameters,**cfg);assert not optimizer.state
            with fitting.FittingMonitor(visualize=False,**{k:v for k,v in cfg.items() if k!='visualize'}) as monitor:
                closure=monitor.create_fitting_closure(optimizer,body,camera,target[...,:2],loss,joints_conf=target[...,2],joint_weights=weights,
                    return_verts=True,return_full_pose=True,use_vposer=True,vposer=decoder,pose_embedding=latent,create_graph=create_graph)
                reported=monitor.run_fitting(optimizer,closure,parameters,body,use_vposer=True,pose_embedding=latent,vposer=decoder)
                evaluations=monitor.steps
            after,obj_after=evaluate();assert np.isfinite(obj_after) and all(np.isfinite(v).all() for v in after.values() if v.dtype.kind in 'fci')
            for name in p['fixed']:
                if name in saved:assert np.array_equal(saved[name],after[name]),name
            for name,_ in body.named_parameters():
                if name not in ACTIVE:assert np.array_equal(saved[name],after[name]),name
            limb_delta=float(np.max(np.abs(after['vertices_camera'][:,limb_vertices]-before['vertices_camera'][:,limb_vertices])))
            target_np=target[0].cpu().numpy();old_metrics=metrics(baseline,target_np);prior_metrics=metrics(saved,target_np);current_metrics=metrics(after,target_np)
            protections={group:dict(delta_px=current_metrics['groups'][group]['median']-old_metrics['groups'][group]['median'],tolerance_px=tol,
                passed=current_metrics['groups'][group]['median']-old_metrics['groups'][group]['median']<=tol)
                for group,tol in [('body17',1.),('feet6',1.),('face51',.5)]}
            info=dict(frame=frame,source_sha256=sha(SOURCE/'frames'/frame/'winner-camera-space.npz'),parity=parity,
                original_objective_before=obj_before,original_objective_after=obj_after,objective_nonincreasing=obj_after<=obj_before+.01,
                original_monitor_reported_loss=reported,closure_evaluations=evaluations,elapsed_seconds=time.monotonic()-start,
                baseline=old_metrics,previous=prior_metrics,candidate=current_metrics,protections=protections,
                active_parameters=list(ACTIVE),all_fixed_parameter_values_exact=True,limb_vertex_max_delta_m=limb_delta,
                frozen_limb_geometry_passed=limb_delta<p['limits']['frozen_limb_vertex_delta_m'],
                left_elbow_remaining_ratio=current_metrics['left_elbow_px']/old_metrics['left_elbow_px'],
                parameter_delta={name:float(np.max(np.abs(after[name]-saved[name]))) for name in ACTIVE})
            folder=OUT/'frames'/frame;folder.mkdir(parents=True)
            np.savez_compressed(folder/'candidate-camera-space.npz',**after);save(folder/'result.json',info);rows_result.append(info)
            save(OUT/'progress.json',dict(state='face_only_refinement',pid=os.getpid(),frames_completed=len(rows_result),frame=frame,elapsed_seconds=time.monotonic()-started))
            print(frame,'face',round(prior_metrics['groups']['face51']['median'],4),'->',round(current_metrics['groups']['face51']['median'],4),'limb_delta',limb_delta,flush=True)
        verify();failures=[dict(frame=r['frame'],region=g,**v) for r in rows_result for g,v in r['protections'].items() if not v['passed']]
        elbow=all(r['left_elbow_remaining_ratio']<=.5 for r in rows_result if int(r['frame'][-6:]) in p['protections']['elbow_frames'])
        checks=dict(all_region_protections=not failures,all_six_elbows_halved=elbow,
            objective_nonincreasing=all(r['objective_nonincreasing'] for r in rows_result),frozen_limbs=all(r['frozen_limb_geometry_passed'] for r in rows_result))
        result=dict(updated=now(),state='complete_pending_visual_review',frames=59,rows=rows_result,checks=checks,failures=failures,
            numerical_gate_passed=all(checks.values()),plan_sha256=sha(PLAN),elapsed_seconds=time.monotonic()-started,
            no_optimizer_reused=True,no_camera_or_whole_body_update=True,no_per_frame_selection=True,quality_accepted=False,vhap_admitted=False,new_gaussian_started=False)
        save(OUT/'result.json',result);save(OUT/'progress.json',dict(state='complete',pid=None,frames_completed=59,elapsed_seconds=result['elapsed_seconds']))
        print('FACE_POLISH_COMPLETE',checks,'failures',[(f['frame'],f['region']) for f in failures],flush=True)
    except BaseException as e:
        save(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),frames_completed=len(rows_result)))
        save(OUT/'progress.json',dict(state='failed_preserved',pid=None,frames_completed=len(rows_result)));raise

def run(pid):
    p=verify();assert not OUT.exists() and not HANDOFF.exists()
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid);assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    receipt=dict(started=now(),supervisor_pid=os.getpid(),baseline_pid=pid,identity=identity,command=cmd,state='suspending');save(HANDOFF,receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        env=dict(os.environ,SHUOLEE_CONTROL_BASELINE_PID=str(pid),SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        with Path(__file__).with_suffix('.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=p['runtime_budget_seconds'])
        receipt['refinement_finished']=True
    finally:
        try:
            if process(pid)[2]==identity:
                os.kill(pid,signal.SIGCONT);receipt.update(state='resumed',baseline_state=process(pid)[1])
            else:receipt['state']='baseline_identity_changed_not_signaled'
        except (FileNotFoundError,ProcessLookupError):receipt['state']='baseline_exited_before_resume'
        receipt['finished']=now();save(HANDOFF,receipt);print('BASELINE',receipt['state'],flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','worker','run']);a.add_argument('--pid',type=int);args=a.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':worker()
    else:run(args.pid)
