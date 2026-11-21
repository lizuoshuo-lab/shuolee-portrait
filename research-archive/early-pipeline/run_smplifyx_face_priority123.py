"""One uniform face/body ratio control on the existing original fit pipeline."""
import argparse
import copy
import inspect
from pathlib import Path
import time
import numpy as np
import run_smplifyx_body_weight_sequence123 as previous

warm=previous.warm
BODY=previous.BODY
BASE=warm.forward.BASE
OUT=BODY/'face-priority-guarded-sequence'
PLAN=BODY/'face-priority-guarded-sequence-plan.json'
PREFLIGHT=BODY/'face-priority-preflight.json'
HANDOFF=BODY/'face-priority-guarded-sequence-gpu-handoff.json'
read,save,sha,now=previous.read,previous.save,previous.sha,previous.now


def bind():
    warm.OUT,warm.PLAN,warm.HANDOFF=OUT,PLAN,HANDOFF
    warm.bind()


def face_config(c):
    c=copy.deepcopy(c)
    assert len(c['face_joints_weights'])==2 and c['face_joints_weights']==[0.,2.],c['face_joints_weights']
    c['face_joints_weights']=[2*x for x in c['face_joints_weights']]
    return c


def normalize_selection_weights(weights,anchor):
    w=weights.clone()
    assert tuple(w.shape)==(1,118)
    active=[i for i in range(25) if i not in (1,9,12)]
    assert bool((w[:,[1,9,12]]==0).all())
    assert bool((w[:,active]==(1 if anchor else 2)).all())
    assert bool((w[:,25:67]==2).all())
    assert bool((w[:,67:]==(2 if anchor else 4)).all())
    if not anchor:
        w[:,:25]/=2
        w[:,67:]/=2
    expected=w.new_ones((1,118));expected[:,[1,9,12]]=0;expected[:,25:]=2
    assert bool((w==expected).all())
    return w


def prepare():
    assert not OUT.exists() and not PLAN.exists() and not PREFLIGHT.exists()
    old_plan=BODY/'body-weight-guarded-sequence-plan.json'
    p=copy.deepcopy(read(old_plan))
    for f,h in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(f)==h,f
    assert read(BODY/'face-fixed-body-fp64/review/closeout.json')['state']=='precision_control_rejected_face_protection'
    audit=read(BODY/'face-body-priority-audit.json')
    assert audit['original_face_to_body_squared_ratio']==4 and audit['body_control_face_to_body_squared_ratio']==1
    assert len(p['temporal_roster'])==59 and p['anchor']=='frame_000033'
    p.update(updated=now(),state='prepared_requires_preflight',output=str(OUT),
        hypothesis='Restoring the original relative face/body observation priority may retain the body improvement without the two facial regressions.',
        single_change_from_failed_full59='Non-anchor face_joints_weights in original stages3/4 [0,2]->[0,4]; body amplitude stays2. Squared relative face/body ratio1->4, matching original relative ratio.',
        face_observation_amplitude={'anchor_final':2.,'all_other_final':4.,'previous_all_other_final':2.},
        anchor_policy='Reference33 original five-stage amplitude1/face2 fit unchanged; other58 follow the same new face schedule.',
        unchanged_from_previous=['all59 observations/confidence/K/camera initialization','fixed shared beta and reference33',
            'chronological predecessor/boundary seed and original-score fallback','original stage3/4 priors, hand schedule, float32, LBFGS and stopping/maxiters30','zero hand/face initialization'],
        causal_scope='Only the configured face observation schedule changes relative to the body-weight control. Within-run seeds can change as a consequence; do not claim identical optimized initial states for later frames.',
        score_contract='For both candidates recompute the original final objective after normalizing body2->1 and face4->2. Hand2 and all priors remain original. Never compare differently weighted totals.',
        risks=['Restoring facial priority can worsen limbs or feet.','Wrong or occluded face landmarks may pull pose incorrectly.','Original-score selection and detector medians do not prove3D anatomical truth.'],
        runtime_budget_seconds=3000,expected_seconds=1800,
        budget=dict(new_per_frame_fits=59,alternative_prior_sweeps=0,max_seconds=3000,expected_seconds=1800,automatic_followup=False),
        no_automatic_followup=True,automatic_gaussian_training=False,quality_accepted=False,goal_complete=False)
    p['selection_policy']['candidate0']='New body2/face4 chronological original-tail fit (anchor33 original), scored with original body1/face2.'
    p['source_hashes'][str(Path(__file__).resolve())]=sha(__file__)
    for f in (old_plan,BODY/'body-weight-guarded-sequence/review/closeout.json',BODY/'face-fixed-body-fp64/review/closeout.json',BODY/'face-body-priority-audit.json'):
        p['input_hashes'][str(f)]=sha(f)
    # No original solver or previous completed adapter is edited.
    assert p['source_adapter_delta']==warm.source()[1]
    save(PLAN,p);bind();warm.forward.temporal.verify_plan()
    print('PREPARED_ONE_FACE_PRIORITY_CONTROL',flush=True)


def preflight():
    import os
    device='cpu' if os.environ.get('CUDA_VISIBLE_DEVICES')=='' else 'cuda'
    assert not PREFLIGHT.exists() and not OUT.exists()
    if device=='cuda':warm.forward.guarded.lease.verify_handoff()
    bind();p=warm.forward.temporal.verify_plan();started=time.monotonic()
    import torch,smplx
    import replay_demo_limb_losses123 as b
    from review_smplifyx_orientations123 import load
    entry=warm.forward.temporal.entry
    torch.set_num_threads(2)
    utils,prior,fitting,_=entry.load_official(b.OFFICIAL)
    cfg=entry.load_official_config(b.OFFICIAL)
    tail={k:copy.deepcopy(v) for k,v in cfg.items()}
    if tail.get('data_weights') is None:tail['data_weights']=[1]*5
    for k in warm.FIELDS:tail[k]=tail[k][3:]
    modified=face_config(tail)
    assert [k for k in tail if tail[k]!=modified[k]]==['face_joints_weights']
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    model=smplx.SMPLX(str(b.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
        use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
        joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).to(device).eval()
    model.parents=model.parents.cpu()
    for parameter in model.parameters():parameter.requires_grad_(False)
    l2=lambda:prior.L2Prior()
    loss=fitting.create_loss(loss_type='smplify',rho=cfg['rho'],body_pose_prior=l2(),shape_prior=l2(),expr_prior=l2(),
        angle_prior=prior.SMPLifyAnglePrior(),jaw_prior=l2(),left_hand_prior=l2(),right_hand_prior=l2(),
        use_joints_conf=True,use_face=True,use_hands=True,interpenetration=False,dtype=torch.float32).to(device)
    data=load(BASE/'fit/fitted.npz');ids=list(data['frame_ids'].astype(str))
    records={r['frame']:r for r in read(BASE/'data/training-inputs.json')['records']}
    expected={r['frame']:r['original_saved_objective'] for r in read(warm.forward.guarded.AUDIT)['rows'] if r['arm']=='baseline'}
    checks=[]
    with torch.no_grad():
        for frame in p['processing_order']:
            assert time.monotonic()-started<120
            saved=load(BASE/'frames'/frame/'winner-camera-space.npz');anchor=frame==p['anchor']
            model.reset_params(**{k:saved[k] for k,_ in model.named_parameters()})
            output=model(body_pose=torch.from_numpy(saved['body_pose_axis_angle']).to(device),return_verts=True,return_full_pose=True)
            K=torch.from_numpy(saved['K_compute_float32']).to(device);t=torch.from_numpy(saved['camera_translation']).to(device)[:,None]
            class Camera:
                def __call__(self,points):
                    q=(points+t)@K.T;return q[...,:2]/q[...,2:]
            camera=Camera();target=torch.from_numpy(data['observed_native_keypoints'][ids.index(frame):ids.index(frame)+1]).to(device)
            vw=float(np.max(np.abs((output.vertices+t).cpu().numpy()-saved['vertices_camera'])))
            uv=float(np.max(np.abs(camera(output.joints).cpu().numpy()-saved['joints_projected_native'])))
            assert vw<2e-6 and uv<.002,(frame,vw,uv)
            weights=torch.full((1,118),1. if anchor else 2.,device=device);weights[:,[1,9,12]]=0;weights[:,25:67]=2;weights[:,67:]=2 if anchor else 4
            normalized=normalize_selection_weights(weights,anchor)
            loss.reset_loss_weights(dict(data_weight=1000/records[frame]['wh'][1],body_pose_weight=cfg['body_pose_prior_weights'][-1],
                shape_weight=cfg['shape_weights'][-1],expr_prior_weight=cfg['expr_weights'][-1],bending_prior_weight=3.17*cfg['body_pose_prior_weights'][-1],
                hand_prior_weight=cfg['hand_pose_prior_weights'][-1],jaw_prior_weight=[float(x) for x in cfg['jaw_pose_prior_weights'][-1].split(',')]))
            def value(w):return float(loss(output,camera=camera,gt_joints=target[...,:2],joints_conf=target[...,2],joint_weights=w,
                body_model_faces=model.faces_tensor.view(-1),pose_embedding=torch.from_numpy(saved['pose_embedding']).to(device),use_vposer=True))
            weighted,original=value(weights),value(normalized)
            err=original-expected[frame];assert abs(err)<.01,(frame,err)
            elements=(normalized*target[...,2]).unsqueeze(-1).pow(2)*loss.robustifier(target[...,:2]-camera(output.joints))*loss.data_weight.pow(2)
            expected_extra=0. if anchor else float(3*(elements[:,:25].sum()+elements[:,67:].sum()))
            extra_error=weighted-original-expected_extra;assert abs(extra_error)<.02,(frame,extra_error)
            checks.append(dict(frame=frame,original_loss=original,independent_oracle=expected[frame],oracle_error=err,
                weighted_loss=weighted,expected_extra_body_and_face=expected_extra,extra_error=extra_error,vertex_parity_m=vw,projection_parity_px=uv))
    assert (device=='cuda' or not torch.cuda.is_initialized()) and len(checks)==59
    warm.forward.temporal.verify_plan()
    save(PREFLIGHT,dict(updated=now(),state='passed',device=device,frames=59,optimizer_steps=0,body_forwards=59,checks=checks,
        source_sha256=sha(__file__),plan_sha256=sha(PLAN),only_config_field_changed='face_joints_weights',
        original_tail=tail['face_joints_weights'],candidate_tail=modified['face_joints_weights'],
        max_original_oracle_error=max(abs(x['oracle_error']) for x in checks),elapsed_seconds=time.monotonic()-started,
        exact_registered_seed_schedule_retained=True,quality_accepted=False))
    print('FACE_PRIORITY_PREFLIGHT_PASSED',len(checks),time.monotonic()-started,flush=True)


def worker():
    bind();entry=warm.forward.temporal.entry
    p=warm.forward.temporal.verify_plan();pre=read(PREFLIGHT)
    assert pre['state']=='passed' and pre['source_sha256']==sha(__file__) and pre['plan_sha256']==sha(PLAN)
    original=entry.fit_one;weighted=previous.single.live_config_fit();original_worker_source=warm.forward.worker_source
    def fit(**kwargs):
        frame=Path(kwargs['result_path']).parent.name;anchor=frame==p['anchor']
        if anchor:arrays,report=original(**kwargs)
        else:
            current_config=entry.load_official_config
            entry.load_official_config=lambda root:face_config(current_config(root))
            try:arrays,report=weighted(**kwargs)
            finally:entry.load_official_config=current_config
        context=entry._guard_context;context['joint_weights']=normalize_selection_weights(context['joint_weights'],anchor)
        report.update(body_observation_amplitude=1. if anchor else 2.,face_observation_amplitude=2. if anchor else 4.,
            original_body_observation_amplitude=1.,original_face_observation_amplitude=2.,
            fit_weighted_objective=report['winning_loss'],selection_uses_original_body_and_face_amplitudes=True,
            original_body_numeric_weights_unchanged=anchor,hand_stage_values_unchanged=True,face_stage_values_unchanged=anchor,
            five_stages=anchor,original_stage_indices=list(range(5)) if anchor else [3,4],stage_values_unchanged=anchor,
            local_curriculum_adapter=not anchor,face_schedule_original=[0.,2.] if not anchor else [0.,0.,0.,0.,2.],
            face_schedule_fit=[0.,4.] if not anchor else [0.,0.,0.,0.,2.])
        return arrays,report
    entry.fit_one=fit;warm.forward.worker_source=warm.source
    try:warm.forward.worker()
    finally:entry.fit_one=original;warm.forward.worker_source=original_worker_source


def run(pid):
    bind();pre=read(PREFLIGHT)
    assert pre['state']=='passed' and pre['source_sha256']==sha(__file__) and pre['plan_sha256']==sha(PLAN)
    fn=warm.forward.guarded.run;scope=dict(fn.__globals__,__file__=__file__)
    exec(compile(inspect.getsource(fn),'<existing-exclusive-original-fit-face-ratio-control>','exec'),scope)
    scope['run'](pid)


def preflight_gpu(pid):
    bind();fn=warm.forward.guarded.run
    source=inspect.getsource(fn)
    old="str(Path(__file__).resolve()),'worker'";assert source.count(old)==1
    source=source.replace(old,"str(Path(__file__).resolve()),'preflight'")
    old="Path(__file__).with_suffix('.log').open('x',encoding='utf-8')";assert source.count(old)==1
    source=source.replace(old,"Path(str(__file__)+'.preflight.log').open('x',encoding='utf-8')")
    source=source.replace("timeout=p['runtime_budget_seconds']","timeout=120")
    scope=dict(fn.__globals__,__file__=__file__,HANDOFF=BODY/'face-priority-preflight-gpu-handoff.json')
    exec(compile(source,'<existing-exclusive-gpu-score-preflight>','exec'),scope)
    scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','preflight','preflight-gpu','worker','run']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='preflight':preflight()
    elif args.mode=='preflight-gpu':preflight_gpu(args.pid)
    elif args.mode=='worker':worker()
    else:run(args.pid)
