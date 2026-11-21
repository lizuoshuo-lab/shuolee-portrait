"""One CPU forward per saved face endpoint; no optimizer steps or new fit."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import replay_demo_limb_losses123 as b
import official_smplifyx_entry123 as entry
from review_smplifyx_orientations123 import load

BODY=b.ROOT.parent
SOURCE=BODY/'face-fixed-body-refinement'
PLAN=BODY/'face-fixed-body-stationarity-plan.json'
EXECUTION=BODY/'face-fixed-body-stationarity-execution.json'
OUT=SOURCE/'stationarity-audit'
ACTIVE=('expression','jaw_pose','leye_pose','reye_pose')


def now():return datetime.now(timezone.utc).isoformat()


def native(s):
    s=str(s).replace('\\','/')
    return Path('/mnt/'+s[0].lower()+s[2:]) if len(s)>2 and s[1:3]==':/' else Path(s)


def verify():
    p=b.read(PLAN)
    for path,digest in {**p['inputs'],**p['sources']}.items():assert b.sha(native(path))==digest,path
    if EXECUTION.exists():
        for path,digest in b.read(EXECUTION)['bindings'].items():assert b.sha(path)==digest,path
    return p


def prepare():
    p=verify();assert not EXECUTION.exists() and not OUT.exists()
    import smplx.body_models,smplx.lbs
    files=[PLAN,SOURCE/'result.json',BODY/'face-fixed-body-refinement-plan.json',b.ROOT/'fit/fitted.npz',b.ROOT/'data/training-inputs.json',b.ASSETS/'smplx/SMPLX_NEUTRAL.npz',Path(__file__),Path(entry.__file__),Path(b.__file__),Path(smplx.body_models.__file__),Path(smplx.lbs.__file__)]
    files += [b.OFFICIAL/x for x in entry.PINNED]
    files += [SOURCE/'frames'/frame/'candidate-camera-space.npz' for frame in p['frames']]
    b.save(EXECUTION,dict(updated=now(),state='prepared',bindings={str(f.resolve()):b.sha(f) for f in files},
        per_frame_body_forwards=1,forward_budget=59,optimizer_updates=0,cpu_seconds_max=p['cpu_seconds_max'],
        parity_limits={'vertex_max_m':2e-6,'projection_max_px':.002,'loss_absolute_max':.05},
        components=['body_data','hand_data','face_data','expression_prior','jaw_prior','constant_other_priors'],
        original_gradient_test_only=True,no_new_gt_or_fit=True))
    print('PREPARED_ENDPOINT_GRADIENT_AUDIT')


def run():
    p=verify();execution=b.read(EXECUTION);assert not OUT.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    OUT.mkdir();started=time.monotonic()
    b.save(OUT/'progress.json',dict(state='cpu_endpoint_gradients',pid=os.getpid(),frames_completed=0))
    def deadline(signum,frame):raise TimeoutError('CPU stationarity audit budget exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(p['cpu_seconds_max'])
    try:
        import torch,smplx
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert not torch.cuda.is_initialized()
        utils,prior,fitting,factory=entry.load_official(b.OFFICIAL)
        cfg=b.read(BODY/'face-fixed-body-refinement-plan.json')['configuration']
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        model=smplx.SMPLX(str(b.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
            joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cpu().eval()
        model.parents=model.parents.cpu()
        for name,param in model.named_parameters():param.requires_grad_(name in ACTIVE)
        named=[(name,param) for name,param in model.named_parameters() if param.requires_grad]
        assert set(n for n,_ in named)==set(ACTIVE)
        params=[v for _,v in named]
        opt,_=factory.create_optimizer(params,**cfg)
        settings={k:v for k,v in opt.param_groups[0].items() if k!='params'}
        assert not opt.state
        l2=lambda:prior.L2Prior()
        loss=fitting.create_loss(loss_type='smplify',rho=cfg['rho'],body_pose_prior=l2(),shape_prior=l2(),expr_prior=l2(),
            angle_prior=prior.SMPLifyAnglePrior(),jaw_prior=l2(),left_hand_prior=l2(),right_hand_prior=l2(),
            use_joints_conf=True,use_face=True,use_hands=True,interpenetration=False,dtype=torch.float32)
        aggregate=load(b.ROOT/'fit/fitted.npz');ids=list(aggregate['frame_ids'].astype(str))
        records={r['frame']:r for r in b.read(b.ROOT/'data/training-inputs.json')['records']}
        previous={r['frame']:r for r in b.read(SOURCE/'result.json')['rows']}
        rows=[];gradients={};calls=0
        for frame in p['frames']:
            assert time.monotonic()-started<p['cpu_seconds_max']
            saved=load(SOURCE/'frames'/frame/'candidate-camera-space.npz')
            model.reset_params(**{name:saved[name] for name,_ in model.named_parameters()})
            K=torch.tensor(saved['K_integer_centers'],dtype=torch.float32);translation=torch.tensor(saved['camera_translation'])
            class Camera:
                def __call__(self,points):
                    q=(points+translation[:,None])@K.T;return q[...,:2]/q[...,2:]
            camera=Camera();latent=torch.from_numpy(saved['pose_embedding']);pose=torch.from_numpy(saved['body_pose_axis_angle'])
            target=torch.from_numpy(aggregate['observed_native_keypoints'][ids.index(frame):ids.index(frame)+1])
            weights=torch.full((1,118),1. if frame=='frame_000033' else 2.)
            weights[:,[1,9,12]]=0;weights[:,25:67]=cfg['hand_joints_weights'][-1];weights[:,67:]=cfg['face_joints_weights'][-1]
            loss.reset_loss_weights(dict(data_weight=1000/records[frame]['wh'][1],body_pose_weight=cfg['body_pose_prior_weights'][-1],
                shape_weight=cfg['shape_weights'][-1],expr_prior_weight=cfg['expr_weights'][-1],bending_prior_weight=3.17*cfg['body_pose_prior_weights'][-1],
                hand_prior_weight=cfg['hand_pose_prior_weights'][-1],jaw_prior_weight=[float(x) for x in cfg['jaw_pose_prior_weights'][-1].split(',')]))
            output=model(body_pose=pose,return_verts=True,return_full_pose=True);calls+=1
            vertices=(output.vertices+translation[:,None]).detach().numpy();uv=camera(output.joints)
            vp=float(np.max(np.abs(vertices-saved['vertices_camera'])))
            up=float(np.max(np.abs(uv.detach().numpy()-saved['joints_projected_native'])))
            assert vp<execution['parity_limits']['vertex_max_m'] and up<execution['parity_limits']['projection_max_px'],(frame,vp,up)
            total=loss(output,camera=camera,gt_joints=target[...,:2],joints_conf=target[...,2],joint_weights=weights,
                body_model_faces=model.faces_tensor.view(-1),pose_embedding=latent,use_vposer=True)
            lp=abs(float(total.detach())-previous[frame]['original_objective_after']);assert lp<execution['parity_limits']['loss_absolute_max'],(frame,lp)
            elements=(weights*target[...,2]).unsqueeze(-1).pow(2)*loss.robustifier(target[...,:2]-uv)*loss.data_weight.pow(2)
            terms={'body_data':elements[:,:25].sum(),'hand_data':elements[:,25:67].sum(),'face_data':elements[:,67:].sum(),
                'expression_prior':loss.expr_prior(output.expression).sum()*loss.expr_prior_weight.pow(2),
                'jaw_prior':loss.jaw_prior(output.jaw_pose*loss.jaw_prior_weight).sum()}
            other=total-sum(terms.values())
            def gradient(value):
                if not value.requires_grad:return [torch.zeros_like(v) for v in params]
                return [torch.zeros_like(v) if g is None else g for g,v in zip(torch.autograd.grad(value,params,retain_graph=True,allow_unused=True),params)]
            all_grad=gradient(total);term_grad={name:gradient(value) for name,value in terms.items()}
            total_flat=torch.cat([v.reshape(-1) for v in all_grad]);sum_flat=sum(torch.cat([v.reshape(-1) for v in values]) for values in term_grad.values())
            mismatch=float((total_flat-sum_flat).abs().max())
            assert mismatch<.05+float(total_flat.abs().max())*1e-4,(frame,mismatch)
            blocks={name:dict(max_abs=float(g.abs().max()),l2=float(g.norm()),original_monitor_abs_of_max=float(g.max().abs()),
                original_monitor_gtol_pass=bool(g.max().abs()<cfg['gtol'])) for (name,_),g in zip(named,all_grad)}
            components={name:dict(value=float(value.detach()),gradient_l2=float(torch.cat([v.reshape(-1) for v in term_grad[name]]).norm())) for name,value in terms.items()}
            components['constant_other_priors']=dict(value=float(other.detach()),gradient_residual_max=mismatch)
            gd=torch.cat([v.reshape(-1) for v in term_grad['body_data']])+torch.cat([v.reshape(-1) for v in term_grad['hand_data']])+torch.cat([v.reshape(-1) for v in term_grad['face_data']])
            gp=torch.cat([v.reshape(-1) for v in term_grad['expression_prior']])+torch.cat([v.reshape(-1) for v in term_grad['jaw_prior']])
            cosine=float(torch.dot(gd,gp)/(gd.norm()*gp.norm()).clamp_min(1e-20))
            rows.append(dict(frame=frame,parity=dict(vertex_max_m=vp,projection_max_px=up,loss_abs=lp),objective=float(total.detach()),
                gradient_max_abs=float(total_flat.abs().max()),gradient_l2=float(total_flat.norm()),parameter_blocks=blocks,
                optimizer_gradient_stop_pass=bool(total_flat.abs().max()<=settings['tolerance_grad']),
                monitor_gradient_stop_pass=all(v['original_monitor_gtol_pass'] for v in blocks.values()),
                components=components,data_prior_gradient_cosine=cosine,gradient_sum_parity_max=mismatch,
                previous_closure_evaluations=previous[frame]['closure_evaluations'],previous_parameter_delta=previous[frame]['parameter_delta'],
                previous_face_median=previous[frame]['candidate']['groups']['face51']['median'],historical_stop_reason='not_recorded_not_inferred'))
            gradients[frame+'__total']=total_flat.detach().numpy()
            for name,values in term_grad.items():gradients[frame+'__'+name]=torch.cat([v.reshape(-1) for v in values]).detach().numpy()
            if len(rows)%10==0 or len(rows)==59:b.save(OUT/'progress.json',dict(state='cpu_endpoint_gradients',pid=os.getpid(),frames_completed=len(rows),elapsed_seconds=time.monotonic()-started))
        verify();assert calls==execution['forward_budget'] and not opt.state and not torch.cuda.is_initialized()
        np.savez_compressed(OUT/'gradients-only.npz',**gradients)
        report=dict(updated=now(),state='complete',frames=59,body_forwards=calls,optimizer_updates=0,elapsed_seconds=time.monotonic()-started,
            optimizer_settings=settings,monitor_settings={name:cfg[name] for name in ('gtol','ftol','maxiters')},parameter_order=[name for name,_ in named],rows=rows,
            summary=dict(optimizer_gradient_pass_count=sum(r['optimizer_gradient_stop_pass'] for r in rows),monitor_gradient_pass_count=sum(r['monitor_gradient_stop_pass'] for r in rows),
                max_vertex_parity_m=max(r['parity']['vertex_max_m'] for r in rows),max_projection_parity_px=max(r['parity']['projection_max_px'] for r in rows),
                gradient_max_abs_median=float(np.median([r['gradient_max_abs'] for r in rows])),gradient_max_abs_max=max(r['gradient_max_abs'] for r in rows)),
            plan_sha256=b.sha(PLAN),execution_sha256=b.sha(EXECUTION),
            limits=['Endpoint gradients alone do not recover the actual historical stop branch.',
                'A nonstationary endpoint does not prove the same model can meet geometric quality gates.',
                'CPU float32 parity is bounded, not bitwise GPU execution parity.',
                'No candidates, optimizer changes or new observations were generated.'],quality_accepted=False)
        b.save(OUT/'result.json',report);b.save(OUT/'progress.json',dict(state='complete',pid=None,frames_completed=59,elapsed_seconds=report['elapsed_seconds']))
        print('STATIONARITY_AUDIT_COMPLETE',report['summary'],flush=True)
        for row in rows:
            if row['frame'] in ('frame_000051','frame_000059'):print(row['frame'],row['gradient_max_abs'],row['components'],flush=True)
    except BaseException as error:
        b.save(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        b.save(OUT/'progress.json',dict(state='failed_preserved',pid=None));raise
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()
