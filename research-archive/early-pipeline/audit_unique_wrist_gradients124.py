"""One bounded CPU-only decomposition of saved final-stage objectives."""
import os
import inspect
import time
from pathlib import Path
import numpy as np
import control_unique_wrist124 as c
from smplify_temporal124 import rotation_energy

a,entry=c.a,c.entry
OUT=c.OUT/'gradient-audit'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not OUT.exists()
    assert c.read(c.OUT/'conclusion.json')['actual_visual_review']
    row,original_targets=c.base.target()
    paths=[Path(__file__),Path(c.__file__),Path(a.__file__),Path(entry.__file__),
        Path(inspect.getsourcefile(rotation_energy)),c.OUT/'conclusion.json',c.OUT/'plan.json',
        c.OUT/'trajectory.npz',c.OUT/'result.json',c.OUT/'winner-camera-space.npz',
        c.base.OUT/'trajectory.npz',c.base.OUT/'result.json',a.frame(a.RUN,17),
        a.RUN/'plan.json',a.SRC/'data/inputs.json',a.SRC/'data/observations.npz',
        a.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    paths += [a.OFFICIAL/name for name in entry.PINNED]
    frozen={str(p):c.sha(p) for p in paths};OUT.mkdir()
    c.save(OUT/'plan.json',dict(updated=c.now(),hypothesis='At the same saved geometry, isolate duplicated wrist-root gradient from face,40 fingers,body/feet,original priors and the existing temporal term. Then inspect the original and changed endpoints for residual shoulder/latent coupling. Local gradients are diagnostics, not evidence of a correct physical pose.',
        budget=dict(cpu_seconds=120,states=4,evaluations=5,optimizer_steps=0,new_fitter_calls=0),
        evaluations=['old_stage3_original_input','old_stage3_unique_input','unique_stage3_unique_input','old_final_original_input','unique_final_unique_input'],
        gates=['Saved projection parity within0.02px and vertices within1e-5 model meters.','Original and unique final objective relative parity below1e-4.','Gradient sum matches the complete objective; finite values.'],
        forbidden=['new fit','pose updates','weight or threshold sweep','old geometry as a new initialization','Gaussian admission'],
        interpretation='For projected shoulder squared-error gradient gS and component gradient gL, -dot(gS,gL) is the local directional derivative for descent of that component. Normalized cosine is scale-free but is not the actual LBFGS update or finite-step causality.',
        frozen=frozen))
    import torch,smplx
    torch.set_num_threads(1);started=time.monotonic()
    utils,prior,fitting,_=entry.load_official(a.OFFICIAL)
    config=entry.load_official_config(a.OFFICIAL)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(a.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
        use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
        create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32)
    loader=entry.module(a.PROJECT/'src/avatar_core/shuolee_vposer.py','_unique_gradient_vposer124')
    loader.MODEL,loader.CODE=a.ASSETS/'vposer/V02_05',a.PROJECT/'third_party/human_body_prior'
    vposer,_,_=loader.load_vposer('cpu');vposer.eval()
    output=a.load_npz(c.OUT/'winner-camera-space.npz')
    modified_targets=output['fitting_native_observations']
    previous=a.load_npz(a.frame(a.RUN,17));reference=torch.zeros(1,165)
    reference[:,:3]=torch.from_numpy(previous['global_orient']);reference[:,3:66]=torch.from_numpy(previous['body_pose_axis_angle'])
    dt=abs(row['time_seconds']-c.read(a.SRC/'data/inputs.json')['records'][16]['time_seconds'])
    nominal=c.read(a.RUN/'plan.json')['nominal_dt_seconds']
    K=torch.from_numpy(output['K_compute_float32']);translation=torch.from_numpy(output['camera_translation'])
    def camera(points):
        q=(points+translation[:,None])@K.T
        return q[...,:2]/q[...,2:]
    archives={label:a.load_npz(path/'trajectory.npz') for label,path in [('old',c.base.OUT),('unique',c.OUT)]}
    results={label:c.read(path/'result.json') for label,path in [('old',c.base.OUT),('unique',c.OUT)]}
    tasks=[('old',0,False),('old',0,True),('unique',0,True),('old',1,False),('unique',1,True)]
    records=[];parity={}
    for source,stage_slot,modified in tasks:
        assert time.monotonic()-started<120
        stages=[s for s in results[source]['stages'] if s['kind']=='body'];index=stages[stage_slot]['after_state']
        archive=archives[source];prefix=f'state_{index:03d}'
        body.reset_params(**{n:archive[prefix+'_parameter_'+n] for n,_ in body.named_parameters()})
        latent=torch.from_numpy(archive[prefix+'_pose_embedding'].copy()).requires_grad_()
        geometry=body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
        projected=camera(geometry.joints)
        differences=dict(vertices=float((geometry.vertices+translation[:,None]-torch.from_numpy(archive[prefix+'_vertices_camera'])).abs().max()),
            projection=float((projected-torch.from_numpy(archive[prefix+'_joints_projected_native'])).abs().max()))
        assert differences['vertices']<1e-5 and differences['projection']<.02,differences
        target=torch.from_numpy(modified_targets if modified else original_targets)
        jw=torch.ones(1,118);jw[:,[1,9,12]]=0;jw[:,25:67]=config['hand_joints_weights'][4];jw[:,67:]=config['face_joints_weights'][4]
        weights=dict(data_weight=1000/row['wh'][1],body_pose_weight=config['body_pose_prior_weights'][4],shape_weight=config['shape_weights'][4],
            bending_prior_weight=3.17*config['body_pose_prior_weights'][4],hand_prior_weight=config['hand_pose_prior_weights'][4],
            expr_prior_weight=config['expr_weights'][4],jaw_prior_weight=[float(x) for x in config['jaw_pose_prior_weights'][4].split(',')])
        l2=prior.L2Prior
        loss=fitting.SMPLifyLoss(interpenetration=False,rho=config['rho'],use_hands=True,use_face=True,
            body_pose_prior=l2(),shape_prior=l2(),angle_prior=prior.SMPLifyAnglePrior(),left_hand_prior=l2(),right_hand_prior=l2(),expr_prior=l2(),jaw_prior=l2(),**weights)
        original=loss(geometry,camera,target[...,:2],target[...,2],None,jw,use_vposer=True,pose_embedding=latent)
        temporal=rotation_energy(geometry.full_pose,reference,dt,nominal)
        terms=((jw*target[...,2]).square()[...,None]*loss.data_weight.square()*loss.robustifier(projected-target[...,:2])).sum(-1)
        components=dict(body_feet=terms[:,:25].sum(),duplicate_roots=terms[:,[25,46]].sum(),
            fingers=terms[:,np.r_[26:46,47:67]].sum(),face=terms[:,67:].sum(),priors=original-terms.sum(),temporal=temporal)
        total=original+temporal
        gradients={name:torch.autograd.grad(value,(latent,body.global_orient),retain_graph=True,allow_unused=True) for name,value in components.items()}
        total_gradient=torch.autograd.grad(total,(latent,body.global_orient),retain_graph=True)
        gradient_sum=[]
        for param in (0,1):
            subtotal=sum((gs[param] for gs in gradients.values() if gs[param] is not None),torch.zeros_like(total_gradient[param]))
            error=float((subtotal-total_gradient[param]).abs().max())
            tolerance=1e-5*max(1.,float(total_gradient[param].abs().max()))
            assert error<tolerance,(error,tolerance)
            gradient_sum.append(dict(max_error=error,tolerance=tolerance))
        # No residual-based data selection: these same protected joints caused the preregistered failure.
        joints={}
        for joint,name in [(2,'right_shoulder'),(3,'right_elbow'),(5,'left_shoulder'),(13,'left_knee')]:
            error=(projected[0,joint]-torch.from_numpy(original_targets[0,joint,:2])).square().sum()
            gs=torch.autograd.grad(error,(latent,body.global_orient),retain_graph=True)
            sensitivity={}
            for term,grad in gradients.items():
                sensitivity[term]={}
                for pindex,pname in enumerate(('body_latent','root')):
                    g=grad[pindex] if grad[pindex] is not None else torch.zeros_like(gs[pindex])
                    dot=float((gs[pindex]*g).sum());denom=float(gs[pindex].norm()*g.norm())
                    sensitivity[term][pname]=dict(descent_error_derivative=-dot,descent_error_cosine=-dot/denom if denom>1e-12 else None)
            joints[name]=dict(error_px=float(error.sqrt()),sensitivity=sensitivity)
        rec=dict(source=source,saved_stage=stage_slot+3,input='unique' if modified else 'original',
            geometry_parity=differences,total=float(total),losses={k:float(v.detach()) for k,v in components.items()},
            gradient_norms={k:{pname:float(gs[j].norm()) if gs[j] is not None else 0. for j,pname in enumerate(('body_latent','root'))} for k,gs in gradients.items()},
            gradient_sum=gradient_sum,joints=joints)
        if stage_slot==1:
            expected=results[source]['original_report']['winning_loss']
            rel=abs(rec['total']-expected)/max(1.,abs(expected));assert rel<1e-4,rel
            parity[source]=dict(saved_loss=expected,recomputed=rec['total'],relative_difference=rel)
        records.append(rec)
    for path,digest in frozen.items():assert c.sha(path)==digest,path
    c.save(OUT/'result.json',dict(updated=c.now(),state='complete',cpu_only=True,optimizer_steps=0,
        elapsed_seconds=time.monotonic()-started,records=records,final_loss_parity=parity,
        source_hashes_verified=True,quality_accepted=False,gaussian_admitted=False,
        interpretation='Local gradients only; do not treat negative cosine as proof of a finite optimizer trajectory or detector ground truth.'))
    print('UNIQUE_WRIST_GRADIENT_AUDIT_COMPLETE',parity,flush=True)


if __name__=='__main__':
    main()
