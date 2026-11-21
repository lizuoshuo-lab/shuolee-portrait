"""Evaluate the stopped sofa objective at fixed states. Never fit or select poses."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS
import official_smplifyx_entry123 as entry
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster
from smplify_temporal124 import rotation_energy

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
RUN=ROOT/'sofa-temporal'
OUT=RUN/'objective-audit'
SRC=ROOT/'sofa-input'
OFFICIAL=ROOT.parent/'smplifyx-reference-control/official'

def prepare():
    assert not OUT.exists()
    assert read(RUN/'temporal-progress.json')['pid'] is None
    assert not read(RUN/'quality-stop-decision.json')['continue_same_candidate']
    inputs=[SRC/'data/inputs.json',SRC/'data/observations.npz',RUN/'plan.json',RUN/'quality-stop-decision.json']
    inputs += [RUN/f'official-fit/frames/frame_{i:06d}/{name}' for i in (32,33) for name in ('winner-camera-space.npz','result.json')]
    sources=[Path(__file__),Path(entry.__file__),Path(__file__).with_name('smplify_temporal124.py'),
        Path(__file__).with_name('smplify_observation_validity124.py'),OFFICIAL/'smplifyx/fitting.py',OFFICIAL/'cfg_files/fit_smplx.yaml']
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),hypothesis='Distinguish a poor original-stage endpoint from a final objective that prefers the hip-depth flip. Evaluate33 under the exact saved32 temporal target, fixed camera and observations.',
        states=['fitted33','body/root32 with all other fitted33 parameters','actual body/root32 initializer with original zero face/hands'],
        budget=dict(cpu_only=True,states=3,original_stages=5,max_seconds=120,optimizer_steps=0),
        forbidden=['new fit','select or substitute a pose','weight scan','Gaussian admission'],
        previous_relevant_evidence='Demo stage19 trajectory and59-frame warmtail control already exist; this audit does not establish a new stage-tail fix.',
        sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs}))
    print('SOFA_OBJECTIVE_REGISTERED',flush=True)

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (OUT/'result.json').exists()
    p=read(OUT/'plan.json')
    for path,digest in {**p['sources'],**p['inputs']}.items():assert sha(path)==digest,path
    import torch,smplx
    torch.set_num_threads(1);start=time.monotonic()
    rows=read(SRC/'data/inputs.json')['records'];row=rows[32]
    assert row['frame']=='frame_000033' and rows[31]['frame']=='frame_000032'
    with np.load(SRC/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    targets=np.zeros((1,118,3),np.float32)
    dst=np.r_[entry.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    source=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    targets[0,dst,:2]=obs['keypoints'][32,source];targets[0,dst,2]=confidence[32,source]
    targets,inside=mask_outside_raster(targets,row['wh'])
    utils,prior,fitting,_=entry.load_official(OFFICIAL);config=entry.load_official_config(OFFICIAL)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
        use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
        create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32)
    loader=entry.module(PROJECT/'src/avatar_core/shuolee_vposer.py','_audit_sofa124_vposer')
    loader.MODEL=ASSETS/'vposer/V02_05';loader.CODE=PROJECT/'third_party/human_body_prior'
    vposer,_,_=loader.load_vposer('cpu');vposer.eval()
    with np.load(RUN/'official-fit/frames/frame_000033/winner-camera-space.npz',allow_pickle=False) as z:end=dict(z)
    with np.load(RUN/'official-fit/frames/frame_000032/winner-camera-space.npz',allow_pickle=False) as z:previous=dict(z)
    receipt=read(RUN/'official-fit/frames/frame_000033/result.json')
    assert np.array_equal(receipt['continuation']['initial_body_latent'],previous['pose_embedding'])
    assert np.array_equal(end['betas'],previous['betas'])
    reference=torch.zeros(1,165);reference[:,:3]=torch.from_numpy(previous['global_orient']);reference[:,3:66]=torch.from_numpy(previous['body_pose_axis_angle'])
    dt=row['time_seconds']-rows[31]['time_seconds'];nominal=read(RUN/'plan.json')['nominal_dt_seconds']
    K=torch.from_numpy(end['K_compute_float32']);translation=torch.from_numpy(end['camera_translation'])
    gt=torch.from_numpy(targets[:,:,:2]);conf=torch.from_numpy(targets[:,:,2]);records=[];parity={}
    def camera(points):
        q=(points+translation[:,None])@K.T
        return q[...,:2]/q[...,2:]
    for state in ('fitted33','body_root32_other_parameters33','actual_seed32'):
        params={k:end[k] for k,_ in body.named_parameters()}
        if state!='fitted33':params['global_orient']=previous['global_orient']
        if state=='actual_seed32':params=dict(betas=end['betas'],global_orient=previous['global_orient'])
        body.reset_params(**params)
        latent=torch.from_numpy((end if state=='fitted33' else previous)['pose_embedding'].copy()).requires_grad_()
        for stage in range(5):
            assert time.monotonic()-start<120
            output=body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
            if stage==0 and state=='fitted33':
                parity=dict(vertices_max=float((output.vertices+translation[:,None]-torch.from_numpy(end['vertices_camera'])).abs().max()),
                    projection_max=float((camera(output.joints)-torch.from_numpy(end['joints_projected_native'])).abs().max()))
                assert parity['vertices_max']<1e-5 and parity['projection_max']<.02,parity
            jw=torch.ones(1,118);jw[:,[1,9,12]]=0
            jw[:,25:67]=config['hand_joints_weights'][stage];jw[:,67:]=config['face_joints_weights'][stage]
            weights=dict(data_weight=1000/row['wh'][1],body_pose_weight=config['body_pose_prior_weights'][stage],
                shape_weight=config['shape_weights'][stage],bending_prior_weight=3.17*config['body_pose_prior_weights'][stage],
                hand_prior_weight=config['hand_pose_prior_weights'][stage],expr_prior_weight=config['expr_weights'][stage],
                jaw_prior_weight=[float(x) for x in config['jaw_pose_prior_weights'][stage].split(',')])
            l2=prior.L2Prior
            loss=fitting.SMPLifyLoss(interpenetration=False,rho=config['rho'],use_hands=True,use_face=True,
                body_pose_prior=l2(),shape_prior=l2(),angle_prior=prior.SMPLifyAnglePrior(),left_hand_prior=l2(),right_hand_prior=l2(),expr_prior=l2(),jaw_prior=l2(),**weights)
            original=loss(output,camera,gt,conf,None,jw,use_vposer=True,pose_embedding=latent)
            temporal=rotation_energy(output.full_pose,reference,dt,nominal)
            data=(jw[...,None]*conf[...,None]).square()*loss.robustifier(gt-camera(output.joints))*weights['data_weight']**2
            terms=dict(original=original,temporal=temporal,data=data.sum(),vposer=latent.square().sum()*weights['body_pose_weight']**2)
            gradients={}
            for term,value in terms.items():
                gs=torch.autograd.grad(value,(latent,body.global_orient),retain_graph=True,allow_unused=True)
                gradients[term]={k:float(g.norm()) if g is not None else 0. for k,g in zip(('latent_l2','root_l2'),gs)}
            records.append(dict(state=state,stage=stage,weights=weights,losses={k:float(v.detach()) for k,v in terms.items()},
                total=float((original+temporal).detach()),gradient_norms=gradients,
                data_groups={label:float(data[:,lo:hi].sum().detach()) for label,lo,hi in [('body_feet',0,25),('hands',25,67),('face',67,118)]}))
    final=next(r for r in records if r['state']=='fitted33' and r['stage']==4)
    rel=abs(final['total']-receipt['winning_loss'])/max(1.,abs(receipt['winning_loss']))
    save(OUT/'result.json',dict(updated=now(),elapsed_seconds=time.monotonic()-start,records=records,geometry_parity=parity,
        saved_winner_loss=receipt['winning_loss'],saved_loss_relative_difference=rel,replay_final_loss_accepted=rel<1e-4,
        optimization_steps=0,cpu_only=True,causal_conclusion_reviewed=False,quality_accepted=False,gaussian_admitted=False,source_hashes_verified=True))
    assert rel<1e-4,(rel,final['total'],receipt['winning_loss'])
    print('SOFA_OBJECTIVE_COMPLETE',rel,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args();(prepare if args.mode=='prepare' else run)()
