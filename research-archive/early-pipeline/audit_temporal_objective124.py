"""Read-only CPU loss/gradient audit at frozen failed winterhat poses."""
import argparse
import os
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read,save,sha,now,PROJECT,ASSETS
import official_smplifyx_entry123 as entry
from smplify_score_contract123 import relative_confidence
from smplify_temporal124 import rotation_energy

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
OUT=ROOT/'winterhat-continuation/objective-audit'
OFFICIAL=ROOT.parent/'smplifyx-reference-control/official'
SRC=ROOT/'winterhat-input'


def prepare():
    assert not OUT.exists()
    inputs=[SRC/'data/inputs.json',SRC/'data/observations.npz']
    inputs += [ROOT/run/'official-fit/frames'/frame/'winner-camera-space.npz' for run,frame in
        [('winterhat-continuation','frame_000049'),('winterhat-continuation','frame_000050'),('winterhat-temporal','frame_000049')]]
    inputs += [ROOT/'winterhat-continuation/official-fit/frames/frame_000049/result.json']
    sources=[Path(__file__),Path(entry.__file__),Path(__file__).with_name('smplify_temporal124.py'),
        OFFICIAL/'smplifyx/fitting.py',OFFICIAL/'smplifyx/fit_single_frame.py',OFFICIAL/'cfg_files/fit_smplx.yaml']
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),hypothesis='Find whether the actual soft temporal objective competes with original staged priors/data at the observed49 flip. Do not infer causality from loss magnitudes alone.',
        states=['saved continuation49','saved baseline49','actual49 body/root initializer from same-run50 with original zero face/hands'],
        fixed='Same native49 observations, full K, shared beta,49 camera translation and50 detached temporal target; evaluate all five original weight stages.',
        budget=dict(cpu_only=True,states=3,stages=5,max_seconds=120),optimization_steps=0,
        forbidden=['new fitting','weight scan','frame substitution','Gaussian training'],
        sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs}))
    print('OBJECTIVE_AUDIT_REGISTERED',flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (OUT/'result.json').exists()
    p=read(OUT/'plan.json')
    for path,digest in {**p['sources'],**p['inputs']}.items():assert sha(path)==digest,path
    import time,torch,smplx
    torch.set_num_threads(1);start=time.monotonic()
    rows=read(SRC/'data/inputs.json')['records'];slot=next(i for i,r in enumerate(rows) if r['frame']=='frame_000049');row=rows[slot]
    with np.load(SRC/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    target=np.zeros((1,118,3),np.float32)
    dst=np.r_[entry.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    source=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    target[0,dst,:2]=obs['keypoints'][slot,source];target[0,dst,2]=confidence[slot,source]
    utils,prior,fitting,_=entry.load_official(OFFICIAL)
    config=entry.load_official_config(OFFICIAL)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
        use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
        create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32)
    loader=entry.module(PROJECT/'src/avatar_core/shuolee_vposer.py','_audit124_vposer')
    loader.MODEL=ASSETS/'vposer/V02_05';loader.CODE=PROJECT/'third_party/human_body_prior'
    vposer,_,_=loader.load_vposer('cpu');vposer.eval()
    fits={}
    for name,run,frame in [('new49','winterhat-continuation','frame_000049'),('old49','winterhat-temporal','frame_000049'),('seed50','winterhat-continuation','frame_000050')]:
        with np.load(ROOT/run/'official-fit/frames'/frame/'winner-camera-space.npz',allow_pickle=False) as z:fits[name]=dict(z)
    for key in ('betas','camera_translation','K_integer_centers'):
        assert np.array_equal(fits['new49'][key],fits['old49'][key]),key
    reference=torch.zeros(1,165);reference[:,:3]=torch.from_numpy(fits['seed50']['global_orient']);reference[:,3:66]=torch.from_numpy(fits['seed50']['body_pose_axis_angle'])
    dt=abs(row['time_seconds']-rows[slot+1]['time_seconds']);nominal=read(ROOT/'winterhat-continuation/plan.json')['nominal_dt_seconds']
    K=torch.from_numpy(fits['new49']['K_compute_float32']);translation=torch.from_numpy(fits['new49']['camera_translation'])
    gt=torch.from_numpy(target[:,:,:2]);conf=torch.from_numpy(target[:,:,2]);records=[];parities={}
    def camera(points):
        q=(points+translation[:,None])@K.T
        return q[...,:2]/q[...,2:]
    for name in ('new49','old49','seed50'):
        f=fits[name]
        body.reset_params(**({k:f[k] for k,_ in body.named_parameters()} if name!='seed50' else dict(betas=f['betas'],global_orient=f['global_orient'])))
        latent=torch.from_numpy(f['pose_embedding'].copy()).requires_grad_()
        for stage in range(5):
            assert time.monotonic()-start<120
            output=body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
            if stage==0 and name!='seed50':
                parities[name]=dict(vertices_max=float((output.vertices+translation[:,None]-torch.from_numpy(f['vertices_camera'])).abs().max()),
                    projection_max=float((camera(output.joints)-torch.from_numpy(f['joints_projected_native'])).abs().max()))
                assert parities[name]['vertices_max']<1e-5 and parities[name]['projection_max']<.02,parities
            jw=torch.ones(1,118);jw[:,[1,9,12]]=0
            jw[:,25:67]=config['hand_joints_weights'][stage];jw[:,67:]=config['face_joints_weights'][stage]
            l2=prior.L2Prior
            weights=dict(data_weight=1000/row['wh'][1],body_pose_weight=config['body_pose_prior_weights'][stage],
                shape_weight=config['shape_weights'][stage],bending_prior_weight=3.17*config['body_pose_prior_weights'][stage],
                hand_prior_weight=config['hand_pose_prior_weights'][stage],expr_prior_weight=config['expr_weights'][stage],
                jaw_prior_weight=[float(x) for x in config['jaw_pose_prior_weights'][stage].split(',')])
            loss=fitting.SMPLifyLoss(interpenetration=False,rho=config['rho'],use_hands=True,use_face=True,
                body_pose_prior=l2(),shape_prior=l2(),angle_prior=prior.SMPLifyAnglePrior(),
                left_hand_prior=l2(),right_hand_prior=l2(),expr_prior=l2(),jaw_prior=l2(),**weights)
            original=loss(output,camera,gt,conf,None,jw,use_vposer=True,pose_embedding=latent)
            temporal=rotation_energy(output.full_pose,reference,dt,nominal)
            data=(jw[...,None]*conf[...,None]).square()*loss.robustifier(gt-camera(output.joints))*weights['data_weight']**2
            body_prior=latent.square().sum()*weights['body_pose_weight']**2
            terms=dict(original=original,temporal=temporal,data=data.sum(),vposer=body_prior)
            gradients={}
            for term,value in terms.items():
                gs=torch.autograd.grad(value,(latent,body.global_orient),retain_graph=True,allow_unused=True)
                gradients[term]={k:float(g.norm()) if g is not None else 0. for k,g in zip(('latent_l2','root_l2'),gs)}
            records.append(dict(state=name,stage=stage,weights=weights,losses={k:float(v.detach()) for k,v in terms.items()},
                total=float((original+temporal).detach()),data_groups={label:float(data[:,lo:hi].sum().detach()) for label,lo,hi in [('body_feet',0,25),('hands',25,67),('face',67,118)]},gradient_norms=gradients))
    saved=read(ROOT/'winterhat-continuation/official-fit/frames/frame_000049/result.json')
    end=next(r for r in records if r['state']=='new49' and r['stage']==4)
    rel=abs(end['total']-saved['winning_loss'])/max(1.,abs(saved['winning_loss']))
    save(OUT/'result.json',dict(updated=now(),elapsed_seconds=time.monotonic()-start,records=records,geometry_parity=parities,
        saved_winner_loss=saved['winning_loss'],saved_loss_relative_difference=rel,orientation_count=saved['orientation_count'],
        replay_final_loss_accepted=rel<1e-4,optimization_steps=0,cpu_only=True,causal_conclusion_reviewed=False,
        quality_accepted=False,gaussian_admitted=False,source_hashes_verified=True))
    assert rel<1e-4,(rel,end['total'],saved['winning_loss'])
    print('OBJECTIVE_AUDIT_COMPLETE',len(records),rel,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();(prepare if a.mode=='prepare' else run)()
