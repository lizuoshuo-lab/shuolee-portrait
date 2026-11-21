"""Read-only replay of frozen grass poses; no optimizer or pose selection."""
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
from run_grass_temporal124 import transport, ROOT, SRC, OUT as RUN, BASE

OUT = RUN/'objective-audit'
OFFICIAL = ROOT.parent/'smplifyx-reference-control/official'


def prepare():
    assert not OUT.exists()
    assert read(RUN/'temporal-progress.json')['pid'] is None
    assert not read(RUN/'conclusion.json')['quality_accepted']
    files = [SRC/'data/inputs.json', SRC/'data/observations.npz', RUN/'plan.json', RUN/'conclusion.json']
    files += [RUN/f'official-fit/frames/frame_{i:06d}/{name}' for i in (32,33)
              for name in ('winner-camera-space.npz','result.json')]
    files += [BASE/f'frames/frame_000033/{name}' for name in ('winner-camera-space.npz','result.json')]
    sources = [Path(__file__), Path(entry.__file__), Path(__file__).with_name('smplify_temporal124.py'),
               Path(__file__).with_name('run_grass_temporal124.py'),
               Path(__file__).with_name('smplify_observation_validity124.py'),
               Path(__file__).with_name('smplify_score_contract123.py')]
    sources += [OFFICIAL/name for name in entry.PINNED]
    save(OUT/'plan.json',dict(updated=now(),
        hypothesis='Distinguish stage/local optimization failure from final objective preference for the wrong arm depth. Evaluate the old33, temporal33 and transported32 seed at exactly the same33 input. Inspect, but do not optimize, the camera-translation gradient frozen after torso initialization.',
        states=['anchor32','independent33','temporal33','transported32_body_root_other_parameters33'],
        budget=dict(cpu_only=True,states=4,original_stages=5,max_seconds=120,optimizer_steps=0),
        protected='Frozen input hashes, native K, shared beta and saved camera translation; exact geometry and saved-objective replay; no re-fit, new initializers, weight scan, pose substitution, point cloud or quality admission.',
        sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in files}))
    print('GRASS_OBJECTIVE_REGISTERED',flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (OUT/'result.json').exists()
    plan=read(OUT/'plan.json')
    def verify():
        for p,h in {**plan['sources'],**plan['inputs']}.items():assert sha(p)==h,p
    verify()
    import torch,smplx
    torch.set_num_threads(1); start=time.monotonic()
    save(OUT/'progress.json',dict(state='running_readonly',pid=os.getpid(),started=now()))
    rows=read(SRC/'data/inputs.json')['records']
    with np.load(SRC/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    utils,prior,fitting,_=entry.load_official(OFFICIAL);config=entry.load_official_config(OFFICIAL)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
        use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
        create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32)
    loader=entry.module(PROJECT/'src/avatar_core/shuolee_vposer.py','_grass_objective124_vposer')
    loader.MODEL=ASSETS/'vposer/V02_05';loader.CODE=PROJECT/'third_party/human_body_prior'
    vposer,_,_=loader.load_vposer('cpu');vposer.eval()
    def load(p):
        with np.load(p,allow_pickle=False) as z:return dict(z)
    anchor=load(RUN/'official-fit/frames/frame_000032/winner-camera-space.npz')
    final=load(RUN/'official-fit/frames/frame_000033/winner-camera-space.npz')
    old=load(BASE/'frames/frame_000033/winner-camera-space.npz')
    assert np.array_equal(old['camera_translation'],final['camera_translation'])
    assert np.array_equal(old['betas'],final['betas']) and np.array_equal(anchor['betas'],final['betas'])
    root=transport(anchor['global_orient'],rows[31]['w2c'],rows[32]['w2c'])
    reference=torch.zeros(1,165);reference[:,:3]=torch.from_numpy(root)
    reference[:,3:66]=torch.from_numpy(anchor['body_pose_axis_angle'])
    nominal=read(RUN/'plan.json')['nominal_dt_seconds'];dt=rows[32]['time_seconds']-rows[31]['time_seconds']
    receipt=read(RUN/'official-fit/frames/frame_000033/result.json')
    assert np.array_equal(root,receipt['continuation']['initial_state']['body_fit_initial_root'])
    records=[];parity={};receipts={}
    states=[('anchor32',31,anchor),('independent33',32,old),('temporal33',32,final),
            ('transported32_body_root_other_parameters33',32,final)]
    for state,idx,saved in states:
        row=rows[idx];target=np.zeros((1,118,3),np.float32)
        dst=np.r_[entry.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
        src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
        target[0,dst,:2]=obs['keypoints'][idx,src];target[0,dst,2]=confidence[idx,src]
        target,_=mask_outside_raster(target,row['wh']);gt=torch.from_numpy(target[...,:2]);conf=torch.from_numpy(target[...,2])
        params={k:saved[k] for k,_ in body.named_parameters()}
        is_seed=state.startswith('transported')
        if is_seed:params['global_orient']=root
        body.reset_params(**params)
        latent=torch.from_numpy((anchor if is_seed else saved)['pose_embedding'].copy()).requires_grad_()
        translation=torch.from_numpy(saved['camera_translation'].copy()).requires_grad_()
        K=torch.from_numpy(saved['K_compute_float32'])
        def camera(points):
            q=(points+translation[:,None])@K.T
            return q[...,:2]/q[...,2:]
        for stage in range(5):
            assert time.monotonic()-start<120
            output=body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
            if stage==0 and not is_seed:
                parity[state]=dict(vertices_max=float((output.vertices+translation[:,None]-torch.from_numpy(saved['vertices_camera'])).abs().max().detach()),
                    projection_max=float((camera(output.joints)-torch.from_numpy(saved['joints_projected_native'])).abs().max().detach()))
                assert parity[state]['vertices_max']<1e-5 and parity[state]['projection_max']<.02
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
            temporal=rotation_energy(output.full_pose,reference,dt,nominal) if idx==32 else output.full_pose.sum()*0
            data=(jw[...,None]*conf[...,None]).square()*loss.robustifier(gt-camera(output.joints))*weights['data_weight']**2
            terms=dict(original=original,temporal=temporal,data=data.sum(),vposer=latent.square().sum()*weights['body_pose_weight']**2,
                       bending=loss.angle_prior(output.full_pose[:,3:66]).sum()*weights['bending_prior_weight'])
            gradients={}
            for name,value in terms.items():
                gs=torch.autograd.grad(value,(latent,body.global_orient,translation),retain_graph=True,allow_unused=True)
                gradients[name]={label:float(g.norm()) if g is not None else 0. for label,g in zip(('latent_l2','root_l2','translation_l2'),gs)}
            records.append(dict(state=state,frame=row['frame'],stage=stage,weights=weights,
                losses={k:float(v.detach()) for k,v in terms.items()},total=float((original+temporal).detach()),gradient_norms=gradients,
                data_groups={label:float(data[:,lo:hi].sum().detach()) for label,lo,hi in [('body_feet',0,25),('hands',25,67),('face',67,118)]},
                right_elbow_term=float(data[:,3].sum().detach()),right_wrist_term=float(data[:,4].sum().detach()),
                data_camera_translation_gradient=torch.autograd.grad(terms['data'],translation)[0].detach().tolist()))
        if not is_seed:
            path=BASE/'frames/frame_000033/result.json' if state=='independent33' else RUN/f'official-fit/frames/{row["frame"]}/result.json'
            expected=read(path)['winning_loss'];actual=records[-1]['total'] if state=='temporal33' else records[-1]['losses']['original']
            relative=abs(actual-expected)/max(1,abs(expected));assert relative<1e-4,(state,actual,expected)
            receipts[state]=dict(replayed=actual,saved=expected,relative_difference=relative)
    verify()
    save(OUT/'result.json',dict(updated=now(),elapsed_seconds=time.monotonic()-start,records=records,geometry_parity=parity,
        loss_replay=receipts,optimizer_steps=0,cpu_only=True,source_hashes_unchanged=True,quality_accepted=False,
        gaussian_admitted=False,goal_complete=False,conclusion_reviewed=False))
    save(OUT/'progress.json',dict(state='complete_readonly',pid=None,completed_states=4))
    print('GRASS_OBJECTIVE_COMPLETE',time.monotonic()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args()
    (prepare if args.mode=='prepare' else run)()
