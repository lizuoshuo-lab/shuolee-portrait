"""Read-only CPU replay of frame19's seeded state and two saved candidates.

No optimizer, fitting, parameter selection, or change to the live59-frame run.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import numpy as np
import replay_demo_limb_losses123 as base
import official_smplifyx_entry123 as entry

ROOT = base.ROOT.parent/'forward-guarded-sequence'
PLAN = ROOT/'review/stage19-replay-plan.json'
OUT = ROOT/'review/stage19-replay-result.json'
read, save, sha = base.read, base.save, base.sha


def prepare():
    assert not PLAN.exists() and not OUT.exists()
    frame = ROOT/'frames/frame_000019'
    receipt = read(frame/'result.json')
    assert receipt['temporal_seed']['seed'] == 'frame_000018'
    assert read(ROOT/'frames/frame_000018/objective-selection.json')['selected_index'] == 0
    assert read(frame/'objective-selection.json')['selected_index'] == 1
    inputs = [Path(receipt['seed_path']), frame/'result.json', frame/'candidate-00-temporal.npz',
        frame/'objective-selection.json', base.ROOT/'frames/frame_000019/winner-camera-space.npz',
        base.ROOT/'fit/fitted.npz', base.ROOT/'data/training-inputs.json',
        base.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    sources = [Path(__file__), Path(entry.__file__), Path(base.__file__)]
    sources += [base.OFFICIAL/p for p in entry.PINNED]
    save(PLAN, dict(updated=datetime.now(timezone.utc).isoformat(), state='prepared',
        frame='frame_000019', hypothesis='Does the good preceding pose already miss frame19 after the unchanged camera initialization, or does the final fitted candidate move away? Compare fixed states under all five original objective stages; this does not trace the optimizer trajectory.',
        states=['exact_seeded_body_root_zero_hands_expression', 'new_candidate', 'independent_candidate'],
        body_forwards_max=3, loss_evaluations_max=15, wall_seconds_max=120,
        device='cpu', optimizer_updates=0, fitter_calls=0, no_output_fit=True,
        parity_tolerance_m=2e-5, parity_tolerance_px=.005,
        criteria=['Verify current camera translation and seed identity against actual receipt.',
            'Replay saved candidate vertices/projection within registered CPU/GPU tolerances before interpreting any losses.',
            'Report every original-stage data/prior term and both elbow residuals for all three fixed states.',
            'Original weights, observations, complete K and all live files stay unchanged. No acceptance or new candidate from this diagnostic.'],
        limits=['Fixed-state losses do not establish the actual stage where the optimizer lost the arm.',
            'Original detector residuals are not independent physical truth.'],
        input_hashes={str(p):sha(p) for p in inputs}, source_hashes={str(p):sha(p) for p in sources},
        quality_accepted=False, goal_complete=False))
    print('PREPARED_CPU_STAGE19_REPLAY', flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUT.exists()
    p=read(PLAN); started=time.monotonic()
    for path,digest in {**p['input_hashes'],**p['source_hashes']}.items(): assert sha(path)==digest,path
    import torch
    import smplx
    torch.set_num_threads(1)
    cfg=entry.load_official_config(base.OFFICIAL)
    utils,prior,fitting,_=entry.load_official(base.OFFICIAL)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(base.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,
        num_expression_coeffs=10,use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,
        create_body_pose=False,create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32)
    body.eval()
    loader=entry.module(base.PROJECT/'src/avatar_core/shuolee_vposer.py','_forward_stage19_cpu_vposer')
    loader.MODEL,loader.CODE=base.ASSETS/'vposer/V02_05',base.PROJECT/'third_party/human_body_prior'
    vposer,_,vposer_meta=loader.load_vposer('cpu'); vposer.eval()
    for model in (body,vposer):
        for parameter in model.parameters(): parameter.requires_grad_(False)
    l2=lambda:prior.L2Prior()
    loss=fitting.create_loss(loss_type='smplify',rho=cfg['rho'],body_pose_prior=l2(),shape_prior=l2(),
        expr_prior=l2(),angle_prior=prior.SMPLifyAnglePrior(),jaw_prior=l2(),left_hand_prior=l2(),
        right_hand_prior=l2(),use_joints_conf=True,use_face=True,use_hands=True,interpenetration=False,dtype=torch.float32)
    def load(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    frame='frame_000019'; receipt=read(ROOT/'frames'/frame/'result.json')
    seed=load(receipt['seed_path']); new=load(ROOT/'frames'/frame/'candidate-00-temporal.npz')
    original=load(base.ROOT/'frames'/frame/'winner-camera-space.npz')
    assert np.array_equal(seed['global_orient'],receipt['initial_state']['body_fit_initial_root'])
    assert np.array_equal(new['camera_translation'],receipt['initial_state']['camera_translation'])
    with np.load(base.ROOT/'fit/fitted.npz',allow_pickle=False) as z:
        index=z['frame_ids'].astype(str).tolist().index(frame)
        target=torch.as_tensor(z['observed_native_keypoints'][index:index+1].copy())
    record=next(r for r in read(base.ROOT/'data/training-inputs.json')['records'] if r['frame']==frame)
    K=torch.as_tensor(new['K_integer_centers'],dtype=torch.float32); translation=torch.as_tensor(new['camera_translation'])
    def camera(points):
        q=(points+translation[:,None])@K.T
        return q[...,:2]/q[...,2:]
    rows=[]
    with torch.no_grad():
        for name,saved in zip(p['states'],[None,new,original]):
            assert time.monotonic()-started < p['wall_seconds_max']
            if saved is None:
                body.reset_params(betas=new['betas'],global_orient=seed['global_orient'])
                latent=torch.as_tensor(seed['pose_embedding'])
                for key in ['left_hand_pose','right_hand_pose','jaw_pose','expression','leye_pose','reye_pose']:
                    assert torch.count_nonzero(getattr(body,key)) == 0,key
            else:
                body.reset_params(**{key:saved[key] for key,_ in body.named_parameters()})
                latent=torch.as_tensor(saved['pose_embedding'])
            result=body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
            projected=camera(result.joints); parity=None
            if saved is not None:
                vd=float((result.vertices+translation[:,None]-torch.as_tensor(saved['vertices_camera'])).abs().max())
                pd=float((projected-torch.as_tensor(saved['joints_projected_native'])).abs().max())
                assert vd<p['parity_tolerance_m'] and pd<p['parity_tolerance_px'],(name,vd,pd)
                parity=dict(vertex_max_m=vd,projection_max_px=pd)
            residual=projected-target[...,:2]; distance=torch.linalg.vector_norm(residual,dim=-1)[0]
            stages=[]
            for stage in range(5):
                weights=torch.ones((1,118)); weights[:,[1,9,12]]=0
                weights[:,25:67]=cfg['hand_joints_weights'][stage]; weights[:,67:]=cfg['face_joints_weights'][stage]
                values=dict(data_weight=1000/record['wh'][1],body_pose_weight=cfg['body_pose_prior_weights'][stage],
                    shape_weight=cfg['shape_weights'][stage],expr_prior_weight=cfg['expr_weights'][stage],
                    bending_prior_weight=3.17*cfg['body_pose_prior_weights'][stage],hand_prior_weight=cfg['hand_pose_prior_weights'][stage],
                    jaw_prior_weight=[float(x) for x in cfg['jaw_pose_prior_weights'][stage].split(',')])
                loss.reset_loss_weights(values)
                total=loss(result,camera=camera,gt_joints=target[...,:2],joints_conf=target[...,2],
                    body_model_faces=body.faces_tensor.view(-1),joint_weights=weights,pose_embedding=latent,use_vposer=True)
                terms=((weights*target[...,2]).square()[...,None]*loss.data_weight.square()*loss.robustifier(residual)).sum(-1)[0]
                stages.append(dict(stage=stage,total=float(total),body_data=float(terms[:19].sum()),feet_data=float(terms[19:25].sum()),
                    hands_data=float(terms[25:67].sum()),face_data=float(terms[67:].sum()),prior_total=float(total-terms.sum()),left_elbow_data=float(terms[6])))
            rows.append(dict(state=name,left_elbow_px=float(distance[6]),right_elbow_px=float(distance[3]),parity=parity,stages=stages))
    assert not torch.cuda.is_initialized() and len(rows)==3
    for path,digest in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(path)==digest,path
    save(OUT,dict(state='readonly_cpu_replay_complete',updated=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        rows=rows,body_forwards=3,loss_evaluations=15,optimizer_updates=0,fitter_calls=0,cuda_initialized=False,
        plan_sha256=sha(PLAN),vposer=vposer_meta,quality_accepted=False,goal_complete=False,limits=p['limits']))
    for row in rows:print(row['state'],row['left_elbow_px'],[(s['stage'],round(s['total'],2)) for s in row['stages']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()
