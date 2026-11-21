"""Read-only replay of pinned SMPLify-X final losses on three saved fits.

Exactly three body forwards, no optimizer, no fitter call, no saved-fit edit.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time
import traceback

import numpy as np
import official_smplifyx_entry123 as entry

H=Path(__file__).resolve().parent
BASE=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT=BASE/'demo-vhap-initialization/fullbody-smplifyx/full-sequence'
OFFICIAL=BASE/'smplifyx-reference-control/official'
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
SELECTED=('frame_000016','frame_000018','frame_000033')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(q,p)


def run():
    import fcntl
    with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(ROOT/'progress.json')['state']=='complete'
        output=ROOT/'limb-loss-replay.json'
        assert not output.exists(), 'Preserve completed replay.'
        source_hash=sha(__file__)
        progress=ROOT/'limb-loss-replay-progress.json'
        save(progress,dict(state='replaying_saved_final_loss',pid=os.getpid(),frames=list(SELECTED),optimizer_updates=0))
        start=time.monotonic()
        try:
            entry.adapted_entry_source(OFFICIAL)
            config=entry.load_official_config(OFFICIAL)
            receipt=read(ROOT/'fit/official-fit-result.json')
            assert sha(ROOT/'fit/fitted.npz')==receipt['fit_sha256']
            assert sha(ROOT/'data/training-inputs.json')==receipt['training_inputs_sha256']
            with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as z:aggregate=dict(z)
            rows=read(ROOT/'data/training-inputs.json')['records']
            assert list(aggregate['frame_ids'].astype(str))==[r['frame'] for r in rows]
            import torch
            import smplx
            utils,prior,fitting,_=entry.load_official(OFFICIAL)
            mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
            body=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,
                num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,
                flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
                joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cuda()
            body.parents=body.parents.cpu();body.betas.requires_grad_(False)
            loader=entry.module(PROJECT/'src/avatar_core/shuolee_vposer.py','_readonly_limb123_vposer')
            loader.MODEL=ASSETS/'vposer/V02_05';loader.CODE=PROJECT/'third_party/human_body_prior'
            vposer,_,vposer_meta=loader.load_vposer('cuda')
            vposer.eval()
            for p in vposer.parameters():p.requires_grad_(False)
            l2=lambda:prior.L2Prior().cuda()
            loss=fitting.create_loss(loss_type='smplify',rho=config['rho'],
                body_pose_prior=l2(),shape_prior=l2(),expr_prior=l2(),angle_prior=prior.SMPLifyAnglePrior().cuda(),
                jaw_prior=l2(),left_hand_prior=l2(),right_hand_prior=l2(),use_joints_conf=True,
                use_face=True,use_hands=True,interpenetration=False,dtype=torch.float32).cuda()
            samples=[]
            for frame in SELECTED:
                slot=[r['frame'] for r in rows].index(frame);row=rows[slot]
                saved=ROOT/'frames'/frame/'winner-camera-space.npz'
                with np.load(saved,allow_pickle=False) as z:a=dict(z)
                assert np.array_equal(a['vertices_camera'][0],aggregate['vertices_camera'][slot])
                body.reset_params(**{name:a[name] for name,_ in body.named_parameters()})
                latent=torch.tensor(a['pose_embedding'],device='cuda',requires_grad=True)
                aa=vposer.decode(latent)['pose_body'].reshape(1,63)
                result=body(body_pose=aa,return_verts=True,return_full_pose=True)
                K=torch.tensor(a['K_integer_centers'],device='cuda',dtype=torch.float32)
                translation=torch.tensor(a['camera_translation'],device='cuda')
                class Camera:
                    def __call__(self,points):
                        q=(points+translation[:,None])@K.T
                        return q[...,:2]/q[...,2:]
                camera=Camera()
                vertex_delta=float((result.vertices+translation[:,None]-torch.tensor(a['vertices_camera'],device='cuda')).abs().max())
                projection_delta=float((camera(result.joints)-torch.tensor(a['joints_projected_native'],device='cuda')).abs().max())
                assert vertex_delta<2e-6 and projection_delta<.002
                target=torch.tensor(aggregate['observed_native_keypoints'][slot:slot+1],device='cuda')
                weights=torch.ones((1,118),device='cuda');weights[:,[1,9,12]]=0
                weights[:,25:67]=config['hand_joints_weights'][-1];weights[:,67:]=config['face_joints_weights'][-1]
                final=dict(data_weight=1000/row['wh'][1],body_pose_weight=config['body_pose_prior_weights'][-1],
                    shape_weight=config['shape_weights'][-1],expr_prior_weight=config['expr_weights'][-1],
                    bending_prior_weight=3.17*config['body_pose_prior_weights'][-1],
                    hand_prior_weight=config['hand_pose_prior_weights'][-1],
                    jaw_prior_weight=[float(x) for x in config['jaw_pose_prior_weights'][-1].split(',')])
                kwargs=dict(body_model_output=result,camera=camera,gt_joints=target[...,:2],joints_conf=target[...,2],
                    body_model_faces=body.faces_tensor.view(-1),joint_weights=weights,use_vposer=True,pose_embedding=latent)
                params=[('pose_embedding',latent)]+[(n,p) for n,p in body.named_parameters() if p.requires_grad]
                def gradients(value):
                    grads=torch.autograd.grad(value,[p for _,p in params],retain_graph=True,allow_unused=True)
                    return {n:dict(l2=float(g.norm()),max_abs=float(g.abs().max())) if g is not None else None
                            for (n,_),g in zip(params,grads)}
                loss.reset_loss_weights(final);total=loss(**kwargs);total_gradient=gradients(total)
                components={}
                for selected in final:
                    isolated={k:(v if k==selected else [0,0,0] if k=='jaw_prior_weight' else 0.) for k,v in final.items()}
                    loss.reset_loss_weights(isolated);value=loss(**kwargs)
                    components[selected]=dict(value=float(value),gradients=gradients(value))
                difference=abs(float(total)-sum(c['value'] for c in components.values()))
                assert difference<.01
                report=read(ROOT/'frames'/frame/'result.json')
                samples.append(dict(frame_id=frame,fit_sha256=sha(saved),vertex_parity_max_m=vertex_delta,
                    projection_parity_max_px=projection_delta,effective_final_weights=final,
                    original_final_loss_at_saved_parameters=float(total),upstream_reported_last_step_loss=report['winning_loss'],
                    total_gradients=total_gradient,components=components,component_sum_difference=difference))
                save(progress,dict(state='replaying_saved_final_loss',pid=os.getpid(),completed=len(samples),optimizer_updates=0))
            assert sha(__file__)==source_hash and sha(ROOT/'fit/fitted.npz')==receipt['fit_sha256']
            report=dict(state='complete_readonly_loss_replay',finished=datetime.now(timezone.utc).isoformat(),
                source_sha256=source_hash,fit_sha256=receipt['fit_sha256'],vposer=vposer_meta,frames=samples,
                body_forwards=3,optimizer_updates=0,fit_files_modified=False,original_loss_class=True,
                interpretation='Positive prior values and gradients prove these terms are active at saved fits. The replay does not establish a unique failure cause or try new weights, iterations, initialization or constraints.',
                quality_accepted=False,elapsed_seconds=time.monotonic()-start)
            save(output,report);save(progress,dict(state='complete',pid=None,report_sha256=sha(output)))
            print(json.dumps(dict(state=report['state'],seconds=report['elapsed_seconds'],
                frames=[dict(frame=x['frame_id'],loss=x['original_final_loss_at_saved_parameters'],
                    body_prior=x['components']['body_pose_weight']['value'],angle_prior=x['components']['bending_prior_weight']['value'],
                    pose_gradient=x['total_gradients']['pose_embedding']) for x in samples])))
        except BaseException as error:
            save(ROOT/'limb-loss-replay-failure.json',dict(error=repr(error),traceback=traceback.format_exc(),optimizer_updates=0))
            save(progress,dict(state='failed_preserved',pid=None));raise


if __name__=='__main__':run()
