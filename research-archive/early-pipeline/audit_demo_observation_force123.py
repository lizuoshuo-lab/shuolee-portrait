"""Three saved-fit forwards: original joint influence and latent-gradient audit.

No optimizer or fitting call. Baseline geometry and source files are immutable.
"""
from pathlib import Path
import json
import os
import time

import numpy as np
import replay_demo_limb_losses123 as base
import official_smplifyx_entry123 as entry

FRAMES = ('frame_000017', 'frame_000018', 'frame_000033')
OUT = base.ROOT.parent/'observation-force-audit'


def main():
    import fcntl
    import torch
    import smplx
    started = time.monotonic()
    with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = base.read(base.ROOT.parent/'observation-force-plan.json')
        assert base.sha(__file__) == plan['source_sha256']
        assert base.sha(base.ROOT/'fit/fitted.npz') == plan['fit_sha256']
        OUT.mkdir(exist_ok=False)
        base.save(OUT/'progress.json', dict(state='readonly_saved_gradient_audit', pid=os.getpid()))
        entry.adapted_entry_source(base.OFFICIAL)
        config = entry.load_official_config(base.OFFICIAL)
        utils, prior, fitting, _ = entry.load_official(base.OFFICIAL)
        mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
            use_face_contour=False, openpose_format='coco25')
        body = smplx.SMPLX(str(base.ASSETS/'smplx/SMPLX_NEUTRAL.npz'), batch_size=1,
            num_betas=10, num_expression_coeffs=10, use_pca=True, num_pca_comps=12,
            flat_hand_mean=False, use_face_contour=False, create_body_pose=False, create_transl=False,
            joint_mapper=utils.JointMapper(mapping), dtype=torch.float32).cuda()
        body.parents = body.parents.cpu(); body.betas.requires_grad_(False)
        loader = entry.module(base.PROJECT/'src/avatar_core/shuolee_vposer.py', '_force123_vposer')
        loader.MODEL=base.ASSETS/'vposer/V02_05'; loader.CODE=base.PROJECT/'third_party/human_body_prior'
        vposer, _, meta = loader.load_vposer('cuda')
        vposer.eval()
        for p in vposer.parameters():
            p.requires_grad_(False)
        l2=lambda:prior.L2Prior().cuda()
        loss = fitting.create_loss(loss_type='smplify', rho=config['rho'],
            body_pose_prior=l2(), shape_prior=l2(), expr_prior=l2(), angle_prior=prior.SMPLifyAnglePrior().cuda(),
            jaw_prior=l2(), left_hand_prior=l2(), right_hand_prior=l2(), use_joints_conf=True,
            use_face=True, use_hands=True, interpenetration=False, dtype=torch.float32).cuda()
        with np.load(base.ROOT/'fit/fitted.npz', allow_pickle=False) as saved:
            aggregate={k:saved[k].copy() for k in ('frame_ids','observed_native_keypoints')}
        rows=base.read(base.ROOT/'data/training-inputs.json')['records']
        ids=aggregate['frame_ids'].astype(str).tolist()
        assert ids==[r['frame'] for r in rows]
        results=[]
        for frame in FRAMES:
            slot=ids.index(frame); row=rows[slot]
            path=base.ROOT/'frames'/frame/'winner-camera-space.npz'
            with np.load(path, allow_pickle=False) as z:
                saved=dict(z)
            body.reset_params(**{name:saved[name] for name,_ in body.named_parameters()})
            latent=torch.tensor(saved['pose_embedding'],device='cuda',requires_grad=True)
            output=body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
            K=torch.tensor(saved['K_integer_centers'],device='cuda',dtype=torch.float32)
            translation=torch.tensor(saved['camera_translation'],device='cuda')
            def camera(points):
                q=(points+translation[:,None])@K.T
                return q[...,:2]/q[...,2:]
            predicted=camera(output.joints)
            vertex_error=float((output.vertices+translation[:,None]-torch.tensor(saved['vertices_camera'],device='cuda')).abs().max())
            projection_error=float((predicted-torch.tensor(saved['joints_projected_native'],device='cuda')).abs().max())
            assert vertex_error<2e-6 and projection_error<.002
            target=torch.tensor(aggregate['observed_native_keypoints'][slot:slot+1],device='cuda')
            weights=torch.ones((1,118),device='cuda'); weights[:,[1,9,12]]=0
            weights[:,25:67]=config['hand_joints_weights'][-1]; weights[:,67:]=config['face_joints_weights'][-1]
            final=dict(data_weight=1000/row['wh'][1],body_pose_weight=config['body_pose_prior_weights'][-1],
                shape_weight=config['shape_weights'][-1],expr_prior_weight=config['expr_weights'][-1],
                bending_prior_weight=3.17*config['body_pose_prior_weights'][-1],
                hand_prior_weight=config['hand_pose_prior_weights'][-1],
                jaw_prior_weight=[float(v) for v in config['jaw_pose_prior_weights'][-1].split(',')])
            loss.reset_loss_weights(final)
            total=loss(body_model_output=output,camera=camera,gt_joints=target[...,:2],
                joints_conf=target[...,2],body_model_faces=body.faces_tensor.view(-1),
                joint_weights=weights,use_vposer=True,pose_embedding=latent)
            residual=predicted-target[...,:2]
            effective=(weights*target[...,2]).square()*loss.data_weight.square()
            terms=(effective[...,None]*loss.robustifier(residual)).sum(-1)[0]
            groups=dict(body_other=[i for i in range(19) if i!=6],holding_left_elbow=[6],
                feet=list(range(19,25)),hands=list(range(25,67)),face=list(range(67,118)))
            pieces={name:terms[index].sum() for name,index in groups.items()}
            pieces.update(data=terms.sum(),vposer=latent.square().sum()*loss.body_pose_weight.square(),
                bending=loss.angle_prior(output.full_pose[:,3:66]).sum()*loss.bending_prior_weight,total=total)
            gradients={name:torch.autograd.grad(value,latent,retain_graph=True)[0].detach().cpu().numpy().ravel()
                       for name,value in pieces.items()}
            disjoint=sum((gradients[name] for name in groups),np.zeros(32,np.float32))
            assert np.allclose(disjoint,gradients['data'],atol=.01,rtol=1e-4)
            assert np.allclose(gradients['data']+gradients['vposer']+gradients['bending'],gradients['total'],atol=.01,rtol=1e-4)
            rho=float(config['rho'])
            r=residual.detach().cpu().numpy()[0]
            eff=effective.detach().cpu().numpy()[0]
            pixel_force=2*eff[:,None]*rho**4*r/(r*r+rho*rho)**2
            def relation(left,right):
                g,h=gradients[left].astype(float),gradients[right].astype(float)
                norm=float(np.linalg.norm(g)*np.linalg.norm(h))
                return dict(dot=float(g@h),cosine=float(g@h/norm) if norm else None)
            joint_rows=[dict(index=i,residual_px=float(np.linalg.norm(r[i])),
                confidence=float(target[0,i,2]),effective_loss_coefficient=float(eff[i]),
                term=float(terms[i]),pixel_gradient=pixel_force[i].astype(float).tolist(),
                pixel_gradient_norm=float(np.linalg.norm(pixel_force[i]))) for i in range(118)]
            results.append(dict(frame_id=frame,fit_sha256=base.sha(path),vertex_error_m=vertex_error,
                projection_error_px=projection_error,rho=rho,original_loss=float(total),
                groups={name:dict(loss=float(value),latent_gradient=gradients[name].astype(float).tolist(),
                    latent_gradient_norm=float(np.linalg.norm(gradients[name]))) for name,value in pieces.items()},
                joints=joint_rows,relations={f'{l}_versus_{r}':relation(l,r) for l,r in
                    [('holding_left_elbow','vposer'),('holding_left_elbow','bending'),('holding_left_elbow','total'),('data','vposer')]},
                gradient_sum_max=float(np.abs(gradients['data']+gradients['vposer']+gradients['bending']-gradients['total']).max())))
        assert base.sha(base.ROOT/'fit/fitted.npz')==plan['fit_sha256'] and base.sha(__file__)==plan['source_sha256']
        result=dict(state='complete_readonly_observation_influence',frames=results,body_forwards=3,
            optimizer_updates=0,source_sha256=base.sha(__file__),fit_sha256=plan['fit_sha256'],
            elapsed_seconds=time.monotonic()-started,vposer=meta,quality_accepted=False,
            scope='Original final objective at saved parameters. Local gradients and heuristic confidence are not physical visibility or proof of a unique global failure cause.')
        base.save(OUT/'result.json',result)
        base.save(OUT/'progress.json',dict(state='complete',pid=None,optimizer_updates=0,result_sha256=base.sha(OUT/'result.json')))
        print(json.dumps(dict(state=result['state'],seconds=result['elapsed_seconds'],
            frames=[dict(frame=x['frame_id'],elbow=x['joints'][6],relations=x['relations']) for x in results])),flush=True)


if __name__=='__main__':main()
