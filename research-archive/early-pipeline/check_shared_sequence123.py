"""One bounded real-SMPLX implementation control, no real-person fit."""
from pathlib import Path
import argparse, copy, fcntl, json, os, sys, time, traceback
import numpy as np
import torch
import run_smplify_reference123 as r

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.sequence_fitting import (SharedIdentitySequence, SequenceObservations,
    camera_to_world, project_complete_k, sequence_residuals, staged_weights)

OUT=r.O/'framewise-canonical-route/shared-sequence-control'
CODE=PROJECT/'src/shuolee_portrait/sequence_fitting.py'

def prepare():
    assert not OUT.exists(), 'Registered implementation suite runs once; repairs retain failure evidence.'
    OUT.mkdir()
    r.save(OUT/'plan.json',dict(registered=r.now(),state='registered_not_executed',
      purpose='Actual SMPLX and VPoser sequence interface with one shared shape and explicit cameras; known geometry only.',
      source_hashes={str(p):r.sha(p) for p in [Path(__file__),CODE,Path(r.__file__)]},
      frame_ids=['synthetic:0','synthetic:1','synthetic:2'],
      checks=['Independent full-K projection including skew/noncentral cx/cy',
              'Independent inverse world/camera coordinates',
              'One trainable beta tensor and exact summed beta gradients',
              'Batch geometry versus independent one-frame forward calls',
              'Official objective versus mean sequence residual norm',
              'Held/unknown/semantic-unverified face coordinates do not affect fitting value or gradient',
              'Hand root duplicates are inactive, facial weights unchanged by hand policy',
              'Known per-frame root recovery with known distinct body poses',
              'Directional derivative for every optimized variable block',
              'No mutation of K or extrinsics; reference snapshot equals selected shared sequence frame'],
      budget=dict(seconds=180,root_closure_evaluations=180,real_fits=0,gaussian_steps=0),
      acceptance='All algebraic checks finite, projection/covariate checks <=1e-8, known root reprojection max<0.01px; no physical acceptance inferred.',
      no_repeat_parameter_sweep=True, physical_accuracy_passed=False))

def make_model(parts):
    n=3;device='cuda';dtype=torch.float64
    ref=r.Ref(parts,np.eye(3),dict(betas=np.zeros(10),body_pose=np.zeros(63),global_orient=[np.pi,0,0],transl=[0,0,4]))
    K=np.array([[[1180.,1.7,1261.25],[0,1100.,731.75],[0,0,1.]],
                [[1111.,0,1340.5],[0,1095.,771.5],[0,0,1.]],
                [[1155.,-.7,1310.25],[0,1120.,732.25],[0,0,1.]]])
    E=np.tile(np.eye(4),(n,1,1));a=.2
    E[1,:3,:3]=[[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]
    E[1,:3,3]=[.4,-.1,.2];E[2,:3,3]=[-.3,.2,-.1]
    initial=dict(global_orient=np.array([[np.pi,0,0],[3.05,.12,-.09],[3.08,-.15,.05]]),
                 transl=np.array([[0,0,4],[.1,.02,4.2],[-.12,-.02,4.4]]),
                 latent=.08*np.sin(np.arange(n*32).reshape(n,32)+1),
                 jaw_pose=np.zeros((n,3)),expression=np.zeros((n,10)))
    model=SharedIdentitySequence(ref.body,copy.deepcopy(parts[4]),['synthetic:0','synthetic:1','synthetic:2'],K,E,initial)
    return model,ref

def observations(model, uv):
    n=len(model.frame_ids)
    return SequenceObservations(model.frame_ids,uv.detach().clone(),torch.ones((n,118),device=uv.device,dtype=uv.dtype),
        np.full((n,51),'observed',dtype='<U10'),np.full((n,51),'FIT',dtype='<U10'),
        np.ones((n,51),bool),np.ones((n,51),bool),np.tile(np.array([f'synthetic:material:{i}' for i in range(51)]),(n,1)))

def run():
    p=r.read(OUT/'plan.json');assert not (OUT/'progress.json').exists()
    for source,h in p['source_hashes'].items():assert r.sha(source)==h
    started=time.monotonic();r.save(OUT/'progress.json',dict(state='running',pid=os.getpid(),started=r.now()))
    parts=r.setup();model,ref=make_model(parts);checks={}
    with torch.no_grad():model.state['betas'].copy_(model.K.new_tensor(.15*np.cos(np.arange(10))).reshape(1,10))
    theta=model.flatten();out=model.forward_state();uv=model.project(out)
    K0=model.K.clone();E0=model.w2c.clone()
    xyz=out.joints.detach().cpu().numpy();K=model.K.cpu().numpy()
    q=np.einsum('nij,nkj->nki',K,xyz);expected=q[...,:2]/q[...,2:]
    checks['numpy_projection_max_px']=float(np.max(np.abs(expected-uv.detach().cpu().numpy())))
    assert checks['numpy_projection_max_px']<1e-8
    world=model.world_vertices(out);vc=out.vertices.detach().cpu().numpy();E=model.w2c.cpu().numpy()
    expected_world=np.stack([(vc[i]-E[i,:3,3])@E[i,:3,:3] for i in range(3)])
    checks['numpy_world_inverse_max']=float(np.max(np.abs(world.detach().cpu().numpy()-expected_world)))
    roundtrip=torch.einsum('nij,nkj->nki',model.w2c[:,:3,:3],world)+model.w2c[:,None,:3,3]
    checks['world_camera_roundtrip_max']=float((roundtrip-out.vertices).abs().max())
    assert max(checks['numpy_world_inverse_max'],checks['world_camera_roundtrip_max'])<1e-10
    assert model.state['betas'].shape==(1,10)
    checks['one_shared_identity_parameter']=True
    # Exercise the exact same SMPL model in unbatched calls, using explicit arguments.
    per=[];s=model.state
    for i in range(3):
        x=model.body(betas=s['betas'],body_pose=model.vposer.decode(s['latent'][i:i+1])['pose_body'].reshape(1,63),
          global_orient=s['global_orient'][i:i+1],transl=s['transl'][i:i+1],jaw_pose=s['jaw_pose'][i:i+1],expression=s['expression'][i:i+1],
          left_hand_pose=model.hand_pca[i:i+1],right_hand_pose=model.hand_pca[i:i+1],leye_pose=model.eye_pose[i:i+1],reye_pose=model.eye_pose[i:i+1],return_verts=True,return_full_pose=True)
        per.append(x)
    checks['batched_vs_single_vertices_max']=float((out.vertices-torch.cat([x.vertices for x in per])).abs().max())
    assert checks['batched_vs_single_vertices_max']<1e-9
    combined=uv.square().mean((1,2)).sum();g=torch.autograd.grad(combined,model.state['betas'],retain_graph=True)[0]
    separate=torch.stack([torch.autograd.grad(uv[i].square().mean(),model.state['betas'],retain_graph=True)[0] for i in range(3)]).sum(0)
    checks['shared_beta_gradient_sum_relative_error']=float((g-separate).abs().max())/max(1.,float(g.abs().max()))
    assert checks['shared_beta_gradient_sum_relative_error']<1e-9
    obs=observations(model,uv);obs.targets+=1.3*torch.sin(torch.arange(708,device='cuda',dtype=torch.float64).reshape(3,118,2)+1)
    conf=obs.fitting_confidence();face0=conf[:,67:].clone()
    alt=obs.fitting_confidence(hand_policy='fit_without_duplicate_wrists')
    assert torch.equal(face0,alt[:,67:]) and not alt[:,[25,46]].any() and not conf[:,25:67].any()
    checks['hand_policy_has_no_facial_weight_change_and_no_duplicate_wrists']=True
    # Compare the complete mean objective with direct pinned upstream calls.
    values=[];ref.loss.double()
    for stage in range(5):
        w=staged_weights(stage,theta);jw=torch.ones((3,118),device='cuda',dtype=torch.float64);jw[:,67:]*=w['face']
        ref.loss.reset_loss_weights(dict(data_weight=1000/1520,body_pose_weight=w['body'],shape_weight=w['shape'],bending_prior_weight=w['bending'],hand_prior_weight=w['hand'],expr_prior_weight=w['expression'],jaw_prior_weight=w['jaw']))
        official=[]
        for i,one in enumerate(per):
            camera=lambda joints,i=i:project_complete_k(joints,model.K[i:i+1])
            official.append(ref.loss(one,camera,obs.targets[i:i+1],conf[i:i+1],model.body.faces_tensor,jw[i:i+1],use_vposer=True,pose_embedding=s['latent'][i:i+1]))
        rr,_=sequence_residuals(model,theta,obs,stage,[1520]*3);lhs=rr.square().sum();rhs=torch.stack(official).mean()
        err=float((lhs-rhs).abs())/max(1.,abs(float(rhs)));values.append(err);assert err<1e-9
    checks['all5_mean_official_objective_relative_errors']=values
    # Whole semantic-family holdout across all frames, not a pointwise split.
    obs.face_split[:,0]='HELD';obs.face_visibility[:,1]='unknown';obs.face_semantic_verified[:,2]=False
    probe=theta.detach().requires_grad_(True)
    rr,_=sequence_residuals(model,probe,obs,4,[1520]*3);value=rr.square().sum();grad=torch.autograd.grad(value,probe)[0]
    changed=copy.deepcopy(obs);changed.targets[:,67:70]+=1e5
    # The COCO face aliases are deliberately inactive, too.
    changed.targets[:,[0,15,16,17,18]]-=1e5
    rr,_=sequence_residuals(model,probe,changed,4,[1520]*3);value2=rr.square().sum();grad2=torch.autograd.grad(value2,probe)[0]
    assert torch.equal(value,value2) and torch.equal(grad,grad2)
    checks['held_unknown_unverified_no_value_or_gradient_leakage']=True
    checks['all_parameter_directional_derivatives']={}
    for name,(a,b,shape) in model.layout().items():
        direction=torch.zeros_like(theta);direction[a:b]=torch.cos(torch.arange(b-a,device='cuda',dtype=torch.float64)+1);direction/=direction.norm()
        eps=1e-5
        aa=sequence_residuals(model,theta+eps*direction,obs,4,[1520]*3)[0].square().sum()
        bb=sequence_residuals(model,theta-eps*direction,obs,4,[1520]*3)[0].square().sum()
        fd=float((aa-bb)/(2*eps));ad=float(grad@direction)
        error=abs(fd-ad)/max(1.,abs(fd),abs(ad));checks['all_parameter_directional_derivatives'][name]=error
        assert error<2e-4,(name,error)
    # Recovery concerns root only, with three known different articulated poses.
    truth=theta.clone();root_keys=['global_orient','transl'];root_indices=[]
    for name in root_keys:
        a,b,_=model.layout()[name];root_indices+=list(range(a,b))
    idx=torch.tensor(root_indices,device='cuda');q=torch.nn.Parameter(theta[idx].clone())
    with torch.no_grad():q+=.015*torch.cos(torch.arange(len(q),device='cuda',dtype=torch.float64))
    truth_uv=uv.detach();count=0
    opt=torch.optim.LBFGS([q],lr=1,max_iter=120,max_eval=180,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        nonlocal count
        if count>=180 or time.monotonic()-started>180:raise RuntimeError('Registered control budget exhausted.')
        count+=1;opt.zero_grad();candidate=theta.scatter(0,idx,q);pred=model.project(model.forward_state(model.unpack(candidate)))
        loss=(pred-truth_uv).square().mean();loss.backward();return loss
    opt.step(closure)
    fitted=theta.scatter(0,idx,q.detach());model.restore(fitted)
    final=model.project(model.forward_state());checks['known_three_root_recovery_max_px']=float(torch.linalg.vector_norm(final-truth_uv,dim=-1).max())
    checks['known_root_evaluations']=count
    assert checks['known_three_root_recovery_max_px']<.01
    model.restore(truth);snap=model.native_snapshot('synthetic:1')
    assert torch.equal(snap['reference_vertices_world'],model.world_vertices(model.forward_state())[1])
    assert torch.equal(K0,model.K) and torch.equal(E0,model.w2c)
    checks['reference_snapshot_exact_and_cameras_unchanged']=True
    assert time.monotonic()-started<180
    for source,h in p['source_hashes'].items():assert r.sha(source)==h
    r.save(OUT/'result.json',dict(updated=r.now(),state='passed',checks=checks,seconds=time.monotonic()-started,
      actual_SMPLX_VPoser_GPU=True,source_hashes=p['source_hashes'],real_fit=False,gaussian_steps=0,
      physical_face_accuracy_passed=False,real_input_review_required=True))
    r.save(OUT/'progress.json',dict(state='complete',pid=None,finished=r.now()))
    print(json.dumps(checks),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    else:
        lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception:
            r.save(OUT/'failure.json',dict(updated=r.now(),traceback=traceback.format_exc()));r.save(OUT/'progress.json',dict(state='failed',pid=None));raise
