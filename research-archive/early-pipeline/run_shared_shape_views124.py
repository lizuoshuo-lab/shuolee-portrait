"""One shared10D shape in the existing selected-view upstream fitting interface."""
from pathlib import Path
import inspect,sys
import numpy as np
import run_grass_selected_views124 as original
from run_official_gaussians123 import read,save,sha,now

OUT=original.ROOT/'grass-selected-shape124'
H=Path(__file__).resolve().parent


def make_models(n):
    import torch
    body,vposer=original.make_models(n)
    # SMPL-X already expands singleton betas to the body batch in forward().
    # One Parameter prevents independent per-image body identities.
    body.betas=torch.nn.Parameter(body.betas.detach()[:1].clone())
    return body,vposer


scope=original.__dict__.copy();scope.update(OUT=OUT,make_models=make_models)
run_source=inspect.getsource(original.run)
assert run_source.count('freeze_betas=True,device=')==1
run_source=run_source.replace('freeze_betas=True,device=','freeze_betas=False,device=')
run_source=run_source.replace("assert np.array_equal(arrays['betas'],np.repeat(beta,8,0))",
    "assert arrays['betas'].shape==(1,10) and np.isfinite(arrays['betas']).all()")
run_source=run_source.replace("sharedkeys={'faces','K_integer_centers'", "sharedkeys={'betas','faces','K_integer_centers'")
exec(compile(run_source,'<same-eight-view-shared-shape-run>','exec'),scope)


def prepare():
    assert not OUT.exists()
    baseline=original.OUT
    old=read(baseline/'plan.json');rows,target=original.data()
    guard=read(original.ROOT/'selected-observation-audit124/wrist-preselection/grass.json')
    byid={r['frame']:r for r in guard['records']}
    assert [r['frame'] for r in rows]==old['frames']
    assert all(not byid[f]['screen_flags'] for f in old['frames'])
    with np.load(baseline/'selected-fit.npz') as z:assert np.array_equal(z['observed_native_keypoints'],target)
    decision=original.REC/'smplifyx-reference-control/official-first-cycle/shared-shape-control/decision.json'
    assert read(decision)['shared_beta_promoted'] is False
    paths=[Path(__file__),Path(original.__file__),Path(original.adapter.__file__),Path(original.entry.__file__),
        H/'smplify_score_contract123.py',H/'smplify_observation_validity124.py',decision,
        baseline/'plan.json',baseline/'result.json',baseline/'selected-fit.npz',
        original.ROOT/'selected-observation-audit124/wrist-preselection/grass.json',
        original.SRC/'data/inputs.json',original.SRC/'data/observations.npz',original.SRC/'selection.json',
        original.SRC/'reference-fit/winner-camera-space.npz',original.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    paths += [original.OFFICIAL/f for f in original.entry.PINNED]
    import smplx
    paths += [Path(smplx.__file__).parent/'body_models.py',Path(smplx.__file__).parent/'lbs.py']
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered',subject='grass',frames=old['frames'],rows=rows,
        reference_for_root_seed=old['reference_for_root_seed'],baseline=str(baseline),
        hypothesis='One shared body identity should be estimated jointly from these8views instead of permanently fixing the unaccepted single-reference32 beta. Test whether original shape optimization improves visible anatomy without losing chest-arm depth.',
        user_basis='User allowed one shared shape to update when unsuitable; later delegated routine research decisions. This is not a per-frame beta model.',
        only_causal_change='Original10D beta is one shared learnable Parameter. Identical8observations,root initialization,shared VPoser latent,per-view root/placement/face/fingers,fullK,original five stages/max30/weights/LBFGS.',
        earlier_control='A single-side63shape update failed front32protection. Do not reuse that rejected beta. This candidate uses original reference32 beta only as seed and jointly fits all8views.',
        forbidden=['No new thresholds','No discarded points','No old optimized pose reuse','No per-frame shape','No cameraK update','No Gaussian launch from loss alone'],
        static_gesture_assumption=True,shared_betas_frozen=False,old_optimized_pose_used=False,
        physical_camera_calibration=False,risks=['Shape can absorb observation or camera errors.','Face,feet or chest-depth may regress.','Shared static pose does not prove actual motion is absent.'],
        budget=dict(candidates=1,selected_views=8,GPU_seconds=600,new_gaussian_steps=0),
        quality_accepted=False,gaussian_admitted=False,source_hashes={str(p):sha(p) for p in paths}))
    (OUT/'runner-interface.txt').write_text(run_source)
    print('SHARED_SHAPE_EIGHT_REGISTERED',flush=True)


def preflight():
    import fcntl,torch
    assert not (OUT/'preflight.json').exists() and not (OUT/'progress.json').exists()
    plan=read(OUT/'plan.json')
    for p,h in plan['source_hashes'].items():assert sha(p)==h,p
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    with np.load(original.SRC/'reference-fit/winner-camera-space.npz') as z:seed=z['betas'].copy()
    checks={};results=[]
    for n in (1,8):
        body,vposer=make_models(n)
        body.reset_params(betas=seed,global_orient=np.tile([0,np.pi,0],(n,1)))
        latent=torch.zeros((1,32),device='cuda',requires_grad=True)
        pose=vposer.decode(latent)['pose_body'].reshape(1,63)
        out=body(body_pose=pose,return_verts=True,return_full_pose=True)
        assert body.betas.shape==(1,10) and out.betas.shape==(n,10)
        assert torch.equal(out.betas,body.betas.expand(n,-1))
        K=torch.tensor(plan['rows'][0]['K_integer_centers'],device='cuda',dtype=torch.float32)
        jc=out.joints+torch.tensor([0,0,3.],device='cuda');u=jc@K.T;uv=u[...,:2]/u[...,2:]
        objective=uv[:,[2,3,4,5,6,7,11,14]].square().sum()/n
        gradient=torch.autograd.grad(objective,body.betas,retain_graph=True)[0]
        prior_gradient=torch.autograd.grad(out.betas.square().sum()/n,body.betas,retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().max()>0
        assert torch.allclose(prior_gradient,2*body.betas,rtol=1e-5,atol=1e-5)
        results.append((uv.detach().cpu().numpy(),gradient.detach().cpu().numpy()))
        if n==8:
            modules=original.entry.load_official(original.OFFICIAL)
            opt,_=modules[3].create_optimizer(list(body.parameters()),optim_type='lbfgsls',lr=1.,maxiters=30)
            checks['one10D_beta_in_original_optimizer']=sum(p is body.betas for group in opt.param_groups for p in group['params'])==1
            checks['single_shared_shape_exact']=True
            checks['original_shape_prior_not_divided_by8']=True
        del body,vposer,out,pose,uv,jc,u
    checks['duplicate_view_projection_parity']=np.max(np.abs(results[0][0]-results[1][0]))<.001
    checks['shared_beta_gradient_duplicate_parity']=np.allclose(results[0][1],results[1][1],rtol=2e-5,atol=.05)
    checks['source_input_unchanged']=all(sha(p)==h for p,h in plan['source_hashes'].items())
    assert all(checks.values()),checks
    checks={key:bool(value) for key,value in checks.items()}
    save(OUT/'preflight.json',dict(updated=now(),checks=checks,optimizer_steps=0,quality_accepted=False,
        native_projection_max_abs_px=float(np.max(np.abs(results[0][0]-results[1][0]))),
        beta_gradient_max_abs=float(np.max(np.abs(results[0][1]-results[1][1])))))
    print('SHARED_SHAPE_GPU_PREFLIGHT_PASSED',checks,flush=True)


def run():
    scope['run']()
    result=read(OUT/'result.json')
    with np.load(OUT/'selected-fit.npz') as z:beta=z['betas'].copy()
    with np.load(original.SRC/'reference-fit/winner-camera-space.npz') as z:seed=z['betas'].copy()
    result.update(shared_beta_parameter_count=10,shared_shape_updated=True,beta_before=seed.tolist(),beta_after=beta.tolist(),
        beta_delta_l2=float(np.linalg.norm(beta-seed)),freeze_betas=False,
        upstream_loss_monitor_optimizer_unchanged=False,
        actual_reuse='Pinned loss,optimizer,monitor,weights and5stages reused via the existing shared-body mean-view interface; only one shared beta Parameter is newly unfrozen.',
        quality_accepted=False,gaussian_admitted=False)
    save(OUT/'result.json',result)


if __name__=='__main__':{'prepare':prepare,'preflight':preflight,'run':run}[sys.argv[1]]()
