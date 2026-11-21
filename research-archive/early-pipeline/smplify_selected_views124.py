"""Small selected-view interface to the pinned SMPLify-X five-stage solver.

One VPoser latent is decoded once and broadcast to selected views. Root pose,
placement, expression and finger parameters remain per-view. No camera
calibration or new optimizer is implemented here. A static local body pose is
an explicit experiment assumption, not the default video reconstruction mode.
"""
import difflib
import inspect
import numpy as np
import torch
import official_smplifyx_entry123 as entry

BASE_SOURCE=entry.adapted_entry_source
BASE_LOAD=entry.load_official


def replace_once(text, old, new):
    assert text.count(old)==1, old
    return text.replace(old,new)


def source(root):
    original,delta=BASE_SOURCE(root)
    code=replace_once(original,
        'joints_conf = keypoint_data[:, :, 2].reshape(1, -1)',
        'joints_conf = keypoint_data[:, :, 2]')
    # Keep the same three existing placement scalars per view in the original
    # optimizer, as already implemented in the single-view placement adapter.
    code=replace_once(code,'                body_params = list(body_model.parameters())',
        '                body_params = list(body_model.parameters())\n                body_params.append(camera.translation)')
    code=replace_once(code,"        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):",
        "        _placement_start = camera.translation.detach().clone()\n        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):\n            with torch.no_grad():\n                camera.translation.copy_(_placement_start)")
    # All selected views are one optimization problem, not a combinatorial
    # search over per-image orientation winners. The original N=1 branch stays.
    code=replace_once(code,'try_both_orient = shoulder_dist.item() < side_view_thsh',
        'try_both_orient = keypoint_data.shape[0] == 1 and shoulder_dist.item() < side_view_thsh')
    compile(code,'<upstream-selected-view-input-adapter>','exec')
    delta+='\n'+''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),
        fromfile='pinned-single-view-interface',tofile='selected-views-shared-body-interface'))
    return code,delta


def mean_loss_class(base):
    class MeanViewsLoss(base):
        def forward(self,body_model_output,*args,**kwargs):
            n=body_model_output.joints.shape[0]
            if kwargs.get('use_vposer') and kwargs.get('pose_embedding') is not None:
                latent=kwargs['pose_embedding']
                assert latent.shape[0]==1
                kwargs['pose_embedding']=latent.expand(n,-1)
            # All terms now have one contribution per view, including the
            # broadcast shared VPoser prior. Divide once, not just the data term.
            return super().forward(body_model_output,*args,**kwargs)/n
    return MeanViewsLoss


def load_official(root):
    utils,prior,fitting,optimizer=BASE_LOAD(root)
    original=fitting.create_loss
    body_loss=mean_loss_class(fitting.SMPLifyLoss)
    camera_loss=mean_loss_class(fitting.SMPLifyCameraInitLoss)
    def factory(loss_type='smplify',**kwargs):
        if loss_type=='smplify':return body_loss(**kwargs)
        if loss_type=='camera_init':return camera_loss(**kwargs)
        return original(loss_type=loss_type,**kwargs)
    fitting.create_loss=factory
    return utils,prior,fitting,optimizer


def make_entry(view_count,root_init):
    assert view_count>=1 and np.asarray(root_init).shape==(view_count,3)
    original=inspect.getsource(entry.fit_one)
    code=replace_once(original,'def fit_one(', 'def fit_selected_views(')
    code=replace_once(code,'torch.zeros((1,3), dtype=dtype, device=dev)',
        'torch.zeros((_view_count,3), dtype=dtype, device=dev)')
    code=replace_once(code,"body_model.reset_params(**({'betas': shared} if shared is not None else {}))",
        "body_model.reset_params(global_orient=_root_init, **({'betas': shared} if shared is not None else {}))")
    code=replace_once(code,'        body.reset_params(**parameters)',
        "        if 'global_orient' not in parameters:\n            parameters['global_orient'] = _root_init\n        body.reset_params(**parameters)")
    scope=entry.__dict__.copy()
    scope.update(_view_count=view_count,_root_init=np.asarray(root_init,dtype=np.float32),
        adapted_entry_source=source,load_official=load_official)
    exec(compile(code,'<pinned-entry-selected-view-camera-size>','exec'),scope)
    return scope['fit_selected_views'], ''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),
        fromfile='existing-upstream-entry',tofile='selected-view-entry'))


def broadcast_body_pose(body,view_count):
    original=body.forward
    def forward(*args,**kwargs):
        pose=kwargs.get('body_pose')
        if pose is not None:
            assert pose.shape==(1,63)
            kwargs['body_pose']=pose.expand(view_count,-1)
        return original(*args,**kwargs)
    body.forward=forward


def verify(official_root,device='cuda'):
    from types import SimpleNamespace
    _,prior,fitting,_=BASE_LOAD(official_root)
    torch.manual_seed(124)
    dtype=torch.float64
    p=torch.randn(1,165,device=device,dtype=dtype).mul(.05).requires_grad_()
    j=torch.randn(1,118,3,device=device,dtype=dtype).requires_grad_()
    latent=torch.randn(1,32,device=device,dtype=dtype).mul(.03).requires_grad_()
    beta=torch.randn(1,10,device=device,dtype=dtype).mul(.02).requires_grad_()
    x=SimpleNamespace(full_pose=p,joints=j,betas=beta,body_pose=p[:,3:66])
    options=dict(interpenetration=False,use_face=False,use_hands=False,
        body_pose_prior=prior.L2Prior(),shape_prior=prior.L2Prior(),angle_prior=prior.SMPLifyAnglePrior(),
        data_weight=.7,body_pose_weight=4.78,shape_weight=5.,bending_prior_weight=15.1526,dtype=dtype)
    original=fitting.SMPLifyLoss(**options).to(device)
    extended=mean_loss_class(fitting.SMPLifyLoss)(**options).to(device)
    gt=torch.randn(1,118,2,device=device,dtype=dtype)
    conf=torch.ones(1,118,device=device,dtype=dtype);jw=conf.clone()
    def camera(t):return t[...,:2]
    def evaluate(module,obj,truth,confidence):
        loss=module(obj,camera,truth,confidence,None,jw,use_vposer=True,pose_embedding=latent)
        grad=torch.autograd.grad(loss,(p,j,latent,beta),retain_graph=True)
        return loss,grad
    one,g0=evaluate(original,x,gt,conf);new,g1=evaluate(extended,x,gt,conf)
    checks=dict(single_view_loss_exact=torch.equal(one,new),
        single_view_all_parameter_gradients_exact=all(torch.equal(a,b) for a,b in zip(g0,g1)))
    n=8;xx=SimpleNamespace(full_pose=p.expand(n,-1),joints=j.expand(n,-1,-1),
        betas=beta.expand(n,-1),body_pose=p[:,3:66].expand(n,-1))
    eight,gg=evaluate(extended,xx,gt.expand(n,-1,-1),conf.expand(n,-1))
    checks['duplicate_views_loss_parity']=torch.allclose(one,eight,rtol=1e-12,atol=1e-10)
    checks['duplicate_views_shared_gradient_parity']=all(torch.allclose(a,b,rtol=1e-12,atol=1e-10) for a,b in zip(g0,gg))
    # Disagreement in a second view must actually influence the shared pose;
    # duplicate-view parity alone could pass an adapter that reads only view0.
    changed=gt.expand(n,-1,-1).clone();changed[1,4,0]+=20
    disagree,gd=evaluate(extended,xx,changed,conf.expand(n,-1))
    checks['other_view_changes_joint_gradient']=not torch.allclose(gg[1],gd[1])
    checks['all_gradients_finite']=all(bool(torch.isfinite(v).all()) for v in gd)
    # All root translations are genuinely parameters of the existing optimizer.
    _,_,_,factory=BASE_LOAD(official_root)
    placement=torch.nn.Parameter(torch.zeros(8,3,device=device))
    opt,_=factory.create_optimizer([placement],optim_type='lbfgsls',lr=1.,maxiters=30)
    checks['original_optimizer_contains_all_placements']=opt.param_groups[0]['params'][0] is placement and placement.numel()==24
    assert all(checks.values()),checks
    return dict(checks={k:bool(v) for k,v in checks.items()},device=device,optimizer_steps=0,
        scope='N=1 loss/gradient equality, duplicate-view normalization and multi-view gradient flow. No physical pose acceptance.',
        single_view_loss=float(one.detach()),eight_duplicate_loss=float(eight.detach()))
