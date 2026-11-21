"""Small temporal-loss extension to pinned SMPLify-X; upstream solver unchanged.

Rotation matrices avoid axis-angle branch jumps. A neighbour is a detached
estimate from this same run, never independent 3D truth or an initialization.
"""
import torch
from smplx.lbs import batch_rodrigues


def rotation_energy(full_pose, neighbour, dt, nominal_dt, body_weight=30., root_weight=30.):
    if dt <= 0 or nominal_dt <= 0:
        raise ValueError('Positive observed timestamp intervals are required.')
    rotations = batch_rodrigues(full_pose[:, :66].reshape(-1, 3)).reshape(-1,22,3,3)
    target = batch_rodrigues(neighbour[:, :66].reshape(-1,3)).reshape(-1,22,3,3).detach()
    # 0.5||Ra-Rb||_F^2 approximates squared relative angle near zero;
    # the exact periodic chordal penalty stays finite at pi and 2pi.
    per_joint = .5*(rotations-target).square().sum(dim=(-2,-1))
    factor = (nominal_dt/dt)**2
    return factor*(body_weight*per_joint[:,1:].sum()+root_weight*per_joint[:,0].sum())


def extend_loss_class(base_class, neighbour, dt, nominal_dt, receipt):
    class TemporalLoss(base_class):
        def forward(self, body_model_output, *args, **kwargs):
            original = super().forward(body_model_output, *args, **kwargs)
            if neighbour is None:
                return original
            pose = body_model_output.full_pose
            target = torch.as_tensor(neighbour, device=pose.device, dtype=pose.dtype)
            temporal = rotation_energy(pose, target, dt, nominal_dt)
            if 'gradient_checked' not in receipt:
                grad = torch.autograd.grad(temporal, pose, retain_graph=True)[0]
                receipt.update(gradient_checked=True,
                    initial_temporal_gradient_max=float(grad.abs().max().detach()),
                    temporal_gradient_finite=bool(torch.isfinite(grad).all()))
            receipt['calls'] = receipt.get('calls',0)+1
            receipt['_last_original'] = original.detach()
            receipt['_last_temporal'] = temporal.detach()
            return original + temporal
    return TemporalLoss


def install_loss_factory(fitting, neighbour, dt, nominal_dt, receipt):
    original_factory = fitting.create_loss
    extended = extend_loss_class(fitting.SMPLifyLoss, neighbour, dt, nominal_dt, receipt)
    # Keep the upstream global SMPLifyLoss name: its explicit super(Class,self)
    # otherwise recurses when that global is replaced with a subclass.
    def factory(loss_type='smplify', **kwargs):
        if loss_type == 'smplify':
            return extended(**kwargs)
        return original_factory(loss_type=loss_type, **kwargs)
    fitting.create_loss = factory


def verify():
    checks = {}
    p = torch.zeros((1,165), dtype=torch.float64, requires_grad=True)
    q = torch.zeros_like(p).detach()
    e = rotation_energy(p,q,.2,.2)
    g = torch.autograd.grad(e,p)[0]
    checks['identical_zero'] = float(e)==0 and float(g.abs().max())==0
    q[0,7] = 2*torch.pi
    checks['two_pi_same_rotation'] = float(rotation_energy(p,q,.2,.2))<1e-12
    q[0,7] = .6
    a=rotation_energy(p,q,.2,.2);b=rotation_energy(p,q,.4,.2)
    checks['actual_dt_scaling'] = abs(float(a/b)-4)<1e-12
    g=torch.autograd.grad(a,p)[0]
    checks['finite_nonzero_gradient'] = bool(torch.isfinite(g).all()) and float(g.abs().max())>0
    checks['face_fingers_not_temporally_forced'] = float(g[:,66:].abs().max())==0
    # Verify the original objective is retained exactly at identical poses.
    import official_smplifyx_entry123 as entry
    from pathlib import Path
    root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official')
    from types import SimpleNamespace
    _,prior,fitting,_=entry.load_official(root)
    kwargs=dict(interpenetration=False,use_face=False,use_hands=False,
        body_pose_prior=prior.L2Prior(),shape_prior=prior.L2Prior(),angle_prior=prior.SMPLifyAnglePrior(),
        data_weight=.5,body_pose_weight=4.78,shape_weight=5,bending_prior_weight=15.1526)
    torch.manual_seed(124)
    pose=(torch.randn(1,165)*.03).requires_grad_()
    obj=SimpleNamespace(full_pose=pose,joints=torch.randn(1,118,3),betas=torch.zeros(1,10))
    args=(lambda j:j[...,:2],torch.randn(1,118,2),torch.ones(1,118),None,torch.ones(1,118))
    extra=dict(use_vposer=True,pose_embedding=torch.ones(1,32)*.2)
    original=fitting.SMPLifyLoss(**kwargs)
    record={};adapted=extend_loss_class(fitting.SMPLifyLoss,pose.detach().clone(),.2,.2,record)(**kwargs)
    checks['original_loss_parity'] = torch.equal(original(obj,*args,**extra),adapted(obj,*args,**extra))
    null=extend_loss_class(fitting.SMPLifyLoss,None,.2,.2,{})(**kwargs)
    checks['anchor_original_loss_parity'] = torch.equal(original(obj,*args,**extra),null(obj,*args,**extra))
    install_loss_factory(fitting,pose.detach().clone(),.2,.2,{})
    actual = fitting.create_loss('smplify',**kwargs)
    checks['actual_factory_original_loss_parity'] = torch.equal(original(obj,*args,**extra),actual(obj,*args,**extra))
    camera_loss = fitting.create_loss('camera_init',init_joints_idxs=torch.tensor([2,5,9,12],dtype=torch.long))
    checks['camera_init_class_unchanged'] = isinstance(camera_loss,fitting.SMPLifyCameraInitLoss)
    assert all(checks.values()),checks
    return checks
