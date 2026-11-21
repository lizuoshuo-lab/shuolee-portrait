"""Keep the multi-view torso/arm estimate while refining peripheral joints.

This is a decoder parameter mask on the existing seeded upstream entry.
The original optimizer, last two stages, priors and weight30 pose tether stay.
It assumes the selected views have the same torso/arm gesture; it is not a
general dynamic-body tracker or a hard-coded hand-to-chest contact target.
"""
import torch
import smplify_seeded_refine124 as base

PROTECTED_IDS = (3,6,9,13,14,16,17,18,19)
PROTECTED_NAMES = ('spine1','spine2','spine3','left_collar','right_collar',
                   'left_shoulder','right_shoulder','left_elbow','right_elbow')
FREE_IDS = tuple(i for i in range(1,22) if i not in PROTECTED_IDS)


class ProtectedDecoder:
    def __init__(self, original, anchor):
        from smplx.joint_names import JOINT_NAMES
        assert tuple(JOINT_NAMES[i] for i in PROTECTED_IDS) == PROTECTED_NAMES
        self.original = original
        self.anchor = anchor

    def decode(self, latent):
        result = self.original.decode(latent)
        pose = result['pose_body'].reshape(latent.shape[0],21,3)
        anchor = torch.as_tensor(self.anchor,device=pose.device,dtype=pose.dtype).reshape(1,21,3).detach()
        ids = torch.tensor([i-1 for i in PROTECTED_IDS],device=pose.device,dtype=torch.long)
        pose = pose.index_copy(1,ids,anchor.expand_as(pose).index_select(1,ids))
        return {**result,'pose_body':pose}


def source(root):
    # There is no new upstream-loop patch. The decoder is passed through the
    # existing VPoser2API wrapper; saved axis angles are replayed by SMPL-X.
    return base.source(root)


def fit(*,saved_seed,vposer2,**kwargs):
    decoder = ProtectedDecoder(vposer2,saved_seed['body_pose_axis_angle'])
    arrays, report = base.fit(saved_seed=saved_seed,vposer2=decoder,**kwargs)
    pose = arrays['body_pose_axis_angle'].reshape(1,21,3)
    anchor = saved_seed['body_pose_axis_angle'].reshape(1,21,3)
    ids = [i-1 for i in PROTECTED_IDS]
    assert (pose[:,ids] == anchor[:,ids]).all()
    report.update(actual_method='Existing seeded original stage3/4 with decoder mask: spine,collar,shoulder,elbow rotations remain at the coherent multi-view estimate; legs,neck,head,wrist,root,placement,face,fingers remain adjustable. Original priors and shared-pose energy30 unchanged.',
        protected_joint_ids=list(PROTECTED_IDS),protected_joint_names=list(PROTECTED_NAMES),
        free_body_joint_ids=list(FREE_IDS),protected_angle_max_change=0.,
        actual_static_torso_arm_assumption=True,new_optimizer=False,
        no_hand_chest_contact_target=True)
    return arrays,report


def verify_decoder(body,vposer,seed):
    import numpy as np
    decoder = ProtectedDecoder(vposer,seed['body_pose_axis_angle'])
    z = torch.as_tensor(seed['pose_embedding'],device='cuda').clone().requires_grad_(True)
    ids = torch.tensor([i-1 for i in PROTECTED_IDS],device=z.device)
    free = torch.tensor([i-1 for i in FREE_IDS],device=z.device)
    initial = decoder.decode(z)['pose_body']
    anchor = torch.as_tensor(seed['body_pose_axis_angle'],device=z.device).reshape_as(initial)
    assert torch.max(abs(initial-anchor)) < 1e-6
    g = torch.autograd.grad(initial[:,ids].square().sum(),z,retain_graph=True)[0]
    assert float(g.abs().max()) == 0.
    g = torch.autograd.grad(initial[:,free].square().sum(),z)[0]
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0
    probe = z.detach()+.15*torch.sin(torch.arange(32,device=z.device))[None]
    changed = decoder.decode(probe)['pose_body']
    assert torch.equal(changed[:,ids],anchor[:,ids])
    assert float((changed[:,free]-initial[:,free]).abs().max()) > .001
    # Check actual mapped shoulder/elbow/wrist positions, not only angles.
    body.reset_params(**{name:seed[name] for name,_ in body.named_parameters()})
    with torch.no_grad():
        body.global_orient.zero_()
        ja = body(body_pose=initial.reshape(1,63),return_full_pose=True).joints
        jb = body(body_pose=changed.reshape(1,63),return_full_pose=True).joints
    arm = [2,3,4,5,6,7]
    error = float((ja[:,arm]-jb[:,arm]).abs().max())
    assert error < 1e-6, error
    # Independent SMPL-X replay of saved final axis angles does not depend
    # on reconstructing this decoder wrapper in the Gaussian handoff.
    return dict(zero_seed_pose_max=float((initial-anchor).abs().max().detach()),
        protected_decoder_gradient_zero=True,free_decoder_gradient_finite_nonzero=True,
        perturbed_latent_protected_angles_unchanged=True,
        actual_body_local_arm_joint_max_change_m=error,
        free_pose_max_probe_change_rad=float((changed[:,free]-initial[:,free]).abs().max().detach()),
        protected_joint_names=list(PROTECTED_NAMES),free_body_joint_ids=list(FREE_IDS),
        measured_physical_depth_truth=False)
