"""One original-prior reduction control; no new solver or hand/angle losses.

Relative to the matched VPoser normalized-Laplacian control, change only the
32D body prior reduction: dimension mean -> dimension sum, batch mean kept.
The coefficient stays 0.01, so this is explicitly a 32x effective-weight test,
not a claim that the fork's original mean was an implementation error.
"""
import inspect
import textwrap
import types
import torch
import fit_smplify_vhap_normalized_laplacian123 as base
from run_official_gaussians123 import read, save, sha, now

joint = base.mask.joint
OUT = joint.ROOT / 'smplify-vhap-pose-prior-sum'
PLAN = joint.ROOT / 'smplify-vhap-pose-prior-sum-plan.json'


def make_tracker(out):
    t, rows = base.make_tracker(out)
    assert t.cfg.model.use_vposer and t.pose_embedding.shape == (59, 32)
    assert t.cfg.w.reg_vposer == .01
    original = t.compute_joint_L2_energy
    source = textwrap.dedent(inspect.getsource(original.__func__))
    old = '(self.pose_embedding[timesteps] ** 2).mean() * self.cfg.w.reg_vposer'
    new = '(self.pose_embedding[timesteps] ** 2).sum(dim=-1).mean() * self.cfg.w.reg_vposer'
    assert source.count(old) == 1
    candidate = source.replace(old, new, 1)
    namespace = dict(original.__func__.__globals__)
    exec(compile(candidate, '<single-vposer-reduction-control>', 'exec'), namespace)
    changed = types.MethodType(namespace['compute_joint_L2_energy'], t)
    before = t.pose_embedding.detach().clone()
    try:
        with torch.no_grad():
            t.pose_embedding[0] = torch.linspace(-.5, .5, 32, device='cuda')
        old_value = original([0])
        new_value = changed([0])
        expected_extra = 31 * t.cfg.w.reg_vposer * t.pose_embedding[0].square().mean()
        assert float((new_value - old_value - expected_extra).detach().abs()) < 1e-6
        old_grad, = torch.autograd.grad(old_value, t.pose_embedding)
        new_grad, = torch.autograd.grad(new_value, t.pose_embedding)
        assert torch.allclose(new_grad, old_grad * 32, atol=1e-7, rtol=1e-6)
        assert torch.equal(new_grad[1:], torch.zeros_like(new_grad[1:]))
    finally:
        with torch.no_grad():
            t.pose_embedding.copy_(before)
    assert torch.equal(t.pose_embedding.detach(), before)
    t.compute_joint_L2_energy = changed
    (out / 'original-joint-prior.py').write_text(source)
    (out / 'candidate-joint-prior.py').write_text(candidate)
    save(out / 'prior-interface-check.json', dict(updated=now(), state='passed',
         changed_parameter_family='pose_embedding only', reduction_before='mean over batch and 32 dimensions',
         reduction_after='sum over 32 dimensions, mean over batch', coefficient=.01,
         effective_body_prior_multiplier=32, loss_delta_parity=True, gradient_multiplier_parity=True,
         preflight_parameters_restored=True, optimizer_updates=0, added_bending_prior=False,
         hand_face_root_priors_unchanged=True, production_default_changed=False))
    return t, rows


def main():
    plan = read(PLAN)
    assert plan['wrapper_sha256'] == sha(__file__)
    assert plan['changes'] == ['VPoser latent dimension reduction: mean32 to sum32; batch mean retained.']
    joint.OUT, joint.PLAN, joint.make_tracker = OUT, PLAN, make_tracker
    joint.main()
    for name in ('result.json', 'progress.json'):
        result = read(OUT / name)
        result.update(updated=now(), original_optimizer_loss_scheduler=False,
            original_optimizer_and_scheduler=True, controlled_change=plan['changes'][0],
            body_prior_coefficient=.01, effective_body_prior_multiplier=32,
            angular_prior_added=False, initial_roster_unchanged=True, production_default_changed=False,
            added_mask_term_unchanged=True, quality_accepted=False, gaussian_training_started=False)
        save(OUT / name, result)


if __name__ == '__main__':
    main()
