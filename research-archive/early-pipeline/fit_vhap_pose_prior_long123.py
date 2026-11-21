"""Bounded all59 VHAP long fit using the existing original-repository stages.

Only extend the completed pose-prior-sum trial from 30 to 120 global epochs
and stretch the exponential LR schedule by four. Preserve original sources,
all inputs/losses/parameter groups, and a fresh original SMPLify-X start.
"""
import inspect
import math
import textwrap
import types
import fit_vhap_pose_prior_sum123 as base
from run_official_gaussians123 import read, save, sha, now

joint = base.joint
OUT = joint.ROOT / 'smplify-vhap-pose-prior-long'
PLAN = joint.ROOT / 'smplify-vhap-pose-prior-long-plan.json'
EPOCHS = 120
GLOBAL_UPDATES = 59 * EPOCHS
GAMMA = .9 ** (30 / EPOCHS)


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after, 1)


def make_tracker(out):
    t, rows = base.make_tracker(out)
    assert t.cfg.pipeline.rgb_global_tracking.num_epochs == 30 and len(rows) == 59
    t.cfg.pipeline.rgb_global_tracking.num_epochs = EPOCHS
    original = t.optimize_stage.__func__
    source = textwrap.dedent(inspect.getsource(original))
    candidate = replace_once(source, 'gamma=0.9', f'gamma={GAMMA!r}')
    candidate = replace_once(candidate, '(epoch_i + 1) % 10 == 0', '(epoch_i + 1) % 30 == 0')
    namespace = dict(original.__globals__)
    exec(compile(candidate, '<original-vhap-stage-long-schedule>', 'exec'), namespace)
    t.optimize_stage = types.MethodType(namespace['optimize_stage'], t)
    (out / 'original-optimize-stage.py').write_text(source)
    (out / 'long-optimize-stage.py').write_text(candidate)
    assert math.isclose(GAMMA ** EPOCHS, .9 ** 30, rel_tol=1e-13)
    save(out / 'schedule-check.json', dict(updated=now(), state='passed',
        original_epochs=30, epochs=EPOCHS, original_gamma=.9, gamma=GAMMA,
        checkpoint_epochs=[30, 60, 90, 120], global_updates=GLOBAL_UPDATES,
        total_updates=500 + GLOBAL_UPDATES, lr_scale=.1,
        end_lr_fraction=GAMMA ** EPOCHS, original_end_lr_fraction=.9 ** 30,
        original_optimizer_and_per_step_update=True, inputs_and_losses_unchanged=True,
        root_rotation_representation_unchanged=True, bending_prior_added=False,
        optimizer_updates=0, quality_accepted=False))
    return t, rows


def main():
    plan = read(PLAN)
    assert plan['wrapper_sha256'] == sha(__file__)
    assert plan['epochs'] == EPOCHS and plan['updates'] == 500 + GLOBAL_UPDATES
    assert plan['fresh_initialization'] and not OUT.exists()
    joint.OUT, joint.PLAN, joint.make_tracker = OUT, PLAN, make_tracker
    # Adapt only the existing runner's budget assertions; preserve it on disk.
    source = textwrap.dedent(inspect.getsource(joint.main))
    changed = replace_once(source, 'rgb_global_tracking.num_epochs==30', 'rgb_global_tracking.num_epochs==120')
    assert changed.count('>1800') == 2
    changed = changed.replace('>1800', '>2400').replace('Joint trial30-minute budget reached.', 'Long trial40-minute budget reached.')
    changed = replace_once(changed, 't.global_step==1770+init_updates', 't.global_step==7080+init_updates')
    changed = replace_once(changed, 'global_refinement_updates=1770', 'global_refinement_updates=7080')
    changed = replace_once(changed, 'original_optimizer_loss_scheduler=True', 'original_optimizer_loss_scheduler=False')
    namespace = dict(joint.main.__globals__)
    exec(compile(changed, '<existing-vhap-runner-long-budget>', 'exec'), namespace)
    namespace['main']()
    (OUT / 'original-runner.py').write_text(source)
    (OUT / 'long-runner.py').write_text(changed)
    for name in ('result.json', 'progress.json'):
        result = read(OUT / name)
        result.update(updated=now(), epochs=EPOCHS, gamma=GAMMA,
            original_optimizer_and_per_step_update=True, same_losses_as='smplify-vhap-pose-prior-sum',
            body_prior_coefficient=.01, effective_body_prior_multiplier=32,
            controlled_change='30 to 120 global epochs with proportionally stretched LR decay',
            fresh_initialization=True, optimizer_state_reused=False,
            angular_prior_added=False, root_rotation_representation_unchanged=True,
            production_default_changed=False, quality_accepted=False, gaussian_training_started=False)
        save(OUT / name, result)


if __name__ == '__main__':
    main()
