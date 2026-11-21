"""One-variable wrapper:10x only the global direct-body Adam group LR.

Reuse the complete direct63 driver and original VHAP optimizer/scheduler. This
module never edits their source and does not authorize its own execution.
"""
from pathlib import Path
import copy
import time
import types
import torch
import fit_smplify_vhap_direct_body123 as direct


OUT = direct.ROOT / 'smplify-vhap-body-lr'
PLAN = direct.ROOT / 'smplify-vhap-body-lr-plan.json'
MULTIPLIER = 10.


def json_option(value):
    if isinstance(value, (tuple, list)):
        return [json_option(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unexpected non-scalar Adam group option: ' + repr(type(value)))


def checked_optimizer(self, original, params, lr_scale, checks):
    """Split one existing parameter, before scheduler construction or updates."""
    assert lr_scale in (1., .1), 'Unexpected stage scale in the frozen two-stage driver.'
    flat = [p for group in params.values() for p in group]
    assert len(flat) == len({id(p) for p in flat}), 'Input parameter duplicated.'
    body = self.body_pose
    assert body is not None and body.shape == (59, 63)
    assert sum(p is body for p in flat) == 1
    # Full CPU clones verify values without consuming RNG or modifying a tensor.
    # In particular, the texture parameter is not sampled for this check.
    snapshots = {id(p): p.detach().cpu().clone() for p in flat}
    optimizer = original(params, lr_scale=lr_scale)
    assert type(optimizer) is torch.optim.Adam
    assert not optimizer.state, 'Original optimizer must be new, with empty state.'
    before = {id(p): (float(group['lr']), copy.deepcopy({k: v for k, v in group.items() if k != 'params'}))
              for group in optimizer.param_groups for p in group['params']}
    initial_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(initial_ids) == len(set(initial_ids)) == len(flat)
    assert set(initial_ids) == set(snapshots)
    body_groups = [group for group in optimizer.param_groups if any(p is body for p in group['params'])]
    assert len(body_groups) == 1
    source = body_groups[0]
    assert float(source['lr']) == float(self.cfg.lr.base * lr_scale)
    expected_body_lr = float(source['lr'])
    if lr_scale == .1:
        new_group = {k: copy.deepcopy(v) for k, v in source.items() if k != 'params'}
        new_group['params'] = [body]
        new_group['lr'] = float(source['lr']) * MULTIPLIER
        source['params'] = [p for p in source['params'] if p is not body]
        optimizer.add_param_group(new_group)
        expected_body_lr *= MULTIPLIER
    after_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(after_ids) == len(set(after_ids)) == len(flat)
    assert set(after_ids) == set(initial_ids)
    assert not optimizer.state, 'Group split must precede all optimizer updates.'
    rows = []
    for group_index, group in enumerate(optimizer.param_groups):
        options = {k: v for k, v in group.items() if k != 'params'}
        for parameter in group['params']:
            old_lr, old_options = before[id(parameter)]
            is_body = parameter is body
            assert float(group['lr']) == (expected_body_lr if is_body else old_lr)
            assert {k: v for k, v in options.items() if k != 'lr'} == {
                k: v for k, v in old_options.items() if k != 'lr'}
            assert torch.equal(parameter.detach().cpu(), snapshots[id(parameter)]), 'Parameter value changed while regrouping.'
            aliases = [key for key, values in params.items() if any(p is parameter for p in values)]
            rows.append(dict(parameter_groups=aliases, shape=list(parameter.shape),
                             dtype=str(parameter.dtype), is_body_pose=is_body,
                             optimizer_group=group_index, lr_before=old_lr,
                             lr_after=float(group['lr']),
                             options={k: json_option(v) for k, v in options.items()},
                             values_unchanged=True))
    receipt = dict(stage='rgb_init_offset' if lr_scale == 1. else 'rgb_global_tracking',
                   lr_scale=lr_scale, multiplier=MULTIPLIER if lr_scale == .1 else 1.,
                   fresh_adam=True, optimizer_state_entries=len(optimizer.state),
                   each_parameter_present_exactly_once=True, all_tensor_values_unchanged=True,
                   all_nonbody_learning_rates_unchanged=True,
                   all_nonlr_group_options_unchanged=True,
                   body_lr_before=before[id(body)][0], body_lr_after=expected_body_lr,
                   scheduler_not_yet_constructed=True, parameter_receipts=rows)
    checks.append(receipt)
    direct.save(OUT / 'optimizer-group-checks.json', dict(updated=direct.now(),
                original_configure_optimizer_source=str(Path(direct.REPO) / 'vhap/model/tracker_smplx.py'),
                original_driver_source=str(Path(direct.__file__)),
                wrapper_sha256=direct.sha(__file__), calls=checks,
                only_global_body_lr_multiplier=MULTIPLIER, optimizer_updates_at_each_check=0,
                quality_accepted=False, gaussian_training_started=False))
    return optimizer


def main():
    # The underlying driver continues to verify its own source_sha256. The
    # wrapper has a separate, mandatory bound input entry in the parent's plan.
    plan = direct.read(PLAN)
    wrapper = Path(__file__).resolve()
    assert any(Path(path).resolve() == wrapper and digest == direct.sha(__file__)
               for path, digest in plan['input_hashes'].items()), 'Bind wrapper SHA in input_hashes.'
    assert plan['source_sha256'] == direct.sha(direct.__file__)
    original_make = direct.base.make_tracker
    original_out, original_plan = direct.OUT, direct.PLAN
    checks = []

    def make_tracker(output):
        tracker, rows = original_make(output)
        original_optimizer = tracker.configure_optimizer

        def configure(self, params, lr_scale=1.):
            return checked_optimizer(self, original_optimizer, params, lr_scale, checks)

        tracker.configure_optimizer = types.MethodType(configure, tracker)
        return tracker, rows

    began = time.monotonic()
    direct.OUT, direct.PLAN = OUT, PLAN
    direct.base.make_tracker = make_tracker
    try:
        direct.main()
        assert [x['stage'] for x in checks] == ['rgb_init_offset', 'rgb_global_tracking']
        assert [x['body_lr_after'] for x in checks] == [.005, .005]
        result = direct.read(OUT / 'result.json')
        assert result['steps'] == 2270 and result['frames'] == 59
        result.update(only_global_body_lr_multiplier=MULTIPLIER,
                      comparison_control='smplify-vhap-direct-body',
                      causal_variable='Only global-stage body_pose Adam group LR is multiplied by10.',
                      other_parameter_lrs_unchanged=True,
                      init_stage_lrs_unchanged=True, scheduler_gamma_unchanged=.9,
                      optimizer_group_check_sha256=direct.sha(OUT / 'optimizer-group-checks.json'),
                      wrapper_sha256=direct.sha(__file__),
                      wrapper_elapsed_seconds=time.monotonic() - began,
                      quality_accepted=False, gaussian_training_started=False)
        direct.save(OUT / 'result.json', result)
        direct.save(OUT / 'progress.json', dict(**result, pid=None))
        print(direct.json.dumps(result), flush=True)
    finally:
        direct.base.make_tracker = original_make
        direct.OUT, direct.PLAN = original_out, original_plan


if __name__ == '__main__':
    main()
