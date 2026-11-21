"""Read-only reuse of the frozen selection, protection, SO3, and seed audits."""
import inspect
import json
import audit_gaussianavatars_params123 as a
import audit_smplifyx_forward_guard123 as forward


def invoke(module):
    source = inspect.getsource(module.main)
    assert 'guarded-root-sequence' in source
    source = source.replace('guarded-root-sequence', 'warmtail-guarded-sequence')
    scope = dict(module.main.__globals__)
    exec(compile(source, '<warmtail-frozen-selection-and-criteria>', 'exec'), scope)
    scope['main']()


def main():
    source = inspect.getsource(forward.main)
    assert 'forward-guarded-sequence' in source
    source = source.replace('forward-guarded-sequence', 'warmtail-guarded-sequence')
    scope = dict(forward.main.__globals__)
    scope['invoke'] = invoke
    exec(compile(source, '<warmtail-exact-chronological-seeds>', 'exec'), scope)
    scope['main']()
    body = a.REC / 'demo-vhap-initialization/fullbody-smplifyx'
    run = body / 'warmtail-guarded-sequence'
    plan = a.read(body / 'warmtail-guarded-sequence-plan.json')
    rows = []
    for item in plan['schedule']:
        path = run / 'frames' / item['frame'] / 'result.json'
        if not a.lp(path).exists():
            break
        result = a.read(path)
        expected = [0, 1, 2, 3, 4] if item['frame'] == plan['anchor'] else [3, 4]
        assert result['original_stage_indices'] == expected, item['frame']
        assert result['stage_values_unchanged'] is True
        assert result['upstream_solver_loss_optimizer_unchanged'] is True
        rows.append(dict(frame=item['frame'], original_stage_indices=expected,
                         recorded_stage_contract_verified=True))
    a.save(run / 'audit-scope.json', dict(
        selection_and_criteria_reused=True, frames=len(rows), plan_sha256=a.sha(body / 'warmtail-guarded-sequence-plan.json'),
        rows=rows, candidate_modified=False, quality_accepted=False, goal_complete=False))
    print(json.dumps(dict(audit_frames=len(rows), candidate_modified=False)))


if __name__ == '__main__':
    main()
