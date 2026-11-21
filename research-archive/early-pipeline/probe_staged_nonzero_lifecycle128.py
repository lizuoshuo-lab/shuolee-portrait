"""Boundary9/end23 wrapper around the frozen native lifecycle fixture.

Only engineering checkpoint/end literals change. The original18000 clock,
target mask radius24, density/reset schedule, model and losses are preserved.
No CUDA work occurs in audit/build_run/patched_run_source or compare.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

H = Path(__file__).resolve().parent
BASE = H/'probe_staged_lifecycle128.py'
BASE_SHA = '0dbab3f629fbd20708195f026fb384f03beecf6d2e7671d225750f12d62f2ed1'
SUBDIR = 'nonzero-boundary9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _base_module():
    assert sha(BASE) == BASE_SHA, 'Frozen original probe changed'
    spec = importlib.util.spec_from_file_location('_frozen_lifecycle_fixture128', BASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patched_run_source():
    """Return (compilable run-function source, bounded adaptation receipt)."""
    assert sha(BASE) == BASE_SHA
    tree = ast.parse(BASE.read_text(encoding='utf-8'))
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    fn = copy.deepcopy(original)
    counts = {'stop_assignment': 0, 'checkpoint_list': 0, 'checkpoint_filename': 0,
              'source_freeze_insertion': 0, 'moment_activity_insertion': 0}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id == 'stop':
            assert ast.unparse(node.value) == "8 if mode == 'stage' else 24"
            node.value = ast.parse("9 if mode == 'stage' else 23", mode='eval').body
            counts['stop_assignment'] += 1
        if node.targets[0].id == 'checkpoint_steps':
            assert ast.unparse(node.value) == "[8, 24] if mode == 'continuous' else [stop]"
            node.value = ast.parse("[9, 23] if mode == 'continuous' else [stop]", mode='eval').body
            counts['checkpoint_list'] += 1
    class CheckpointName(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 'chkpnt8.pth':
                counts['checkpoint_filename'] += 1
                return ast.copy_location(ast.Constant('chkpnt9.pth'), node)
            return node
    fn = CheckpointName().visit(fn)
    body = []
    for statement in fn.body:
        body.append(statement)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == 'life':
            body.append(ast.parse('_freeze_actual_sources(life, locals())').body[0])
            counts['source_freeze_insertion'] += 1
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func) == 'save':
            # The original fixture has exactly one result-JSON save in run().
            body.append(ast.parse('_write_nonzero_moment_activity(dest, model, stop)').body[0])
            counts['moment_activity_insertion'] += 1
    fn.body = body
    ast.fix_missing_locations(fn)
    assert counts == {'stop_assignment': 1, 'checkpoint_list': 1, 'checkpoint_filename': 2,
                      'source_freeze_insertion': 1, 'moment_activity_insertion': 1}, counts
    # Exact option assignments and synthetic target construction must survive.
    def selected(n, wanted):
        return [ast.dump(x, include_attributes=False) for x in ast.walk(n)
                if isinstance(x, ast.Assign) and any(
                    (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'opt')
                    or (isinstance(t, ast.Name) and t.id in wanted) for t in x.targets)]
    assert selected(original, {'masks', 'pixels'}) == selected(fn, {'masks', 'pixels'})
    source = ast.unparse(fn)+'\n'
    compile(source, str(BASE), 'exec')
    receipt = dict(schema='bounded-original-probe-wrapper128-v1', original_probe=str(BASE),
        original_probe_sha256=BASE_SHA, wrapper_sha256=sha(__file__),
        stage_step=9, end_step=23, total_iterations=18000,
        original_optimization_and_targets_AST_exact=True, adaptation_counts=counts,
        patched_run_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        no_GPU_executed_by_construction=True, quality_accepted=False)
    return source, receipt


def _actual_source_paths(local):
    """Bind the actual loaded local implementations, not guessed repo aliases."""
    paths = {BASE, Path(__file__), H/'staged_training_lifecycle128.py',
             H/'native_target_cache124.py', H/'paired_training_background124.py',
             H/'run_official_gaussians123.py', H/'train_demo_body_upstream123.py'}
    for name in ('official', 'args', 'runtime'):
        paths.add(Path(local[name].__file__))
    functions = [local['camera'], local['render'], local['l1_loss'], local['ssim'],
                 local['compute_face_orientation'], local['rotmat_to_unitquat'],
                 local['quat_xyzw_to_wxyz'], local['SMPLXGaussianModel'].densify_and_split,
                 local['SMPLXGaussianModel'].get_covariance]
    for function in functions:
        paths.add(Path(inspect.getsourcefile(function) or inspect.getfile(function)))
    renderer_globals = local['render'].__globals__
    rasterizer = renderer_globals.get('GaussianRasterizer')
    if rasterizer is not None:
        raster_module = sys.modules[rasterizer.__module__]
        paths.add(Path(raster_module.__file__))
        backend = getattr(raster_module, '_C', None)
        if backend is not None:
            paths.add(Path(backend.__file__))
    # Utility functions used by the actual GaussianModel module also belong to
    # the resumed implementation (including scheduler/quaternion transforms).
    for value in vars(local['official']).values():
        if inspect.isfunction(value) and value.__module__.startswith('utils.'):
            paths.add(Path(inspect.getsourcefile(value) or inspect.getfile(value)))
    return sorted({p.resolve() for p in paths}, key=str)


def _freeze_actual_sources(life, local):
    assert sha(BASE) == BASE_SHA
    extra = {str(p): sha(p) for p in _actual_source_paths(local)}
    life.source_hashes.update(extra)
    assert len(life.source_hashes) >= 15, 'Actual local dependency binding incomplete'
    life._verify_sources()


def _write_nonzero_moment_activity(dest, model, stop):
    import torch
    groups = model.optimizer.param_groups
    assert [g['name'] for g in groups] == ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'static_offset']
    rows = []
    for group in groups:
        parameter = group['params'][0]
        state = model.optimizer.state[parameter]
        row = dict(name=group['name'], parameter_elements=parameter.numel(),
                   step=float(state['step']), empty_SH0_group=group['name'] == 'f_rest',
                   exp_avg_nonzero=int(torch.count_nonzero(state['exp_avg'])),
                   exp_avg_sq_nonzero=int(torch.count_nonzero(state['exp_avg_sq'])))
        assert row['step'] > 0
        if group['name'] == 'f_rest':
            assert parameter.numel() == 0
        else:
            assert parameter.numel() > 0 and row['exp_avg_nonzero'] > 0 and row['exp_avg_sq_nonzero'] > 0, row
        rows.append(row)
    path = Path(dest)/f'nonzero-moment-activity-{stop:02d}.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(step=stop, groups=rows,
        nonempty_gaussian_groups_with_actual_nonzero_moments=5,
        shared_offset_moments_actually_nonzero=True, SH0_rest_intentionally_empty=True,
        total_clock_unchanged=18000, quality_accepted=False), indent=2)+'\n', encoding='utf-8')


def build_run():
    """Return (callable, namespace, receipt); callers may compile this same source.

    In particular, zero-update restore wrappers should retain this namespace's
    __file__ and source-freezing hook so the saved stage9 contract is identical.
    Their own audit-wrapper SHA belongs in their separate audit receipt.
    """
    base = _base_module()
    source, receipt = patched_run_source()
    namespace = vars(base).copy()
    namespace.update(OUT=base.OUT/SUBDIR, _freeze_actual_sources=_freeze_actual_sources,
                     _write_nonzero_moment_activity=_write_nonzero_moment_activity)
    assert namespace['__file__'] == str(BASE)
    exec(compile(source, str(BASE), 'exec'), namespace)
    return namespace['run'], namespace, receipt


def compare():
    """CPU numerical/discrete report, with no forced bitwise-pass assertion."""
    import diagnose_lifecycle_numerical128 as diag
    _, namespace, receipt = build_run()
    out = namespace['OUT']
    files = {k: out/v for k, v in {
        'continuous9': 'continuous/chkpnt9.pth', 'staged9': 'staged/chkpnt9.pth',
        'continuous23': 'continuous/actual-state-23.pt', 'staged23': 'staged/actual-state-23.pt',
        'stage_state9': 'staged/actual-state-09.pt',
        'continuous_render23': 'continuous/actual-render-23.pt',
        'staged_render23': 'staged/actual-render-23.pt'}.items()}
    hashes = {str(p): sha(p) for p in files.values()}
    states = {k: diag.load(p) for k, p in files.items()}
    a, b = states['continuous23'], states['staged23']
    prefix = diag.trace_stats(a['counters']['traces'][:9], states['stage_state9']['counters']['traces'])
    trace = diag.trace_stats(a['counters']['traces'], b['counters']['traces'])
    result = dict(schema='nonzero-boundary9-numerical-control128-v1',
        source_hashes=hashes, adaptation_receipt=receipt, cpu_only=True,
        pre_resume9_capture=diag.capture_stats(states['continuous9'][0], states['staged9'][0]),
        pre_resume9_trace=prefix, complete23_trace=trace,
        final23_capture=diag.capture_stats(a['capture'], b['capture']),
        static_offset=diag.tensor_stats(a['static_offset'], b['static_offset']),
        discrete_states_exact={k: diag.same(a[k], b[k]) for k in
            ('python_rng', 'numpy_rng', 'cpu_rng', 'cuda_rng', 'paired_rng', 'paired_calls')},
        render=diag.tensor_stats(states['continuous_render23'], states['staged_render23']),
        strict_final_state_exact=diag.same(a, b),
        numerical_difference_is_not_a_relaxed_pass=True, quality_accepted=False,
        previous_boundary8_strict_failure_preserved=True)
    assert hashes == {str(p): sha(p) for p in files.values()}
    path = out/'comparison23.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(path), sha256=sha(path),
                         strict_final_state_exact=result['strict_final_state_exact'],
                         all_trace_discrete_exact=trace['all_discrete_exact'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['audit', 'continuous', 'stage', 'resume', 'compare'])
    mode = parser.parse_args().mode
    if mode == 'audit':
        _, namespace, receipt = build_run()
        print(json.dumps(dict(receipt, output_root=str(namespace['OUT']),
            namespace_file=namespace['__file__']), indent=2))
    elif mode == 'compare':
        compare()
    else:
        run, _, _ = build_run()
        run(mode)
