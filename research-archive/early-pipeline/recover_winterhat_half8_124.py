"""One explicit, non-exact recovery of the timed-out winterhat1000 candidate.

``audit`` uses CPU only and writes one independent preparation receipt beside
this helper. ``train`` is deliberately separate: the owner must review this
helper before launching it. The pinned adapter/repo/plan/old1000 stay unchanged.
"""
import ast
import copy
import difflib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from datetime import datetime

import numpy as np
from PIL import Image
import cross_subject_sh0124 as gs

OUT = gs.ROOT / 'winterhat-half8-paired18k124'
RUN = OUT / 'gaussian-run'
SNAPSHOT = RUN / 'step-001000'
CHECKPOINT = SNAPSHOT / 'checkpoint.pt'
AUDIT_PATH = Path(__file__).with_name('recover_winterhat_half8_124_cpu_audit.json')
RUNTIME = OUT / 'recovery124'
SEED = 1241000
LIMIT = 18000
TOTAL_SECONDS = 2400
CHECKPOINT_SHA = 'e89d99a8f0fdbdcbbb6fb388c733d7fc0366cae55549a1f5475f2c7cf6ef4569'
GROUP_NAMES = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'static_offset']
PARAM_CAPTURE_INDICES = [1, 2, 3, 6, 4, 5]


def verify_inputs():
    plan = gs.read(OUT / 'plan.json')
    for path, digest in plan['source_hashes'].items():
        assert gs.sha(path) == digest, path
    assert gs.sha(CHECKPOINT) == CHECKPOINT_SHA
    receipt = gs.read(OUT / 'smplify-gaussian-interface-receipt.json')
    assert gs.sha(OUT / 'fit/fitted.npz') == receipt['fit_sha256']
    assert gs.sha(OUT / 'data/training-inputs.json') == receipt['training_inputs_sha256']
    sampling = gs.read(SNAPSHOT / 'sampling.json')
    assert sampling['counts'] == [125] * 8 and sampling['total_samples'] == 1000
    background = gs.read(SNAPSHOT / 'background-receipt.json')
    assert background['training_calls'] == 1000 and background['seed'] == 124
    assert gs.sha(SNAPSHOT / 'world.ply') == gs.read(SNAPSHOT / 'result.json')['world_ply_sha256']
    first_launch = gs.read(OUT / 'launch.json')
    failure = gs.read(OUT / 'train-failure.json')
    assert 'Review timeout; checkpoint retained' in failure['error']
    prior_seconds = (datetime.fromisoformat(failure['updated']) -
                     datetime.fromisoformat(first_launch['updated'])).total_seconds()
    assert 600 < prior_seconds < TOTAL_SECONDS
    return plan, sampling, background, prior_seconds


def recursive_equal(a, b, torch):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(recursive_equal(a[k], b[k], torch) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(recursive_equal(x, y, torch) for x, y in zip(a, b))
    return a == b


def inspect_cpu_checkpoint(torch):
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert set(saved) == {'step', 'capture', 'static_offset'} and saved['step'] == 1000
    cap = saved['capture']
    assert len(cap) == 14 and cap[0] == 0 and cap[13] == 1.
    assert cap[1].shape == (35514, 3) and cap[7].shape == (35514,)
    assert cap[8].shape == (20908,) and saved['static_offset'].shape == (10475, 3)
    assert all(torch.isfinite(cap[i]).all() for i in range(1, 12))
    assert torch.isfinite(saved['static_offset']).all()
    assert all(cap[i].grad is None for i in PARAM_CAPTURE_INDICES)
    assert saved['static_offset'].grad is None
    assert torch.equal(torch.bincount(cap[7], minlength=20908).to(cap[8].dtype), cap[8])
    state = cap[12]
    assert [g['name'] for g in state['param_groups']] == GROUP_NAMES
    params = [cap[i] for i in PARAM_CAPTURE_INDICES] + [torch.nn.Parameter(saved['static_offset'].clone())]
    opt = torch.optim.Adam([dict(params=[p], lr=g['lr'], name=g['name'])
                            for p, g in zip(params, state['param_groups'])], lr=0., eps=1e-15)
    opt.load_state_dict(state)
    assert recursive_equal(state, opt.state_dict(), torch)
    assert [int(state['state'][g['params'][0]]['step']) for g in state['param_groups']] == [988] * 6 + [999]
    return saved


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def patched_source(prior_seconds, sampling):
    original = (OUT / 'adapter.py').read_text()
    source = original
    edits = [
        ('SNAPSHOTS = [1000,10000,18000]', 'SNAPSHOTS = [10000,18000]'),
        ('run.mkdir(exist_ok=False)', "assert run.is_dir() and not (run/'step-010000').exists() and not (run/'step-018000').exists()"),
        ('started = time.monotonic()', 'started = RECOVERY_START - PRIOR_SECONDS'),
        ('random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)',
         'random.seed(RECOVERY_SEED); np.random.seed(RECOVERY_SEED); torch.manual_seed(RECOVERY_SEED); torch.cuda.manual_seed_all(RECOVERY_SEED)'),
        ('counts = np.zeros(8, dtype=int)', f'counts = np.asarray({sampling["counts"]!r}, dtype=int)'),
        ('saved = []', 'saved = [1000]'),
        ("save(run/'offset-initialization.json', offset_receipt)", "save(RECOVERY_DIR/'offset-initialization.json', offset_receipt)"),
        ('snapshot_step=step, optimizer_updates=step-1, configured_schedule_steps=opt.iterations,',
         'snapshot_step=step, optimizer_updates=step-2, configured_schedule_steps=opt.iterations, recovery_non_exact=True, discarded_step1000_update=True,'),
        ("(run/'upstream-training-function.py').write_text(fn_source+'\\n')", "assert (run/'upstream-training-function.py').read_text()==fn_source+'\\n'"),
        ("save(run/'training-cameras.json', dict(records=records,", "save(RECOVERY_DIR/'training-cameras.json', dict(records=records,"),
        ("save(run/'effective-optimization.json', vars(opt))", "save(RECOVERY_DIR/'effective-optimization.json', vars(opt))"),
        ("save(run/'provenance.json', dict(updated=now(), upstream_commit=COMMIT,", "save(RECOVERY_DIR/'provenance.json', dict(updated=now(), upstream_commit=COMMIT,"),
        ('from_scratch=True, input_mesh_sha256=receipt[\'fit_sha256\']', "from_scratch=False, recovered_same_candidate=True, recovery_non_exact=True, recovery_seed=RECOVERY_SEED, checkpoint_sha256=RECOVERY_CHECKPOINT_SHA, input_mesh_sha256=receipt['fit_sha256']"),
        ('full_official_reproduction=False, old_checkpoint_loaded=False)', 'full_official_reproduction=False, old_checkpoint_loaded=True)'),
        ("status('training_starting', 0)", "status('recovery_starting', 1000)"),
        ("namespace['training'](dataset,opt,pipe,[],SNAPSHOTS,[],None,-1)",
         "recovery_context = locals()\n        original_restore = Body.restore\n        def restore_same_candidate(self, model_args, training_args):\n            recovery_restore(self, model_args, training_args, original_restore, recovery_context)\n        Body.restore = restore_same_candidate\n        namespace['torch'] = checkpoint_torch_proxy(torch)\n        namespace['training'](dataset,opt,pipe,[],SNAPSHOTS,[],RECOVERY_TOKEN,-1)"),
    ]
    for old, new in edits:
        source = replace_once(source, old, new)
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert 'optimizer_updates=step-2' in source
    assert not (RUN / 'step-010000').exists() and not (RUN / 'step-018000').exists()
    compile(source, str(RUNTIME / 'executed-adapter.py'), 'exec')
    diff = ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
        fromfile='pinned-unchanged-adapter.py', tofile='explicit-recovery-interface.py'))
    return source, diff


def audit():
    import torch
    assert not torch.cuda.is_initialized()
    plan, sampling, background, prior = verify_inputs()
    saved = inspect_cpu_checkpoint(torch)
    source, diff = patched_source(prior, sampling)
    result = dict(updated=gs.now(), mode='CPU_preparation_only', passed=True,
        helper_sha256=gs.sha(__file__), pinned_adapter_sha256=gs.sha(OUT / 'adapter.py'),
        checkpoint_sha256=gs.sha(CHECKPOINT), checkpoint_step=1000,
        points=len(saved['capture'][1]), groups=GROUP_NAMES, adam_state_cpu_remap_exact=True,
        source_hash_count=len(plan['source_hashes']), all_source_hashes_match=True,
        saved_counts=sampling['counts'], saved_background_calls=background['training_calls'],
        prior_launch_to_failure_seconds=prior, aggregate_budget_seconds=TOTAL_SECONDS,
        recovery_seed=SEED, starts_at_step=1001, final_shared_offset_optimizer_updates=17998,
        exact_continuation=False, discarded_gradient_update_step=1000, discarded_densification_statistics_step=1000,
        original_adapter_repo_plan_and1000_untouched=True, recovery_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        recovery_diff=diff, cuda_initialized=torch.cuda.is_initialized(),
        gpu_preflight_passed=False, training_started=False)
    gs.save(AUDIT_PATH, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'recovery_diff'}, indent=2), flush=True)


class CheckpointTorchProxy:
    def __init__(self, torch, cap, token):
        self.real, self.cap, self.token = torch, cap, token
        self.loads = 0

    def __getattr__(self, key):
        return getattr(self.real, key)

    def load(self, path, *args, **kwargs):
        assert path is self.token and not args and not kwargs
        self.loads += 1
        assert self.loads == 1
        return self.cap, 1000


def restore_and_preflight(model, model_args, training_args, original_restore, ctx, saved, cpu_saved):
    """Called exactly once by the pinned upstream checkpoint branch."""
    import torch
    assert model_args is saved['capture']
    with torch.no_grad():
        model.static_offset.copy_(saved['static_offset'])
    original_restore(model, model_args, training_args)
    assert all(p.grad is None for g in model.optimizer.param_groups for p in g['params'])
    before = copy.deepcopy(model.capture())
    before_offset = model.static_offset.detach().clone()
    assert recursive_equal(before, cpu_saved['capture'], torch)
    assert recursive_equal(before_offset, cpu_saved['static_offset'], torch)
    for i, name in zip(PARAM_CAPTURE_INDICES, GROUP_NAMES):
        group = next(g for g in model.optimizer.param_groups if g['name'] == name)
        assert group['params'][0] is model.capture()[i]
    assert model.optimizer.param_groups[-1]['params'][0] is model.static_offset
    assert model.optimizer.param_groups[-1]['name'] == 'static_offset'
    old_timestep = int(model.timestep)
    counts_before = ctx['counts'].copy()
    generator_before = ctx['paired_background'].generator.get_state().clone()
    calls_before = ctx['paired_background'].calls
    results = []
    with torch.no_grad():
        for cam, row in zip(ctx['cams'], ctx['rows']):
            model.select_mesh_by_timestep(cam.timestep)
            output = ctx['render'](cam, model, ctx['pipe'], ctx['background'])['render']
            png = torch.as_tensor(np.array(Image.open(SNAPSHOT / (row['frame'] + '_gaussian.png')).convert('RGB')),
                device=output.device, dtype=output.dtype).permute(2, 0, 1) / 255.
            delta = (output.clamp(0, 1) - png).abs()
            value = float(delta.max())
            assert value <= 1 / 255. + 1.e-6, (row['frame'], value)
            results.append(dict(frame=row['frame'], max_abs_vs_saved_8bit=value, mean_abs=float(delta.mean())))
            del output, png, delta
        model.select_mesh_by_timestep(0)
        plain = ctx['official'].GaussianModel(0)
        plain.load_ply(str(SNAPSHOT / 'world.ply'))
        plain.active_sh_degree = model.active_sh_degree
        delta = (ctx['render'](ctx['cams'][0], model, ctx['pipe'], ctx['background'])['render'] -
                 ctx['render'](ctx['cams'][0], plain, ctx['pipe'], ctx['background'])['render']).abs()
        parity = dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()))
        assert parity['mean_abs'] < 2.e-6 and parity['max_abs'] < .005, parity
        del plain, delta
        model.select_mesh_by_timestep(old_timestep)
        geometry = ctx['clothing_geometry'](model)
        assert geometry['protected_face_motion_m'] == geometry['protected_face_gradient'] == 0.
        assert geometry['new_relative_normal_reversals'] == 0
        assert geometry['minimum_area_ratio'] > .1 and geometry['max_added_offset_m'] < .02
    assert recursive_equal(before, model.capture(), torch)
    assert recursive_equal(before_offset, model.static_offset, torch)
    assert np.array_equal(counts_before, ctx['counts'])
    assert torch.equal(generator_before, ctx['paired_background'].generator.get_state())
    assert calls_before == ctx['paired_background'].calls == 0
    assert all(p.grad is None for g in model.optimizer.param_groups for p in g['params'])
    background = ctx['paired_background']
    for _ in range(1000):
        torch.rand(3, device='cuda', generator=background.generator)
    background.calls = 1000
    background.first_colors = gs.read(SNAPSHOT / 'background-receipt.json')['first_colors']
    # A declared deterministic branch, not the absent original RNG state.
    gs.random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        model.select_mesh_by_timestep(old_timestep)
    gs.save(RUNTIME / 'gpu-zero-update-preflight.json', dict(updated=gs.now(), passed=True,
        checkpoint_sha256=gs.sha(CHECKPOINT), restore_calls=1, optimizer_updates=0,
        all_model_and_adam_tensors_exact=True, parameter_group_references_verified=True,
        eight_source_render_quantization=results, native_world_ply_parity=parity, geometry=geometry,
        original_thresholds_unchanged=True, old_timestep_restored=old_timestep,
        training_counts_unchanged=counts_before.tolist(), background_calls_restored=background.calls,
        background_generator_seed=124, background_draws_skipped=1000,
        global_recovery_seed=SEED, exact_continuation=False,
        missing_step1000_gradient_and_statistics_discarded=True, quality_accepted=False))
    print('RECOVERY_ZERO_UPDATE_PREFLIGHT_PASSED', json.dumps(parity), flush=True)
    gs.save(OUT / 'recovery-result.json', dict(updated=gs.now(), state='restored_training',
        actual_zero_update_preflight_passed=True, exact_continuation=False,
        finding_zh='模型与Adam已恢复；8帧保存图像、原生PLY及全部优化器状态在零新增更新下核验通过。从1001步继续，明确跳过1000步未保存的梯度／统计并使用登记的新随机种子；不是精确连续训练。',
        quality_accepted=False))


def train():
    recovery_start = time.monotonic()
    import torch
    assert not torch.cuda.is_initialized()
    plan, sampling, background, prior = verify_inputs()
    cpu_saved = inspect_cpu_checkpoint(torch)
    source, diff = patched_source(prior, sampling)
    audit_receipt = gs.read(AUDIT_PATH)
    assert audit_receipt['passed'] and audit_receipt['helper_sha256'] == gs.sha(__file__)
    assert audit_receipt['recovery_source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    decision = gs.read(OUT / 'review-1000-decision.json')
    assert decision['decision'] == 'continue' and decision['actual_visual_review'] is True
    assert decision['world_ply_sha256'] == gs.sha(SNAPSHOT / 'world.ply')
    first_launch = gs.read(OUT / 'launch.json')
    pid_stat = Path('/proc') / str(first_launch['pid']) / 'stat'
    assert not pid_stat.exists() or pid_stat.read_text().split()[21] != first_launch['proc_start_identity'], 'Original trainer still exists.'
    assert not (OUT / 'recovery124-launch.json').exists(), 'One explicit recovery attempt only.'
    lease = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert TOTAL_SECONDS - prior - (time.monotonic() - recovery_start) > 0
    RUNTIME.mkdir(exist_ok=False)
    (RUNTIME / 'executed-adapter.py').write_text(source)
    (RUNTIME / 'recovery-interface.diff').write_text(diff)
    preserved = {str(p): gs.sha(p) for p in [OUT / 'launch.json', OUT / 'train-failure.json', OUT / 'adapter.py', OUT / 'plan.json', *sorted(SNAPSHOT.glob('*'))] if p.is_file()}
    gs.save(OUT / 'recovery124-launch.json', dict(updated=gs.now(), pid=os.getpid(), command=sys.argv,
        proc_start_identity=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], single_gpu_lock=True,
        helper_sha256=gs.sha(__file__), checkpoint_sha256=gs.sha(CHECKPOINT),
        executed_adapter_sha256=gs.sha(RUNTIME / 'executed-adapter.py'), original_source_hashes=plan['source_hashes'],
        preserved_files=preserved, original_launch_to_failure_seconds=prior, aggregate_budget_seconds=TOTAL_SECONDS,
        recovery_seed=SEED, first_iteration=1001, exact_continuation=False,
        discarded_gradient_update_step=1000, discarded_densification_statistics_step=1000,
        final_shared_offset_optimizer_updates=17998))
    remaining = TOTAL_SECONDS - prior - (time.monotonic() - recovery_start)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Aggregate registered2400seconds')))
    signal.alarm(max(1, math.ceil(remaining)))
    try:
        saved = torch.load(CHECKPOINT, map_location='cuda', weights_only=False)
        token = object()
        proxy = CheckpointTorchProxy(torch, saved['capture'], token)
        ns = dict(gs.__dict__, OUT=OUT, LIMIT=LIMIT, SNAPSHOTS=[10000, 18000],
            suspended_baseline=lambda: None, RECOVERY_START=recovery_start, PRIOR_SECONDS=prior,
            RECOVERY_SEED=SEED, RECOVERY_DIR=RUNTIME, RECOVERY_CHECKPOINT_SHA=CHECKPOINT_SHA,
            RECOVERY_TOKEN=token, checkpoint_torch_proxy=lambda _: proxy)
        ns['recovery_restore'] = lambda model, args, opt, original, ctx: restore_and_preflight(model, args, opt, original, ctx, saved, cpu_saved)
        exec(compile(source, str(RUNTIME / 'executed-adapter.py'), 'exec'), ns)
        ns['train']()
        assert proxy.loads == 1 and gs.read(RUNTIME / 'gpu-zero-update-preflight.json')['passed']
        for path, digest in preserved.items():
            assert gs.sha(path) == digest, path
        result = gs.read(RUN / 'result.json')
        assert result['state'] in ('bounded_control_completed', 'geometry_stopped'), result
        if result['state'] == 'bounded_control_completed':
            assert result['optimizer_updates'] == 17998 and result['snapshot_step'] == 18000
        gs.save(OUT / 'recovery124-process-exit.json', dict(updated=gs.now(), exit_code=0,
            state=result['state'], exact_continuation=False, prior_segment_seconds=prior,
            recovery_segment_seconds=time.monotonic() - recovery_start,
            aggregate_seconds=prior + time.monotonic() - recovery_start, preserved_originals_verified=True))
        gs.save(OUT / 'recovery-result.json', dict(updated=gs.now(), state=result['state'],
            actual_zero_update_preflight_passed=True, exact_continuation=False,
            finding_zh=('本次同实验恢复已完成18000步，实际共享偏移更新17998次；跳过1000步未保存的梯度／统计，随机序列重新登记。原1000步和超时失败已保留。后续质量、相机配套和查看器仍需核验。' if result['state']=='bounded_control_completed' else '同实验恢复后触发几何保护并停止，保留实际失败状态，不自动重试。'),
            quality_accepted=False))
    except BaseException as error:
        gs.save(OUT / 'recovery124-failure.json', dict(updated=gs.now(), error=repr(error),
            traceback=traceback.format_exc(), auto_retry=False,
            aggregate_seconds=prior + time.monotonic() - recovery_start))
        gs.save(OUT / 'recovery124-process-exit.json', dict(updated=gs.now(), exit_code=1,
            error=repr(error), exact_continuation=False))
        progress = gs.read(OUT / 'progress.json')
        progress.update(updated=gs.now(), state='recovery_failed', pid=None,
            optimizer_waiting=False, error=repr(error), quality_accepted=False)
        gs.save(OUT / 'progress.json', progress)
        raise
    finally:
        signal.alarm(0)
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('audit', 'train')
    {'audit': audit, 'train': train}[sys.argv[1]]()
