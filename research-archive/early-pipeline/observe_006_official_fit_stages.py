"""Observe stock CPU SMPLify-X stages on four frozen 006 views without changing the fit."""
import argparse
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import benchmark_006_wsl_fit_workers as stock
from review_bound_alpha_ab import pin


COHORT = (1, 4, 7, 10)
WORKERS_RESULT = stock.LAB/'shuolee-smpl-fit-workers006/result.json'
WORKERS_SHA = 'a2f08b91319104c7891652df738a7e725caf542891847f15d87aba561d68c7f4'
LEASE = Path('/tmp/shuolee-native122-gpu.lock')


def monitor_for(run_fitting, stages):
    """Delegate all optimizer and monitor decisions to the original implementation."""
    import torch

    class ObservedMonitor:
        def run_fitting(self, optimizer, closure, params, body_model, **kwargs):
            slot = len(stages)
            orientation, stage = divmod(slot-1, 5) if slot else (None, 'camera_init')
            original_step = optimizer.step
            active = optimizer.param_groups[0]['params'][0]
            outer = []
            start_steps = self.steps

            def read_counts():
                state = optimizer.state.get(active, {})
                return state.get('n_iter', 0), state.get('func_evals', 0)

            def observe_step(fit_closure):
                n_iter, func_evals = read_counts()
                before = [p.detach().clone() for p in params]
                calls, closure_seconds = 0, 0.

                def observe_closure(*args, **kwargs):
                    nonlocal calls, closure_seconds
                    start = perf_counter()
                    loss = fit_closure(*args, **kwargs)
                    closure_seconds += perf_counter()-start
                    calls += 1
                    return loss

                start = perf_counter()
                loss = original_step(observe_closure)
                elapsed = perf_counter()-start
                new_iters, new_evals = read_counts()
                delta = torch.linalg.vector_norm(torch.cat([
                    (param.detach()-value).reshape(-1) for param, value in zip(params, before, strict=True)]))
                outer.append(dict(call=len(outer)+1, entry_loss=float(loss.detach()), closure_calls=calls,
                    lbfgs_iterations=new_iters-n_iter, lbfgs_func_evals=new_evals-func_evals,
                    closure_seconds=closure_seconds, lbfgs_step_seconds=elapsed,
                    lbfgs_nonclosure_seconds=elapsed-closure_seconds, parameter_delta_l2=float(delta)))
                return loss

            with patch.object(optimizer, 'step', new=observe_step):
                start = perf_counter()
                result = run_fitting(self, optimizer, closure, params, body_model, **kwargs)
                elapsed = perf_counter()-start
            monitor_calls = self.steps-start_steps
            stock.need(monitor_calls == sum(row['closure_calls'] for row in outer)
                       == sum(row['lbfgs_func_evals'] for row in outer), 'Stock closure counters differ.')
            stages.append(dict(stage=stage, orientation=orientation, seconds=elapsed, outer=outer,
                monitor_maxiters=self.maxiters, monitor_ftol=self.ftol, monitor_gtol=self.gtol,
                optimizer_max_iter=optimizer.param_groups[0]['max_iter'], monitor_closure_calls=monitor_calls))
            return result

    return ObservedMonitor.run_fitting


def prepare(output):
    stock.need(not output.exists() and not output.is_relative_to(stock.RUN.parent)
               and not output.is_relative_to(stock.PROJECT), 'Use a new external observation output.')
    pin(WORKERS_RESULT, WORKERS_SHA)
    plan, inputs, all_rows, beta, latent, official, assets, archived = stock.prepare()
    rows = [all_rows[i] for i in COHORT]
    frames = [row['frame'] for row in rows]
    result = stock.seq.rt.read(WORKERS_RESULT)
    stock.need(result['exact_parity'] and all(check['exact'] for frame in frames
              for check in result['comparisons'][frame].values()) and not set(frames) & set(plan['protected_holdouts']),
               'The observed 006 cohort must be protected and exactly benchmarked.')
    plan.update(scope='Four deterministic midpoint views of the frozen 12-frame cohort, CPU observer only.',
        cohort_indices=list(COHORT), frame_ids=frames, source_frame_indices=[row['source_frame_index'] for row in rows],
        worker_counts=[1], observer_sha256=stock.seq.rt.sha(Path(__file__).resolve()),
        workers_result_sha256=WORKERS_SHA, new_optimizer_rule=False,
        entry_loss_scope='Original LBFGSLs returns the pre-step loss. No terminal loss is inferred.')
    return plan, inputs, rows, beta, latent, official, assets, archived


def observe(prepared, output):
    seq = stock.seq
    plan, inputs, rows, beta, latent, official, assets, archived = prepared
    seq.rt.configure(assets)
    seq._init_cpu_fit(official, assets)
    _, _, vposer, _ = seq._CPU_FIT
    pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    stock.need(sha256(pose.tobytes()).hexdigest() == plan['shared_body_target_sha256'], 'Frozen shared pose differs.')
    output.mkdir(parents=True, exist_ok=False)
    seq.rt.save(output/'preregistration.json', plan)
    loader = seq.official.load_official
    records = {}
    for row in rows:
        frame = row['frame']
        slot = inputs['ids'].index(frame)
        target, K = inputs['targets'][slot:slot+1], inputs['K'][slot]
        phases = []

        def observed_load(path):
            utils, prior, fitting, optimizer = loader(path)
            monitor = fitting.FittingMonitor
            monitor.run_fitting = monitor_for(monitor.run_fitting, phases)
            return utils, prior, fitting, optimizer

        job = (row, output/frame, target, K, beta, True, pose)
        with patch.object(seq.official, 'load_official', new=observed_load):
            arrays, receipt = seq._fit_cpu_job(job)
        report = seq.rt.read(receipt['path'])
        seq.check_result(report, row)
        scores, flags, _, _ = stock.stable_selection._candidate(row, arrays, target[0], K, beta, None, report, True)
        old_arrays, old_report, old_flags = archived[frame]
        saved = seq.rt.read(stock.RUN/'fit/selection.json')['reasons'][frame]
        same = (arrays.keys() == old_arrays.keys() and all(stock.exact(arrays[k], old_arrays[k]) for k in arrays)
                and report == old_report and scores == saved['scores'] and flags == old_flags == saved['screen_flags'])
        stock.need(len(phases) == 1+5*report['orientation_count'] and all(p['outer'] for p in phases),
                   f'Incomplete stock monitor phase trace: {frame}')
        records[frame] = dict(exact_archive_parity=same, source_frame_index=row['source_frame_index'],
                              screen_flags=flags, phases=phases)
        seq.rt.save(output/'result.json', dict(state='running' if same else 'parity_failed', records=records,
            optimizer_rule_changed=False, gaussian_training_started=False, quality_accepted=False))
        stock.need(same, f'Observer changed stock fit/receipt/screen: {frame}')
    seq.check_bound(plan['source_pins'])
    pin(WORKERS_RESULT, WORKERS_SHA)
    result = dict(state='completed', exact_stock_parity=True, optimizer_rule_changed=False, records=records,
                  preregistration_sha256=seq.rt.sha(output/'preregistration.json'), gaussian_training_started=False,
                  cold_to_export_measured=False, quality_accepted=False)
    seq.rt.save(output/'result.json', result)
    print(json.dumps(dict(state=result['state'], frames=list(records), exact_stock_parity=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=stock.LAB/'shuolee-smpl-fit-stages006')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    prepared = prepare(output)
    if args.preflight:
        plan = prepared[0]
        print(json.dumps(dict(state='stock_observer_inputs_verified', frames=plan['frame_ids'],
            source_frame_indices=plan['source_frame_indices'], protected_holdouts=plan['protected_holdouts'],
            pins=len(plan['source_pins']), fits_started=0, gpu_calls=0)))
        return
    with LEASE.open('rb') as lease, stock.LOCK.open('rb') as local:
        for lock in (lease, local):
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stock.guard_machine()
        observe(prepared, output)


if __name__ == '__main__':
    main()
