"""Observe one unchanged original frame18 fit; never alter stopping decisions."""
from pathlib import Path
import argparse
import math
import sys
import time

import numpy as np
import official_smplifyx_entry123 as entry
import run_official_sequence123 as sequence
from run_official_sequence123_topology import cache_static_topology


def scalar(value):
    if value is None:
        return None
    value = float(value.detach().cpu()) if hasattr(value, 'detach') else float(value)
    return value if math.isfinite(value) else str(value)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frame-ids', nargs='+', required=True)
    parser.add_argument('--inspect-only', action='store_true')
    args, _ = parser.parse_known_args()
    if args.frame_ids != ['frame_000018']:
        raise ValueError('One observational replay of frame18 only.')
    output = Path(args.output_dir)
    plan = sequence.read(output.parent / 'stopping-trace-plan.json')
    source = Path(__file__).resolve()
    if entry.sha(source) != plan['wrapper_sha256']:
        raise ValueError('Trace wrapper differs from the registered plan.')
    old_load, old_fit = entry.load_official, entry.fit_one
    codes, stages, live = set(), [], {}
    started = time.monotonic()

    def load(root):
        modules = old_load(root)
        if entry.load_official_config(root)['maxiters'] != 30:
            raise ValueError('Replay must preserve the original maxiters30.')
        codes.add(modules[2].FittingMonitor.run_fitting.__code__)
        return modules

    def local_trace(frame, event, arg):
        values = frame.f_locals
        stage = live[id(frame)]
        if event == 'line':
            # Original pinned source: these are actual executed branches,
            # not an inferred explanation from the final loss.
            line = frame.f_lineno
            if line == 177:
                stage['outer_steps'].append(dict(
                    outer_index=int(values['n']), previous_loss=scalar(values['prev_loss']),
                    step_returned_loss=scalar(values['loss'])))
            elif line in (179, 183, 189, 193):
                stage['executed_break_line'] = line
                stage['reason'] = {179: 'nan_loss', 183: 'infinite_loss',
                    189: 'relative_loss_change', 193: 'gradient_threshold'}[line]
        elif event == 'return':
            monitor = values['self']
            stage.update(returned_loss=scalar(arg), last_loss=scalar(values.get('loss')),
                relative_loss_change=scalar(values.get('loss_rel_change')),
                ftol=monitor.ftol, gtol=monitor.gtol, maxiters=monitor.maxiters)
            stage.setdefault('reason', 'maxiters_exhausted')
            gradients = []
            names = {id(p): name for name, p in values['body_model'].named_parameters()}
            names[id(values.get('pose_embedding'))] = 'pose_embedding'
            for index, parameter in enumerate(values['params']):
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().cpu().numpy()
                gradients.append(dict(name=names.get(id(parameter), 'parameter_' + str(index)),
                    official_abs_of_max=float(abs(grad.max())), true_absmax=float(abs(grad).max())))
            stage['gradient_at_return'] = gradients
            stage['optimizer_state'] = [dict(n_iter=int(state.get('n_iter', 0)),
                func_evals=int(state.get('func_evals', 0))) for state in values['optimizer'].state.values()]
            stages.append(stage)
            del live[id(frame)]
        return local_trace

    def dispatch(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            caller = frame.f_back.f_locals
            live[id(frame)] = dict(role='body' if 'opt_idx' in caller else 'camera',
                orientation=caller.get('or_idx'), stage=caller.get('opt_idx'), outer_steps=[])
            return local_trace
        return None

    def fit(**kwargs):
        topology = cache_static_topology(kwargs['body_model'])
        if sys.gettrace() is not None:
            raise ValueError('Do not replace an existing debugger/trace.')
        sys.settrace(dispatch)
        try:
            arrays, report = old_fit(**kwargs)
        finally:
            sys.settrace(None)
            sequence.save(output / 'stage-trace.json', dict(stages=stages,
                unchanged_source=True, diagnostics_only=True))
        report['runtime_optimization'] = topology
        report['observational_trace'] = dict(wrapper_sha256=entry.sha(source),
            solver_loss_config_initialization_unchanged=True)
        return arrays, report

    entry.load_official, entry.fit_one = load, fit
    try:
        sequence.main()
    finally:
        entry.load_official, entry.fit_one = old_load, old_fit
    if args.inspect_only:
        return
    baseline = output.parent / 'full-sequence/frames/frame_000018/winner-camera-space.npz'
    replay = output / 'frames/frame_000018/winner-camera-space.npz'
    if entry.sha(baseline) != plan['baseline_frame_sha256']:
        raise ValueError('Baseline changed; cannot accept replay parity.')
    with np.load(baseline, allow_pickle=False) as before, np.load(replay, allow_pickle=False) as after:
        keys = ('vertices_camera', 'joints_camera', 'joints_projected_native', 'global_orient',
                'camera_translation', 'body_pose_axis_angle', 'pose_embedding', 'betas',
                'expression', 'jaw_pose', 'left_hand_pose', 'right_hand_pose')
        parity = {key: dict(exact=bool(np.array_equal(before[key], after[key])),
            max_abs_delta=float(np.abs(before[key] - after[key]).max())) for key in keys}
    accepted = all(value['exact'] for value in parity.values())
    result = dict(state='trace_completed_exact_parity' if accepted else 'trace_not_baseline_equivalent',
        updated=sequence.now(), elapsed_seconds=time.monotonic()-started,
        baseline_sha256=entry.sha(baseline), replay_sha256=entry.sha(replay), parity=parity,
        trace_accepted=accepted, stages=stages, Gaussian_training_started=False,
        quality_accepted=False, source_sha256=entry.sha(source))
    sequence.save(output / 'result.json', result)
    print('TRACE_RESULT ' + __import__('json').dumps(dict(state=result['state'],
        stage_count=len(stages), reasons=[s['reason'] for s in stages],
        outer_steps=[len(s['outer_steps']) for s in stages])), flush=True)
    if not accepted:
        raise ValueError('Replay differs from original frame; retain trace but do not assert causality.')


if __name__ == '__main__':
    main()
