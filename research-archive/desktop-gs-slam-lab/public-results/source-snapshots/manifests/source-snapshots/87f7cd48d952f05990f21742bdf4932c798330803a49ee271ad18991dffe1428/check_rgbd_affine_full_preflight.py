"""CPU mock-only serial/cutoff/failure gates for the new four-round full driver.

No real subprocess or registration is run. Mock outputs are labelled, isolated
below this control directory and never eligible as actual full-sequence results.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys
import traceback
import run_rgbd_affine_full as driver

pipeline = driver.pipeline
ROOT = pipeline.ROOT


def checks(output, protocol_path):
    sources = pipeline.source_checks(); raw, protocol = pipeline.read_protocol(protocol_path, pipeline.sha(protocol_path))
    saved = (pipeline.ROOT, pipeline.source_checks, pipeline.before_deadline, driver.subprocess.Popen, sys.argv)
    fixture_protocol = output/'fixture_protocol.json'; fixture_protocol.write_bytes(raw); digest = pipeline.sha(fixture_protocol)
    summaries = {}
    scenarios = ['serial_success', 'cutoff_before_first', 'cutoff_before_second', 'first_child_exit1',
                 'second_child_exit1', 'interrupt_started_child', 'terminal_configuration_mismatch',
                 'terminal_frame_missing', 'terminal_source_mismatch', 'occupied_fourth_output', 'wrong_protocol_sha']
    try:
        for name in scenarios:
            sandbox = output/name; (sandbox/'outputs').mkdir(parents=True)
            pipeline.ROOT = sandbox; pipeline.source_checks = lambda: sources
            events = []; children = []; guards = 0
            if name == 'occupied_fourth_output': (sandbox/'outputs'/protocol['rounds'][3]['run']).mkdir()
            def guard():
                nonlocal guards
                guards += 1
                stamp = pipeline.DEADLINE-dt.timedelta(microseconds=1)
                if name == 'cutoff_before_first' or (name == 'cutoff_before_second' and guards >= 2): stamp = pipeline.DEADLINE
                events.append('guard:'+stamp.isoformat())
                return saved[2](stamp)
            pipeline.before_deadline = guard
            class Child:
                def __init__(self, argv):
                    self.run = argv[argv.index('--run')+1]; self.waits = 0
                def wait(self):
                    self.waits += 1; events.append('wait:'+self.run+':'+str(self.waits))
                    if name == 'interrupt_started_child' and self.waits == 1: raise KeyboardInterrupt('Synthetic parent interruption')
                    row = next(r for r in protocol['rounds'] if r['run'] == self.run)
                    folder = sandbox/'outputs'/self.run; folder.mkdir(exist_ok=True)
                    packet = {'status': 'completed', 'frames_processed': row['configuration']['frames'],
                        'frame_metrics': [{'frame_id': i} for i in row['configuration']['frame_ids']],
                        'configuration': copy.deepcopy(row['configuration']), 'source_sha256': copy.deepcopy(sources),
                        'protocol_sha256': digest, 'GT_pose_used_in_optimizer': False,
                        'GT_pose_evaluation_after_all_estimates_saved': True, 'GPU_runs': 0, 'failure_frame_count': 0,
                        'scope': 'MOCK SYNTHETIC TERMINAL PACKET ONLY; no trajectory, sensor, optimizer or real child'}
                    if name == 'terminal_configuration_mismatch': packet['configuration']['start_frame'] = 200
                    if name == 'terminal_frame_missing': packet['frame_metrics'].pop()
                    if name == 'terminal_source_mismatch': packet['source_sha256']['rgbd_affine_photometric_odometry.py'] = '0'*64
                    if not (folder/'metrics.json').exists(): pipeline.write(folder/'metrics.json', packet)
                    return 1 if name == 'first_child_exit1' or (name == 'second_child_exit1' and len(children) == 2) else 0
            def popen(argv, **kwargs):
                pipeline.require(kwargs['shell'] is False and kwargs['cwd'] == str(sandbox), 'Actual argv/shell/cwd')
                pipeline.require(not children or children[-1].waits > 0, 'Every previous child collected')
                events.append('Popen:'+argv[argv.index('--run')+1]); child = Child(argv); children.append(child); return child
            driver.subprocess.Popen = popen
            actual_digest = '0'*64 if name == 'wrong_protocol_sha' else digest
            sys.argv = ['mock-full-driver', '--protocol', str(fixture_protocol), '--protocol-sha', actual_digest, '--run', 'mock-batch']
            error = None
            try: driver.main()
            except BaseException as exc: error = type(exc).__name__
            terminal_path = sandbox/'outputs/mock-batch'/('metrics.json' if error is None else 'failure.json')
            terminal = pipeline.read(terminal_path) if terminal_path.exists() else None
            if name == 'serial_success':
                pipeline.require(error is None and len(children) == terminal['runs_started'] == 4 and
                    [r['returncode'] for r in terminal['run_records']] == [0]*4, 'Exactly4 serial successes')
            elif name in ('occupied_fourth_output', 'wrong_protocol_sha'):
                pipeline.require(error == 'ValueError' and len(children) == 0 and terminal is None, 'Prelaunch all-output/hash guard')
            elif name == 'cutoff_before_first':
                pipeline.require(error == 'ValueError' and len(children) == terminal['runs_started'] == 0, 'Equality cutoff before first')
            elif name == 'cutoff_before_second':
                pipeline.require(error == 'ValueError' and len(children) == terminal['runs_started'] == 1 and terminal['run_records'][0]['returncode'] == 0, 'Started child collected; no second at cutoff')
            elif name == 'interrupt_started_child':
                pipeline.require(error == 'KeyboardInterrupt' and len(children) == 1 and children[0].waits == 2 and
                    terminal['run_records'][0]['collected_after_interrupt'], 'Started child collected after interrupt, no more rounds')
            else:
                expected = 2 if name == 'second_child_exit1' else 1
                pipeline.require(error == 'ValueError' and len(children) == terminal['runs_started'] == expected, 'Failure/mismatch stops without retry')
                if 'exit1' in name: pipeline.require(terminal['run_records'][-1]['returncode'] == 1, 'Nonzero exit retained despite mock complete metrics')
            summaries[name] = {'status': 'passed_mock_control', 'events': events, 'mock_children_started': len(children),
                'exception_type': error, 'terminal_status': terminal['status'] if terminal else None,
                'terminal_sha256': pipeline.sha(terminal_path) if terminal else None}
    finally:
        pipeline.ROOT, pipeline.source_checks, pipeline.before_deadline, driver.subprocess.Popen, sys.argv = saved
    pipeline.require(pipeline.source_checks() == sources and not any(n in sys.modules for n in ('torch', 'gsplat')), 'Source/CPU postguard')
    result = {'status': 'passed_all_mock_full_driver_controls', 'source_sha256': sources,
        'protocol_sha256': pipeline.sha(protocol_path), 'checker_sha256': pipeline.sha(Path(__file__)), 'fixtures': summaries,
        'real_children_started': 0, 'real_pair_registration_runs': 0, 'GPU_calls': 0,
        'limitations': ['Mocks test actual driver control flow, not OS scheduling or numerical solver behavior.',
                       'No successful mock packet represents a real run or grants execution permission.']}
    pipeline.write(output/'metrics.json', result)
    pipeline.write(output/'completion.json', {'status': 'completed_mock_only',
        'files': {str(p.relative_to(output)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': pipeline.sha(p)}
                  for p in sorted(output.rglob('*')) if p.is_file()}, 'GPU_calls': 0})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT/'outputs/rgbd-affine-full-cpu-preparation-001/protocol.json')
    p.add_argument('--run', default='rgbd-affine-full-driver-cpu-controls-001'); args = p.parse_args()
    pipeline.require(args.run and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Simple unique control name')
    output = ROOT/'outputs'/args.run; output.mkdir(exist_ok=False)
    (output/'source_script.py').write_bytes(Path(__file__).read_bytes())
    try: checks(output, args.protocol)
    except BaseException as exc:
        pipeline.write(output/'failure.json', {'status': 'mock_control_failure_preserved', 'type': type(exc).__name__,
            'message': str(exc), 'traceback': traceback.format_exc(), 'GPU_calls': 0}); raise
    print(json.dumps({'status': 'passed_all_mock_full_driver_controls', 'real_registration_runs': 0, 'GPU_calls': 0}))


if __name__ == '__main__': main()
