"""CPU-only driver failure/cutoff fixtures, never a real pair registration.

Uses mock child processes and synthetic terminal packets in a unique fixture
directory. These fixtures demonstrate collection/order handling; they are not
real validation results and cannot authorize running the reviewed protocol.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys

import run_rgbd_affine_photogeometric_validation as driver

pipeline = driver.pipeline
ROOT = pipeline.ROOT


def checks(output, protocol_path):
    protocol = pipeline.read(protocol_path); sources = pipeline.source_checks()
    pipeline.require(protocol['source_sha256'] == sources, 'Actual prepared source versions')
    original_root, original_sources, original_configuration, original_deadline = pipeline.ROOT, pipeline.source_checks, pipeline.configuration_for, pipeline.original.before_deadline
    original_popen, original_argv = driver.subprocess.Popen, sys.argv
    fixture_protocol = output / 'fixture_protocol.json'; pipeline.write(fixture_protocol, protocol)
    digest = pipeline.sha(fixture_protocol); summaries = {}
    now = dt.datetime(2026, 10, 8, 22, tzinfo=dt.timezone.utc)
    try:
        for name in ('serial_success', 'cutoff_before_first', 'cutoff_before_second', 'first_child_exit1', 'interrupt_started_child'):
            sandbox = output / name; (sandbox / 'outputs').mkdir(parents=True)
            pipeline.ROOT = sandbox; pipeline.source_checks = lambda: sources
            pipeline.configuration_for = lambda scene: copy.deepcopy(next(row['configuration'] for row in protocol['rounds'] if row['scene'] == scene))
            events = []; children = []; clock_calls = 0
            def guard():
                nonlocal clock_calls
                clock_calls += 1
                if (name == 'cutoff_before_first' and clock_calls == 1) or (name == 'cutoff_before_second' and clock_calls == 2):
                    raise ValueError('Injected actual-start cutoff rejection')
                events.append('guard' + str(clock_calls)); return now.isoformat()
            pipeline.original.before_deadline = guard
            class Child:
                def __init__(self, argv): self.argv = argv; self.waits = 0; self.run = argv[argv.index('--run') + 1]
                def wait(self):
                    self.waits += 1; events.append('wait:' + self.run + ':' + str(self.waits))
                    if name == 'interrupt_started_child' and self.waits == 1: raise KeyboardInterrupt('Injected interrupt after process start')
                    destination = sandbox / 'outputs' / self.run
                    destination.mkdir(exist_ok=True)
                    packet = {'status': 'completed', 'frames_processed': 100, 'failure_frame_count': 0,
                              'scope': 'SYNTHETIC MOCK TERMINAL PACKET; no real trajectory or registration'}
                    if not (destination / 'metrics.json').exists(): pipeline.write(destination / 'metrics.json', packet)
                    return 1 if name == 'first_child_exit1' else 0
            def popen(argv, **kwargs):
                pipeline.require(kwargs['shell'] is False and kwargs['cwd'] == str(sandbox), 'Actual shellfree/cwd call')
                pipeline.require(not children or children[-1].waits >= 1, 'Previous child collected before new spawn')
                events.append('Popen:' + argv[argv.index('--run') + 1]); child = Child(argv); children.append(child); return child
            driver.subprocess.Popen = popen
            sys.argv = ['fixture_driver', '--protocol', str(fixture_protocol), '--protocol-sha', digest, '--run', 'mock-batch']
            error = None
            try: driver.main()
            except BaseException as exc: error = type(exc).__name__
            folder = sandbox / 'outputs/mock-batch'; terminal = pipeline.read(folder / ('metrics.json' if error is None else 'failure.json'))
            if name == 'serial_success':
                pipeline.require(error is None and len(children) == terminal['runs_started'] == 2 and [r['returncode'] for r in terminal['run_records']] == [0, 0], 'Exactly2 serial mocked success')
            elif name == 'cutoff_before_first': pipeline.require(error == 'ValueError' and len(children) == terminal['runs_started'] == 0, 'No child at first cutoff')
            elif name == 'cutoff_before_second': pipeline.require(error == 'ValueError' and len(children) == terminal['runs_started'] == 1, 'No second child after cutoff')
            elif name == 'first_child_exit1': pipeline.require(error == 'ValueError' and len(children) == terminal['runs_started'] == 1 and terminal['run_records'][0]['returncode'] == 1, 'Nonzero child retained/no retry')
            else: pipeline.require(error == 'KeyboardInterrupt' and len(children) == terminal['runs_started'] == 1 and children[0].waits == 2 and terminal['run_records'][0]['collected_after_interrupt'], 'Started child collected on interrupt/no continuation')
            summaries[name] = {'status': 'passed_CPU_mock_fixture', 'events': events, 'mock_children_started': len(children),
                               'terminal_status': terminal['status'], 'exception_type': error,
                               'mock_terminal_sha256': pipeline.sha(folder / ('metrics.json' if error is None else 'failure.json'))}
    finally:
        pipeline.ROOT, pipeline.source_checks, pipeline.configuration_for, pipeline.original.before_deadline = original_root, original_sources, original_configuration, original_deadline
        driver.subprocess.Popen, sys.argv = original_popen, original_argv
    pipeline.require(pipeline.source_checks() == sources and not any(k in sys.modules for k in ('torch', 'gsplat')), 'Fixture source preservation/CPU only')
    pipeline.write(output / 'metrics.json', {'status': 'passed_mock_driver_fixtures_only', 'source_sha256': sources,
                   'checker_sha256': pipeline.sha(Path(__file__)), 'actual_prepared_protocol_sha256': pipeline.sha(protocol_path),
                   'fixtures': summaries, 'real_children_started': 0, 'real_pair_registration_runs': 0, 'GPU_calls': 0,
                   'limits': ['Mocks check driver order/collection and cutoff boundary, not OS scheduling or actual optimization behavior.',
                              'Synthetic terminal packets are explicitly labelled and remain only inside this fixture directory.']})
    (output / 'source_script.py').write_bytes(Path(__file__).read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', default='rgbd-affine-photogeometric-driver-cpu-controls-001')
    p.add_argument('--protocol', type=Path, default=ROOT / 'outputs/rgbd-affine-photogeometric-real-cpu-preparation-001/protocol.json'); args = p.parse_args()
    pipeline.require(args.run and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Unique simple fixture name')
    output = ROOT / 'outputs' / args.run; output.mkdir(exist_ok=False); checks(output, args.protocol)
    print(json.dumps({'status': 'passed_mock_driver_fixtures_only', 'output': str(output), 'real_pair_registration_runs': 0, 'GPU_calls': 0}))


if __name__ == '__main__': main()
