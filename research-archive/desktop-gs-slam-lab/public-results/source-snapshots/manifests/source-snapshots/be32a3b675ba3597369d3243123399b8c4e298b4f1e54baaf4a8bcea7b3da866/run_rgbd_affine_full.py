"""Exactly four declared CPU full-sequence children, serial, no retry.

No process is spawned at/after the declared deadline. A started process is
collected even if time advances past it or the parent receives an interrupt.
Preparation and synthetic fixtures do not call this driver's real subprocess.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
import track_rgbd_affine_full as pipeline


def terminal_check(destination, row, protocol, digest):
    m = pipeline.read(destination/'metrics.json')
    pipeline.require(m['status'] in ('completed', 'completed_with_pair_rejections') and
        m['frames_processed'] == row['configuration']['frames'] and
        len(m['frame_metrics']) == m['frames_processed'] and
        [r['frame_id'] for r in m['frame_metrics']] == row['configuration']['frame_ids'] and
        m['configuration'] == row['configuration'] and m['source_sha256'] == protocol['source_sha256'] and
        m['protocol_sha256'] == digest and m['GT_pose_used_in_optimizer'] is False and
        m['GT_pose_evaluation_after_all_estimates_saved'] is True and m['GPU_runs'] == 0,
        'Complete exact terminal full trajectory/config/source/GT boundary')
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True); p.add_argument('--protocol-sha', required=True)
    p.add_argument('--run', default='rgbd-affine-full-validation-001'); args = p.parse_args()
    pipeline.require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', args.run) is not None and
                     re.fullmatch(r'[a-f0-9]{64}', args.protocol_sha) is not None, 'Exact batch name/hash')
    raw, protocol = pipeline.read_protocol(args.protocol, args.protocol_sha)
    # Reject any occupied child destination before launching even the first child.
    for row in protocol['rounds']:
        pipeline.require(not (pipeline.ROOT/'outputs'/row['run']).exists(), 'All4 unique outputs absent before first launch')
    output = pipeline.ROOT/'outputs'/args.run; output.mkdir(exist_ok=False)
    (output/'protocol_used.json').write_bytes(raw)
    ledger = {'status': 'ready', 'protocol_sha256': args.protocol_sha, 'source_sha256': protocol['source_sha256'],
              'runs_started': 0, 'run_records': [], 'current_run': None, 'GPU_runs': 0}
    pipeline.update(output/'progress.json', ledger)
    child = None; current = None; log_path = None
    try:
        for row in protocol['rounds']:
            pipeline.require(pipeline.source_checks() == protocol['source_sha256'], 'Sources before each child')
            destination = pipeline.ROOT/'outputs'/row['run']; pipeline.require(not destination.exists(), 'No overwrite')
            argv = [sys.executable, str(pipeline.ROOT/'scripts/track_rgbd_affine_full.py'), '--execute', '--protocol', str(args.protocol.resolve()),
                    '--protocol-sha', args.protocol_sha, '--round-id', row['round_id'], '--run', row['run']]
            parsed = pipeline.arguments(argv[2:]); pipeline.require(parsed.run == row['run'], 'Actual parsed CLI')
            current = {k: row[k] for k in ('order', 'round_id', 'scene', 'model_dimension', 'run', 'configuration')}; current['argv'] = argv
            log_path = output/(row['round_id']+'.log')
            with log_path.open('x', encoding='utf-8') as log:
                stamp = pipeline.before_deadline()  # immediately before actual Popen, after all preparatory I/O
                child = subprocess.Popen(argv, cwd=str(pipeline.ROOT), stdout=log, stderr=subprocess.STDOUT, shell=False)
                current.update(status='running', started_utc=stamp); ledger['run_records'].append(current)
                ledger.update(status='running', runs_started=ledger['runs_started']+1, current_run=row['run'])
                pipeline.update(output/'progress.json', ledger); code = child.wait(); child = None
            current.update(returncode=code, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), log_sha256=pipeline.sha(log_path))
            pipeline.require(code == 0, 'Nonzero child retained; no retry or later round')
            metrics = terminal_check(destination, row, protocol, args.protocol_sha)
            current.update(status=metrics['status'], metrics_sha256=pipeline.sha(destination/'metrics.json'),
                           failure_frame_count=metrics['failure_frame_count'], frames_processed=metrics['frames_processed'])
            ledger['current_run'] = None; pipeline.update(output/'progress.json', ledger)
        ledger.update(status='all_four_predeclared_CPU_full_sequence_rounds_completed', current_run=None)
        pipeline.write(output/'metrics.json', ledger); pipeline.update(output/'progress.json', ledger)
    except BaseException as error:
        if child is not None:
            code = child.wait()
            current.update(returncode=code, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), collected_after_interrupt=True)
            if current not in ledger['run_records']: ledger['run_records'].append(current)
            if log_path: current['log_sha256'] = pipeline.sha(log_path)
        ledger.update(status='failed_or_cutoff_no_more_rounds_started', exception_type=type(error).__name__,
                      exception_message=str(error), traceback=traceback.format_exc())
        pipeline.write(output/'failure.json', ledger); pipeline.update(output/'progress.json', ledger); raise
    print(json.dumps({'status': ledger['status'], 'runs_started': ledger['runs_started'], 'GPU_runs': 0}))


if __name__ == '__main__': main()
