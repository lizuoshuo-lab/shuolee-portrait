"""Serial exactly-two-round CPU affine validation driver; no automatic launch.

Requires reviewed source/protocol bytes, unique outputs and cutoff immediately
before each Popen. Any started child is collected; failure stops without retry.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import track_rgbd_affine_photogeometric as pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-sha', required=True); p.add_argument('--run', default='rgbd-affine-photogeometric-real-validation-001')
    args = p.parse_args()
    pipeline.require(args.run and len(args.run) <= 120 and args.run[0].isalnum() and
                     all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Simple unique batch name')
    raw = args.protocol.read_bytes(); pipeline.require(hashlib.sha256(raw).hexdigest() == args.protocol_sha, 'Exact reviewed protocol bytes')
    protocol = json.loads(raw)
    pipeline.require(protocol['source_sha256'] == pipeline.source_checks() and protocol['status'] == 'ready_for_explicit_root_review_then_CPU_execution' and
                     protocol['round_order'] == [r['round_id'] for r in pipeline.ROUNDS] and len(protocol['rounds']) == 2, 'Reviewed frozen exact2-round protocol')
    for row, spec in zip(protocol['rounds'], pipeline.ROUNDS):
        pipeline.require({k: row[k] for k in spec} == spec and row['configuration'] == pipeline.configuration_for(row['scene']), 'All2rows config/order exact before launch')
    output = pipeline.ROOT / 'outputs' / args.run; output.mkdir(exist_ok=False)
    ledger = {'status': 'ready_to_execute', 'protocol_sha256': args.protocol_sha, 'source_sha256': protocol['source_sha256'],
              'runs_started': 0, 'current_run': None, 'run_records': [], 'GPU_runs': 0}
    pipeline.update(output / 'progress.json', ledger); child = None; current_record = None; log_path = None
    try:
        for row in protocol['rounds']:
            pipeline.require(protocol['source_sha256'] == pipeline.source_checks(), 'Source guard before each process')
            destination = pipeline.ROOT / 'outputs' / row['run']; pipeline.require(not destination.exists(), 'No output overwrite')
            argv = [sys.executable, str(pipeline.ROOT / 'scripts/track_rgbd_affine_photogeometric.py'), '--execute', '--protocol', str(args.protocol.resolve()),
                    '--protocol-sha', args.protocol_sha, '--round-id', row['round_id'], '--run', row['run']]
            parsed = pipeline.arguments(argv[2:]); pipeline.require(parsed.round_id == row['round_id'] and parsed.run == row['run'], 'Declared actual child CLI')
            current_record = {'round_id': row['round_id'], 'run': row['run'], 'order': row['order'], 'configuration': row['configuration'], 'argv': argv}
            log_path = output / (row['round_id'] + '.log')
            with log_path.open('x', encoding='utf-8') as log:
                started = pipeline.original.before_deadline()  # actual cutoff after log-open, immediately before spawn
                child = subprocess.Popen(argv, cwd=str(pipeline.ROOT), stdout=log, stderr=subprocess.STDOUT, shell=False)
                current_record.update(status='running', started_utc=started); ledger['run_records'].append(current_record)
                ledger.update(status='running', current_run=row['run'], runs_started=ledger['runs_started'] + 1)
                pipeline.update(output / 'progress.json', ledger); returncode = child.wait(); child = None
            current_record.update(returncode=returncode, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), log_sha256=pipeline.sha(log_path))
            pipeline.require(returncode == 0, 'Child failed; stop without retry/new round')
            metrics = pipeline.read(destination / 'metrics.json')
            pipeline.require(metrics['status'] in ('completed', 'completed_with_pair_rejections') and metrics['frames_processed'] == 100, 'Complete terminal100 collected')
            current_record.update(status=metrics['status'], metrics_sha256=pipeline.sha(destination / 'metrics.json'), failure_frame_count=metrics['failure_frame_count'])
            ledger['current_run'] = None; pipeline.update(output / 'progress.json', ledger)
        ledger.update(status='all_two_predeclared_CPU_rounds_completed', current_run=None)
        pipeline.write(output / 'metrics.json', ledger); pipeline.update(output / 'progress.json', ledger)
    except BaseException as error:
        if child is not None:
            returncode = child.wait()
            current_record.update(returncode=returncode, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), collected_after_interrupt=True)
            if current_record not in ledger['run_records']: ledger['run_records'].append(current_record)
            if log_path is not None: current_record['log_sha256'] = pipeline.sha(log_path)
        ledger.update(status='failed_or_interrupted_no_more_rounds_started', exception_type=type(error).__name__, exception_message=str(error), traceback=traceback.format_exc())
        pipeline.write(output / 'failure.json', ledger); pipeline.update(output / 'progress.json', ledger); raise
    print(json.dumps({'status': ledger['status'], 'runs_started': ledger['runs_started'], 'GPU_runs': 0}))


if __name__ == '__main__': main()
