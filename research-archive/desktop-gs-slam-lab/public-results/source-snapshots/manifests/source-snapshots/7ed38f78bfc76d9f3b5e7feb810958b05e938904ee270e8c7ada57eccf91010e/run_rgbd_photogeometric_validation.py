"""Serial execution of exactly four predeclared CPU odometry rounds.

This script is an explicit execution entrypoint, not a preparation action.
Starts only before the fixed deadline, waits for each process, stops on process
failure and retains the child's failure/rejected-frame artifacts. No GPU calls.
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

import track_rgbd_photogeometric as pipeline

ROOT=pipeline.ROOT


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True);parser.add_argument('--run',default='rgbd-photogeometric-real-validation-001');args=parser.parse_args()
    pipeline.require(args.run and all(c in'abcdefghijklmnopqrstuvwxyz0123456789-'for c in args.run),'simple output name')
    pipeline.require(sha(args.protocol)==args.protocol_sha,'reviewed protocol exact hash')
    protocol=json.loads(args.protocol.read_text());pipeline.require(protocol['source_sha256']==pipeline.source_checks(),'frozen source bytes')
    pipeline.require(protocol['round_order']==[r['round_id']for r in pipeline.ROUNDS]and len(protocol['rounds'])==4,'fixed four-round order')
    output=ROOT/'outputs'/args.run;output.mkdir(exist_ok=False)
    ledger={'status':'ready_to_execute','protocol_sha256':args.protocol_sha,'source_sha256':protocol['source_sha256'],
        'runs_started':0,'current_run':None,'run_records':[],'GPU_runs':0}
    pipeline.update(output/'progress.json',ledger)
    child=None
    try:
        for row in protocol['rounds']:
            pipeline.require(protocol['source_sha256']==pipeline.source_checks(),'source bytes before every process')
            destination=ROOT/'outputs'/row['run'];pipeline.require(not destination.exists(),'never replace previous run')
            argv=[sys.executable,str(ROOT/'scripts/track_rgbd_photogeometric.py'),'--execute','--protocol',str(args.protocol.resolve()),
                  '--protocol-sha',args.protocol_sha,'--round-id',row['round_id'],'--run',row['run']]
            record={'round_id':row['round_id'],'run':row['run'],'order':row['order'],'configuration':row['configuration']}
            with (output/(row['round_id']+'.log')).open('x',encoding='utf-8')as log:
                start=pipeline.before_deadline()  # Immediately before Popen, after log open.
                child=subprocess.Popen(argv,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,shell=False)
                record.update(started_utc=start,status='running');ledger['run_records'].append(record)
                ledger.update(status='running',runs_started=ledger['runs_started']+1,current_run=row['run'])
                pipeline.update(output/'progress.json',ledger)
                returncode=child.wait();child=None
            record.update(returncode=returncode,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),log_sha256=sha(output/(row['round_id']+'.log')))
            if returncode!=0:raise RuntimeError(f'CPU process failed: {row["run"]}, returncode{ returncode}')
            metrics=json.loads((destination/'metrics.json').read_text())
            pipeline.require(metrics['status']in('completed','completed_with_pair_rejections')and metrics['frames_processed']==100,'terminal round collected')
            record.update(status=metrics['status'],metrics_sha256=sha(destination/'metrics.json'),failure_frame_count=metrics['failure_frame_count'])
            ledger['current_run']=None;pipeline.update(output/'progress.json',ledger)
        ledger.update(status='all_four_predeclared_CPU_rounds_completed',current_run=None)
        pipeline.write(output/'metrics.json',ledger);pipeline.update(output/'progress.json',ledger)
    except BaseException as error:
        if child is not None:
            # Do not abandon or start another process after an interruption.
            returncode=child.wait();ledger['run_records'][-1].update(returncode=returncode,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),collected_after_interrupt=True)
        ledger.update(status='failed_or_interrupted_no_more_rounds_started',exception_type=type(error).__name__,exception_message=str(error),traceback=traceback.format_exc())
        pipeline.write(output/'failure.json',ledger);pipeline.update(output/'progress.json',ledger)
        raise
    print(json.dumps({'status':ledger['status'],'runs_started':ledger['runs_started'],'GPU_runs':0},indent=2))


if __name__=='__main__':main()
