"""Continue the authorized experiment serially after its active trainer exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from shuolee_pgsr_evaluate import RUN, OUT, dump, read, sha


def main(resume_waiting=False,resume_metrics=False):
    status_file=RUN/'followthrough.json'
    scripts=Path(__file__).resolve().parent
    targets=[scripts/'shuolee_pgsr_evaluate.py',scripts/'shuolee_pgsr_score.py']
    if resume_waiting or resume_metrics:
        status=read(status_file)
        if resume_metrics:
            assert status['status']=='failed' and status['failed_stage']=='metrics_and_comparison'
            assert [s['returncode'] for s in status['stages']]==[0,0,1]
            amendment=read(OUT/'development-masks/protocol.json')
            assert amendment['status']=='source_only_annotation_reviewed'
            assert sha(OUT/'development-masks/score-before-mask-fix.py')==status['source_sha256'][str(targets[1])]
            status['original_source_sha256']=dict(status['source_sha256'])
            status['source_sha256'][str(targets[1])]=sha(targets[1])
            status['evaluation_amendment_sha256']=sha(OUT/'development-masks/protocol.json')
        else:
            assert status['status']=='waiting_for_training' and not status['stages']
        try:
            os.kill(status['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Recorded follow-through process still exists; inspect before resuming')
        assert read(RUN/'train/manifest.json')['status']=='complete'
        if not resume_metrics:
            assert not any((RUN/(s+'.log')).exists() for s in
                ['baseline_render','pgsr_render','metrics_and_comparison'])
        status.setdefault('recovery_history',[]).append(dict(
            previous_pid=status['pid'],resumed=time.time(),
            reason=('Missing development masks corrected using source-only annotation; frozen renderer outputs unchanged.' if resume_metrics else
                'Training completed after the waiter observed the trainer process exit. No evaluation stage had started.')))
        status['pid']=os.getpid()
    else:
        assert not status_file.exists(),'Do not create a second follow-through process'
        status=dict(status='waiting_for_training',pid=os.getpid(),started=time.time(),
            source_sha256={str(p):sha(p) for p in targets},
            protocol_sha256=sha(OUT/'protocol.json'),stages=[],accepted_as_best=False)
    dump(status_file,status);deadline=time.time()+6*3600
    process_missing_since=None
    while True:
        train=read(RUN/'train/manifest.json')
        if train['status']=='complete':break
        assert train['status'] in ['running','starting'],train['status']
        if train.get('pid'):
            try:
                os.kill(train['pid'],0)
                process_missing_since=None
            except ProcessLookupError:
                # The trainer's final manifest may become visible after process exit.
                process_missing_since=process_missing_since or time.time()
                if time.time()-process_missing_since>60:
                    raise RuntimeError('Trainer exited without a complete manifest; inspect before recovery')
        assert time.time()<deadline,'Training observation limit reached; inspect process before recovery'
        time.sleep(30)
    # Training wrapper has written completion after the child exited and PLY validation.
    assert train['validation']['saved_parameters_finite'] and train['validation']['geometry_terms_active']
    assert sha(OUT/'protocol.json')==status['protocol_sha256']
    for path,digest in status['source_sha256'].items():assert sha(Path(path))==digest
    env=os.environ.copy();env['OMP_NUM_THREADS']='4'
    schedule=[
        ('baseline_render',[sys.executable,str(targets[0]),'baseline']),
        ('pgsr_render',[sys.executable,str(targets[0]),'pgsr']),
        ('metrics_and_comparison',[sys.executable,str(targets[1])])]
    if resume_metrics:schedule=schedule[-1:]
    for stage,command in schedule:
        status.update(status=stage,updated=time.time());dump(status_file,status)
        log_name=stage+('_mask_repair' if resume_metrics else '')+'.log'
        with (RUN/log_name).open('x') as log:
            p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            status.update(child_pid=p.pid);dump(status_file,status)
            code=p.wait()
        status['stages'].append(dict(stage=stage,returncode=code))
        if code:
            status.update(status='failed',failed_stage=stage,updated=time.time());dump(status_file,status)
            raise RuntimeError(f'{stage} failed; inspect its log before retrying')
    status.update(status='complete_pending_visual_review',finished=time.time(),human_reviews=0)
    status.pop('failed_stage',None)
    dump(status_file,status)
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--resume-waiting',action='store_true')
    group.add_argument('--resume-metrics',action='store_true')
    args=parser.parse_args()
    main(args.resume_waiting,args.resume_metrics)
