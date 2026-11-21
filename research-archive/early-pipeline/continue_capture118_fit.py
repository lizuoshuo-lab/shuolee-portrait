"""Continue only the previously unexecuted fresh prior/fit stages of video 118."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl, hashlib, json, os, signal, subprocess, sys, time

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
RECORD=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
PYTHON='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python'
ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');q.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    lock=open('/tmp/shuolee-new-capture-118.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait.cli import gpu_lock
    with gpu_lock():
        q=read(RECORD/'queue-state.json')
        assert q['state']=='preflight_reviewed_training_not_admitted',q['state']
        assert not (ROOT/'inference').exists() and not (ROOT/'fit').exists()
        assert not (ROOT/'gaussian-run').exists()
        assert read(ROOT/'run.json')['video_sha256']=='871d55a34424c942dab7c51755dc24bc1cf8eb0a649df2b422bbdabc800cd79b'
        audit=read(ROOT/'camera-audit.json')
        assert audit['registered']==audit['frames']==89 and audit['reserved_measurements_used']==0
        plan=dict(created=now(),reason='User explicitly requested continuation of local new-video 118.',
            stages=['prior','fit','fit-gate'],same_run_preprocessing_reused=True,
            fresh_person_fit=True,old_person_fit_loaded=False,old_gaussian_loaded=False,
            initial_gaussians_planned=200000,iterations_planned=18000,
            source_hashes={str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'src/shuolee_portrait').glob('*.py')},
            fixed_camera_sha256=sha(ROOT/'data/training-inputs.json'),
            stopped_prior_to_gaussian_allocation=True,
            questions=['Does the newly fitted shared static person project correctly over this complete sampled orbit?',
                'Are remaining residuals detector/visibility errors or real conflicting static observations?'],
            limits=['89 real sampled cameras, not all 884 source frames.',
                'GVHMR temporal cadence at stride10 is only an initialization prior; its geometry is not ground truth.',
                'Keep original frozen 256-track diagnostic including the identified false correspondence.',
                'No permissive training flag or edited validation scores.'],
            stop_conditions=['Any stage exception or nonfinite output.','Prior 20min; fit 25min; fit-gate 5min.',
                'Review actual input/mesh projections after fit-gate before deciding Gaussian training.'])
        with (RECORD/'fit-continuation-plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
        done=list(q['completed'])
        env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
        for stage,timeout in [('prior',1200),('fit',1500),('fit-gate',300)]:
            command=[PYTHON,str(PROJECT/'run.py'),'_stage','--stage',stage,'--output',str(ROOT),'--assets',ASSETS]
            started=time.monotonic()
            with (ROOT/'logs'/f'{stage}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                print(json.dumps(dict(stage=stage,pid=proc.pid,log=str(ROOT/'logs'/f'{stage}.log'))),flush=True)
                timed_out=False
                while proc.poll() is None:
                    save(RECORD/'queue-state.json',dict(updated=now(),state='fresh_body_preparation_running',stage=stage,
                        pid=proc.pid,supervisor_pid=os.getpid(),command=command,completed=done,
                        seconds=time.monotonic()-started,run_root=str(ROOT),training_started=False,quality_accepted=False))
                    if time.monotonic()-started>timeout:
                        timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                        break
                    time.sleep(4)
                code=proc.wait()
            if code:
                save(RECORD/'queue-state.json',dict(updated=now(),state='fresh_body_preparation_failed',stage=stage,
                    pid=None,completed=done,exit_code=code,timed_out=timed_out,run_root=str(ROOT),
                    training_started=False,quality_accepted=False,log=str(ROOT/'logs'/f'{stage}.log')))
                raise SystemExit(code)
            done.append(stage)
        save(RECORD/'queue-state.json',dict(updated=now(),state='fresh_body_fit_pending_visual_review',stage='fit-gate',
            pid=None,completed=done,run_root=str(ROOT),training_started=False,quality_accepted=False))
        print('Fresh prior and shared static mesh fitting completed; Gaussian training has not started.',flush=True)
if __name__=='__main__':main()
