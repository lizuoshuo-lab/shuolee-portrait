"""Durable child runner for the currently requested MSV2 comparison."""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,sys,time,traceback

SCRIPT=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/standard_msv2_demo.py')
REVIEW=SCRIPT.with_name('review_standard_msv2_demo.py')
PROFILE=SCRIPT.with_name('profile_standard_msv2_full4_runtime.py')
spec=importlib.util.spec_from_file_location('standard_experiment',SCRIPT)
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)

def finish_comparison(start_four=False,fresh_all=False):
    """Finish this authorized comparison; stop on any failed validation."""
    if fresh_all:
        assert ex.CONFIG and ex.EXPECTED_FRAMES is not None
        assert ex.read(ex.PREP/'state.json')['state']=='completed'
        assert len(ex.native_rows())==ex.EXPECTED_FRAMES
        if not ex.ROOT.exists():ex.initialize()
        for name in ex.NAMES.values():
            assert not (ex.ROOT/name/'model').exists(), 'All variants require fresh models'
    path=ex.ROOT/'job-finish.json'
    state=dict(state='running',job='all' if fresh_all else ('finish4' if start_four else 'finish'),pid=os.getpid(),started=ex.now(),
        completed=['review2','viewer_2'] if start_four else [])
    def update(action):
        state.update(action=action,updated=ex.now());ex.save(path,state)
    def wait_job(name):
        update('waiting_for_'+name)
        while True:
            current=ex.read(ex.ROOT/f'job-{name}.json')
            if current['state']=='completed':return
            if current['state']!='running':raise RuntimeError(f'{name}: {current}')
            try:os.kill(current['pid'],0)
            except ProcessLookupError:raise RuntimeError(f'{name}: worker process is absent')
            time.sleep(10)
    def job(name):
        update(name)
        subprocess.run([sys.executable,__file__,name],check=True)
        state['completed'].append(name);ex.save(path,state)
    def viewer(scale):
        update('publish_viewer_'+str(scale))
        python='/mnt/c/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe'
        script='C:/Users/Administrator/Documents/New project/shuolee-slide-update/publish_standard_msv2_viewer.py'
        command=[python,script,'--scale',str(scale)]
        if ex.CONFIG_INPUT:command.extend(['--config',ex.CONFIG_INPUT])
        subprocess.run(command,check=True)
        state['completed'].append('viewer_'+str(scale));ex.save(path,state)
    try:
        if fresh_all:
            ex.seed_superresolution()
            sr_log=ex.ROOT/'parallel-sr4.log'
            with sr_log.open('w') as log:
                sr_process=subprocess.Popen([sys.executable,__file__,'sr4'],stdout=log,stderr=subprocess.STDOUT)
            state['parallel_superresolution_pid']=sr_process.pid;ex.save(path,state)
            job('pixels1');job('pixels2')
            for scale in (1,2):
                job('train'+str(scale));job('review'+str(scale));viewer(scale)
        elif start_four:
            for name in ('original','interpolation2x'):
                assert ex.read(ex.RECORD/name/'metrics.json')['execution']['state']=='training_completed'
                assert ex.read(ex.RECORD/name/'delivery.json')['parity_passed']
        else:
            wait_job('train2');job('review2');viewer(2)
        wait_job('sr4')
        assert ex.read(ex.ROOT/'swinir4x/superresolution.json')['state']=='completed'
        assert ex.read(ex.ROOT/'swinir4x/pixels.json')['count']==len(ex.native_rows())
        job('train4');job('review4');viewer(4)
        update('publish_comparison')
        subprocess.run([sys.executable,str(REVIEW),'publish'],check=True)
        state.update(state='completed',finished=ex.now());ex.save(path,state)
    except BaseException as error:
        state.update(state='failed',error=repr(error),finished=ex.now());ex.save(path,state);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=['pixels1','pixels2','cache2','sr4','train1','train2','train4','review1','review2','review4','runtime4','profile4','finish','finish4','all']);a=p.parse_args()
    if a.job in ('finish','finish4','all'):
        finish_comparison(start_four=a.job=='finish4',fresh_all=a.job=='all');return
    jobs={'pixels1':[('pixels',1)],'pixels2':[('pixels',2)],'cache2':[('cameras',2)],'sr4':[('sr',4)],
          'runtime4':[('validate_runtime',4)],
          'profile4':[('profile',4)],
          'train1':[('cameras',1),('preflight',1),('train',1)],
          'train2':[('cameras',2),('preflight',2),('train',2)],
          'train4':[('cameras',4),('profile',4),('preflight',4),('train',4)],
          'review1':[('export',1),('evaluate',1)],'review2':[('export',2),('evaluate',2)],'review4':[('export',4),('evaluate',4)]}
    stamp=time.strftime('%Y%m%dT%H%M%S');state=dict(state='running',job=a.job,pid=os.getpid(),started=ex.now(),commands=[])
    path=ex.ROOT/f'job-{a.job}.json';ex.save(path,state)
    env=os.environ.copy();env.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    try:
        for action,scale in jobs[a.job]:
            log=ex.ROOT/f'{a.job}-{stamp}-{action}.log'
            command=([sys.executable,str(PROFILE)] if action=='profile' else
                [sys.executable,str(REVIEW if action in ('export','evaluate') else SCRIPT),action,'--scale',str(scale)])
            state.update(action=action,log=str(log),updated=ex.now());state['commands'].append(command);ex.save(path,state)
            with log.open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        state.update(state='completed',finished=ex.now());ex.save(path,state)
    except BaseException as error:
        state.update(state='failed',error=repr(error),finished=ex.now());ex.save(path,state);raise

if __name__=='__main__':main()
