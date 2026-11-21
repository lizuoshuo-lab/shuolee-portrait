"""Sequential native training/evaluation pairs for seeds1/2. Restartable per job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-001');REPO=RUNTIME/'repo'

def main():
    queue_path=ROOT/'native-repeat-queue.json'
    queue=dict(status='running',started=time.time(),seeds=[1,2],jobs=[])
    def save():queue_path.write_text(json.dumps(queue,indent=2))
    for seed in [1,2]:
        for kind in ['baseline','motion-rgb']:
            name=f'native-{kind}-18000-s{seed}';out=ROOT/name;mp=out/'manifest.json'
            if (out/'evaluation-summary.json').exists() and json.loads(mp.read_text())['status']=='complete':
                queue['jobs'].append(dict(name=name,status='previously_complete'));continue
            if out.exists():raise RuntimeError(f'Inspect incomplete run first: {name}')
            out.mkdir()
            src=REPO/('msv2/train.py' if kind=='baseline' else 'msv2/train_motion_rgb.py')
            text=src.read_text();needle='    safe_state(args.quiet)';assert text.count(needle)==1
            text=text.replace(needle,needle+f'\n    import random as native_seed_random\n    native_seed_random.seed({seed})\n    np.random.seed({seed})\n    torch.manual_seed({seed})')
            script=REPO/f'msv2/train_{kind.replace("-","_")}_seed{seed}.py';script.write_text(text)
            (out/script.name).write_text(text)
            config=out/'config';config.write_text((REPO/'config/fast').read_text())
            env=os.environ.copy();env.update(PYTHONPATH=str(RUNTIME/'deps')+':'+str(REPO),OMP_NUM_THREADS='4',MINIGS_MOTION_MASKS=str(ROOT/'motion-masks-001'))
            cmd=[sys.executable,str(script),'-s',str(RUNTIME/'dataset'),'-m',str(out/'model'),'-r','1','--eval',
                 '--imp_metric','indoor','--config_path',str(config),'--test_iterations','-1','--port','6119']
            manifest=dict(status='running',seed=seed,steps=18000,command=cmd,started=time.time(),
                intervention='none' if kind=='baseline' else 'reviewed moving-background RGB supervision mask only',
                source_commit='5ec9202c2db7d900728bfc09733e85467cff885e',
                training_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                scope='full native topology evolution; seed only added after official safe_state initialization',
                depth_reinitialization_masked=False)
            mp.write_text(json.dumps(manifest,indent=2));queue['current']=name;save()
            with (out/'train.log').open('w') as log:result=subprocess.run(cmd,cwd=REPO/'msv2',env=env,stdout=log,stderr=subprocess.STDOUT)
            manifest.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,elapsed_seconds=time.time()-manifest['started'])
            ply=out/'model/point_cloud/iteration_18000/point_cloud.ply'
            if result.returncode==0:
                assert ply.exists();manifest.update(ply=str(ply),ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest())
            mp.write_text(json.dumps(manifest,indent=2))
            if result.returncode:
                queue['status']='failed';save();return result.returncode
            # Use the ordinary gsplat environment, while the native render child
            # configures its own isolated extension path in the evaluator.
            with (out/'evaluate.log').open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_native_stage.py')),name],stdout=log,stderr=subprocess.STDOUT)
            queue['jobs'].append(dict(name=name,training_complete=True,evaluation_returncode=result.returncode));save()
            if result.returncode:
                queue['status']='evaluation_failed';save();return result.returncode
    queue.update(status='complete',finished=time.time());save()

if __name__=='__main__':raise SystemExit(main())
