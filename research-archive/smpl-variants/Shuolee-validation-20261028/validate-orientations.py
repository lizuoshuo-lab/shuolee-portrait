"""Real fresh landscape/portrait reconstruction from Git through video upload."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, time, traceback

base=Path('/opt/gs-slam-lab/smpl-x-orientations-20261028')
repo=Path('/mnt/d/Projects/Shuolee-validation-20261028/smpl-x')
art=Path('/mnt/d/Projects/Shuolee-validation-20261028')
sources={'portrait':Path('/mnt/d/00/工作/数据集/视频-777d211114ab4f39a1b5612466dda6d5.mp4'),
         'landscape':Path('/mnt/d/00/工作/数据集/819080de-8b00.mp4')}
service='smpl-x-orientations.service'
assert not base.exists(), 'Preserve existing acceptance runs.'
assert subprocess.check_output(['systemctl','show','-p','LoadState','--value',service],text=True).strip()=='not-found'
base.mkdir(parents=True)
receipt=dict(state='running',source_git=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
             entrypoint='deploy.sh',trigger='video upload only',service=service,jobs={},old_runs_preserved=True)
def save():
    receipt['updated']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (art/'orientation-validation.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
env=os.environ.copy()
env.update(SMPLX_ROOT=str(base),SMPLX_ENV_DIR='/opt/gs-slam-lab/shuolee-portrait-clean-001/env',
           SMPLX_ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',SMPLX_SERVICE_NAME=service[:-8])
save()
try:
    with (art/'orientation-deploy.log').open('w') as log:
        subprocess.run(['bash',str(repo/'deploy.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert not (repo/'apps/viewer').exists()
    receipt.update(deployment_passed=True,viewer_deployed=False)
    for name,source in sources.items():
        partial=base/'videos'/f'{name}.mp4.filepart'
        shutil.copyfile(source,partial)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        receipt['jobs'][name]=dict(source=str(source),source_sha256=digest,state='uploading')
    time.sleep(35)
    assert not list((base/'queue/jobs').glob('*.json'))
    receipt['partial_upload_ignored']=True
    (base/'videos/portrait.mp4.filepart').rename(base/'videos/portrait.mp4')
    save()
    print('DEPLOYED; PORTRAIT_UPLOADED',flush=True)
    started=time.monotonic();last={};landscape_uploaded=False
    while time.monotonic()-started < 10800:
        for name,row in receipt['jobs'].items():
            jobfile=base/'queue/jobs'/(row['source_sha256']+'.json')
            if not jobfile.exists():continue
            job=json.loads(jobfile.read_text())
            run=Path(job['run'])
            status=json.loads((run/'status.json').read_text()) if (run/'status.json').is_file() else {}
            marker=(job['state'],status.get('stage'))
            row.update(state=job['state'],job=job,run=str(run),status=status)
            if marker!=last.get(name):
                print(json.dumps(dict(video=name,state=marker[0],stage=marker[1],run=str(run))),flush=True)
                last[name]=marker
                if job['state'] in ('failed','interrupted'):
                    print(Path(job['log']).read_text()[-7000:],flush=True)
            if (run/'intrinsic-contract.json').is_file():
                intrinsic=json.loads((run/'intrinsic-contract.json').read_text())
                assert intrinsic['mode']=='rectified_center_crop' and intrinsic['orientation']==name
                assert not intrinsic['distortion_applied'] and not intrinsic['images_warped']
                assert intrinsic['reference_wh']==[4000,3000]
                row['intrinsics']=intrinsic
            if job['state']=='completed':
                result=json.loads((run/'result.json').read_text())
                assert result['steps']==18000 and result['format_passed'] is True
                row['result']=result
        if receipt['jobs']['portrait']['state'] not in ('uploading',) and not landscape_uploaded:
            (base/'videos/landscape.mp4.filepart').rename(base/'videos/landscape.mp4')
            landscape_uploaded=True
            print('LANDSCAPE_UPLOADED; SERIAL_QUEUE',flush=True)
        save()
        if all(row['state'] in ('completed','failed','interrupted') for row in receipt['jobs'].values()):break
        time.sleep(5)
    else:raise TimeoutError('Orientation validation exceeded three hours; evidence retained.')
    assert all(hashlib.sha256(sources[name].read_bytes()).hexdigest()==row['source_sha256'] for name,row in receipt['jobs'].items())
    if any(row['state']!='completed' for row in receipt['jobs'].values()):
        raise RuntimeError('At least one orientation failed; both results preserved.')
    subprocess.run(['systemctl','restart',service],check=True)
    time.sleep(40)
    status=json.loads((base/'queue/status.json').read_text())
    assert status['active'] is None and status['counts']=={'completed':2}
    receipt.update(state='passed',restart_did_not_retrain=True,queue_after_restart=status)
except BaseException:
    receipt.update(state='failed',error=traceback.format_exc())
    raise
finally:
    stop=subprocess.run(['systemctl','disable','--now',service],capture_output=True,text=True)
    receipt['local_validation_service_stopped']=stop.returncode==0
    save()
    print(json.dumps(dict(state=receipt['state'],jobs={k:v['state'] for k,v in receipt['jobs'].items()},evidence=str(art/'orientation-validation.json'))),flush=True)
