"""Local acceptance harness: deploy from Git, then upload only a video."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, sys, time
base=Path('/opt/gs-slam-lab/shuolee-folder119-validation-20261027')
repo=Path('/mnt/d/Projects/Shuolee-validation-20261027/reconstruction-only')
art=Path('/mnt/d/Projects/Shuolee-validation-20261027')
service='shuolee119-validation.service'
source=Path('/mnt/d/00/工作/数据集/819080de-8b00.mp4')
jobfiles=list((base/'queue/jobs').glob('*.json'))
assert not jobfiles, 'Preserve existing job evidence; do not restart this acceptance run.'
subprocess.run(['systemctl','stop',service],check=True)
subprocess.run(['systemctl','disable',service],check=True)
env=os.environ.copy()
env.update(SHUOLEE_ROOT=str(base),SHUOLEE_ENV_DIR='/opt/gs-slam-lab/shuolee-portrait-clean-001/env',SHUOLEE_ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',SHUOLEE_SERVICE_NAME='shuolee119-validation')
with (art/'folder119-deploy.log').open('w') as log:
    subprocess.run(['bash',str(repo/'deploy119.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('DEPLOY_READY',flush=True)
inbox=base/'videos'
partial=inbox/'sofa119.mp4.filepart'
shutil.copyfile(source,partial)
print('PARTIAL_UPLOADED',flush=True)
time.sleep(36)
assert not list((base/'queue/jobs').glob('*.json')), 'A temporary upload was admitted.'
assert subprocess.run(['systemctl','is-active','--quiet',service]).returncode==0
partial.rename(inbox/'sofa119.mp4')
print('UPLOAD_FINALIZED',flush=True)
started=time.monotonic(); last=None; duplicate=False
while time.monotonic()-started<3600:
    jobs=list((base/'queue/jobs').glob('*.json'))
    assert len(jobs)<=1, 'Duplicate video started another job.'
    if jobs:
        job=json.loads(jobs[0].read_text())
        run=Path(job['run'])
        stage=json.loads((run/'status.json').read_text()).get('stage') if (run/'status.json').exists() else None
        marker=(job['state'],stage)
        if marker!=last: print(json.dumps(dict(state=marker[0],stage=marker[1],run=str(run))),flush=True); last=marker
        if not duplicate:
            shutil.copyfile(source,inbox/'renamed-copy.mp4.filepart')
            (inbox/'renamed-copy.mp4.filepart').rename(inbox/'renamed-copy.mp4')
            duplicate=True
            print('DUPLICATE_UPLOADED',flush=True)
        if job['state'] in ('failed','interrupted'):
            print(Path(job['log']).read_text()[-10000:],flush=True)
            raise SystemExit('Folder-triggered run failed; evidence retained.')
        if job['state']=='completed': break
    time.sleep(5)
else: raise SystemExit('Validation timed out; service and output retained.')
result=json.loads((run/'result.json').read_text())
assert result['steps']==18000 and result['format_passed'] is True
subprocess.run(['systemctl','restart',service],check=True)
time.sleep(40)
jobs=list((base/'queue/jobs').glob('*.json'))
assert len(jobs)==1 and json.loads(jobs[0].read_text())['state']=='completed'
status=json.loads((base/'queue/status.json').read_text())
assert status['active'] is None and status['counts']=={'completed':1}
assert subprocess.check_output(['systemctl','is-enabled',service],text=True).strip()=='enabled'
receipt=dict(state='passed',date=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_git=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),entrypoint='deploy119.sh',trigger='video upload only; no train command',partial_upload_ignored=True,renamed_content_deduplicated=True,restart_did_not_retrain=True,systemd_enabled_verified=True,run=str(run),video_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),job=job,export=result,queue_after_restart=status,local_validation_service_stopped_after_test=True)
(art/'folder119-integration.json').write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run(['systemctl','disable','--now',service],check=True)
print(json.dumps(receipt),flush=True)
