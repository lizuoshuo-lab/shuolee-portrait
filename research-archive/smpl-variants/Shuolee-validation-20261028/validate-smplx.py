"""Local validation from a clean Git checkout and a real 00 portrait video."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, time, traceback

base = Path('/opt/gs-slam-lab/smpl-x-validation-20261028')
repo = Path('/mnt/d/Projects/Shuolee-validation-20261028/smpl-x')
art = Path('/mnt/d/Projects/Shuolee-validation-20261028')
source = Path('/mnt/d/00/工作/数据集/视频-777d211114ab4f39a1b5612466dda6d5.mp4')
service = 'smpl-x.service'
python = '/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python'
assert not base.exists(), 'Existing evidence must be preserved.'
assert subprocess.check_output(['systemctl', 'show', '-p', 'LoadState', '--value', service], text=True).strip() == 'not-found'
base.mkdir(parents=True)
receipt = dict(state='running', source_git=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(), source_video=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), service=service, entrypoint='deploy.sh', full_training=False)

def save():
    receipt['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (art/'smplx-validation.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')

env = os.environ.copy()
env.update(SMPLX_ROOT=str(base), SMPLX_ENV_DIR=str(Path(python).parent.parent), SMPLX_ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
save()
try:
    with (art/'smplx-deploy.log').open('w') as log:
        subprocess.run(['bash', str(repo/'deploy.sh')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert not (repo/'apps/viewer').exists()
    receipt.update(deployment_passed=True, viewer_deployed=False)
    manual_env = env.copy()
    manual_env.pop('SMPLX_ENV_DIR', None)
    manual_env.pop('SHUOLEE_ENV_DIR', None)
    helper = subprocess.run(['bash', str(repo/'scripts/train.sh'), '--help'], env=manual_env, capture_output=True, text=True, check=True)
    assert 'smpl-x train' in helper.stdout
    receipt['manual_command_uses_saved_environment'] = True
    save()
    print('SMPL-X_DEPLOYED_AND_MANUAL_HELP_PASSED', flush=True)
    partial = base/'videos/portrait.mp4.filepart'
    shutil.copyfile(source, partial)
    time.sleep(35)
    assert not list((base/'queue/jobs').glob('*.json'))
    receipt['partial_upload_ignored'] = True
    partial.rename(base/'videos/portrait.mp4')
    print('PORTRAIT_VIDEO_UPLOADED', flush=True)
    started, last = time.monotonic(), None
    while time.monotonic()-started < 5400:
        jobs = list((base/'queue/jobs').glob('*.json'))
        assert len(jobs) <= 1
        if jobs:
            job = json.loads(jobs[0].read_text())
            run = Path(job['run'])
            status = json.loads((run/'status.json').read_text()) if (run/'status.json').is_file() else {}
            marker = (job['state'], status.get('stage'))
            if marker != last:
                print(json.dumps(dict(state=marker[0],stage=marker[1],run=str(run))),flush=True)
                last = marker
            receipt.update(job=job, status=status, run=str(run))
            save()
            if job['state'] in ('failed', 'interrupted'):
                print(Path(job['log']).read_text()[-7000:],flush=True)
                raise RuntimeError('Portrait reconstruction failed; original evidence retained.')
            if job['state'] == 'completed':
                result = json.loads((run/'result.json').read_text())
                assert result['steps'] == 18000 and result['format_passed'] is True
                receipt.update(state='passed', full_training=True, result=result)
                break
        time.sleep(5)
    else:
        raise TimeoutError('Local portrait validation exceeded 90 minutes.')
    subprocess.run(['systemctl','restart',service],check=True)
    time.sleep(40)
    jobs = list((base/'queue/jobs').glob('*.json'))
    assert len(jobs) == 1 and json.loads(jobs[0].read_text())['state'] == 'completed'
    assert subprocess.check_output(['systemctl','is-enabled',service],text=True).strip() == 'enabled'
    receipt.update(restart_did_not_retrain=True, service_enabled=True)
except BaseException:
    receipt.update(state='failed', error=traceback.format_exc())
    raise
finally:
    stop = subprocess.run(['systemctl','disable','--now',service],capture_output=True,text=True)
    receipt['local_validation_service_stopped'] = stop.returncode == 0
    save()
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
