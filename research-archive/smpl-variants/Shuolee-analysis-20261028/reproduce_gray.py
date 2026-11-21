"""Run the published reconstruction locally, preserving original inputs/results."""
from pathlib import Path
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import traceback

REPO = Path('/mnt/d/Projects/Shuolee')
OUTPUT = Path('/mnt/d/Projects/Shuolee-analysis-20261028')
RUN = Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
VIDEO = Path('/mnt/d/00/工作/数据集/视频-f5420380646f495f872c9ce9bcce5448.mp4')
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
os.environ.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
sys.path.insert(0, str(REPO / 'src'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(receipt):
    receipt['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    target = OUTPUT / 'gray-reproduction.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)

if RUN.exists():
    raise FileExistsError(f'Preserve existing local run: {RUN}')
OUTPUT.mkdir(parents=True, exist_ok=True)
receipt = dict(state='waiting_for_local_gpu', run=str(RUN), video=str(VIDEO),
               video_sha256=sha(VIDEO), source_git=subprocess.check_output(
                   ['/mnt/c/Program Files/Git/cmd/git.exe', '-C', 'D:/Projects/Shuolee', 'rev-parse', 'HEAD'], text=True).strip(),
               original_sources_preserved=True, server_results_loaded=False,
               user_requested_local_training=True, quality_accepted=False,
               source_sha256={str(p.relative_to(REPO)): sha(p)
                              for p in (REPO / 'src/shuolee_portrait').glob('*.py')})
assert receipt['video_sha256'] == '7b1c0fd30822c3de879bd6f8649d63175d4d7fa9e2484f0e76d3c763452d79bb'
save(receipt)
with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    lock.seek(0)
    lock.truncate()
    lock.write(json.dumps(dict(pid=os.getpid(), task='smpl-x alignment reproduction', run=str(RUN))))
    lock.flush()
    receipt['state'] = 'running'
    save(receipt)
    with (OUTPUT / 'gray-reproduction.log').open('w', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                from shuolee_portrait.route119 import reconstruct
                cfg = json.loads((REPO / 'configs/video119.json').read_text())
                reconstruct(VIDEO, RUN, ASSETS, cfg)
                receipt['state'] = 'completed'
            except BaseException:
                receipt['state'] = 'failed'
                receipt['error'] = traceback.format_exc()
                traceback.print_exc()
            finally:
                for name, path in [('run_status', RUN / 'status.json'),
                                   ('training', RUN / 'gaussian-run/result.json'),
                                   ('result', RUN / 'result.json')]:
                    if path.exists():
                        receipt[name] = json.loads(path.read_text())
                receipt['sources_unchanged'] = all(sha(REPO / path) == digest
                                                  for path, digest in receipt['source_sha256'].items())
                receipt['input_unchanged'] = sha(VIDEO) == receipt['video_sha256']
                save(receipt)
    fcntl.flock(lock, fcntl.LOCK_UN)
print(json.dumps({key: receipt[key] for key in ('state', 'run', 'sources_unchanged', 'input_unchanged')}, ensure_ascii=False))
sys.exit(0 if receipt['state'] == 'completed' else 1)
