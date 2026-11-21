from pathlib import Path
import json

root = Path('/opt/gs-slam-lab/smpl-x-orientations-20261028')
art = Path('/mnt/d/Projects/Shuolee-validation-20261028')

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None

receipt = read(art / 'orientation-validation.json') or {}
summary = dict(state=receipt.get('state'), gpu=(read(art / 'gpu-coordination.json') or {}).get('state'), jobs={})
for name, job in receipt.get('jobs', {}).items():
    row = dict(state=job.get('state'), stage=job.get('status', {}).get('stage'))
    if job.get('run'):
        run = Path(job['run'])
        for relative in ['video119-progress.json', 'progress.json', 'fit/progress.json', 'gaussian-run/progress.json']:
            value = read(run / relative)
            if value:
                row[relative] = {key: value[key] for key in ['stage', 'component', 'done', 'total', 'step', 'steps', 'target_steps', 'points', 'views_seen', 'registered', 'seconds'] if key in value}
    summary['jobs'][name] = row
peer = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-supervision/progress.json')
peer_state = read(peer) or {}
summary['existing_gpu_task'] = {key: peer_state[key] for key in ['state', 'done', 'total'] if key in peer_state}
print(json.dumps(summary, ensure_ascii=False))
