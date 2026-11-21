from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-camera')
assert not (O/'progress.json').exists()
assert not (O/'plan-before-path-repair.json').exists()
shutil.copy2(O/'plan.json',O/'plan-before-path-repair.json')
p=json.loads((O/'plan.json').read_text());key=str(H/'probe_residual_camera122.py')
repair=dict(updated=datetime.now(timezone.utc).isoformat(),error='Windows backslashes not normalized by source-hash preflight WSL path conversion; FileNotFoundError before progress/CPU/CUDA.',
            changed_file=key,old_sha256=p['source_hashes'][key],new_sha256=hashlib.sha256(Path(key).read_bytes()).hexdigest(),
            fixture_changed=False,threshold_changed=False,cuda_executed=False,repair='Normalize path separators before C:/ to /mnt/c conversion')
p['source_hashes'][key]=repair['new_sha256'];p['preflight_repair']=repair
for name,v in [('preflight-failure.json',repair),('plan.json',p)]:
    (O/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Path preflight repair recorded; no numerical run occurred.')
