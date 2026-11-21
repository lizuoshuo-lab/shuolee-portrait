"""Finish only the aborted CPU negative control; never repeat the CUDA suite."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np
from plyfile import PlyData, PlyElement
from native_gaussian_ply122 import read_standard_ply

O = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/covariance-sh-ply')
def save(name, value):
    (O/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
original=json.loads((O/'result.json').read_text(encoding='utf-8'))
if (O/'result-completed.json').exists():
    raise RuntimeError('CPU control already completed; do not repeat')
assert original['error']=='ValueError' and original['message']=='field with empty name'
assert len(original['checks'])==9 and all(c['passed'] for c in original['checks'])
save('fixture-repair-plan.json',{
    'created':datetime.now(timezone.utc).isoformat(),
    'cause':'NumPy selecting a subset of structured fields retains an unnamed padding slot; plyfile rejects that test fixture before the reader is called.',
    'fix':'Repack exactly the same 61 intended fields into a contiguous structured dtype, then call the unchanged strict reader.',
    'scope':'Only the missing-f_rest_44 negative CPU control. Preserve original failed result and all nine completed controls.',
    'CUDA_rerun':False,'renderer_changed':False,'acceptance_limits_changed':False,
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
vertices=PlyData.read(O/'point_cloud.ply')['vertex'].data
names=[n for n in vertices.dtype.names if n!='f_rest_44']
damaged=np.empty(len(vertices),dtype=[(n,'<f4') for n in names])
for name in names:
    damaged[name]=vertices[name]
path=O/'deliberately_incomplete.ply'
assert not path.exists()
PlyData([PlyElement.describe(damaged,'vertex')],text=False,byte_order='<').write(path)
rejected=False;message=None
try:
    read_standard_ply(path)
except ValueError as e:
    rejected=True;message=str(e)
check={'name':'incomplete_SH_rejected','passed':rejected,'error':message,'CUDA_rerun':False}
save('fixture-repair-result.json',check)
result={k:v for k,v in original.items() if k not in ('error','message','traceback','passed','finished','checks')}
result.update(passed=rejected,checks=original['checks']+[check],
              initial_run_result='result.json',supplemental_control='fixture-repair-result.json',
              correction='CPU-only invalid-field fixture construction fixed; renderer and serialization code unchanged',
              finished=datetime.now(timezone.utc).isoformat())
save('result-completed.json',result)
save('progress.json',{'phase':'completed' if rejected else 'failed','pid':None,
                      'result':'result-completed.json','initial_run_preserved':'result.json',
                      'updated':datetime.now(timezone.utc).isoformat(),'gpu_running':False})
print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'CPU_only':True,'error_rejected':message}))
