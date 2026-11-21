"""CPU-only addendum for an inter-step mutable-camera gap found in source review."""
from pathlib import Path
import copy,json,hashlib
import numpy as np
from native_training122_v3 import create_view,assert_frozen_view,render_view,export_native_checkpoint
from native_residual_camera122 import ResidualRenderContract
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-camera')
if (O/'frozen-view-addendum.json').exists():raise FileExistsError('Addendum already ran')
c=json.loads((O/'cameras-fixture.json').read_text())[0];h,w=c['height'],c['width'];checks=[]
def make():return create_view(c,np.zeros((h,w,3),'float32'),np.ones((h,w,1),'float32'),np.ones((h,w,1),bool),np.ones((h,w,1),bool),device='cpu')
v=make();assert_frozen_view(v);checks.append(dict(name='Fresh view matches immutable camera digest',passed=True))
for name,edit in [('radial',lambda x:x['residual_projection']['radial'].__setitem__(0,.2)),
                  ('tangential',lambda x:x['residual_projection']['tangential'].__setitem__(1,.05)),
                  ('thin_prism',lambda x:x['residual_projection']['thin_prism'].__setitem__(2,.05)),
                  ('K',lambda x:x['K'][0].__setitem__(2,44)),('E',lambda x:x['w2c'][0].__setitem__(3,.1))]:
    v=make();edit(v.camera)
    for stage,call in [('render',lambda:render_view(None,v,ResidualRenderContract('synthetic_known_camera'))),
                       ('export',lambda:export_native_checkpoint(O/'MUST_NOT_CREATE',None,[v],ResidualRenderContract('synthetic_known_camera'),scope='synthetic-control-only'))]:
        try:call()
        except ValueError as e:
            if 'changed after view creation' not in str(e):raise
            checks.append(dict(name=name+' mutation rejected before '+stage,passed=True))
        else:raise AssertionError(name+' '+stage)
assert not (O/'MUST_NOT_CREATE').exists()
source=Path(__file__).with_name('native_training122_v3.py')
plan=json.loads((O/'plan.json').read_text());old=[v for p,v in plan['source_hashes'].items() if p.endswith('native_training122_v3.py')][0]
result=dict(scope='CPU-only post-review freeze invariant addendum',checks=checks,passed=True,
    original_gpu_result_preserved=True,new_gpu_run=False,numerical_render_code_changed=False,
    cause='Per-step snapshots did not detect D dictionary changes between steps; create-time digest now guards render andexport.',
    old_trainer_sha256=old,new_trainer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),physical_quality_accepted=False)
(O/'frozen-view-addendum.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),GPU_run=False)))
