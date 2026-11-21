"""CPU checks of the actual serialized-camera replay parser, with negative controls."""
import ast,copy
from types import SimpleNamespace
import numpy as np
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now

source=a.H/'verify_demo_head_export123.py'
tree=ast.parse(a.lp(source).read_text(encoding='utf-8'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='serialized_camera')
ns={'np':np,'rt':SimpleNamespace(read=a.read)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),ns)
parse=ns['serialized_camera'];root=a.REC/'demo-vhap-initialization/head-gaussian-first-cycle'
row=a.read(root/'inputs.json')['records'][32];rotation=np.diag([1.,-1.,-1.]);cases=[]
for step in [1000,18000]:
    out=root/'delivery'/f'step-{step:06d}';mapped=parse(out,row,rotation)
    cases.append(dict(case=f'actual_{step}_paired_camera',passed=True,
        camera_sha256=a.sha(out/'cameras.json'),K=mapped['K'],w2c=mapped['w2c']))
original=a.read(root/'delivery/step-018000/cameras.json')
def check(name,mutate):
    records=copy.deepcopy(original);mutate(records)
    ns['rt']=SimpleNamespace(read=lambda _:records)
    try:parse(root,row,rotation)
    except (ValueError,KeyError) as e:cases.append(dict(case=name,passed=True,rejected=str(e)))
    else:raise AssertionError('Wrong camera accepted: '+name)
check('duplicate_camera',lambda x:x.append(copy.deepcopy(x[0])))
check('wrong_frame',lambda x:x[0].update(img_name='frame_000001'))
check('one_pixel_principal_point_shift',lambda x:x[0].update(cx=x[0]['cx']+1))
check('wrong_crop_resolution',lambda x:x[0].update(width=173))
check('wrong_pixel_origin',lambda x:x[0].update(pixelConvention='integer_centers'))
check('camera_translation_shift',lambda x:x[0].update(position=[.01,0,0]))
check('left_handed_rotation',lambda x:x[0].update(rotation=np.diag([1,1,-1]).tolist()))
a.save(a.H/'head-serialized-camera-check123.json',dict(updated=now(),helper_sha256=a.sha(source),
    tests=cases,passed=len(cases),GPU_started=False,native_replay_executed=False,
    scope='CPU parser reads the actual paired camera and rejects identity/K/pose/pixel-origin substitutions. The updated CUDA replay awaits live300k GPU completion; previous replay evidence is not upgraded by these CPU checks.'))
print('SERIALIZED_CAMERA_CPU_CONTROLS_PASSED',len(cases))
