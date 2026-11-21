"""Read saved new-video features and inspect the original bounded focal objective."""
from pathlib import Path
import json, sys
import cv2
import numpy as np
import scipy.optimize

repo=Path('/mnt/d/Projects/Shuolee-validation-20261028/smpl-x')
root=Path('/opt/gs-slam-lab/smpl-x-validation-20261028/runs/portrait-32be30636adabfc4')
out=Path('/mnt/d/Projects/Shuolee-validation-20261028/focal-diagnostic.json')
sys.path.insert(0,str(repo/'src'))
from shuolee_portrait.video119 import estimate_focal_intrinsics
xy=[np.load(p)['xy'] for p in sorted((root/'camera').glob('features-*.npz'))]
records=json.loads((root/'camera/appearance-tracks.json').read_text())['records']
wh=json.loads((root/'decoded.json').read_text())['records'][0]['wh']
original=scipy.optimize.minimize_scalar
report=dict(scope='Read-only focal objective diagnosis; no new camera, fitting or training',wh=wh)
def inspect(fun,**kwargs):
    bounds=kwargs['bounds']
    samples=np.linspace(*bounds,81)
    values=[fun(x) for x in samples]
    result=original(fun,**kwargs)
    report.update(samples=[dict(focal=float(np.exp(x)),cost=float(y)) for x,y in zip(samples,values)],bounded_result=dict(success=bool(result.success),focal=float(np.exp(result.x)),cost=float(result.fun)),grid_best=dict(focal=float(np.exp(samples[np.argmin(values)])),cost=float(min(values))))
    return result
scipy.optimize.minimize_scalar=inspect
cv2.setRNGSeed(119)
try:
    K,evidence=estimate_focal_intrinsics(xy,records,wh)
    report.update(K=K.tolist(),evidence=evidence)
except Exception as exc:
    report['error']=repr(exc)
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
