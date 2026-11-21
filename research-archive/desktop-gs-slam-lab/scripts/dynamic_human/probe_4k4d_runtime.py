from pathlib import Path
import json, importlib, traceback, subprocess
OUT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-4k4d-official-035')
report={}
for name in ['torch','pytorch3d','open3d','easyvolcap.utils.console_utils','easyvolcap.models.samplers.r4dv_sampler']:
    try:
        m=importlib.import_module(name)
        report[name]={'status':'imported','file':str(m.__file__),'version':getattr(m,'__version__',None)}
    except Exception:
        report[name]={'status':'failed','error':traceback.format_exc()}
        print(report[name]['error'],flush=True)
import torch
from pytorch3d.ops import knn_points
p=torch.randn(1,100,3,device='cuda',requires_grad=True)
v=knn_points(p,p.detach()+.01,K=2).dists.sum(); v.backward()
report['pytorch3d_knn_backward']={'loss':v.item(),'finite_gradient':bool(torch.isfinite(p.grad).all()),'device':torch.cuda.get_device_name(),'torch':torch.__version__}
(OUT/'runtime-import-probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
