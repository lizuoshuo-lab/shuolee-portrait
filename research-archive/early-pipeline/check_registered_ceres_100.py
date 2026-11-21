import sys,json,datetime
sys.path.insert(0,'/opt/gs-slam-lab/shuolee-bundle-audit/pyceres-2.3')
import pyceres
import pycolmap
from pathlib import Path
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/shared-point-position-preflight-100')
o=pycolmap.BundleAdjustmentOptions()
out=dict(created=datetime.datetime.now().astimezone().isoformat(),pycolmap=pycolmap.__version__,
         pyceres=getattr(pyceres,'__version__','2.3'),isolated_install=True,training_environment_packages_unchanged=True)
try:
    solver=o.solver_options
    out.update(registered=True,solver_type=str(type(solver)),solver_fields=[x for x in dir(solver) if not x.startswith('_')])
    out['defaults']={name:getattr(solver,name) for name in ['max_num_iterations','function_tolerance','gradient_tolerance','parameter_tolerance','max_solver_time_in_seconds','num_threads']}
except Exception as e:
    out.update(registered=False,error=type(e).__name__+': '+str(e))
(O/'solver-options-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
