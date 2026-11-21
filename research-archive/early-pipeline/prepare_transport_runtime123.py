"""One isolated build/load of the existing native backend; no render/fit/train."""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,json,hashlib,time,traceback,subprocess

OUT=Path(sys.argv[1]).resolve()
plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
need=lambda ok,msg:None if ok else (_ for _ in ()).throw(ValueError(msg))
need(not (OUT/'progress.json').exists() and not (OUT/'result.json').exists(),'Already started: inspect evidence.')
need(sys.executable==plan['python'],'Wrong runtime')
need(os.environ.get('TORCH_EXTENSIONS_DIR')==plan['cache'],'Wrong isolated cache')
repo=Path(plan['gsplat_repo']);sys.path.insert(0,str(repo))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
for p,d in plan['source_sha256'].items():need(sha(p)==d,'Changed source: '+p)
start=time.monotonic();stamp=datetime.now(timezone.utc).isoformat()
with (OUT/'progress.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),state='building_or_loading',started=stamp),f)
result=dict(passed=False,pid=os.getpid(),started=stamp,render_calls=0,model_forwards=0,gaussian_steps=0,
    source_sha256=plan['source_sha256'],cache=plan['cache'],plan_sha256=sha(OUT/'plan.json'))
try:
    import torch,gsplat
    need(torch.__version__=='2.7.1+cu128' and gsplat.__version__=='1.5.3','Unexpected backend version')
    need(Path(gsplat.__file__).resolve().is_relative_to(repo),'Unexpected gsplat location')
    from gsplat.cuda._backend import _C
    need(_C is not None,'Native backend did not load')
    for name in ['CameraModelType','projection_ut_3dgs_fused','rasterize_to_pixels_eval3d_fwd','rasterize_to_pixels_eval3d_bwd']:
        need(hasattr(_C,name),'Missing native symbol: '+name)
    artifact=Path(_C.__file__).resolve()
    need(artifact.is_relative_to(Path(plan['cache']).resolve()),'Loaded outside isolated cache')
    for p,d in plan['source_sha256'].items():need(sha(p)==d,'Changed source: '+p)
    result.update(passed=True,python=sys.executable,torch=torch.__version__,gsplat=gsplat.__version__,
        extension=str(artifact),extension_sha256=sha(artifact),extension_bytes=artifact.stat().st_size)
except BaseException as exc:
    result.update(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
result.update(finished=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start)
write(OUT/'result.json',result);write(OUT/'progress.json',dict(pid=os.getpid(),state='completed' if result['passed'] else 'failed',finished=result['finished']))
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2),flush=True)
sys.exit(0 if result['passed'] else 1)
