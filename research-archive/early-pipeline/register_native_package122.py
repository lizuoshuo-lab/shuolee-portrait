from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

H=Path(__file__).parent
R=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\robust-portrait-122')
O=R/'native-package-loader'
assert not (O/'progress.json').exists() and not (R/'native-package-loader-implementation.json').exists()
O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
names=['native_scene_package122.py','validate_native_package122.py','native_gaussian_ply122.py','native_portrait_renderer122.py','run_native_package122.sh']
inputs=['native-package-loader-plan.json']+['covariance-sh-ply/'+n for n in ['native-renderer.json','original-parameters.npz','point_cloud.ply','cameras.json','result-completed.json']]
reg={'registered':datetime.now(timezone.utc).isoformat(),'sources':{n:sha(H/n) for n in names},
     'inputs':{n:sha(R/n) for n in inputs},'GPU_executed':False,
     'validation':'21 strict malformed package rejection controls, positive rectified-pinhole metadata, exact recorded-camera loading, six native-versus-frozen numeric renders with original 3e-6 maximum RGB/alpha limits.',
     'limits':'One GPU task, six Gaussians, three cameras, two projections, <=300 seconds; no person data, parameter fitting, preview fallback or threshold tuning.'}
save(R/'native-package-loader-implementation.json',reg)
q=json.loads((R/'queue-state.json').read_text(encoding='utf-8-sig'))
q.update(updated=reg['registered'],phase='R1c_native_package_loader_validation',running=None,
         next='Execute pre-registered R1c once; inspect numeric output and actual images. No human training admission.')
save(R/'queue-state.json',q)
print(json.dumps({'registered':str(R/'native-package-loader-implementation.json'),'training_started':False}))
