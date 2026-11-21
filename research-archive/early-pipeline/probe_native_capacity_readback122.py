"""T7b: read-only rendering of saved T7 parameters; no optimizer or rescaling."""
from pathlib import Path
import gc,hashlib,json,os,signal,sys,time,traceback
from datetime import datetime,timezone
H=Path(__file__).resolve().parent
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-capacity/readback-diagnosis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def mount(s):
 s=s.replace('\\','/')
 return Path('/mnt/'+s[0].lower()+s[2:]) if len(s)>2 and s[1]==':' else Path(s)
def stamp():return datetime.now(timezone.utc).isoformat()
def main():
 if (O/'result.json').exists() or (O/'progress.json').exists():raise RuntimeError('No repeated T7b')
 plan=read(O/'plan.json')
 if plan['driver_sha256']!=sha(__file__):raise RuntimeError('Unregistered script')
 if os.environ.get('SHUOLEE_T7_SHARED_LOCK')!='held-by-flock':raise RuntimeError('Shared GPU lock required')
 start=time.monotonic();stage='source_hashes'
 def progress(value):
  nonlocal stage
  stage=value;save(O/'progress.json',dict(state='running',pid=os.getpid(),stage=stage,updated=stamp()));print(stage,flush=True)
 def alarm(a,b):raise TimeoutError('T7b 120sec wall limit')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(120);progress(stage)
 try:
  assert all(sha(mount(k))==v for k,v in plan['source_sha256'].items())
  sys.path.insert(0,'/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
  import numpy as np,torch
  from native_scene_package122_v2 import load_scene,render_recorded,write_scene
  from native_portrait_renderer122_v2 import NativeCameraContractV2,render_native
  scene_path=R/'native-capacity/portrait-native-capacity/package/scene.json'
  scene=load_scene(scene_path);cam=scene.camera('portrait-native-capacity')
  parameters={k:scene.parameters[k].copy() for k in ['means','quats','scales','opacities','sh']}
  np.savez(O/'exact-native-parameters.npz',**parameters)
  with np.load(O/'exact-native-parameters.npz') as z:exact={k:z[k] for k in z.files}
  exact_equal={k:bool(np.array_equal(v,exact[k])) for k,v in parameters.items()}
  contract=NativeCameraContractV2('pinhole','synthetic_known_camera')
  p2=write_scene(O/'second-ply-package',parameters,[cam],contract=contract,scope='synthetic-control-only')
  scene2=load_scene(p2)
  attributes={k:dict(max_abs=float(np.abs(v-scene2.parameters[k]).max()),changed_elements=int(np.count_nonzero(v!=scene2.parameters[k]))) for k,v in parameters.items()}
  def delta(a,b):
   d=np.abs(a-b)
   return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),changed_elements=int(np.count_nonzero(d)),over_2e_5_elements=int(np.count_nonzero(d>2e-5)))
  progress('repeat_identical_recorded_package')
  rgb,alpha,_=render_recorded(scene,cam['id'],'pinhole')
  with np.load(R/'native-capacity/portrait-native-capacity/native-float-response.npz') as z:
   repeated=dict(rgb=delta(rgb,z['rgb']),alpha=delta(alpha,z['alpha']))
  progress('exact_npz_input_render')
  t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
  with torch.no_grad():
   rr,aa,_=render_native(t(exact['means']),t(exact['quats']),t(exact['scales']),t(exact['opacities']),t(exact['sh']),t([cam['w2c']]),t([cam['K']]),cam['width'],cam['height'],contract=contract,sh_degree=3)
   torch.cuda.synchronize();rnp=rr[0].cpu().numpy();anp=aa[0].cpu().numpy()
  exact_response=dict(rgb=delta(rgb,rnp),alpha=delta(alpha,anp));del rr,aa;gc.collect();torch.cuda.empty_cache()
  progress('second_ply_generation_render')
  r2,a2,_=render_recorded(scene2,cam['id'],'pinhole')
  ply_response=dict(rgb=delta(rgb,r2),alpha=delta(alpha,a2))
  # Keep the full reference, delta and every original parameter for causal follow-up.
  np.savez_compressed(O/'response-comparison.npz',reference_rgb=rgb,reference_alpha=alpha,ply_rgb=r2,ply_alpha=a2)
  result=dict(state='complete_pending_review',updated=stamp(),elapsed=time.monotonic()-start,
   native_parameter_NPZ_bit_identical=exact_equal,repeat_same_package=repeated,exact_npz_render=exact_response,
   second_PLY_parameter_changes=attributes,second_PLY_response=ply_response,
   sources_unchanged=all(sha(mount(k))==v for k,v in plan['source_sha256'].items()),
   old_T7_reference_not_retained=True,scope='New saved-package diagnostic. Does not reconstruct the lost direct updated T7 arrays or uniquely explain the first failed comparison.',
   optimizer_steps=0,new_calibration=False,physical_quality_accepted=False)
  save(O/'result.json',result);save(O/'progress.json',dict(state='complete',pid=None,updated=stamp()));print(json.dumps(result),flush=True)
 except BaseException as exc:
  save(O/'result.json',dict(state='failed',stage=stage,error=repr(exc),traceback=traceback.format_exc(),elapsed=time.monotonic()-start))
  save(O/'progress.json',dict(state='failed',pid=None,updated=stamp()));raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
