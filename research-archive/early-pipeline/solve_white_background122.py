"""N12g: one fresh fixed-K, reviewed-background-only seven-frame diagnostic."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,sys,traceback
import numpy as np
import cv2
from scipy.sparse import coo_matrix
from audit_shared_solver122 import Problem,huber_cost
from repair_shared_solver122 import schur_step,linear_control
from shared_background122 import project,triangulate,track_initial_point,evaluate,camera_center
sys.path.insert(0,'/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5')
import poselib
I=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
F=I/'clip03-roof-review';O=I/'clip03-fresh-geometry';P=I/'clip03-fresh-geometry-plan.json';G=I/'clip03-grid-coverage/array-cache-repair'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def mat(p):
 m=np.eye(4);m[:3,:3]=p.R;m[:3,3]=p.t;return m
def register():
 assert not P.exists() and not O.exists()
 review=read(F/'visual-review.json');assert review['gate']['counts_passed'] and review['gate']['per_frame_8fit_3bins_passed']
 inputs=[F/'reviewed-support.json',F/'visual-review.json',G/'visual-index.json',G/'visual-review.json',I/'output-intrinsics-audit/result.json']
 sources=[Path(__file__)]+[Path(__file__).parent/n for n in ['audit_shared_solver122.py','repair_shared_solver122.py','shared_background122.py']]
 K=read(I/'output-intrinsics-audit/result.json')['variants'][0]['K']
 save(P,dict(stage='N12g',registered=now(),hypothesis='Reviewed near-ground and distant opaque structural observations may establish a shared camera path with the existing conditional4000K. This is the first whiteT camera candidate, not a repeat of hat geometry or a claim thatK is exact.',
  variable='Fresh seven-view fixedK geometry on the frozen, physically reviewed N12d/N12f support. Observation collection and review changed together relative to prior hat experiments; no single-factor cross-person claim.',
  no_old_pose_or_geometry=True,frames=[184,200,216,232,248,264,280],initial_pair=[216,232],initial_pair_rule='Earliest adjacent pair with >=12sharedFIT and >=6nongroundFIT; chosen from image-support counts before residuals.',registration_order=[200,184,248,264,280],
  K_integer_centers=K,K_source='N4 H0 firstYAML4000reference, ideal halfpixelresize+centercrop to1520x2704. It is the existing conditionalK, not verified exporterK.',
  gauge='Initial216camera identity and216->232baseline unit during solve; output rebased to184 identity only after actual registration. No point/pose copied from earlier candidates.',
  initialization=dict(relative_method='PoseLib calibrated5point RANSAC',relative_px=1.,absolute_method='PoseLib P3P RANSAC',PnP_px=2.,min_consensus=8,min_iterations=100,max_iterations=10000,seed=1227,positive_depth=True,min_ray_angle_degrees=.2),
  optimization=dict(method='Existing analytically checked componentwiseHuber1px dampedSchur',iterations=80,damping_trials=8,lambda_start=.001,accepted_divide=3,rejected_multiply=10),
  measurements='Original SIFT native pixels only; N12e localphotometric positions never substituted. No heldout/person/camera-result reads for optimization. Each family contributes one frozen representative.',
  validation='All15heldout background families and all6coarseface families evaluated leave-one-observation-out, plus retained person/feet diagnostic rows. Every target/failure retained. Initial andfinal evaluation aftersolve only. No residual-based deletion or candidate selection.',
  stop='Missingcamera, consensus<8, invalidinput or wallbudget -> preserve failure and stop; no alternatepair/threshold/candidate. Poor heldout/depth evidence -> no training. Sevenframes never substitute for464fullvideo.',
  budget=dict(runs=1,CPU_wall_seconds=180,GPU=0,training=0),input_hashes={str(p):sha(p) for p in inputs},code_hashes={str(p):sha(p) for p in sources},physical_subpixel_accepted=False))
 print('N12G_REGISTERED',flush=True)

class FreshProblem(Problem):
 def __init__(self,K,order,cameras,tracks,points):
  # Do not invoke Problem.__init__: it loads historical optimized parameters.
  self.K=K;self.frames=order;self.tids=sorted(points);self.slices={};params=[]
  for i,f in enumerate(order[1:],1):
   rv=cv2.Rodrigues(cameras[f][:3,:3])[0].ravel();t=cameras[f][:3,3]
   val=np.r_[rv,np.arctan2(t[1],t[0]),np.arcsin(np.clip(t[2],-1,1))] if i==1 else np.r_[rv,t]
   self.slices[f]=slice(len(params),len(params)+len(val));params.extend(val)
  self.ncam=len(params);self.x0=np.r_[params,np.array([points[t] for t in self.tids]).ravel()]
  lookup={tid:i for i,tid in enumerate(self.tids)}
  obs=[(o['frame'],lookup[t['track']],o['uv']) for t in tracks if t['track'] in lookup for o in t['observations']]
  self.target=np.array([uv for f,i,uv in obs]);self.indices=np.array([i for f,i,uv in obs]);self.groups={f:np.flatnonzero([ff==f for ff,i,uv in obs]) for f in order}
  rr=[];cc=[]
  for j,(f,i,uv) in enumerate(obs):
   cols=list(range(self.ncam+3*i,self.ncam+3*i+3))
   if f in self.slices:cols+=list(range(self.slices[f].start,self.slices[f].stop))
   for row in [2*j,2*j+1]:rr.extend([row]*len(cols));cc.extend(cols)
  self.sparsity=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(2*len(obs),len(self.x0))).tocsr()

def directional_control(p):
 x=p.x0;j=p.jacobian(x);rng=np.random.default_rng(1227);v=rng.normal(size=len(x));v/=np.linalg.norm(v);h=1e-6
 nv=(p.residual(x+h*v)-p.residual(x-h*v))/(2*h);av=j@v
 e=float(np.linalg.norm(av-nv)/max(np.linalg.norm(nv),1e-12));assert e<1e-5
 return e
def synthetic_control(K):
 rng=np.random.default_rng(1227);X=rng.uniform([-1,-1,4],[1,1,8],size=(30,3));cams={0:np.eye(4),1:np.eye(4)}
 cams[1][:3,:3]=cv2.Rodrigues(np.array([.01,.08,-.02]))[0];cams[1][:3,3]=np.array([1.,0,0])
 tracks=[dict(track=i,observations=[dict(frame=f,uv=project(K,cams[f],X[i:i+1])[0][0]) for f in [0,1]]) for i in range(len(X))]
 p=FreshProblem(K,[0,1],cams,tracks,{i:x for i,x in enumerate(X)})
 terr=max(np.linalg.norm(triangulate(K,cams,[(o['frame'],o['uv']) for o in t['observations']])-X[t['track']]) for t in tracks)
 assert terr<1e-8 and max(abs(p.residual(p.x0)))<1e-8
 return dict(analytic_directional_relative=directional_control(p),triangulation_max=float(terr),schur_dense_max=linear_control())
def run():
 assert not O.exists();p=read(P)
 for path,h in p['input_hashes'].items():assert sha(Path(path))==h
 for path,h in p['code_hashes'].items():assert sha(Path(path))==h
 O.mkdir();start=time.monotonic();cv2.setNumThreads(2)
 K=np.array(p['K_integer_centers']);controls=synthetic_control(K)
 def progress(state,**kw):
  assert time.monotonic()-start<180,'Registered wallbudget exceeded'
  save(O/'progress.json',dict(state=state,pid=os.getpid(),updated=now(),elapsed_seconds=time.monotonic()-start,**kw))
 support=read(F/'reviewed-support.json')['tracks'];fit=[t for t in support if t['split']=='fit'];assert len(fit)==40
 camdesc=dict(model='PINHOLE',width=1520,height=2704,params=[K[0,0],K[1,1],K[0,2],K[1,2]])
 a,b=p['initial_pair'];common=[t for t in fit if {a,b}<=set(o['frame'] for o in t['observations'])]
 uv=[np.array([next(o['uv'] for o in t['observations'] if o['frame']==f) for t in common]) for f in [a,b]]
 ro=dict(max_epipolar_error=1.,min_iterations=100,max_iterations=10000,seed=1227)
 progress('fresh_relative_pose',shared_fit=len(common))
 pose,info=poselib.estimate_relative_pose(*uv,camdesc,camdesc,ro,{})
 save(O/'relative-initialization.json',dict(frames=[a,b],families=[t['family'] for t in common],inliers=[bool(x) for x in info['inliers']],R=pose.R.tolist(),t=pose.t.tolist()))
 assert info['num_inliers']>=8,'Relative consensus below preregistered8'
 cameras={a:np.eye(4),b:mat(pose)};cameras[b][:3,3]/=np.linalg.norm(cameras[b][:3,3]);points={};initializations=[]
 for f in p['registration_order']:
  for t in fit:
   if t['track'] not in points:
    X=track_initial_point(K,cameras,t)
    if X is not None:points[t['track']]=X
  avail=[(t,o) for t in fit if t['track'] in points for o in t['observations'] if o['frame']==f]
  progress('fresh_absolute_pose',frame=f,available=len(avail))
  assert len(avail)>=8,'Insufficient fresh2D3D'
  xyz=np.array([points[t['track']] for t,o in avail]);pix=np.array([o['uv'] for t,o in avail])
  pose,info=poselib.estimate_absolute_pose(pix,xyz,camdesc,dict(max_reproj_error=2.,min_iterations=100,max_iterations=10000,seed=1227),{})
  row=dict(frame=f,available=len(avail),consensus=int(info['num_inliers']),families=[t['family'] for t,o in avail],inliers=[bool(x) for x in info['inliers']]);initializations.append(row)
  save(O/'PnP-progress.json',initializations)
  assert info['num_inliers']>=8,'PnP consensus below preregistered8'
  cameras[f]=mat(pose)
 points={t['track']:X for t in fit if (X:=track_initial_point(K,cameras,t)) is not None}
 exclusions=[dict(track=t['track'],family=t['family'],reason='nonfinite/negative-depth or<0.2deg initialparallax; originalobservationsretained') for t in fit if t['track'] not in points]
 assert len(points)>=20,'Too few valid freshFIT structure points'
 order=[a,b]+p['registration_order'];problem=FreshProblem(K,order,cameras,fit,points);controls['actual_analytic_directional_relative']=directional_control(problem)
 x=problem.x0.copy();cost=huber_cost(problem.residual(x));cost0=cost;damping=.001;history=[];reason='max_iterations'
 save(O/'initialization.json',dict(PnP=initializations,FIT_points=len(points),excluded=exclusions,controls=controls,initial_cameras=[dict(frame=f,w2c=cameras[f].tolist()) for f in p['frames']]))
 for it in range(80):
  progress('fresh_fixedK_bundle',iteration=it,cost=cost)
  r=problem.residual(x);j=problem.jacobian(x);accepted=False
  for trial in range(8):
   dx,g,d=schur_step(j,r,problem.ncam,damping);xr=x+dx;cr,xx=problem.decode(xr);rr=problem.residual(xr)
   positive=all(np.all(project(K,cr[f],xx[problem.indices[ix]])[1]>0) for f,ix in problem.groups.items())
   tc=huber_cost(rr) if np.isfinite(rr).all() and positive else float('inf')
   if tc<cost:
    previous=cost;x=xr;cost=tc;accepted=True;used=damping;damping=max(damping/3,1e-12);break
   damping*=10
  history.append(dict(iteration=it,cost=float(cost),accepted=accepted,trials=trial+1,damping=float(used if accepted else damping),gradient_inf=float(max(abs(g)))))
  if not accepted:reason='no_descent_with_registered_damping';break
  if max(abs(dx))<1e-9 or max(abs(g))<1e-5:reason='numerical_stationarity';break
  if previous-cost<1e-9*max(1,previous):reason='relative_cost_change';break
 final,xyz=problem.decode(x)
 # Evaluate all frozen target observations after the one candidate was fit.
 ov=read(G/'visual-index.json');vr=read(G/'visual-review.json');held=[t for t in support if t['split']=='heldout']
 face=[ov['rows'][r['row']] for r in vr['rows'] if r['reviewed_material']=='face_surface_coarse']
 person=[ov['rows'][r['row']] for r in vr['rows'] if r['row']>=51]
 groups=dict(background=held,background_ground=[t for t in held if not t['nonground']],background_nonground=[t for t in held if t['nonground']],face_all_diagnostic=face,face_heldout=[t for t in face if t['split']=='heldout'],person_diagnostic=person,feet_diagnostic=[ov['rows'][r['row']] for r in vr['rows'] if r['reviewed_material'] in ['shoe','sock']])
 vals={label:{name:evaluate(K,cams,ts) for name,ts in groups.items()} for label,cams in [('initial',cameras),('final',final)]}
 # Coordinate-only world gauge change, with an explicit projection equality control.
 inv=np.linalg.inv(final[p['frames'][0]]);rebased={f:M@inv for f,M in final.items()};baseM=final[p['frames'][0]]
 worldxyz=xyz@baseM[:3,:3].T+baseM[:3,3]
 err=max(float(np.max(abs(project(K,final[f],xyz)[0]-project(K,rebased[f],worldxyz)[0]))) for f in final);assert err<1e-7
 save(O/'validation.json',vals)
 np.savez_compressed(O/'diagnostic-structure.npz',points=worldxyz,track_ids=problem.tids,parameters=x)
 result=dict(stage='N12g',state='complete_pending_review',updated=now(),elapsed_seconds=time.monotonic()-start,controls=dict(**controls,gauge_projection_max_px=err),K_integer_centers=K.tolist(),K_optimized=False,registered_frames=p['frames'],all464_registered=False,initial_pair=[a,b],FIT_families=40,FIT_initialized=len(points),structural_exclusions=exclusions,heldout_families=15,cameras=[dict(frame=f,w2c=rebased[f].tolist()) for f in p['frames']],optimizer=dict(reason=reason,initial_cost=cost0,final_cost=cost,history=history),validation={l:{g:r['summary'] for g,r in gr.items()} for l,gr in vals.items()},validation_sha256=sha(O/'validation.json'),old_pose_or_geometry_used=False,training_started=False,physical_subpixel_accepted=False)
 save(O/'result.json',result);save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=now()))
 print(json.dumps(dict(elapsed=result['elapsed_seconds'],fit_points=len(points),optimizer=reason,validation=result['validation'])),flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except Exception as e:
  if O.exists():
   save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
   save(O/'progress.json',dict(state='failed_preserved',pid=None,updated=now()))
  raise
