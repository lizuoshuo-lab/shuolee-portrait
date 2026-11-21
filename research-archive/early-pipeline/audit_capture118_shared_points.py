"""CPU-only diagnostic: identity/projection contract and static free-joint residuals."""
from pathlib import Path
import argparse,sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).parent))
from capture118_all_subjects import ROOT,REC,rt

ap=argparse.ArgumentParser();ap.add_argument('--subject',required=True);a=ap.parse_args()
root=ROOT/a.subject;out=REC/(a.subject+'-shared-point-audit-v2');out.mkdir(exist_ok=True)
assert not (out/'result.json').exists(),'Preserve completed diagnostic.'
protocol=dict(created=rt.now(),subject=a.subject,camera_fixed=True,images_unwarped=True,fit_unchanged=True,
 hypothesis='Separate software frame/projection mismatch and SMPL-X fitting restrictions from inability of fixed cameras to support shared static learned landmarks.',
 variable='Only per-joint shared3D location is freed; K, E, image coordinates and observed confidence stay fixed. No per-frame pose or camera parameters.',
 split='Even frame indices fit a free3D point, odd frame indices evaluate it. All reliable held observations are retained.',
 budget='17 joints, max100 evaluations perjoint, CPU only. No Gaussian or camera optimization.',
 pixel_contract='Stored renderer K uses pixel corners. Detectors and fit targets use integer pixel centers. Convert K principal by minus0.5 for both projection and DLT. V1 diagnostic omitted this conversion; V1 remains preserved, not used as projection-mismatch evidence.',
 limitation='Learned joints are not physical material truth. Free-point failure cannot distinguish camera error, detector ambiguity or subject motion. Free-point success cannot prove camera calibration.')
rt.save(out/'protocol.json',protocol)
rows=rt.read(root/'data/training-inputs.json')['records'];raw=rt.read(root/'camera/candidate-cameras.json')['records']
fit=dict(np.load(root/'fit/fitted.npz'));obs=dict(np.load(root/'data/observations.npz'));weights=dict(np.load(root/'fit/observation-weights.npz'))
ids=[r['frame'] for r in rows];assert len(set(ids))==len(ids)
assert np.array_equal(fit['frame_ids'],ids) and np.array_equal(obs['frame_ids'],ids)
assert ids==[r['frame'] for r in raw]
K=np.array([r['K'] for r in rows]);E=np.array([r['w2c'] for r in rows]);n=len(rows)
assert np.array_equal(K,[r['K'] for r in raw]) and np.array_equal(E,[r['w2c'] for r in raw])
K=K.copy();K[:,0,2]-=.5;K[:,1,2]-=.5
def project(X):
 cam=np.einsum('nij,j->ni',E[:,:3,:3],X)+E[:,:3,3]
 uv=np.einsum('nij,nj->ni',K,cam);return uv[:,:2]/uv[:,2,None],cam[:,2]
native=np.stack([project(x)[0] for x in fit['coco17_world'][0]],1)
contract=float(np.max(np.abs(native-fit['projected_coco17'])))
target=weights['body_target'];bw=weights['body']*weights['visibility'];reliable=bw>.25
h=obs['boxes'][:,3]-obs['boxes'][:,1];native_error=np.linalg.norm(native-target,axis=-1)
P=K@E[:,:3];joint_results=[];all_native=[];all_free=[];per_frame=[]
for j in range(17):
 train=np.flatnonzero(reliable[:,j]&(np.arange(n)%2==0));test=np.flatnonzero(reliable[:,j]&(np.arange(n)%2==1))
 if len(train)<6 or len(test)<4:
  joint_results.append(dict(joint=j,state='insufficient_reliable_observations',train=len(train),test=len(test)));continue
 mats=[]
 for i in train:
  u,v=target[i,j];mats.extend([u*P[i,2]-P[i,0],v*P[i,2]-P[i,1]])
 _,_,vh=np.linalg.svd(np.asarray(mats),full_matrices=False);x=vh[-1,:3]/vh[-1,3]
 def fun(x):return ((project(x)[0][train]-target[train,j])*np.sqrt(bw[train,j,None])).ravel()
 sol=least_squares(fun,x,loss='soft_l1',f_scale=3,max_nfev=100)
 uv,z=project(sol.x);free=np.linalg.norm(uv[test]-target[test,j],axis=-1);old=native_error[test,j]
 all_native.extend((old/h[test]).tolist());all_free.extend((free/h[test]).tolist())
 joint_results.append(dict(joint=j,train_frames=train.tolist(),held_frames=test.tolist(),optimizer_success=bool(sol.success),
  native_held_median_px=float(np.median(old)),free_held_median_px=float(np.median(free)),free_held_p90_px=float(np.quantile(free,.9)),
  free_positive_depth_all=bool(np.all(z>0)),free_point=sol.x.tolist(),held_native_px=old.tolist(),held_free_px=free.tolist()))
for i in range(n):
 good=reliable[i];err=native_error[i,good]
 per_frame.append(dict(frame=ids[i],reliable_joints=int(good.sum()),fit_median_px=float(np.median(err)) if len(err) else None,
  fit_p90_px=float(np.quantile(err,.9)) if len(err) else None,box_height=float(h[i]),sum_weights=float(bw[i].sum())))
def stat(x):
 return dict(count=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9))) if len(x) else dict(count=0)
result=dict(created=rt.now(),subject=a.subject,contract_identity_pass=True,numpy_vs_saved_projection_max_px=contract,
 camera_sha256=rt.sha(root/'camera/candidate-cameras.json'),fit_sha256=rt.sha(root/'fit/fitted.npz'),
 native_held_body_fraction=stat(all_native),free_point_held_body_fraction=stat(all_free),joint_results=joint_results,per_frame=per_frame,
 physical_accuracy_proven=False,causal_uniqueness_proven=False)
rt.save(out/'result.json',result)
print({k:v for k,v in result.items() if k not in ['joint_results','per_frame']},flush=True)
