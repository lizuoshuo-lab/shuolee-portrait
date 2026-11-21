"""Training-time reprojection refinement of the continuous articulated prior.

Only even frames enter optimization. Raw cameras and the 062 timing hypothesis
remain fixed. Licensed body parameters and deformation tables remain local.
"""
from pathlib import Path
import json, time, shutil, hashlib
import numpy as np
import torch
import smplx
from smplx.lbs import lbs, blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
SRC = LAB/'outputs/dynamic-human-enerf-continuous-body-063'
SEQ = LAB/'outputs/dynamic-human-enerf-pose-sequence-062'
OLD = LAB/'outputs/dynamic-human-enerf-wholebody-059'
OUT = LAB/'outputs/dynamic-human-enerf-direct-pose-065'
OUT.mkdir(exist_ok=True)
assert not (OUT/'completion.json').exists()
read = lambda p: json.loads(p.read_text())
save = lambda p,x: p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
t = lambda x: torch.as_tensor(x,dtype=torch.float32,device='cuda')
torch.set_num_threads(4); torch.manual_seed(73)
f = dict(np.load(SRC/'estimated_offsets/motion-fit.npz'))
ref = dict(np.load(SRC/'reference.npz'))
model = smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=1).cuda()
for p in model.parameters(): p.requires_grad_(False)
betas=t(f['betas']); scales=t(f['scales']); basepose=t(f['basepose']); basetrans=t(f['basetrans'])
pose0=t(f['posecoef']); trans0=t(f['transcoef'])
posecoef=torch.nn.Parameter(pose0.clone()); transcoef=torch.nn.Parameter(trans0.clone())
mask=torch.ones(72,device='cuda')
for j in [20,21,22,23]: mask[3*j:3*j+3]=0
ids=[24,26,25,28,27,16,17,18,19,20,21,1,2,4,5,7,8,29,30,31,32,33,34]
types=['person_dark_clothing','person_light_clothing']
support={(r['person'],r['index']):{o['camera'] for o in r.get('observations',[]) if o['inlier']} for r in read(OLD/'triangulation.json')['records'] if r['accepted']}
cameras=[str(c) for c in f['cameras']]; offsets=dict(zip(cameras,f['camera_offsets']))
observations=read(SEQ/'observations.json')['records']
contexts=sorted({(r['camera'],r['frame']) for r in observations if r['frame']%2==0})
context_index={x:i for i,x in enumerate(contexts)}
rows=[]
for r in observations:
 if r['frame']%2: continue
 for pi,person in enumerate(types):
  d=next(x for x in r['detections'] if x['person']==person)
  for j in range(23):
   if pi==1 and j<3: continue
   if r['camera'] not in support.get((person,j),set()) or d['scores'][j]<.4: continue
   rows.append(dict(camera=r['camera'],frame=r['frame'],person=pi,joint=j,context=context_index[r['camera'],r['frame']],xy=d['keypoints'][j],P=(np.array(r['K'])@np.array(r['w2c'])[:3]).tolist(),weight=float(min(1,d['scores'][j]))*(.5 if j<5 or j>=17 else 1)))
assert len(contexts)==105 and not any(r['frame']%2 for r in rows)
save(OUT/'protocol.json',dict(status='fixed_before_direct_pose_refinement',source_sha256=sha(SRC/'estimated_offsets/motion-fit.npz'),updates=3000,
 fit_frames=list(range(744,757,2)),validation_frames=list(range(745,756,2)),fit_contexts=105,fit_observations=len(rows),
 optimized='Cubic continuous global/body axis angles and translation. Neck/head temporal rotations now enabled; local wrists/hands frozen. Shape, scale, cameras and timing fixed.',
 targets='Original RTMW predictions, fixed 059 camera-joint inlier support, confidence >=0.4. Light-person nose/eyes excluded as in060; no finger constraints. Head and feet half weighted.',
 loss='Mean confidence-weighted coordinate-wise softL1 with delta3px; +0.5 mean pose change squared +2 mean change of higher-order pose coefficients squared +0.2 mean translation change squared /0.2^2. Changes relative063. No odd-frame targets in fitting.',
 rationale='Direct image-plane fitting avoids equal 3D residual weighting; enabling head motion addresses frozen head orientation. This is a combined pose-model change, not an isolated timing ablation.',
 qualification='Development time validation, predicted keypoints not GT. Projection improvement is not RGB clarity acceptance.',licensed_data='Parameters/mesh/affines local only',goal_complete=False))
save(OUT/'fit-observation-index.json',dict(records=[{k:v for k,v in r.items() if k not in ['P','xy']} for r in rows]))
times=t([frame+offsets[c] for c,frame in contexts]); tau=(times-750)/6
basis=torch.stack([tau**k for k in range(4)],-1)
ci=torch.tensor([r['context'] for r in rows],device='cuda'); pi=torch.tensor([r['person'] for r in rows],device='cuda'); ji=torch.tensor([r['joint'] for r in rows],device='cuda')
P=t([r['P'] for r in rows]); xy=t([r['xy'] for r in rows]); weights=t([r['weight'] for r in rows])
extra=model.vertex_joint_selector.extra_joints_idxs[:11]
vshape=model.v_template+blend_shapes(betas,model.shapedirs)
Jrest=vertices2joints(model.J_regressor,vshape)
posedirs=model.posedirs.reshape(207,6890,3)[:,extra].reshape(207,-1)

def pose_query(b):
 return basepose[:,None]+torch.einsum('tk,pkd->ptd',b,posecoef)*mask,basetrans[:,None]+torch.einsum('tk,pkd->ptd',b,transcoef)

def fast_landmarks(pose,trans):
 # Exact SMPL joint and selected vertex operations, avoiding unused mesh vertices.
 n=pose.shape[1]; batch=2*n; rot=batch_rodrigues(pose.reshape(-1,3)).reshape(batch,24,3,3)
 rest=Jrest[:,None].expand(-1,n,-1,-1).reshape(batch,24,3)
 joints,A=batch_rigid_transform(rot,rest,model.parents,dtype=torch.float32)
 pf=(rot[:,1:]-torch.eye(3,device='cuda')).reshape(batch,207)
 vpos=(pf@posedirs).reshape(batch,11,3)+vshape[:,None,extra].expand(-1,n,-1,-1).reshape(batch,11,3)
 T=(model.lbs_weights[extra][None]@A.reshape(batch,24,16)).reshape(batch,11,4,4)
 v=(T@torch.cat([vpos,torch.ones(batch,11,1,device='cuda')],-1)[...,None])[:,:,:3,0]
 allj=torch.cat([joints,v],1).reshape(2,n,35,3)
 return allj[:,:,ids]*scales[:,None,None,None]+trans[:,:,None,:]

# Full official LBS equality and a genuine gradient equality check before fitting.
ptest,trtest=pose_query(basis[:3]); ptest=ptest.detach().requires_grad_(True); trtest=trtest.detach()
jfast=fast_landmarks(ptest,trtest)
vfull,jfull=lbs(betas[:,None].expand(-1,3,-1).reshape(6,10),ptest.reshape(6,72),model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights)
jfull=model.vertex_joint_selector(vfull,jfull).reshape(2,3,-1,3)[:,:,ids]*scales[:,None,None,None]+trtest[:,:,None,:]
err=float((jfast-jfull).abs().max()); probe=torch.randn_like(jfast)
gfast=torch.autograd.grad((jfast*probe).sum(),ptest)[0]; gfull=torch.autograd.grad((jfull*probe).sum(),ptest)[0]
gerr=float((gfast-gfull).abs().max()); assert err<2e-5 and gerr<2e-4,(err,gerr)
save(OUT/'subset-forward-gradient-audit.json',dict(status='actual_SMPL_full_forward_and_pose_gradient_equal',max_world_difference=err,max_pose_gradient_difference=gerr))
del jfast,jfull,vfull,gfast,gfull,ptest
opt=torch.optim.Adam([dict(params=[posecoef],lr=.003),dict(params=[transcoef],lr=.001)])
history=[]; start=time.monotonic()
for step in range(3000):
 pose,trans=pose_query(basis); X=fast_landmarks(pose,trans)[pi,ci,ji]
 q=(P[:,:,:3]@X[...,None])[...,0]+P[:,:,3]; uv=q[:,:2]/q[:,2:]
 residual=uv-xy; robust=9*(torch.sqrt(1+(residual/3)**2)-1)
 dp=torch.einsum('tk,pkd->ptd',basis,posecoef-pose0)*mask; dt=torch.einsum('tk,pkd->ptd',basis,transcoef-trans0)
 regularizer=.5*dp.square().mean()+2*((posecoef-pose0)[:,2:]*mask).square().mean()+.2*(dt/.2).square().mean()
 loss=(robust*weights[:,None]).sum()/(2*weights.sum())+regularizer
 opt.zero_grad();loss.backward();assert torch.isfinite(loss) and torch.isfinite(posecoef.grad).all();opt.step()
 for group,lr in zip(opt.param_groups,[.003,.001]):group['lr']=lr*.1**((step+1)/3000)
 if (step+1)%250==0:
  e=torch.linalg.vector_norm(residual,dim=-1);row=dict(step=step+1,loss=float(loss),median_train_px=float(e.median()),p90_train_px=float(torch.quantile(e,.9)),regularizer=float(regularizer),elapsed_seconds=time.monotonic()-start)
  history.append(row);save(OUT/'progress.json',row);print(json.dumps(row),flush=True)
result={**f,'posecoef':posecoef.detach().cpu().numpy(),'transcoef':transcoef.detach().cpu().numpy(),'mask':mask.cpu().numpy()}
np.savez_compressed(OUT/'motion-fit.npz',**result)
contexts_all=[(c,frame,float(frame+offsets[c])) for c in cameras for frame in range(744,757)]
arrA=[];arrb=[];arrv=[];v0=t(ref['vertices']);inv0=t(ref['inverse_linear'])
with torch.no_grad():
 for begin in range(0,len(contexts_all),16):
  ts=t([r[2] for r in contexts_all[begin:begin+16]]);bb=torch.stack([((ts-750)/6)**k for k in range(4)],-1);n=len(ts);pose,trans=pose_query(bb)
  v,j,M=lbs(betas[:,None].expand(-1,n,-1).reshape(-1,10),pose.reshape(-1,72),model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights,return_T=True)
  v=v.reshape(2,n,6890,3)*scales[:,None,None,None]+trans[:,:,None,:]
  M=M[:,:,:3,:3].reshape(2,n,6890,3,3)*scales[:,None,None,None,None]
  A=M@inv0[:,None];b=v-(A@v0[:,None,:,:,None])[...,0]
  arrA.append(A.permute(1,0,2,3,4).reshape(-1,13780,3,3).cpu().numpy());arrb.append(b.permute(1,0,2,3).reshape(-1,13780,3).cpu().numpy());arrv.append(v.permute(1,0,2,3).reshape(-1,13780,3).cpu().numpy())
np.savez_compressed(OUT/'capture-affines.npz',camera=np.array([r[0] for r in contexts_all]),frame=np.array([r[1] for r in contexts_all]),capture_time=np.array([r[2] for r in contexts_all]),linear=np.concatenate(arrA),translation=np.concatenate(arrb),vertices=np.concatenate(arrv))
save(OUT/'completion.json',dict(status='3000update_direct_multiview_continuous_pose_refinement_complete_independent_checks_pending',history=history,contexts=195,fit_observations=len(rows),motion_sha256=sha(OUT/'motion-fit.npz'),affines_sha256=sha(OUT/'capture-affines.npz'),goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)
