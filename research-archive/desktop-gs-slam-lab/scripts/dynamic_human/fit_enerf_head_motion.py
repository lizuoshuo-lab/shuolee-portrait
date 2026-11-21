"""Isolated neck/head motion refinement; all other pose coefficients are frozen."""
from pathlib import Path
import json,time,hashlib,shutil
import numpy as np,torch,smplx
from smplx.lbs import lbs,blend_shapes,vertices2joints,batch_rodrigues,batch_rigid_transform
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');SRC=LAB/'outputs/dynamic-human-enerf-continuous-body-063';SEQ=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';OLD=LAB/'outputs/dynamic-human-enerf-wholebody-059';OUT=LAB/'outputs/dynamic-human-enerf-head-motion-066';OUT.mkdir(exist_ok=True)
assert not (OUT/'completion.json').exists()
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();t=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
torch.set_num_threads(4);torch.manual_seed(73)
f=dict(np.load(SRC/'estimated_offsets/motion-fit.npz'));ref=dict(np.load(SRC/'reference.npz'))
model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=1).cuda()
for p in model.parameters():p.requires_grad_(False)
betas=t(f['betas']);scales=t(f['scales']);basepose=t(f['basepose']);basetrans=t(f['basetrans']);coef0=t(f['posecoef']);transcoef=t(f['transcoef']);mask=t(f['mask']);change_indices=torch.tensor([36,37,38,45,46,47],device='cuda');mask[change_indices]=1
delta=torch.nn.Parameter(torch.zeros(2,4,6,device='cuda'))
cameras=[str(c) for c in f['cameras']];offsets=dict(zip(cameras,f['camera_offsets']));types=['person_dark_clothing','person_light_clothing'];ids=[24,26,25,28,27]
support={(r['person'],r['index']):{o['camera'] for o in r.get('observations',[]) if o['inlier']} for r in read(OLD/'triangulation.json')['records'] if r['accepted']}
obs=read(SEQ/'observations.json')['records'];contexts=sorted({(r['camera'],r['frame']) for r in obs if r['frame']%2==0});lookup={v:i for i,v in enumerate(contexts)};rows=[]
for r in obs:
 if r['frame']%2:continue
 for pi,person in enumerate(types):
  detection=next(d for d in r['detections'] if d['person']==person)
  for j in range(5):
   if pi==1 and j<3:continue
   if r['camera'] not in support.get((person,j),set()) or detection['scores'][j]<.4:continue
   rows.append(dict(camera=r['camera'],frame=r['frame'],person=pi,joint=j,context=lookup[r['camera'],r['frame']],xy=detection['keypoints'][j],P=(np.array(r['K'])@np.array(r['w2c'])[:3]).tolist(),weight=float(min(1,detection['scores'][j]))))
assert len(rows)==546 and len(contexts)==105
save(OUT/'protocol.json',dict(status='fixed_before_head_only_fit',source_sha256=sha(SRC/'estimated_offsets/motion-fit.npz'),updates=3000,fit_frames=list(range(744,757,2)),validation_frames=list(range(745,756,2)),fit_contexts=105,fit_observations=len(rows),
 optimized='Only 2persons x4polynomialcoefficients x2joints x3axisangle components for neck12/head15; no other optimizer parameters.',frozen='All non-neck/head pose coefficients, global translation, shape, scale, calibration and camera timing exactly063estimated branch used by064 baseline.',
 targets='Same546training-only supported head observations as065; dark5face points, light2ears only; RTMWscore>=0.4 and059fixedcamera support. No body/foot/hand or odd-time observations in fitting.',
 loss='Mean confidence weighted coordinate-wise softL1 delta3px +0.5mean head/neck angle change^2 +2mean higher-order head/neck coefficient change^2. Adam lr0.003 exponential decayto0.0003.',
 caveat='Freezing body joint rotations preserves the skeleton but SMPL pose-corrective blendshapes can slightly move non-head vertices; full mesh differences will be measured. Reconstruction appearance is still learned globally from identical105images.',
 downstream='Identical canonical reference and exact affine covariance renderer, identical15000Gaussianupdates andfixed development-time validation.',licensed_data='Bodyparameters/mesh/deformationtables local only',goal_complete=False))
save(OUT/'fit-observation-index.json',dict(records=[{k:v for k,v in r.items() if k not in ['P','xy']} for r in rows]))
times=t([frame+offsets[c] for c,frame in contexts]);basis=torch.stack([((times-750)/6)**k for k in range(4)],-1);pi=torch.tensor([r['person'] for r in rows],device='cuda');ci=torch.tensor([r['context'] for r in rows],device='cuda');ji=torch.tensor([r['joint'] for r in rows],device='cuda');P=t([r['P'] for r in rows]);xy=t([r['xy'] for r in rows]);weight=t([r['weight'] for r in rows])
extra=model.vertex_joint_selector.extra_joints_idxs[:5];vshape=model.v_template+blend_shapes(betas,model.shapedirs);Jrest=vertices2joints(model.J_regressor,vshape);pd=model.posedirs.reshape(207,6890,3)[:,extra].reshape(207,-1)
def coefficients():return coef0+torch.zeros_like(coef0).index_copy(-1,change_indices,delta)
def query(b):return basepose[:,None]+torch.einsum('tk,pkd->ptd',b,coefficients())*mask,basetrans[:,None]+torch.einsum('tk,pkd->ptd',b,transcoef)
def face_landmarks(pose,translation):
 n=pose.shape[1];batch=2*n;rotation=batch_rodrigues(pose.reshape(-1,3)).reshape(batch,24,3,3);j,A=batch_rigid_transform(rotation,Jrest[:,None].expand(-1,n,-1,-1).reshape(batch,24,3),model.parents,dtype=torch.float32)
 vpos=((rotation[:,1:]-torch.eye(3,device='cuda')).reshape(batch,207)@pd).reshape(batch,5,3)+vshape[:,None,extra].expand(-1,n,-1,-1).reshape(batch,5,3);T=(model.lbs_weights[extra][None]@A.reshape(batch,24,16)).reshape(batch,5,4,4)
 v=(T@torch.cat([vpos,torch.ones(batch,5,1,device='cuda')],-1)[...,None])[:,:,:3,0]
 return v.reshape(2,n,5,3)[:,:,[0,2,1,4,3]]*scales[:,None,None,None]+translation[:,:,None,:]
# Compare actual selected vertices and their pose gradients with full LBS.
pose,tr=query(basis[:3]);pose=pose.detach().requires_grad_(True);tr=tr.detach();short=face_landmarks(pose,tr)
v,j=lbs(betas[:,None].expand(-1,3,-1).reshape(6,10),pose.reshape(6,72),model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights);full=model.vertex_joint_selector(v,j).reshape(2,3,-1,3)[:,:,ids]*scales[:,None,None,None]+tr[:,:,None,:]
probe=torch.randn_like(short);gs=torch.autograd.grad((short*probe).sum(),pose)[0];gf=torch.autograd.grad((full*probe).sum(),pose)[0];verr=float((short-full).abs().max());gerr=float((gs-gf).abs().max());assert verr<2e-5 and gerr<2e-4
save(OUT/'subset-forward-gradient-audit.json',dict(status='actual5face_vertices_and_pose_gradients_match_full_SMPL',max_world_difference=verr,max_pose_gradient_difference=gerr))
del pose,tr,short,full,v,j,gs,gf
opt=torch.optim.Adam([delta],lr=.003);start=time.monotonic();history=[]
for step in range(3000):
 pose,tr=query(basis);X=face_landmarks(pose,tr)[pi,ci,ji];q=(P[:,:,:3]@X[...,None])[...,0]+P[:,:,3];residual=q[:,:2]/q[:,2:]-xy;robust=9*(torch.sqrt(1+(residual/3)**2)-1);change=torch.einsum('tk,pkd->ptd',basis,delta);regularizer=.5*change.square().mean()+2*delta[:,2:].square().mean();loss=(robust*weight[:,None]).sum()/(2*weight.sum())+regularizer
 opt.zero_grad();loss.backward();assert torch.isfinite(loss) and torch.isfinite(delta.grad).all();opt.step();opt.param_groups[0]['lr']=.003*.1**((step+1)/3000)
 if (step+1)%250==0:
  err=torch.linalg.vector_norm(residual,dim=-1);row=dict(step=step+1,loss=float(loss),median_head_fit_px=float(err.median()),p90_head_fit_px=float(torch.quantile(err,.9)),regularizer=float(regularizer),elapsed_seconds=time.monotonic()-start);history.append(row);save(OUT/'progress.json',row);print(json.dumps(row),flush=True)
result={**f,'posecoef':coefficients().detach().cpu().numpy(),'mask':mask.cpu().numpy()};unchanged=np.ones(72,bool);unchanged[change_indices.cpu().numpy()]=False;assert np.array_equal(result['posecoef'][:,:,unchanged],f['posecoef'][:,:,unchanged]) and np.array_equal(result['transcoef'],f['transcoef'])
np.savez_compressed(OUT/'motion-fit.npz',**result);contexts_all=[(c,frame,float(frame+offsets[c])) for c in cameras for frame in range(744,757)];arrA=[];arrb=[];arrv=[];v0=t(ref['vertices']);inv0=t(ref['inverse_linear'])
with torch.no_grad():
 for begin in range(0,195,16):
  ts=t([r[2] for r in contexts_all[begin:begin+16]]);b=torch.stack([((ts-750)/6)**k for k in range(4)],-1);n=len(ts);pose,tr=query(b);v,j,M=lbs(betas[:,None].expand(-1,n,-1).reshape(-1,10),pose.reshape(-1,72),model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights,return_T=True)
  v=v.reshape(2,n,6890,3)*scales[:,None,None,None]+tr[:,:,None,:];M=M[:,:,:3,:3].reshape(2,n,6890,3,3)*scales[:,None,None,None,None];A=M@inv0[:,None];bias=v-(A@v0[:,None,:,:,None])[...,0];arrA.append(A.permute(1,0,2,3,4).reshape(-1,13780,3,3).cpu().numpy());arrb.append(bias.permute(1,0,2,3).reshape(-1,13780,3).cpu().numpy());arrv.append(v.permute(1,0,2,3).reshape(-1,13780,3).cpu().numpy())
np.savez_compressed(OUT/'capture-affines.npz',camera=np.array([r[0] for r in contexts_all]),frame=np.array([r[1] for r in contexts_all]),capture_time=np.array([r[2] for r in contexts_all]),linear=np.concatenate(arrA),translation=np.concatenate(arrb),vertices=np.concatenate(arrv))
save(OUT/'completion.json',dict(status='3000updates_head_only_motion_fit_complete_independent_audit_pending',history=history,optimized_parameters=48,frozen_other_pose_coefficients_exact=True,translation_exact=True,motion_sha256=sha(OUT/'motion-fit.npz'),affines_sha256=sha(OUT/'capture-affines.npz'),goal_complete=False));shutil.copy2(__file__,OUT/Path(__file__).name)
