"""Fit a local SMPL body prior to reviewed training-only triangulated joints."""
from pathlib import Path
import json,time,shutil,hashlib
import numpy as np,torch,smplx
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rodrigues,batch_rigid_transform
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');SRC=LAB/'outputs/dynamic-human-enerf-body-observations-057';OUT=LAB/'outputs/dynamic-human-enerf-body-prior-058'
OUT.mkdir(exist_ok=True);assert not (OUT/'fit.npz').exists()
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
z=np.load(SRC/'triangulated-body-joints.npz');valid=z['valid'];target=np.nan_to_num(z['points']);ids=[16,17,18,19,20,21,1,2,4,5,7,8];torso=[0,1,6,7]
assert valid.sum()==23 and valid[:,torso].all()
save(OUT/'protocol.json',dict(status='fixed_before_fitting',source='057 training-only robust body joints;23accepted, dark leftelbow missing',
 smpl_joint_mapping=ids,optimization='2000AdamCPUupdates, fullbody23axisanglepose,10shapecomponents,globalrotation/translation/positiveuniformscale; joint-only FK fitting checked against full SMPL at end.',
 loss='Huber normalized3Djoint residual delta0.1 +.001meanbodypose^2 +.005meanbeta^2 +.01logscale_relative_to_torso_initial^2',
 limits='Coarse body prior, not ground-truth pose or clothing surface. No appearance optimization, no fingers/toys model, no heldout images. Licensed template and mesh remain local.',goal_complete=False))
torch.set_num_threads(4);torch.manual_seed(73)
model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=2).cpu()
for p in model.parameters():p.requires_grad_(False)
J0=model.J_regressor@model.v_template
Jdirs=torch.einsum('jv,vck->jck',model.J_regressor,model.shapedirs[:,:,:10])
target_t=torch.tensor(target,dtype=torch.float32);weight=torch.tensor(valid,dtype=torch.float32)
rot=[];scale=[]
for i in range(2):
 a=J0[ids].numpy()[torso];b=target[i,torso];ac=a-a.mean(0);bc=b-b.mean(0);u,d,vt=np.linalg.svd(ac.T@bc);R=vt.T@u.T
 if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
 sc=float(np.sum(bc*(ac@R.T))/np.sum(ac*ac));assert sc>0
 rot.append(Rotation.from_matrix(R).as_rotvec());scale.append(sc)
betas=torch.nn.Parameter(torch.zeros(2,10));pose=torch.nn.Parameter(torch.zeros(2,23,3));globalrot=torch.nn.Parameter(torch.tensor(np.array(rot),dtype=torch.float32));logscale=torch.nn.Parameter(torch.tensor(np.log(scale),dtype=torch.float32));scale0=logscale.detach().clone()
def joints():
 J=J0[None]+torch.einsum('bk,jck->bjc',betas,Jdirs)
 R=batch_rodrigues(torch.cat([globalrot[:,None,:],pose],1).reshape(-1,3)).reshape(2,24,3,3)
 p,_=batch_rigid_transform(R,J,model.parents,dtype=torch.float32);return p
with torch.no_grad():trans0=target_t[:,torso].mean(1)-joints()[:,ids][:,torso].mean(1)*logscale.exp()[:,None]
trans=torch.nn.Parameter(trans0);norm=torch.linalg.vector_norm(target_t[:,:2].mean(1)-target_t[:,6:8].mean(1),dim=-1).clamp_min(.05)
opt=torch.optim.Adam([dict(params=[pose,globalrot],lr=.03),dict(params=[betas,logscale,trans],lr=.01)])
history=[];start=time.monotonic()
for step in range(2000):
 pred=joints()[:,ids]*logscale.exp()[:,None,None]+trans[:,None,:];diff=(pred-target_t)/norm[:,None,None]
 data=torch.nn.functional.huber_loss(diff,torch.zeros_like(diff),reduction='none',delta=.1).sum(-1)
 loss=(data*weight).sum()/weight.sum()+.001*pose.square().mean()+.005*betas.square().mean()+.01*(logscale-scale0).square().mean()
 opt.zero_grad();loss.backward();assert torch.isfinite(loss);opt.step()
 if (step+1)%200==0:
  row=dict(step=step+1,loss=float(loss),mean_valid_joint_error_world=float((torch.linalg.vector_norm(pred-target_t,dim=-1)*weight).sum()/weight.sum()),elapsed_seconds=time.monotonic()-start)
  history.append(row);save(OUT/'progress.json',row);print(json.dumps(row),flush=True)
with torch.no_grad():
 result=model(betas=betas,body_pose=pose.reshape(2,69),global_orient=globalrot,return_verts=True)
 J=joints();err=float((J-result.joints[:,:24]).abs().max());assert err<1e-5,err
 vertices=result.vertices*logscale.exp()[:,None,None]+trans[:,None,:];worldJ=J*logscale.exp()[:,None,None]+trans[:,None,:]
np.savez_compressed(OUT/'fit.npz',betas=betas.detach().numpy(),body_pose=pose.detach().numpy(),global_orient=globalrot.detach().numpy(),scale=logscale.exp().detach().numpy(),translation=trans.detach().numpy(),vertices=vertices.numpy(),faces=model.faces,joints=worldJ.numpy(),target=target,valid=valid,smpl_joint_ids=np.array(ids))
save(OUT/'fit-summary.json',dict(status='SMPL_joint_prior_fitted_fullforward_consistency_passed_projection_review_pending',updates=2000,source_sha256=hashlib.sha256((SRC/'triangulated-body-joints.npz').read_bytes()).hexdigest(),
 max_fk_vs_full_smpl_joint_error=err,history=history,scales=logscale.exp().detach().tolist(),betas=betas.detach().tolist(),
 per_joint_error_world=np.linalg.norm(worldJ.numpy()[:,ids]-target,axis=-1).tolist(),scope='Body prior only, no reconstruction or clarity claim.',goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)
