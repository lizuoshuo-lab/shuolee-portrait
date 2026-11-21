"""Refine a local SMPL hypothesis with RTMW body/feet and visible head landmarks."""
from pathlib import Path
import json,time,shutil,hashlib
import numpy as np,torch,smplx
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');SRC=LAB/'outputs/dynamic-human-enerf-wholebody-059';OLD=LAB/'outputs/dynamic-human-enerf-body-prior-058';OUT=LAB/'outputs/dynamic-human-enerf-wholebody-prior-060'
OUT.mkdir(exist_ok=True);assert not (OUT/'fit.npz').exists()
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
z=np.load(SRC/'triangulated-wholebody.npz');target=z['points'][:,:23];valid=z['valid'][:,:23].copy();valid[1,:3]=False
# Verified from local SMPL VertexJointSelector: face= nose,reye,leye,rear,lear; feet=Lbig,Lsmall,Lheel,Rbig,Rsmall,Rheel.
ids=[24,26,25,28,27,16,17,18,19,20,21,1,2,4,5,7,8,29,30,31,32,33,34]
save(OUT/'protocol.json',dict(status='fixed_before_fitting',source='059 robust training-only RTMW triangulation; 23 body/head/foot landmarks per person',
 excluded='Light-clothing nose/eyes excluded after back-facing photo review; all fingertips excluded because basic SMPL lacks finger articulation. No new manual point coordinates.',
 smpl_joint_mapping=ids,updates=2500,initialization='058 fitted SMPL parameters',
 loss='Huber normalized3D residual delta0.1; equal joint weights, except face/feet weight0.5, +0.0002meanpose^2 +0.002meanbeta^2 +0.01logscale change^2. Adam with cosine-like exponential LR decay.',
 scope='SMPL structural prior, not clothed surface truth. Shoes, hair, clothing and toys not represented; no heldout data or appearance training.',licensed_parameters='local only',goal_complete=False))
torch.set_num_threads(4);torch.manual_seed(73)
model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=2).cuda()
for p in model.parameters():p.requires_grad_(False)
assert model.vertex_joint_selector.extra_joints_idxs[:11].tolist()==[332,6260,2800,4071,583,3216,3226,3387,6617,6624,6787]
t=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
old=np.load(OLD/'fit.npz');betas=torch.nn.Parameter(t(old['betas']));pose=torch.nn.Parameter(t(old['body_pose']).reshape(2,69));orient=torch.nn.Parameter(t(old['global_orient']));trans=torch.nn.Parameter(t(old['translation']));logscale=torch.nn.Parameter(t(np.log(old['scale'])));log0=logscale.detach().clone()
target_t=t(np.nan_to_num(target));weight=t(valid);weight[:,:5]*=.5;weight[:,17:]*=.5
norm=torch.linalg.vector_norm(target_t[:,5:7].mean(1)-target_t[:,11:13].mean(1),dim=-1).clamp_min(.05)
def forward():
 r=model(betas=betas,body_pose=pose,global_orient=orient,return_verts=True)
 return r,r.joints*logscale.exp()[:,None,None]+trans[:,None,:]
opt=torch.optim.Adam([dict(params=[pose,orient],lr=.01),dict(params=[betas,logscale,trans],lr=.003)])
history=[];start=time.monotonic()
with torch.no_grad():initial=(forward()[1][:,ids]-target_t).norm(dim=-1).cpu().numpy()
for step in range(2500):
 result,world=forward();diff=(world[:,ids]-target_t)/norm[:,None,None]
 data=torch.nn.functional.huber_loss(diff,torch.zeros_like(diff),reduction='none',delta=.1).sum(-1)
 loss=(data*weight).sum()/weight.sum()+.0002*pose.square().mean()+.002*betas.square().mean()+.01*(logscale-log0).square().mean()
 opt.zero_grad();loss.backward();assert torch.isfinite(loss);opt.step()
 for group,lr0 in zip(opt.param_groups,[.01,.003]):group['lr']=lr0*(.1**((step+1)/2500))
 if (step+1)%250==0:
  row=dict(step=step+1,loss=float(loss),mean_weighted_joint_error_world=float(((world[:,ids]-target_t).norm(dim=-1)*weight).sum()/weight.sum()),elapsed_seconds=time.monotonic()-start)
  history.append(row);save(OUT/'progress.json',row);print(json.dumps(row),flush=True)
with torch.no_grad():
 result,world=forward();verts=result.vertices*logscale.exp()[:,None,None]+trans[:,None,:];errors=(world[:,ids]-target_t).norm(dim=-1).cpu().numpy()
np.savez_compressed(OUT/'fit.npz',betas=betas.detach().cpu().numpy(),body_pose=pose.detach().cpu().numpy().reshape(2,23,3),global_orient=orient.detach().cpu().numpy(),scale=logscale.exp().detach().cpu().numpy(),translation=trans.detach().cpu().numpy(),vertices=verts.cpu().numpy(),faces=model.faces,joints=world.cpu().numpy(),target=target,valid=valid,smpl_joint_ids=np.array(ids))
save(OUT/'fit-summary.json',dict(status='refined_training_only_prior_projection_review_pending',updates=2500,source_sha256=hashlib.sha256((SRC/'triangulated-wholebody.npz').read_bytes()).hexdigest(),initial_source_sha256=hashlib.sha256((OLD/'fit.npz').read_bytes()).hexdigest(),history=history,
 fitting_targets=43,initial_mean_valid_error_world=float(initial[valid].mean()),final_mean_valid_error_world=float(errors[valid].mean()),initial_per_joint_error_world=initial.tolist(),final_per_joint_error_world=errors.tolist(),limits='Errors against predicted landmarks, not GT; world units not independently certified as metres.',goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)
