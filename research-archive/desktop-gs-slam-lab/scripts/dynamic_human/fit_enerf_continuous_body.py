"""Two matched continuous SMPL motion hypotheses from training-only trajectories."""
from pathlib import Path
import json,time,shutil,hashlib
import numpy as np,torch,smplx
from smplx.lbs import lbs
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');SOURCE=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';REF=LAB/'outputs/dynamic-human-enerf-wholebody-prior-060';OUT=LAB/'outputs/dynamic-human-enerf-continuous-body-063';OUT.mkdir(exist_ok=True)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'completion.json').exists();torch.set_num_threads(4);torch.manual_seed(73)
old=np.load(REF/'fit.npz');tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=1).cuda()
for p in model.parameters():p.requires_grad_(False)
basepose=tensor(np.c_[old['global_orient'],old['body_pose'].reshape(2,69)]);basetrans=tensor(old['translation']);betas=tensor(old['betas']);scales=tensor(old['scale']);mask=torch.ones(72,device='cuda')
# No head or hand orientation targets in062; retain these local rotations from060.
for j in [12,15,20,21,22,23]:mask[j*3:(j+1)*3]=0
ids=[16,17,18,19,20,21,1,2,4,5,7,8,29,30,31,32,33,34]
times=torch.arange(742,759,device='cuda',dtype=torch.float32);tau=(times-750)/6;basis=torch.stack([tau**k for k in range(4)],-1);T=len(times)
save(OUT/'protocol.json',dict(status='fixed_before_motion_fitting',branches=['fixed_offsets','estimated_offsets'],
 source='062 cubic3Djoint trajectories from seven even trainingtimes. No RGB or reservedodd-timekeypoints added tofit.',
 parameters='Shared060shape/scale fixed; globalandbodyaxisangles plusworldtranslation as cubicpolynomials. Neck/head/wrist/handlocalrotations frozen060 because no reliableorientationtargets.',
 samples='17uniform physicalframeindexvalues742..758 spanningallcamera capturetimes; targets evaluatedanalytically from062training-only trajectory.',
 loss='Normalized3Djoint Huber delta0.08;feetweight0.5;0.0002posechange^2 +0.0003higherorderposecoeff^2 +0.0001translationchange^2 normalizedbytorsoheight.',
 updates=2500,canonical='Exact060referencepose, shape,scaleandvertices shared by bothbranches. Referenceworldcoordinates usedforGaussianbinding.',
 downstream='Affineat each SMPLvertex: currentblendedlineartransform timesinverse referenceblendedlineartransform, anchored at actual posedvertex. IncludesSMPLposecorrectivevertexmotion; not naive sumofjointrelative transforms.',
 scope='Continuousbody motion prior, not photoreal reconstruction. Clothing/hair/toys remainoutsideSMPLsurface; offsetsarequalifiedhypotheses.',licensed_data='alltemplates,fitparameters,meshesanddeformationtableslocalonly',goal_complete=False))
def skin(beta,pose,scale,trans):
 v,j,mat=lbs(beta,pose,model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights,return_T=True)
 j=model.vertex_joint_selector(v,j);return v*scale[:,None,None]+trans[:,None,:],j*scale[:,None,None]+trans[:,None,:],mat[:,:,:3,:3]*scale[:,None,None,None]
with torch.no_grad():
 v0,j0,M0=skin(betas,basepose,scales,basetrans);reference_error=float(abs(v0-tensor(old['vertices'])).max());assert reference_error<2e-5
 inv0=torch.linalg.inv(M0);identity_error=float(abs(M0@inv0-torch.eye(3,device='cuda')).max());assert identity_error<1e-4
np.savez_compressed(OUT/'reference.npz',vertices=v0.cpu().numpy(),joints=j0.cpu().numpy(),linear=M0.cpu().numpy(),inverse_linear=inv0.cpu().numpy(),faces=model.faces,betas=old['betas'],scale=old['scale'],basepose=basepose.cpu().numpy(),translation=old['translation'])
rows=[]
for label in ['fixed_offsets','estimated_offsets']:
 root=OUT/label;root.mkdir(exist_ok=True);assert not (root/'motion-fit.npz').exists();source=np.load(SOURCE/f'{label}-fit.npz');co=tensor(source['coefficients'].reshape(2,18,4,3));target=torch.einsum('tk,pjkd->ptjd',basis,co)
 posecoef=torch.nn.Parameter(torch.zeros(2,4,72,device='cuda'));transcoef=torch.nn.Parameter(torch.zeros(2,4,3,device='cuda'));opt=torch.optim.Adam([dict(params=[posecoef],lr=.01),dict(params=[transcoef],lr=.003)])
 norm=torch.linalg.vector_norm(target[:,:,0:2].mean(2)-target[:,:,6:8].mean(2),dim=-1).mean(1).clamp_min(.05);weights=torch.ones(18,device='cuda');weights[12:]=.5;history=[];began=time.monotonic()
 def query(b):
  n=len(b);pd=torch.einsum('tk,pkd->ptd',b,posecoef)*mask;td=torch.einsum('tk,pkd->ptd',b,transcoef);pose=basepose[:,None]+pd;trans=basetrans[:,None]+td
  v,j,M=skin(betas[:,None].expand(-1,n,-1).reshape(-1,10),pose.reshape(-1,72),scales[:,None].expand(-1,n).reshape(-1),trans.reshape(-1,3))
  return v.reshape(2,n,6890,3),j.reshape(2,n,-1,3),M.reshape(2,n,6890,3,3),pd,td
 for step in range(2500):
  v,j,M,pd,td=query(basis);diff=(j[:,:,ids]-target)/norm[:,None,None,None];data=torch.nn.functional.huber_loss(diff,torch.zeros_like(diff),delta=.08,reduction='none').sum(-1)
  loss=(data*weights).sum()/(2*T*weights.sum())+.0002*pd.square().mean()+.0003*(posecoef[:,2:]*mask).square().mean()+.0001*(td/norm[:,None,None]).square().mean()
  opt.zero_grad();loss.backward();assert torch.isfinite(loss);opt.step()
  for group,lr in zip(opt.param_groups,[.01,.003]):group['lr']=lr*(.1**((step+1)/2500))
  if (step+1)%250==0:
   row=dict(step=step+1,loss=float(loss),joint_error_world=float(torch.linalg.vector_norm(j[:,:,ids]-target,dim=-1).mean()),elapsed_seconds=time.monotonic()-began);history.append(row);save(root/'progress.json',row);print(label,json.dumps(row),flush=True)
 np.savez_compressed(root/'motion-fit.npz',posecoef=posecoef.detach().cpu().numpy(),transcoef=transcoef.detach().cpu().numpy(),basepose=basepose.cpu().numpy(),basetrans=basetrans.cpu().numpy(),mask=mask.cpu().numpy(),betas=betas.cpu().numpy(),scales=scales.cpu().numpy(),joint_coefficients=source['coefficients'],camera_offsets=source['offsets'],cameras=source['cameras'],smpl_joint_ids=np.array(ids))
 # Precompute exactforward at every existing raw image's capturetime; no imagepixels read here.
 cameras=source['cameras'];context=[(str(c),f,float(f+d)) for c,d in zip(cameras,source['offsets']) for f in range(744,757)];affine=[];shift=[];worldverts=[]
 with torch.no_grad():
  for start in range(0,len(context),16):
   ts=tensor([x[2] for x in context[start:start+16]]);b=torch.stack([((ts-750)/6)**k for k in range(4)],-1);v,j,M,_,_=query(b);A=M@inv0[:,None];bias=v-(A@v0[:,None,:,:,None])[...,0]
   affine.append(A.permute(1,0,2,3,4).reshape(-1,13780,3,3).cpu().numpy());shift.append(bias.permute(1,0,2,3).reshape(-1,13780,3).cpu().numpy());worldverts.append(v.permute(1,0,2,3).reshape(-1,13780,3).cpu().numpy())
 A=np.concatenate(affine);b=np.concatenate(shift);vv=np.concatenate(worldverts);reconstruction=A@v0.reshape(1,13780,3,1).cpu().numpy()+b[...,None];anchor_error=float(abs(reconstruction[...,0]-vv).max());assert anchor_error<2e-5
 np.savez_compressed(root/'capture-affines.npz',camera=np.array([x[0] for x in context]),frame=np.array([x[1] for x in context]),capture_time=np.array([x[2] for x in context]),linear=A,translation=b,vertices=vv)
 save(root/'fit-summary.json',dict(status='2500updates_continuousSMPL_and195capturetime_affines_complete_projection_review_pending',history=history,reference_vertex_error_world=reference_error,reference_identity_max_error=identity_error,vertex_anchor_reconstruction_max_error=anchor_error,source_sha256=sha(SOURCE/f'{label}-fit.npz'),scope='Motion fit to predicted3Dtrajectories only; noRGBclarityclaim.',goal_complete=False));rows.append(dict(branch=label,final=history[-1],captures=len(context)))
save(OUT/'completion.json',dict(status='both_continuous_body_motion_hypotheses_fitted',records=rows,goal_complete=False));shutil.copy2(__file__,OUT/Path(__file__).name)
