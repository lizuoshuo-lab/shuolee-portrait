from pathlib import Path
import json,shutil,hashlib
import numpy as np,torch,smplx
from smplx.lbs import lbs
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-continuous-body-063';SOURCE=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4);model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=1).cpu();ref=np.load(OUT/'reference.npz');rows=[];tensor=lambda x:torch.tensor(x,dtype=torch.float32)
for label in ['fixed_offsets','estimated_offsets']:
 root=OUT/label;f=np.load(root/'motion-fit.npz');archive=np.load(root/'capture-affines.npz');tab={k:archive[k] for k in archive.files};v0=ref['vertices'];M0=ref['linear'];inv0=np.linalg.inv(M0);errors=[];afferrors=[];detmin=1e9;detmax=-1e9
 assert len(tab['camera'])==195 and set(tab['frame'].tolist())==set(range(744,757)) and not {'00','06','12'}&set(tab['camera'].tolist())
 for i in range(195):
  tau=float((tab['capture_time'][i]-750)/6);b=np.array([tau**k for k in range(4)]);pose=f['basepose']+np.einsum('k,pkd->pd',b,f['posecoef'])*f['mask'];trans=f['basetrans']+np.einsum('k,pkd->pd',b,f['transcoef'])
  with torch.no_grad():v,j,M=lbs(tensor(f['betas']),tensor(pose),model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights,return_T=True)
  vv=v.numpy()*f['scales'][:,None,None]+trans[:,None,:];linear=M.numpy()[:,:,:3,:3]*f['scales'][:,None,None,None];A=linear@inv0;shift=vv-(A@v0[...,None])[...,0]
  errors.append(float(abs(vv.reshape(-1,3)-tab['vertices'][i]).max()));afferrors.append(float(abs(A.reshape(-1,3,3)-tab['linear'][i]).max()));afferrors.append(float(abs(shift.reshape(-1,3)-tab['translation'][i]).max()));dets=np.linalg.det(tab['linear'][i]);detmin=min(detmin,float(dets.min()));detmax=max(detmax,float(dets.max()))
 assert max(errors)<3e-5 and max(afferrors)<3e-5 and detmin>0
 row=dict(branch=label,contexts=195,max_CPU_reforward_vertex_error_world=max(errors),max_affine_recompute_error=max(afferrors),affine_determinant_range=[detmin,detmax],motion_fit_sha256=sha(root/'motion-fit.npz'),deformation_table_sha256=sha(root/'capture-affines.npz'));rows.append(row);print(json.dumps(row),flush=True)
save(OUT/'independent-audit.json',dict(status='both195context_affine_tables_reforward_onCPU_verified',records=rows,source='062training-onlycubicfit; targettimesincludeoddqueriesbutnooddRGBobservationswereused',goal_complete=False));shutil.copy2(__file__,OUT/Path(__file__).name)
