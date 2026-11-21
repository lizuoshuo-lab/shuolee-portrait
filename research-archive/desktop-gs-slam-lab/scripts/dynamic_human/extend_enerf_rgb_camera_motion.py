"""Query frozen066 motion for new cameras; preserve every old context exactly."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,smplx
from smplx.lbs import lbs
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout';I=L/'outputs/dynamic-human-enerf-heldout-cameras-068';H=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/head';B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/body'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(4)
assert not (R/'motion-extension-audit.json').exists();protocol=read(R/'protocol.json');assert sha(H/'motion-fit.npz')==protocol['body_motion_sha256']
f=dict(np.load(H/'motion-fit.npz'));ref=dict(np.load(B/'reference.npz'));old=dict(np.load(H/'capture-affines.npz'));clocks=read(R/'timing-summary.json')['offsets_frames'];tensor=lambda x:torch.tensor(x,dtype=torch.float32)
model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=1)
for p in model.parameters():p.requires_grad_(False)
betas=tensor(f['betas']);scales=tensor(f['scales']);basepose=tensor(f['basepose']);basetrans=tensor(f['basetrans']);coef=tensor(f['posecoef']);trcoef=tensor(f['transcoef']);mask=tensor(f['mask']);v0=tensor(ref['vertices']);inv0=tensor(ref['inverse_linear'])
@torch.no_grad()
def query(times):
 ts=tensor(times);basis=torch.stack([((ts-750)/6)**k for k in range(4)],-1);n=len(ts);pose=basepose[:,None]+torch.einsum('tk,pkd->ptd',basis,coef)*mask;tr=basetrans[:,None]+torch.einsum('tk,pkd->ptd',basis,trcoef)
 v,j,M=lbs(betas[:,None].expand(-1,n,-1).reshape(-1,10),pose.reshape(-1,72),model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights,return_T=True)
 v=v.reshape(2,n,6890,3)*scales[:,None,None,None]+tr[:,:,None,:];M=M[:,:,:3,:3].reshape(2,n,6890,3,3)*scales[:,None,None,None,None];A=M@inv0[:,None];b=v-(A@v0[:,None,:,:,None])[...,0]
 return dict(linear=A.permute(1,0,2,3,4).reshape(n,13780,3,3).numpy(),translation=b.permute(1,0,2,3).reshape(n,13780,3).numpy(),vertices=v.permute(1,0,2,3).reshape(n,13780,3).numpy())
ids=[i for i,(c,t) in enumerate(zip(old['camera'],old['frame'])) if c=='01' and t in [744,750,756]];check=query(old['capture_time'][ids]);errors={k:float(np.max(abs(v-old[k][ids]))) for k,v in check.items()};assert max(errors.values())<3e-5,errors
reports={}
for label in ['zero','estimated']:
 contexts=[(c,t,float(t+(clocks[c] if label=='estimated' else 0))) for c in ['00','06','12'] for t in range(744,757)];parts={k:[] for k in ['linear','translation','vertices']}
 for start in range(0,len(contexts),8):
  out=query([r[2] for r in contexts[start:start+8]])
  for k in parts:parts[k].append(out[k])
 new={k:np.concatenate(v) for k,v in parts.items()};new.update(camera=np.array([r[0] for r in contexts]),frame=np.array([r[1] for r in contexts]),capture_time=np.array([r[2] for r in contexts]))
 combined={k:np.concatenate([old[k],new[k]]) for k in old};assert all(np.array_equal(v[:195],old[k]) for k,v in combined.items())
 restored=np.einsum('nvij,vj->nvi',new['linear'],ref['vertices'].reshape(-1,3))+new['translation'];reconstruction=float(np.max(abs(restored-new['vertices'])));assert reconstruction<2e-6
 determinant=np.linalg.det(new['linear']);assert np.isfinite(determinant).all() and determinant.min()>0
 path=R/f'capture-affines-{label}.npz';np.savez_compressed(path,**combined)
 reports[label]=dict(contexts=234,old_contexts_unchanged=195,new_contexts=39,min_determinant=float(determinant.min()),reference_vertex_affine_error=reconstruction,sha256=sha(path),new_physical_frame_range=[min(r[2] for r in contexts),max(r[2] for r in contexts)])
save(R/'motion-extension-audit.json',dict(status='frozen_body_motion_queried_for39newcontexts_old195_preserved_exactly',source_motion_sha256=sha(H/'motion-fit.npz'),source_affines_sha256=sha(H/'capture-affines.npz'),old_CPU_reforward_max_error=errors,branches=reports,licensed_data='Allbodyparameters,meshes,andaffinetablesremainlocal',goal_complete=False))
co=np.load(L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/estimated_offsets-fit.npz')['coefficients'];regions=[]
for c in ['00','06','12']:
 for t in [745,751,755]:
  z=np.load(I/'native-input'/f'camera{c}-frame{t:06d}.npz');P=z['K']@z['w2c'][:3];tau=(t+clocks[c]-750)/6;X=np.einsum('k,nkd->nd',np.array([tau**k for k in range(4)]),co);q=X@P[:,:3].T+P[:,3];uv=q[:,:2]/q[:,2:]
  for name,ji in [('dark_right_forearm',[3,5]),('light_left_forearm',[20,22])]:
   lo=uv[ji].min(0)-32;hi=uv[ji].max(0)+32;b=np.r_[np.maximum([0,0],np.floor(lo)),np.minimum([1920,1080],np.ceil(hi))].astype(int).tolist();assert b[2]>b[0] and b[3]>b[1];regions.append(dict(camera=c,frame=t,name=name,bbox=b))
save(R/'review-protocol.json',dict(status='fixed_before_appearance_rendering',full_person=[dict(camera=c,frame=t) for c in ['00','06','12'] for t in [745,751,755]],regions=regions,region_source='Frozen074RGBbody trajectories and even-time fittedclock; elbow+wrist projection with32pixelmargin, no evaluationRGB selection. Includes hand/background/occlusions; notpartGT.',goal_complete=False))
shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps(reports))
