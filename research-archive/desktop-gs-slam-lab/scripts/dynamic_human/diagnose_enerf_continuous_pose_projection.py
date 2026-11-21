from pathlib import Path
import json,shutil
import numpy as np,torch,smplx
from smplx.lbs import lbs
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-continuous-body-063';SOURCE=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';OLD=LAB/'outputs/dynamic-human-enerf-wholebody-059';read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
obs={(r['camera'],r['frame']):r for r in read(SOURCE/'observations.json')['records']};support={(r['person'],r['index']):{o['camera'] for o in r.get('observations',[]) if o['inlier']} for r in read(OLD/'triangulation.json')['records'] if r['accepted']};types=['person_dark_clothing','person_light_clothing'];model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=1).cpu();torch.set_num_threads(4);tensor=lambda x:torch.tensor(x,dtype=torch.float32);rows=[]
for branch in ['fixed_offsets','estimated_offsets']:
 f=np.load(OUT/branch/'motion-fit.npz');source=np.load(SOURCE/f'{branch}-fit.npz');co=source['coefficients'].reshape(2,18,4,3)
 for ci,c in enumerate(f['cameras']):
  for t in range(744,757):
   tau=float((t+f['camera_offsets'][ci]-750)/6);basis=np.array([tau**k for k in range(4)]);pose=f['basepose']+np.einsum('k,pkd->pd',basis,f['posecoef'])*f['mask'];trans=f['basetrans']+np.einsum('k,pkd->pd',basis,f['transcoef'])
   with torch.no_grad():
    v,j=lbs(tensor(f['betas']),tensor(pose),model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights);j=model.vertex_joint_selector(v,j)
   world=j.numpy()*f['scales'][:,None,None]+trans[:,None,:];target=np.einsum('k,pjkd->pjd',basis,co);r=obs[str(c),t];P=np.array(r['K'])@np.array(r['w2c'])[:3]
   def project(X):
    q=np.einsum('ij,pnj->pni',P[:,:3],X)+P[None,None,:,3];return q[...,:2]/q[...,2:]
   smpluv=project(world[:,f['smpl_joint_ids']]);freeuv=project(target)
   for pi,typ in enumerate(types):
    d=next(d for d in r['detections'] if d['person']==typ)
    for j in range(18):
     idx=j+5
     if c not in support[typ,idx] or d['scores'][idx]<.4:continue
     xy=np.array(d['keypoints'][idx]);rows.append(dict(branch=branch,camera=str(c),frame=t,person=typ,index=idx,split='fit' if t%2==0 else 'validation_time',free_trajectory_error_px=float(np.linalg.norm(freeuv[pi,j]-xy)),SMPL_trajectory_error_px=float(np.linalg.norm(smpluv[pi,j]-xy)),projection_change_px=float(np.linalg.norm(freeuv[pi,j]-smpluv[pi,j]))))
def metric(arr):return dict(count=len(arr),median_px=float(np.median(arr)),rmse_px=float(np.sqrt(np.mean(np.array(arr)**2))),p90_px=float(np.quantile(arr,.9)))
summary={b:{s:{k:metric([r[k] for r in rows if r['branch']==b and r['split']==s]) for k in ['free_trajectory_error_px','SMPL_trajectory_error_px','projection_change_px']} for s in ['fit','validation_time']} for b in ['fixed_offsets','estimated_offsets']}
save(OUT/'projection-diagnostic.json',dict(status='body_model_projection_residual_compared_with_free3Djoint_trajectory',summary=summary,records=rows,limits='Samepredictedkeypointobservations, notGT. Tests distortionintroducedbySMPLmotionfit; notreconstructionqualitymetric.',goal_complete=False));shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(summary))
