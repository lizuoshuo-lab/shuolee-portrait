from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,smplx
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OBS=LAB/'outputs/dynamic-human-enerf-wholebody-059';OUT=LAB/'outputs/dynamic-human-enerf-wholebody-prior-060';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
o=read(OBS/'observations.json');cams={r['camera']:r for r in o['records']};assert len(cams)==15 and not {'00','06','12'}&set(cams)
for c,r in cams.items():
 f=BASE/f'train-camera-{c}.npz';z=np.load(f);assert r['input_sha256']==sha(f)
 assert np.array_equal(r['K'],z['K']) and np.array_equal(r['w2c'],z['w2c'])
 assert len(r['detections'])==2 and all(np.array(d['keypoints']).shape==(133,2) for d in r['detections'])
t=read(OBS/'triangulation.json');valid_errors=[];count=0;max_difference=0
for row in t['records']:
 if 'observations' not in row:continue
 x=np.r_[row['point'],1.]
 for q in row['observations']:
  r=cams[q['camera']];cam=np.array(r['w2c'])@x;p=np.array(r['K'])@cam[:3];err=np.linalg.norm(p[:2]/p[2]-q['xy']) if cam[2]>0 else 1e9
  max_difference=max(max_difference,abs(err-q['error_px']));assert (err<8)==q['inlier'];count+=1
  if row['accepted'] and q['inlier']:valid_errors.append(err)
assert max_difference<1e-8 and sum(r['accepted'] for r in t['records'])==64
save(OBS/'independent-audit.json',dict(status='all15input_hashes_calibration_and_saved_reprojections_recomputed',keypoints_per_person=133,accepted_landmarks=64,total_landmarks=66,reprojected_observations=count,max_saved_error_difference_px=max_difference,median_accepted_inlier_error_px=float(np.median(valid_errors)),observations_sha256=sha(OBS/'observations.json'),triangulation_sha256=sha(OBS/'triangulation.json'),limits='Geometric consistency does not establish visibility or anatomical correctness.',goal_complete=False))
torch.set_num_threads(4);model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=2).cpu();z=np.load(OUT/'fit.npz');tensor=lambda k:torch.tensor(z[k],dtype=torch.float32)
with torch.no_grad():
 r=model(betas=tensor('betas'),body_pose=tensor('body_pose').reshape(2,69),global_orient=tensor('global_orient'),return_verts=True)
 vv=r.vertices.numpy()*z['scale'][:,None,None]+z['translation'][:,None,:];jj=r.joints.numpy()*z['scale'][:,None,None]+z['translation'][:,None,:]
ve=float(abs(vv-z['vertices']).max());je=float(abs(jj-z['joints']).max());assert ve<2e-5 and je<2e-5
ids=z['smpl_joint_ids'];valid=z['valid'];assert valid.sum()==43 and not valid[1,:3].any();errors=np.linalg.norm(jj[:,ids]-z['target'],axis=-1);reported=read(OUT/'fit-summary.json')
assert abs(float(errors[valid].mean())-reported['final_mean_valid_error_world'])<1e-5
save(OUT/'independent-audit.json',dict(status='full_saved_SMPL_reforward_on_CPU_and_joint_errors_recomputed',fit_sha256=sha(OUT/'fit.npz'),vertex_max_difference_world=ve,joint_max_difference_world=je,used_predicted_landmarks=int(valid.sum()),initial_mean_valid_error_world=reported['initial_mean_valid_error_world'],final_mean_valid_error_world=float(errors[valid].mean()),excluded_light_nose_eyes=True,no_finger_fitting=True,limits='Template parameters/mesh remain local. Improved predicted-landmark fit does not certify clothed reconstruction.',goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps(dict(accepted_landmarks=64,observations=count,median_inlier_px=float(np.median(valid_errors)),initial_error=reported['initial_mean_valid_error_world'],refined_error=float(errors[valid].mean()),vertex_max_difference=ve)))
