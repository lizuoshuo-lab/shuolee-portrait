"""CPU full-SMPL verification and reserved-time projection evaluation for065."""
from pathlib import Path
import json,hashlib,shutil,argparse
import numpy as np,torch,smplx
from smplx.lbs import lbs
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-direct-pose-065';SRC=LAB/'outputs/dynamic-human-enerf-continuous-body-063';SEQ=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';OLD=LAB/'outputs/dynamic-human-enerf-wholebody-059'
ap=argparse.ArgumentParser();ap.add_argument('--output-root',type=Path,default=OUT);ap.add_argument('--head-only',action='store_true');args=ap.parse_args();OUT=args.output_root
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.tensor(x,dtype=torch.float32)
assert (OUT/'completion.json').exists()
torch.set_num_threads(4);model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=1).cpu()
ref=dict(np.load(SRC/'reference.npz'));archive=np.load(OUT/'capture-affines.npz');tab={k:archive[k] for k in archive.files};f=dict(np.load(OUT/'motion-fit.npz'));old=dict(np.load(SRC/'estimated_offsets/motion-fit.npz'))
for k in ['betas','scales','basepose','basetrans','camera_offsets','cameras']:assert np.array_equal(f[k],old[k]),k
assert np.array_equal(f['mask'][60:],old['mask'][60:]) and np.all(f['posecoef'][:,:,60:]==0)
if args.head_only:
 unchanged=np.ones(72,bool);unchanged[[36,37,38,45,46,47]]=False
 assert np.array_equal(f['posecoef'][:,:,unchanged],old['posecoef'][:,:,unchanged]) and np.array_equal(f['mask'][unchanged],old['mask'][unchanged]) and np.array_equal(f['transcoef'],old['transcoef'])
obs={(r['camera'],r['frame']):r for r in read(SEQ/'observations.json')['records']};support={(r['person'],r['index']):{o['camera'] for o in r.get('observations',[]) if o['inlier']} for r in read(OLD/'triangulation.json')['records'] if r['accepted']}
types=['person_dark_clothing','person_light_clothing'];ids=[24,26,25,28,27,16,17,18,19,20,21,1,2,4,5,7,8,29,30,31,32,33,34]
fitobs=read(OUT/'fit-observation-index.json')['records'];assert all(r['frame']%2==0 for r in fitobs) and len(fitobs)==read(OUT/'protocol.json')['fit_observations']
if args.head_only:assert len(fitobs)==546 and all(r['joint']<5 for r in fitobs)
rows=[];verrors=[];aerrors=[];dmin=1e9;dmax=-1e9;body_joint_change=[];non_head_vertex_change=[];non_head_vertex=(model.lbs_weights[:,[12,15]].sum(-1)<.01).numpy();body_ids=[j for j in range(24) if j not in [12,15]]
for i,(c,frame,capture) in enumerate(zip(tab['camera'],tab['frame'],tab['capture_time'])):
 assert c not in ['00','06','12'];r=obs[str(c),int(frame)];basis=np.array([((capture-750)/6)**k for k in range(4)]);P=np.array(r['K'])@np.array(r['w2c'])[:3]
 uv={}
 for label,p in [('before',old),('after',f)]:
  pose=p['basepose']+np.einsum('k,pkd->pd',basis,p['posecoef'])*p['mask'];trans=p['basetrans']+np.einsum('k,pkd->pd',basis,p['transcoef'])
  with torch.no_grad():v,j,M=lbs(tensor(p['betas']),tensor(pose),model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights,return_T=True);j=model.vertex_joint_selector(v,j)
  world=j.numpy()*p['scales'][:,None,None]+trans[:,None,:];q=np.einsum('ij,pnj->pni',P[:,:3],world[:,ids])+P[None,None,:,3];assert np.all(q[...,2]>0);uv[label]=q[...,:2]/q[...,2:]
  if label=='before':before_world=world.copy();before_vertices=v.numpy()*p['scales'][:,None,None]+trans[:,None,:]
  if label=='after':
   vv=v.numpy()*p['scales'][:,None,None]+trans[:,None,:];A=(M.numpy()[:,:,:3,:3]*p['scales'][:,None,None,None])@np.linalg.inv(ref['linear']);b=vv-(A@ref['vertices'][...,None])[...,0]
   verrors.append(float(abs(vv.reshape(-1,3)-tab['vertices'][i]).max()));aerrors.extend([float(abs(A.reshape(-1,3,3)-tab['linear'][i]).max()),float(abs(b.reshape(-1,3)-tab['translation'][i]).max())]);det=np.linalg.det(tab['linear'][i]);dmin=min(dmin,float(det.min()));dmax=max(dmax,float(det.max()))
   body_joint_change.append(float(abs(world[:,body_ids]-before_world[:,body_ids]).max()));d=np.linalg.norm(vv[:,non_head_vertex]-before_vertices[:,non_head_vertex],axis=-1);non_head_vertex_change.append(dict(camera=str(c),frame=int(frame),max_world=float(d.max()),mean_world=float(d.mean())))
 for pi,typ in enumerate(types):
  det=next(d for d in r['detections'] if d['person']==typ)
  for j in range(23):
   if pi==1 and j<3:continue
   if c not in support.get((typ,j),set()) or det['scores'][j]<.4:continue
   rows.append(dict(camera=str(c),frame=int(frame),person=typ,joint=j,group='head' if j<5 else 'body_feet',split='fit' if frame%2==0 else 'validation_time',**{label+'_px':float(np.linalg.norm(uv[label][pi,j]-det['keypoints'][j])) for label in uv}))
assert max(verrors)<3e-5 and max(aerrors)<3e-5 and dmin>0
if args.head_only:assert max(body_joint_change)<2e-6
def metric(a):return dict(count=len(a),median_px=float(np.median(a)),rmse_px=float(np.sqrt(np.mean(np.array(a)**2))),p90_px=float(np.quantile(a,.9)))
summary={s:{g:{b:metric([r[b+'_px'] for r in rows if r['split']==s and (g=='all' or r['group']==g)]) for b in ['before','after']} for g in ['all','head','body_feet']} for s in ['fit','validation_time']}
save(OUT/'independent-audit.json',dict(status='full195context_CPU_SMPL_reprojection_and_affine_recomputation_verified',max_vertex_error_world=max(verrors),max_affine_error=max(aerrors),determinant_range=[dmin,dmax],motion_sha256=sha(OUT/'motion-fit.npz'),affines_sha256=sha(OUT/'capture-affines.npz'),fit_observations=len(fitobs),all_fit_times_even=True,frozen_shape_scale_cameras_timing_verified=True,goal_complete=False))
if args.head_only:save(OUT/'head-only-invariance-audit.json',dict(status='nonhead_pose_translation_exact_and_body_skeleton_preserved',max_nonhead_skeleton_coordinate_change_world=max(body_joint_change),nonhead_vertex_selection='SMPL neck12+head15skinningweight sum below0.01',nonhead_vertices_per_person=int(non_head_vertex.sum()),nonhead_mesh_change=non_head_vertex_change,qualification='Frozenbodyrotations do not freezeSMPL posecorrective blendshapes; mesh changes quantified explicitly.',goal_complete=False))
save(OUT/'projection-comparison.json',dict(status='all_original_supported_observations_evaluated_before_after',summary=summary,records=rows,limits='Predicted keypoints and repeatedly inspected development times, not finalGT; no RGB clarity claim.',goal_complete=False));shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(summary))
