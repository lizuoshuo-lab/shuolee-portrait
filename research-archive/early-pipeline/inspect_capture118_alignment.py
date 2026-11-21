"""Read-only native calibration, feature support and surface visibility audit."""
from pathlib import Path
import importlib.util, sys, json
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
BASE=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-support')
RECORD=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
rt.configure(ASSETS,ROOT)

def stats(x):
 x=np.asarray(x,float)
 return dict(n=x.size,median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max())) if x.size else dict(n=0)

def main():
 import cv2, pycolmap
 out=ROOT/'alignment/nose-foot-calibration-audit';out.mkdir(exist_ok=True)
 assert not (out/'result.json').exists(), 'Completed diagnostics are immutable.'
 rt.save(out/'plan.json',dict(created=rt.now(),kind='read_only_diagnostic',
   questions=['What factory fields reached the actual raw/rectified training cameras?',
    'Are detector facial observations on back-facing surfaces included in the fit?',
    'Do fixed-camera static facial rays agree with one point, and with the fitted surface?',
    'Which foot observations are genuinely supported by image bounds and the person mask?'],
   budget='89 original selected frames, one frozen fitted surface, no optimization of cameras or model.',
   no_new_gaussian_training=True,foot_visibility_not_ground_truth=True,
   triangulation_scope='Detector-label correspondence is only a hypothesis; fit/hold views split by fixed parity, never residual.',
   camera_sha256=rt.sha(ROOT/'data/training-inputs.json'),fit_sha256=rt.sha(ROOT/'fit/fitted.npz')))
 raw=pycolmap.Reconstruction(str(BASE/'sfm/models/0'));cam=next(iter(raw.cameras.values()))
 rows=rt.read(ROOT/'data/training-inputs.json')['records']
 obs=dict(np.load(ROOT/'data/observations.npz'));fit=dict(np.load(ROOT/'fit/fitted.npz'))
 weights=dict(np.load(ROOT/'fit/observation-weights.npz'))
 template=dict(np.load(ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=True))
 print('template keys:',sorted(template));print('fit keys:',sorted(fit))
 v=fit['vertices_world'][0];faces=fit['faces'];tri=v[faces]
 lmkfi=template['lmk_faces_idx'].reshape(-1);bary=template['lmk_bary_coords'].reshape(-1,3)
 points=(tri[lmkfi]*bary[:,:,None]).sum(1)
 normal=np.cross(tri[lmkfi,1]-tri[lmkfi,0],tri[lmkfi,2]-tri[lmkfi,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
 E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows]);C=-np.einsum('nji,nj->ni',E[:,:3,:3],E[:,:3,3])
 direction=C[:,None,:]-points[None];direction/=np.linalg.norm(direction,axis=-1)[...,None]
 facing=np.einsum('nlj,lj->nl',direction,normal)>.1
 target=obs['keypoints'][:,40:91];w=weights['face']
 residual=np.linalg.norm(fit['projected_face51']-target,axis=-1)
 face_report=dict(weighted_observations=int((w>.25).sum()),
     back_facing_weight_fraction=float(w[~facing].sum()/max(w.sum(),1e-9)),
     front_errors=stats(residual[(w>.25)&facing]),back_errors=stats(residual[(w>.25)&~facing]),
     scope='Surface-normal sidedness of current mesh, not true visibility; detector labels need image review.')
 # Nose root/tip/base and mouth corners, fixed indices in IBUG17..67.
 selection={'nose_root':10,'nose_tip':13,'nose_left':14,'nose_right':18,'mouth_left':31,'mouth_right':37}
 traces=[]
 P=K@E[:,:3,:]
 def project(x,ids):
  q=np.einsum('nij,j->ni',P[ids],np.r_[x,1.]);return q[:,:2]/q[:,2,None]-.5
 for name,j in selection.items():
  ids=np.flatnonzero((w[:,j]>.25)&facing[:,j]);a=ids[::2];b=ids[1::2]
  rec=dict(name=name,face51_index=j,view_ids=ids.tolist(),fit_views=a.tolist(),held_views=b.tolist(),
           fitted_surface_error=stats(residual[ids,j]))
  if len(a)>=3 and len(b)>=2:
   # Triangulate in the same native pinhole domain; no personal point initial state is used.
   A=[]
   for i in a:
    u=target[i,j]+.5;A.extend([u[0]*P[i,2]-P[i,0],u[1]*P[i,2]-P[i,1]])
   _,_,vh=np.linalg.svd(A);x=vh[-1,:3]/vh[-1,3]
   sol=least_squares(lambda x:(project(x,a)-target[a,j]).reshape(-1),x,loss='soft_l1',f_scale=2.,max_nfev=100)
   rec.update(triangulated_point=sol.x.tolist(),held_error=stats(np.linalg.norm(project(sol.x,b)-target[b,j],axis=1)),
              fit_error=stats(np.linalg.norm(project(sol.x,a)-target[a,j],axis=1)),
              point_to_mesh_landmark_world=float(np.linalg.norm(sol.x-points[j])),
              point_to_mesh_landmark_metres_if_prior_scale_correct=float(np.linalg.norm(sol.x-points[j])/fit['world_scale']))
  # Native crops, no labels and no resizing; the named keypoint is at crop center.
  show=ids[np.unique(np.linspace(0,max(0,len(ids)-1),min(8,len(ids))).astype(int))] if len(ids) else []
  if len(show):
   tiles=[]
   for i in show:
    x,y=np.rint(target[i,j]).astype(int);im=Image.open(rows[i]['image']).convert('RGB');tiles.append(im.crop((x-64,y-64,x+64,y+64)))
   sheet=Image.new('RGB',(len(tiles)*128,128))
   for k,im in enumerate(tiles):sheet.paste(im,(k*128,0))
   namefile=name+'_native_landmark_crops.png';sheet.save(out/namefile);rec['review_image']=namefile;rec['review_frames']=[rows[i]['frame'] for i in show]
  traces.append(rec)
 feet=obs['keypoints'][:,17:23];body=obs['keypoints'][:,:17];footrows=[]
 for i,r in enumerate(rows):
  mask=np.array(Image.open(r['mask']))>127;x,y=np.rint(feet[i]).astype(int).T
  inside=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0]);support=np.zeros(6,bool);support[inside]=mask[y[inside],x[inside]]
  footrows.append(dict(frame=r['frame'],inside_image=int(inside.sum()),on_person_mask=int(support.sum()),fit_positive_weights=int((weights['feet'][i]>.25).sum()),xy=feet[i].tolist()))
 calibration=dict(factory=rt.read(RECORD/'protocol.json')['calibration'],
    supplied_usage=rt.read(BASE/'camera-initialization.json'),
    raw_solved=dict(model=cam.model.name,params=cam.params.tolist(),wh=[cam.width,cam.height]),
    training=dict(K=rows[0]['K'],wh=rows[0]['wh'],source='estimated SIMPLE_RADIAL then native undistortion',factory_distortion_applied=False),
    native_input_to_cuda_math='Existing per-camera numerical consistency check; not a calibration accuracy check.',
    direct_factory_intrinsics_used_for_final_training=False)
 result=dict(updated=rt.now(),calibration=calibration,face=face_report,triangulation=traces,feet=footrows,
     camera_changes=0,training_updates=0,quality_accepted=False,
     dependencies={n:bool(importlib.util.find_spec(n)) for n in ['poselib','pyopengv','cv2']},
     opencv_version=cv2.__version__,world_scale=float(fit['world_scale']))
 rt.save(out/'result.json',result);rt.save(RECORD/'nose-foot-calibration-audit.json',result)
 print(json.dumps(dict(raw_camera=calibration['raw_solved'],face=face_report,triangulation=traces,
    feet_on_mask_counts=stats([r['on_person_mask'] for r in footrows]),dependencies=result['dependencies']),indent=2))

if __name__=='__main__':main()
