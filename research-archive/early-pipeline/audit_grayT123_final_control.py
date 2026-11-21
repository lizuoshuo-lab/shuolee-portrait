"""Read-only review of an already completed other-thread control.

No fitting, feature extraction, camera changes or GPU rendering. The archived
motion is examined as a model variable, never as measured physical motion.
"""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import argparse,hashlib,json,os,signal,sys,time
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

H=Path(__file__).resolve().parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/grayT-final-control-review'
D=Path('/mnt/d/Projects/Shuolee-analysis-20261028')
B=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
N=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/gray-dynamic-face-fan')
ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
now=lambda:datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def stats(x):
 x=np.asarray(x).ravel();x=x[np.isfinite(x)]
 return dict(n=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(x.max()),minimum=float(x.min())) if len(x) else {'n':0}
def ply(path):
 with Path(path).open('rb') as f:
  assert f.readline().strip()==b'ply';props=[];n=None
  while True:
   line=f.readline().decode('ascii').strip()
   if line.startswith('format'):assert line=='format binary_little_endian 1.0'
   if line.startswith('element vertex'):n=int(line.split()[-1])
   if line.startswith('property'):
    _,typ,name=line.split();assert typ=='float';props.append((name,'<f4'))
   if line=='end_header':break
  return np.fromfile(f,dtype=np.dtype(props),count=n)
def register():
 O.mkdir(exist_ok=False)
 sources=[B/'config.json',B/'data/training-inputs.json',B/'fit/fitted.npz',B/'clothing-init/points.npz',N/'protocol.json',N/'result.json',N/'source-provenance.json',N/'code/train_head_motion.py',N/'step-018000/canonical.ply',N/'export/validation.json',N/'export/cameras.json',N/'export/camera_provenance.json',D/'dynamic-face-fan-control/motion.npz',D/'dynamic-face-fan-all-views/report.json',D/'validate_all_views.py',D/'export_canonical_control.py',ASSETS/'smplx/SMPLX_NEUTRAL.npz']
 plan=dict(registered=now(),scope='read_only_completed_control_audit',hypotheses=[
  'Image photometric improvement across training frames is not native physical correspondence accuracy.',
  'Animated training renders cannot be represented by a canonical standard PLY plus only unchanged cameras.',
  'Free point centers may differ from template centers while still containing clothing/body artifacts; nearest-vertex distances are a proxy, not surface truth.'],
  sources=[ref(p) for p in sources],bounds=dict(CPU_seconds=300,GPU=0,new_fit=0,new_camera=0,feature_extraction=0,new_training=0),
  metrics='Recompute all95 saved per-frame summaries; quantify saved model canonical-versus-deformed native projections and affine direction conventions. No refit or acceptance tuning.',
  sample='4096 deterministic evenly spaced final centers across the whole model plus4096 across nearest-template head-weight>0.5 centers. Fixed for all95frames, no residual selection.',
  visuals='All95 existing original/static/deformed head triptychs,12canonical novel head and12body views,6existing full-body snapshots. Preserve aspect/pixels; no text burned into crops.',
  stop='Once-only audit. Invalid source identity stops. No dynamic mainline or new static admission from these results.',
  previous_goal_turn='progress: R7 completed actual bounded engineering verification and publication')
 save(O/'plan.json',plan);print(json.dumps({'registered':str(O),'sources':len(sources)}))
def run():
 if (O/'result.json').exists() or (O/'execution.json').exists():raise FileExistsError('Read retained results; do not repeat')
 plan=read(O/'plan.json');assert all(sha(p['path'])==p['sha256'] for p in plan['sources'])
 save(O/'execution.json',dict(updated=now(),pid=os.getpid(),command=sys.argv,source=ref(__file__),plan=ref(O/'plan.json')))
 started=time.monotonic()
 def deadline(*_):raise TimeoutError('Read-only CPU budget exhausted')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(300)
 report=read(D/'dynamic-face-fan-all-views/report.json');rows=read(B/'data/training-inputs.json')['records'];protocol=read(N/'protocol.json');result=read(N/'result.json')
 assert result['state']=='completed' and result['steps']==18000 and len(rows)==report['frames']==95
 assert protocol['cameras_sha256']==sha(B/'data/training-inputs.json') and protocol['initialization_sha256']==sha(B/'clothing-init/points.npz')
 assert read(N/'source-provenance.json')['training_script_sha256']==sha(N/'code/train_head_motion.py')==report['deformation_source_sha256']
 assert report['source_ply_sha256']==sha(N/'step-018000/canonical.ply')
 assert [r['frame'] for r in rows]==[r['frame'] for r in report['records']]
 metrics={};regressions={}
 for key in ['head_foreground_rgb_l1','head_crop_masked_rgb_l1','whole_foreground_rgb_l1']:
  a=np.array([r['static'][key] for r in report['records']]);b=np.array([r['head_motion'][key] for r in report['records']]);relative=1-b/a
  metrics[key]=dict(static_mean=float(a.mean()),deformed_mean=float(b.mean()),mean_relative_reduction=float(1-b.mean()/a.mean()),improved=int((b<a).sum()),regressed=int((b>a).sum()))
  regressions[key]=[dict(frame=rows[i]['frame'],relative_change=float(relative[i]),static=float(a[i]),deformed=float(b[i])) for i in np.where(b>a)[0]]
  assert abs(metrics[key]['mean_relative_reduction']-report['improvement'][key]['relative_improvement_of_mean'])<1e-12
 fit=dict(np.load(B/'fit/fitted.npz',allow_pickle=False));motion=dict(np.load(D/'dynamic-face-fan-control/motion.npz',allow_pickle=False))
 data=ply(N/'step-018000/canonical.ply');xyz=np.stack([data[k] for k in ['x','y','z']],-1);template=np.load(ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=False)
 assert len(motion['vertices_world'])==96 and int(motion['training_timestep_offset'])==1
 assert np.max(np.abs(motion['vertices_world'][0]-fit['vertices_world'][0]))<1e-7
 tree=cKDTree(fit['vertices_world'][0]);distance,near=tree.query(xyz,k=1,workers=2)
 scale=float(fit['world_scale']);skin_head=template['weights'][:,[15,22,23,24]].sum(1)
 head=np.flatnonzero(skin_head[near]>.5);other=np.flatnonzero(skin_head[near]<=.5)
 select=lambda ids:ids[np.unique(np.linspace(0,len(ids)-1,min(4096,len(ids))).astype(int))]
 ids_whole=select(np.arange(len(xyz)));ids_head=select(head);chosen=np.unique(np.r_[ids_whole,ids_head]);points=xyz[chosen].astype(float)
 distances,vertices=tree.query(points,k=4,workers=2);weights=1/(distances**2+(.015*scale)**2);weights/=weights.sum(1,keepdims=True)
 refvertices=motion['vertices_world'][0][vertices].astype(float)
 is_head=np.isin(chosen,ids_head);is_whole=np.isin(chosen,ids_whole)
 def project(x,row):
  E=np.asarray(row['w2c']);K=np.asarray(row['K']);c=x@E[:3,:3].T+E[:3,3];uv=c[:,:2]/c[:,2:]*[K[0,0],K[1,1]]+K[:2,2]
  return uv,c[:,2]
 disp=[];directions=[];affine_stats=[]
 for i,row in enumerate(rows):
  AA=motion['vertex_affine'][i+1][vertices].astype(float);vv=motion['vertices_world'][i+1][vertices].astype(float)
  A=np.sum(AA*weights[:,:,None,None],axis=1);trans=np.sum((vv-np.einsum('nkij,nkj->nki',AA,refvertices))*weights[:,:,None],axis=1)
  deformed=np.einsum('nij,nj->ni',A,points)+trans
  old,z0=project(points,row);new,z1=project(deformed,row)
  width,height=row['wh'];valid=(z0>0)&(z1>0)&(new[:,0]>=0)&(new[:,0]<width)&(new[:,1]>=0)&(new[:,1]<height)
  uvdelta=np.linalg.norm(new-old,axis=1)
  disp.append(dict(frame=row['frame'],native_width=int(width),native_height=int(height),head=stats(uvdelta[valid&is_head]),whole=stats(uvdelta[valid&is_whole]),world_head_scale_units=stats(np.linalg.norm(deformed-points,axis=1)[is_head]/scale)))
  sv=np.linalg.svd(A,compute_uv=False);affine_stats.append(dict(frame=row['frame'],head_singular_values=stats(sv[is_head]),determinant=stats(np.linalg.det(A)[is_head])))
  E=np.asarray(row['w2c']);center=-E[:3,:3].T@E[:3,3];d=deformed-center
  transposed=np.einsum('nji,nj->ni',A,d);inverse=np.linalg.solve(A,d[...,None])[...,0]
  transposed/=np.linalg.norm(transposed,axis=1,keepdims=True);inverse/=np.linalg.norm(inverse,axis=1,keepdims=True)
  angles=np.degrees(np.arccos(np.clip(np.sum(transposed*inverse,axis=1),-1,1)))
  directions.append(dict(frame=row['frame'],head_direction_disagreement_degrees=stats(angles[is_head]),whole_direction_disagreement_degrees=stats(angles[is_whole])))
  if i in [0,47,94]:save(O/'progress.json',dict(state='reading_completed_results',frame=i+1,total=95,seconds=time.monotonic()-started))
 save(O/'canonical-versus-deformed.json',dict(points_sampled=len(chosen),whole_sample=len(ids_whole),head_sample=len(ids_head),frames=disp,
  measurement='Difference between two saved model states in the same recorded camera. Not reprojection to observed material, not measured motion.',scope='model representation diagnosis'))
 save(O/'affine-direction-convention.json',dict(frames=directions,affines=affine_stats,meaning='A transpose equals inverse for an orthogonal map only. Recorded implementation explicitly uses transpose. Differences quantify appearance-coordinate convention, not geometric error or an isolated cause.'))
 initial=np.load(B/'clothing-init/points.npz',allow_pickle=False);init_keys=list(initial.files)
 surface_proxy=dict(final_points=len(xyz),head_points=len(head),body_points=len(other),nearest_template_vertex_distance_divided_by_world_scale=dict(all=stats(distance/scale),head=stats(distance[head]/scale),body=stats(distance[other]/scale)),
  limitation='Nearest template VERTEX distance is not exact mesh-surface distance or measured clothing distance. No claim that point centers match the true surface.',initial_keys=init_keys)
 if 'xyz' in initial:
  d,_=tree.query(initial['xyz'],k=1,workers=2);surface_proxy['initial_points']=len(d);surface_proxy['initial_nearest_vertex_distance']=stats(d/scale)
 save(O/'surface-proxy.json',surface_proxy)
 # Layout existing pixels only, no resizing or text annotations.
 galleries=[];all_crops=[]
 for row in rows:
  path=D/'dynamic-face-fan-all-views/head-crops'/f'{row["frame"]}.png';assert path.exists();all_crops.append(path)
 for page in range(0,95,8):
  paths=all_crops[page:page+8];ims=[Image.open(p).convert('RGB') for p in paths];w=max(im.width for im in ims);h=sum(im.height for im in ims)+12*(len(ims)-1)
  sheet=Image.new('RGB',(w,h),'#f3f3f3');y=0;layout=[]
  for p,im in zip(paths,ims):sheet.paste(im,(0,y));layout.append(dict(source=ref(p),bounds=[0,y,im.width,im.height]));y+=im.height+12
  file=f'head-training-views-{page//8:02d}.png';sheet.save(O/file);galleries.append(dict(file=file,kind='all-training-head',frames=[p.stem for p in paths],layout=layout,columns=['input','static','FAN deformation'],pixel_domain='899x1600 training raster; unscaled crops, not native1520x2704'))
 for kind in ['head','body']:
  for start in range(0,12,4):
   paths=[N/f'export/review/{kind}_{i:02d}.png' for i in range(start,start+4)];ims=[Image.open(p).convert('RGB') for p in paths]
   w,h=ims[0].size;sheet=Image.new('RGB',(2*w,2*h),'black')
   for j,im in enumerate(ims):assert im.size==(w,h);sheet.paste(im,((j%2)*w,(j//2)*h))
   file=f'canonical-novel-{kind}-{start//4:02d}.png';sheet.save(O/file);galleries.append(dict(file=file,kind='canonical-novel-'+kind,frames=[p.stem for p in paths],sources=[ref(p) for p in paths],pixel_domain='canonical model new views; no novel-view photographic truth'))
 # Six full-body original/animated snapshots, at their actual training resolution.
 for i in np.unique(np.linspace(0,94,6).astype(int)):
  frame=rows[int(i)]['frame'];ps=[N/f'step-018000/{frame}_input.png',B/f'gaussian-run/step-018000/{frame}_render.png',N/f'step-018000/{frame}_render.png']
  ims=[Image.open(p).convert('RGB') for p in ps];w,h=ims[0].size;assert all(im.size==(w,h) for im in ims)
  sheet=Image.new('RGB',(w*3,h));[sheet.paste(im,(j*w,0)) for j,im in enumerate(ims)];file=f'wholebody-{frame}.png';sheet.save(O/file)
  galleries.append(dict(file=file,kind='same-camera-wholebody',frames=[frame],sources=[ref(p) for p in ps],columns=['input','static','FAN deformation'],pixel_domain='unaltered899x1600 training raster'))
 save(O/'index.json',dict(scope='completed-control-readonly-review',images=[],gallery=galleries))
 result=dict(updated=now(),state='numeric_audit_complete_visual_pending',frames=95,steps=18000,photometry=metrics,regressed_frames=regressions,
  observed_training_views_only=True,independent_geometric_validation=False,initialization_reused_for_matched_control=protocol['reused_initial_point_cloud_for_matched_comparison'],
  canonical_standard_ply_does_not_include_per_frame_motion=True,source_files_unchanged=all(sha(p['path'])==p['sha256'] for p in plan['sources']),
  final_points=len(xyz),native_camera_dimensions=sorted(set(tuple(r['wh']) for r in rows)),training_height=read(B/'config.json')['training_height'],
  projection_delta_head_per_frame_median=stats([r['head']['median'] for r in disp if r['head']['n']]),
  projection_delta_whole_per_frame_median=stats([r['whole']['median'] for r in disp if r['whole']['n']]),
  appearance_direction_head_per_frame_median=stats([r['head_direction_disagreement_degrees']['median'] for r in directions]),
  gallery_pages=len(galleries),seconds=time.monotonic()-started,new_training=False,quality_accepted=False,
  limitations=['All image scores are training-view RGB errors, not subpixel material or heldout image accuracy.','Motion/shape observations already previously reviewed; cannot relabel per-frame variables as physical truth.','Standard PLY and cameras represent the canonical scene only.','Rendered canonical novel views have no unseen-view photograph ground truth.','Nearest-template-vertex distances and sampled center displacement are model diagnostics, not recovered physical skin/clothing ground truth.'])
 assert result['source_files_unchanged'];save(O/'result.json',result);signal.alarm(0);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','run']);a=p.parse_args()
 if a.stage=='register':register()
 else:
  try:run()
  except BaseException as e:
   import traceback
   save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise
