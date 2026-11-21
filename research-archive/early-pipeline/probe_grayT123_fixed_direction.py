"""One CPU-only direction consistency test. Neither K nor any pose is optimized."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/grayT-fixed-direction-review')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
 x=np.asarray(x,float)
 return dict(n=int(x.size),median=float(np.median(x)),rms=float(np.sqrt(np.mean(x*x))),maximum=float(x.max())) if x.size else dict(n=0)
def norm(x):return x/np.linalg.norm(x)
def line_fit(xy):
 center=xy.mean(0);_,s,v=np.linalg.svd(xy-center,full_matrices=False)
 n=v[-1];l=np.r_[n,-n@center]
 return l,center,s
def direction(lines,Ks,Rs):
 ns=np.array([norm(R.T@(K.T@l)) for l,K,R in zip(lines,Ks,Rs)])
 _,s,v=np.linalg.svd(ns,full_matrices=False)
 return v[-1],s,ns
def predicted_line(center,d,K,R):
 l=np.cross(np.r_[center,1],K@(R@d));return l/np.linalg.norm(l[:2])
def controls():
 K=np.array([[1500.,0,790],[0,1480,1390],[0,0,1]])
 d=norm(np.array([.03,1.,.2]));Rs=[];ls=[];points=[]
 for f in range(4):
  R=Rotation.from_euler('yxz',[f*13,5+f,2-f],degrees=True).as_matrix()
  for k in range(3):
   x=np.array([-.6+k*.6,.1,4.+.2*k]);w=x+np.linspace(-.4,.4,40)[:,None]*d
   cam=w@R.T+np.array([.1*f,0,1.]);q=cam@K.T;xy=q[:,:2]/q[:,2,None]
   l,c,_=line_fit(xy);Rs.append(R);ls.append(l);points.append(xy)
 recovered,s,_=direction(ls[:9],[K]*9,Rs[:9])
 error=float(min(np.linalg.norm(recovered-d),np.linalg.norm(recovered+d)))
 maxerr=0;changed=[];invariance=[]
 Q=Rotation.from_rotvec(d*np.deg2rad(17)).as_matrix()
 X=Rotation.from_euler('x',2,degrees=True).as_matrix()
 for j in [9,10,11]:
  xy=points[j];c=xy.mean(0);l=predicted_line(c,recovered,K,Rs[j]);q=predicted_line(c,recovered,K,Rs[j]@Q)
  maxerr=max(maxerr,float(np.abs(np.c_[xy,np.ones(len(xy))]@l).max()))
  invariance.append(float(np.abs(np.c_[xy,np.ones(len(xy))]@q).max()))
  changed.extend(np.abs(np.c_[xy,np.ones(len(xy))]@predicted_line(c,recovered,K,X@Rs[j])))
 result=dict(known_direction_vector_error=error,synthetic_held_line_max_px=maxerr,
  two_degree_held_pitch_error=stats(changed),seventeen_degree_world_direction_yaw_max_px=max(invariance),
  pass_=bool(error<1e-10 and maxerr<1e-8 and max(invariance)<1e-8 and max(changed)>.01))
 assert result['pass_'],result
 return result
def measure(gray,seed,cfg):
 a=np.array(seed['a']);b=np.array(seed['b']);centers=np.linspace(a,b,cfg['count_per_line'])
 offsets=np.arange(-cfg['horizontal_search_halfwidth_px'],cfg['horizontal_search_halfwidth_px']+cfg['step_px']/2,cfg['step_px'])
 profiles=[]
 for dy in cfg['tangent_rows']:
  x=centers[:,0,None]+offsets;y=np.broadcast_to(centers[:,1,None]+dy,x.shape)
  profiles.append(map_coordinates(gray,[y,x],order=1,mode='nearest'))
 profile=np.mean(profiles,axis=0);gradient=np.gradient(profile,cfg['step_px'],axis=1)
 xy=[];samples=[]
 for c,g in zip(centers,gradient):
  peak=int(np.argmax(g));reason=[];value=float(g[peak]);delta=0.
  if peak==0 or peak==len(g)-1:reason.append('profile_boundary')
  else:
   denom=g[peak-1]-2*g[peak]+g[peak+1]
   if abs(denom)>1e-10:delta=float(np.clip(.5*(g[peak-1]-g[peak+1])/denom,-.5,.5))*cfg['step_px']
  shift=float(offsets[peak]+delta)
  if value<cfg['minimum_positive_gradient']:reason.append('weak_gradient')
  peaks,_=find_peaks(g)
  other=[g[k] for k in peaks if abs(offsets[k]-offsets[peak])>=cfg['secondary_peak_min_separation_px']]
  if other and max(other)>=cfg['secondary_to_primary_limit']*value:reason.append('competing_edge')
  if abs(shift)>=cfg['max_abs_offset_px']:reason.append('offset_boundary')
  point=c+np.array([shift,0])
  samples.append(dict(seed=c.tolist(),xy=point.tolist(),gradient=value,offset=shift,reasons=reason))
  if not reason:xy.append(point)
 valid=np.array(xy);passed=len(valid)>=cfg['minimum_valid_samples'] and len(valid)/len(centers)>=cfg['minimum_valid_fraction']
 out=dict(id=seed['id'],frame=seed['frame'],role=seed['role'],samples=samples,valid=len(valid),total=len(centers),photometric_pass=passed)
 if len(valid)>=3:
  l,c,s=line_fit(valid);res=np.abs(np.c_[valid,np.ones(len(valid))]@l);n=len(valid)
  lh,_,_=line_fit(valid[:n//2]);lt,_,_=line_fit(valid[n//2:])
  out.update(line=l.tolist(),center=c.tolist(),linear_residual_px=stats(res),
    first_second_half_direction_difference_deg=float(np.rad2deg(np.arccos(np.clip(abs(lh[:2]@lt[:2]),0,1)))),
    endpoint_half_pixel_opposite_jitter_bound_deg=float(np.rad2deg(np.arctan(1/np.linalg.norm(valid[-1]-valid[0])))))
 return out
def main():
 assert not (O/'started.json').exists(),'Do not repeat registered probe'
 p=read(O/'plan.json');start=time.monotonic()
 assert all(sha(Path(x['path']))==x['sha256'] for x in p['sources'])
 save(O/'started.json',dict(updated=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(O/'plan.json'),source_sha256=sha(Path(__file__))))
 unit=controls();save(O/'controls.json',unit)
 prep=read(O/'prepared.json');frame_sources={r['frame']:r for r in prep['crops']};images={};measurements=[]
 for seed in p['lines']:
  if time.monotonic()-start>p['computation']['cpu_seconds']:raise RuntimeError('CPU budget exceeded')
  n=seed['frame']
  if n not in images:images[n]=Image.open(frame_sources[n]['image']).convert('RGB')
  measurements.append(measure(np.asarray(images[n].convert('L'),float),seed,p['measurement']))
 save(O/'measurements.json',dict(updated=datetime.now(timezone.utc).isoformat(),lines=measurements))
 failed=[m['id'] for m in measurements if not m['photometric_pass']]
 runs={};base=None
 if not failed:
  anchor=np.array([m['role']=='ANCHOR' for m in measurements]);lines=np.array([m['line'] for m in measurements]);centers=np.array([m['center'] for m in measurements]);fi=np.array([m['frame']-1 for m in measurements])
  for name,path in p['computation']['camera_runs'].items():
   rows=read(Path(path)/'data/inputs.json')['records']
   if base is None:base=rows
   assert [r['source_frame_index'] for r in rows]==[r['source_frame_index'] for r in base]
   Ks=np.array([r['K_integer_centers'] for r in rows]);scales=np.array([r['wh'] for r in rows])/np.array([1520,2704]);Kn=Ks.copy()
   Kn[:,0,:]/=scales[:,0,None];Kn[:,1,:]/=scales[:,1,None]
   Kn[:,0,2]+=.5/scales[:,0]-.5;Kn[:,1,2]+=.5/scales[:,1]-.5
   Rs=np.array([r['w2c'] for r in rows])[:,:3,:3];E=Rs[fi];K=Kn[fi]
   d,s,ns=direction(lines[anchor],K[anchor],E[anchor]);record=[];allheld=[]
   for m,l,c,k,r in zip(measurements,lines,centers,K,E):
    pred=predicted_line(c,d,k,r);xy=np.array([x['xy'] for x in m['samples'] if not x['reasons']]);err=np.abs(np.c_[xy,np.ones(len(xy))]@pred)
    angle=float(np.rad2deg(np.arccos(np.clip(abs(l[:2]@pred[:2]),0,1))))
    record.append(dict(id=m['id'],frame=m['frame'],role=m['role'],predicted_line=pred.tolist(),direction_difference_deg=angle,direction_only_pixel_residual=stats(err)))
    if m['role']=='HELD':allheld.extend(err)
   # These curves are diagnostic only. Translation never enters the equations.
   invariant=[]
   Q=Rotation.from_rotvec(d*np.deg2rad(7)).as_matrix()
   for c,k,r in zip(centers,K,E):
    one=predicted_line(c,d,k,r);two=predicted_line(c,d,k,r@Q)
    invariant.append(min(np.linalg.norm(one-two),np.linalg.norm(one+two)))
   runs[name]=dict(direction_world=d.tolist(),anchor_plane_singular_values=s.tolist(),anchor_singular_gap=float(s[-2]/max(s[-1],1e-16)),records=record,
      held_direction_only_residual_px=stats(allheld),seven_degree_unobservable_rotation_line_max_abs=float(max(invariant)),camera_updated=False)
 all_pass=not failed
 result=dict(updated=datetime.now(timezone.utc).isoformat(),state='completed_direction_control' if all_pass else 'stopped_photometric_measurement_before_camera_comparison',
    elapsed_seconds=time.monotonic()-start,photometric_pass=all_pass,failed_line_ids=failed,controls=unit,runs=runs,
    source_hashes_unchanged=all(sha(Path(x['path']))==x['sha256'] for x in p['sources']),
    camera_or_point_updates=0,quality_accepted=False,physical_material_correspondence_certified=False,
    interpretation='Direction-only observation under conditional K and approximate parallel-upright assumption. Held residuals do not constrain translation, yaw around the common direction, or absolute face position.')
 assert result['source_hashes_unchanged']
 save(O/'result.json',result)
 gallery=[]
 for n,im in sorted(images.items()):
  box=frame_sources[n]['crop'];raw=im.crop(box);meas=raw.copy();pred=raw.copy();a=ImageDraw.Draw(meas);b=ImageDraw.Draw(pred)
  for m in [x for x in measurements if x['frame']==n]:
   for sample in m['samples']:
    x,y=np.array(sample['xy'])-box[:2];color=(80,225,180) if not sample['reasons'] else (220,90,90)
    a.ellipse((x-1,y-1,x+1,y+1),fill=color)
   for j,(name,run) in enumerate(runs.items()):
    rec=next(x for x in run['records'] if x['id']==m['id']);l=np.array(rec['predicted_line']);c=np.array(m['center']);t=np.array([-l[1],l[0]])
    seed=next(x for x in p['lines'] if x['id']==m['id']);length=np.linalg.norm(np.array(seed['b'])-seed['a'])
    ends=[(c+sgn*.5*length*t-np.array(box[:2])).tolist() for sgn in [-1,1]]
    b.line([tuple(x) for x in ends],fill=(0,185,255) if j==0 else (255,160,50),width=1)
  rawname=f'frame_{n:03d}_raw.png';raw.save(O/rawname)
  canvas=Image.new('RGB',(raw.width*3,raw.height),(255,255,255))
  for j,img in enumerate([raw,meas,pred]):canvas.paste(img,(j*raw.width,0))
  name=f'frame_{n:03d}_raw_edges_fixed_cameras.png';canvas.save(O/name)
  gallery.append(dict(frame=f'frame_{n:06d}',region='原生栏杆直线方向核验，尚未质量验收',file=name,raw=rawname,
    columns=['原图','独立亮度边缘：绿为有效，红为光度歧义','冻结相机方向：研究青线／00橙线；线位置固定于观测中心'],source=frame_sources[n]))
 save(O/'index.json',dict(images=gallery))
 print(json.dumps({k:v for k,v in result.items() if k not in ['runs']},ensure_ascii=False))
 for name,r in runs.items():print(json.dumps(dict(run=name,held=r['held_direction_only_residual_px'],yaw_null=r['seven_degree_unobservable_rotation_line_max_abs'],singular=r['anchor_plane_singular_values'])))
if __name__=='__main__':main()
