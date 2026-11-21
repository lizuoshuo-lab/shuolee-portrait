"""Correct three image-only seed annotations; reuse the other nine measurements."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
H=Path(__file__).parent
spec=importlib.util.spec_from_file_location('direction_measurement',H/'probe_grayT123_fixed_direction.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
O=m.O
assert not (O/'seed-annotation-repair.json').exists() and not (O/'result-completed.json').exists()
p=m.read(O/'plan.json');initial=m.read(O/'result.json');measurement=m.read(O/'measurements.json')
assert initial['state']=='stopped_photometric_measurement_before_camera_comparison' and not initial['runs']
assert initial['failed_line_ids']==['F064_bar1','F064_bar2','F064_bar3']
assert all(m.sha(Path(x['path']))==x['sha256'] for x in p['sources'])
corrected=dict(F064_bar1=1242.,F064_bar2=1306.,F064_bar3=1375.)
changes=[]
for seed in p['lines']:
 if seed['id'] in corrected:
  old=seed['a'].copy();seed['a'][0]=corrected[seed['id']]
  changes.append(dict(id=seed['id'],old_a=old,new_a=seed['a'],b_unchanged=seed['b']))
repair=dict(updated=datetime.now(timezone.utc).isoformat(),reason='The original upper guide endpoints were extrapolated from cast-shadow tips to the lower rod segment. Raw-image and intensity-row inspection confirms a seed annotation error; the intended physical edges are unchanged.',
 original_camera_comparison_executed=False,initial_result_preserved=True,changed_endpoints=changes,
 scope='Only three frame64 starting x-coordinates corrected. Same image rows, same twelve physical rods, same anchor/held assignment, same photometric rules and search width. Nine measurements reused exactly. No camera residual was available when annotating.',
 image_evidence=[dict(path=str(O/x),sha256=m.sha(O/x)) for x in ['frame_064_native_rail_crop.png','annotation-appearance-check.json','measurements.json','result.json']],
 repairs_allowed=1,no_further_repair_or_sweep=True,source_sha256=m.sha(Path(__file__)))
m.save(O/'seed-annotation-repair.json',repair)
m.save(O/'plan-annotation-corrected.json',p)
start=time.monotonic();prep=m.read(O/'prepared.json');fs={x['frame']:x for x in prep['crops']}
gray=np.asarray(Image.open(fs[64]['image']).convert('L'),float)
measurements=[]
for old in measurement['lines']:
 if old['id'] in corrected:
  seed=next(x for x in p['lines'] if x['id']==old['id']);measurements.append(m.measure(gray,seed,p['measurement']))
 else:measurements.append(old)
m.save(O/'measurements-completed.json',dict(updated=datetime.now(timezone.utc).isoformat(),lines=measurements,nine_measurements_reused=True))
failed=[x['id'] for x in measurements if not x['photometric_pass']];runs={}
if not failed:
 anchor=np.array([x['role']=='ANCHOR' for x in measurements]);lines=np.array([x['line'] for x in measurements]);centers=np.array([x['center'] for x in measurements]);fi=np.array([x['frame']-1 for x in measurements]);base=None
 for name,path in p['computation']['camera_runs'].items():
  rows=m.read(Path(path)/'data/inputs.json')['records']
  if base is None:base=rows
  assert [r['source_frame_index'] for r in rows]==[r['source_frame_index'] for r in base]
  Ks=np.array([r['K_integer_centers'] for r in rows]);scales=np.array([r['wh'] for r in rows])/np.array([1520,2704]);Kn=Ks.copy()
  Kn[:,0,:]/=scales[:,0,None];Kn[:,1,:]/=scales[:,1,None];Kn[:,0,2]+=.5/scales[:,0]-.5;Kn[:,1,2]+=.5/scales[:,1]-.5
  Rs=np.array([r['w2c'] for r in rows])[:,:3,:3];E=Rs[fi];K=Kn[fi]
  d,s,ns=m.direction(lines[anchor],K[anchor],E[anchor]);record=[];allheld=[]
  for obs,l,c,k,r in zip(measurements,lines,centers,K,E):
   pred=m.predicted_line(c,d,k,r);xy=np.array([x['xy'] for x in obs['samples'] if not x['reasons']]);err=np.abs(np.c_[xy,np.ones(len(xy))]@pred)
   angle=float(np.rad2deg(np.arccos(np.clip(abs(l[:2]@pred[:2]),0,1))))
   record.append(dict(id=obs['id'],frame=obs['frame'],role=obs['role'],predicted_line=pred.tolist(),direction_difference_deg=angle,direction_only_pixel_residual=m.stats(err)))
   if obs['role']=='HELD':allheld.extend(err)
  Q=Rotation.from_rotvec(d*np.deg2rad(7)).as_matrix();invariant=[]
  for c,k,r in zip(centers,K,E):
   one=m.predicted_line(c,d,k,r);two=m.predicted_line(c,d,k,r@Q)
   invariant.append(min(np.linalg.norm(one-two),np.linalg.norm(one+two)))
  runs[name]=dict(direction_world=d.tolist(),anchor_plane_singular_values=s.tolist(),records=record,
    held_direction_only_residual_px=m.stats(allheld),seven_degree_unobservable_rotation_line_max_abs=float(max(invariant)),camera_updated=False)
result=dict(updated=datetime.now(timezone.utc).isoformat(),state='completed_after_documented_seed_annotation_repair' if not failed else 'stopped_after_only_allowed_annotation_repair',elapsed_seconds=time.monotonic()-start,
 failed_line_ids=failed,photometric_pass=not failed,controls=initial['controls'],runs=runs,
 source_hashes_unchanged=all(m.sha(Path(x['path']))==x['sha256'] for x in p['sources']),
 seed_annotation_repair=str(O/'seed-annotation-repair.json'),nine_original_measurements_reused=True,
 camera_or_point_updates=0,quality_accepted=False,physical_material_correspondence_certified=False,
 interpretation=initial['interpretation'])
assert result['source_hashes_unchanged'] and result['elapsed_seconds']<120
m.save(O/'result-completed.json',result)
index=m.read(O/'index.json')
for img in index['images']:
 img['region']='初次标注测量：第64帧端点误入阴影，尚未执行相机检验'
 img['columns'][-1]='原图保留：这一轮在相机计算前停止'
for n in [56,64,72,80]:
 im=Image.open(fs[n]['image']).convert('RGB');box=fs[n]['crop'];raw=im.crop(box);measure=raw.copy();pred=raw.copy();a=ImageDraw.Draw(measure);b=ImageDraw.Draw(pred)
 for obs in [x for x in measurements if x['frame']==n]:
  for sample in obs['samples']:
   x,y=np.array(sample['xy'])-box[:2];color=(80,225,180) if not sample['reasons'] else (220,90,90)
   a.ellipse((x-1,y-1,x+1,y+1),fill=color)
  for j,(name,run) in enumerate(runs.items()):
   rec=next(x for x in run['records'] if x['id']==obs['id']);l=np.array(rec['predicted_line']);c=np.array(obs['center']);t=np.array([-l[1],l[0]])
   seed=next(x for x in p['lines'] if x['id']==obs['id']);length=np.linalg.norm(np.array(seed['b'])-seed['a'])
   ends=[(c+sgn*.5*length*t-np.array(box[:2])).tolist() for sgn in [-1,1]]
   b.line([tuple(x) for x in ends],fill=(0,185,255) if j==0 else (255,160,50),width=1)
 canvas=Image.new('RGB',(raw.width*3,raw.height),(255,255,255))
 for j,image in enumerate([raw,measure,pred]):canvas.paste(image,(j*raw.width,0))
 name=f'frame_{n:03d}_completed_raw_edges_fixed_cameras.png';canvas.save(O/name)
 index['images'].append(dict(frame=f'frame_{n:06d}',region='标注纠正后，同一栏杆竖直方向检查，未达到相机/人脸验收',file=name,raw=f'frame_{n:03d}_raw.png',
   columns=['原图','独立亮度边缘：绿为有效，红为光度歧义','冻结相机方向：研究青线／00橙线，线位置固定于观测中心'],source=fs[n]))
m.save(O/'index.json',index)
print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False))
for name,r in runs.items():print(json.dumps(dict(run=name,held=r['held_direction_only_residual_px'],yaw_null=r['seven_degree_unobservable_rotation_line_max_abs'],singular=r['anchor_plane_singular_values'])))
