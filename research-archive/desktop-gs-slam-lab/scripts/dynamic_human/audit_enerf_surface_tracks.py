from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
OUT=LAB/'outputs/dynamic-human-enerf-surface-tracks-054';z=np.load(OUT/'tracks.npz');r=json.loads((OUT/'depth-comparison.json').read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
cams={c:np.load(BASE/f'train-camera-{c}.npz') for c in set(z['camera'])}
max_sample=0.;max_reproj=0.;depths={c:{k:v for k,v in np.load(OUT/'rendered'/f'camera{c}-depth.npz').items()} for c in cams}
def bilinear32(a,xy):
 q=np.rint(np.asarray(xy)*32).astype(int);ix,iy=q//32;fx,fy=q%32/32
 return float(a[iy,ix]*(1-fx)*(1-fy)+a[iy,ix+1]*fx*(1-fy)+a[iy+1,ix]*(1-fx)*fy+a[iy+1,ix+1]*fx*fy)
for row in r['records']:
 i=row['observation'];c=row['camera'];cam=cams[c];X=z['points'][z['track_id'][i]];T=cam['w2c'];K=cam['K'];q=K@(T@np.r_[X,1])[:3]
 reproj=np.linalg.norm(q[:2]/q[2]-z['xy'][i]);max_reproj=max(max_reproj,abs(reproj-z['reprojection'][i]))
 assert q[2]>0 and reproj<1 and z['angle'][z['track_id'][i]]>2
 d=bilinear32(depths[c]['depth'],z['xy'][i]);alpha=bilinear32(depths[c]['alpha'],z['xy'][i]);max_sample=max(max_sample,abs(d-row['expected_depth']))
 assert abs(d-row['expected_depth'])<2e-6 and abs(alpha-row['alpha'])<2e-7
 assert abs(row['triangulated_depth']-q[2])<1e-9
 camxyz=np.linalg.solve(K,np.r_[z['xy'][i],1])*row['expected_depth'];world=np.linalg.inv(T)@np.r_[camxyz,1]
 assert np.max(abs(world[:3]-row['world_from_render']))<1e-7
 errs=[]
 for j in np.flatnonzero((z['track_id']==z['track_id'][i])&(z['camera']!=c)):
  other=cams[str(z['camera'][j])];p=other['K']@(other['w2c']@world)[:3];errs.append(np.linalg.norm(p[:2]/p[2]-z['xy'][j]))
 assert abs(np.median(errs)-row['cross_view_median_reprojection_px'])<1e-5
for c in ['01','08','11']:
 cam=cams[c];rgb=cam['rgb'];ys,xs=np.where(cam['matte']>.5);box=[max(0,int(xs.min())-16),max(0,int(ys.min())-16),min(1920,int(xs.max())+17),min(1080,int(ys.max())+17)]
 a,b,cc,d=box;pic=Image.fromarray(rgb[b:d,a:cc].copy());draw=ImageDraw.Draw(pic)
 for xy in z['xy'][z['camera']==c]:
  x,y=xy-[a,b];draw.line((x-3,y,x+3,y),fill=(255,0,0),width=1);draw.line((x,y-3,x,y+3),fill=(255,0,0),width=1)
 pic.save(OUT/'identity-review'/f'camera{c}-track-coverage.png')
save(OUT/'independent-audit.json',dict(status='90tracks375observations_geometry_depth_samples_and_crossview_reprojection_independently_recomputed',
 max_reprojection_recompute_difference=max_reproj,max_depth_sampling_difference=max_sample,depth_maps=len(depths),
 checkpoint_sha256=hashlib.sha256((BASE/'first-frame/resume.pt').read_bytes()).hexdigest(),
 depth_metric_qualification='Agreement to triangulation hypotheses from observed textures, not independent physical depth groundtruth.',goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps(dict(status='verified',max_reprojection_difference=max_reproj,max_depth_sampling_difference=max_sample)))
