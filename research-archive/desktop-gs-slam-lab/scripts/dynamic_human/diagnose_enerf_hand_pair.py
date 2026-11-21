"""Training-only near-synchronous source correspondences, no learned labels."""
from pathlib import Path
import json,hashlib,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-pair-geometry-092';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['04','17'];crop={r['camera']:r['crop'] for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records'] if r['frame']==732 and r['camera'] in names}
z={c:np.load(S/f'native-input/camera{c}-frame000732.npz') for c in names};P,Q=[z[c]['K']@z[c]['w2c'][:3] for c in names];A,B=[z[c]['w2c'] for c in names];R=B[:3,:3]@A[:3,:3].T;t=B[:3,3]-R@A[:3,3]
skew=lambda t:np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
F=np.linalg.inv(z['17']['K']).T@skew(t)@R@np.linalg.inv(z['04']['K'])
_,_,vh=np.linalg.svd(P);F2=skew(Q@vh[-1])@Q@np.linalg.pinv(P);f=F/np.linalg.norm(F);f2=F2/np.linalg.norm(F2);assert min(abs(f-f2).max(),abs(f+f2).max())<1e-9
features={};sift=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.01);matcher=cv2.BFMatcher()
for c in names:
 mask=((z[c]['matte']>.9)&z[c]['valid']).astype('uint8')*255;kp,d=sift.detectAndCompute(cv2.cvtColor(z[c]['rgb'],cv2.COLOR_RGB2GRAY),mask);features[c]=(np.array([k.pt for k in kp]),d)
x,dx=features['04'];y,dy=features['17'];fw={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(dx,dy,k=2) if m.distance<.7*n.distance};bw={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(dy,dx,k=2) if m.distance<.7*n.distance};ij=np.array([(i,j) for i,j in fw.items() if bw.get(j)==i]);xx=x[ij[:,0]];yy=y[ij[:,1]]
hx=np.c_[xx,np.ones(len(xx))];hy=np.c_[yy,np.ones(len(yy))];ly=hx@F.T;lx=hy@F;err=abs(np.sum(hy*ly,1))/np.sqrt((lx[:,:2]**2).sum(1)+(ly[:,:2]**2).sum(1));X=cv2.triangulatePoints(P,Q,xx.T,yy.T).T;X=X/X[:,3:];pa=X@P.T;pb=X@Q.T;reproj=np.stack([np.linalg.norm(pa[:,:2]/pa[:,2:]-xx,axis=1),np.linalg.norm(pb[:,:2]/pb[:,2:]-yy,axis=1)],1);positive=(pa[:,2]>0)&(pb[:,2]>0)
inside=lambda uv,b:(uv[:,0]>=b[0])&(uv[:,0]<b[2])&(uv[:,1]>=b[1])&(uv[:,1]<b[3])
hand_a=inside(xx,crop['04']);hand_b=inside(yy,crop['17']);near=hand_a|hand_b;both=hand_a&hand_b
centers=[-z[c]['w2c'][:3,:3].T@z[c]['w2c'][:3,3] for c in names];rays=[X[:,:3]-c for c in centers];angle=np.rad2deg(np.arccos(np.clip(np.sum(rays[0]*rays[1],1)/np.linalg.norm(rays[0],axis=1)/np.linalg.norm(rays[1],axis=1),-1,1)))
good=both&(err<1)&positive&(reproj.max(1)<1.5)&(angle>1)
np.savez_compressed(O/'source-matches.npz',xy_a=xx,xy_b=yy,F=F,P=P,Q=Q,world_points=X[:,:3],sampson=err,reprojection=reproj,positive=positive,ray_angle_degrees=angle,hand_either=near,hand_both=both,geometry_gate=good)
def stats(m):
 return dict(count=int(m.sum()),median_sampson_px=float(np.median(err[m])) if m.any() else None,below1px=int((m&(err<1)).sum()))
ids=np.where(near)[0];sheets=[]
for start in range(0,len(ids),12):
 subset=ids[start:start+12];im=Image.new('RGB',(1000,3*156),'white');d=ImageDraw.Draw(im)
 for n,k in enumerate(subset):
  tx=n%4*250;ty=n//4*156;d.text((tx+3,ty+3),f'#{k}: {err[k]:.2f}px; gate={int(good[k])}',fill='black')
  for side,c in enumerate(names):
   u,v=np.rint((xx if side==0 else yy)[k]).astype(int);patch=Image.fromarray(z[c]['rgb']).crop((u-30,v-30,u+30,v+30)).resize((120,120),Image.Resampling.NEAREST);im.paste(patch,(tx+side*124,ty+27));cx=tx+side*124+60;cy=ty+87;d.line((cx-3,cy,cx+3,cy),fill='red');d.line((cx,cy-3,cx,cy+3),fill='red')
 fn=f'matches-review-{start//12:02d}.png';im.save(O/fn);sheets.append(dict(file=fn,indices=subset.tolist()))
save(O/'summary.json',dict(status='source_matches_complete_visual_review_pending',pair=names,frame=732,crops=crop,baseline_world_units=float(np.linalg.norm(centers[0]-centers[1])),matching='Native SIFT contrast.01, mutual Lowe.7, foreground matte>.9; descriptor matches selected before calibrated epipolar gates.',foreground=stats(np.ones(len(xx),bool)),hand_either=stats(near),hand_both=stats(both),geometry_gate=stats(good),gate_indices=np.where(good)[0].tolist(),sheets=sheets,input_sha256={c:sha(S/f'native-input/camera{c}-frame000732.npz') for c in names},qualification='Source-match candidates, not finger GT. Near-simultaneity is inherited estimated timing. No camera/time/model updates. Geometry gates cannot establish material identity.',goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);print((O/'summary.json').read_text())
