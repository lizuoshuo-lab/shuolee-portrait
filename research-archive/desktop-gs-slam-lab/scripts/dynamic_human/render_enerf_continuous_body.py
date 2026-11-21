from pathlib import Path
import json,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-continuous-body-063';SOURCE=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';ref=np.load(OUT/'reference.npz');faces=ref['faces'];mat={b:np.load(OUT/b/'capture-affines.npz') for b in ['fixed_offsets','estimated_offsets']};observations={(r['camera'],r['frame']):r for r in json.loads((SOURCE/'observations.json').read_text())['records']};colors=np.array([[235,120,40],[35,170,220]])
def overlay(rgb,vertices,K,W):
 cam=vertices@W[:3,:3].T+W[:3,3];p=cam@K.T;uv=p[...,:2]/p[...,2:];tri=cam[:,faces];n=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]);n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-8);direction=-tri.mean(2);direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-8);shade=.35+.65*abs((n*direction).sum(-1));col=(colors[:,None]*shade[...,None]).astype(np.uint8).reshape(-1,3);polys=np.rint(uv[:,faces]).astype(np.int32).reshape(-1,3,2);order=np.argsort(tri[...,2].mean(-1).ravel())[::-1];canvas=np.zeros_like(rgb);mask=np.zeros(rgb.shape[:2],np.uint8)
 for i in order:cv2.fillConvexPoly(canvas,polys[i],tuple(int(x) for x in col[i]));cv2.fillConvexPoly(mask,polys[i],255)
 out=rgb.copy();m=mask>0;out[m]=np.rint(.7*canvas[m]+.3*rgb[m]).astype(np.uint8);return out
for c in ['01','04','08','17']:
 for t in [744,750,756]:
  r=observations[c,t];rgb=np.load(SOURCE/'native-input'/f'camera{c}-frame{t:06d}.npz')['rgb'];panels=[rgb]
  for b,z in mat.items():
   i=np.where((z['camera']==c)&(z['frame']==t))[0].item();panels.append(overlay(rgb,z['vertices'][i].reshape(2,6890,3),np.array(r['K']),np.array(r['w2c'])))
  boxes=[d['box'] for d in r['detections']];x0=max(0,min(b[0] for b in boxes)-40);y0=max(0,min(b[1] for b in boxes)-40);x1=min(1920,max(b[2] for b in boxes)+40);y1=min(1080,max(b[3] for b in boxes)+40);w=x1-x0;h=y1-y0;im=Image.new('RGB',(3*w,h+32),'white');d=ImageDraw.Draw(im)
  for i,(label,p) in enumerate(zip(['Training photograph','Same frame-index body','Capture-time body'],panels)):im.paste(Image.fromarray(p[y0:y1,x0:x1]),(i*w,32));d.text((i*w+5,10),label,fill='black')
  im.save(OUT/f'camera{c}-frame{t:06d}-motion-priors.png')
shutil.copy2(__file__,OUT/Path(__file__).name);print('12 matched original/candidate body projection panels rendered')
