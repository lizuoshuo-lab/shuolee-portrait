from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-body-prior-058';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
z=np.load(OUT/'fit.npz');verts=z['vertices'];faces=z['faces'];rows=[];basecolors=np.array([[235,120,40],[35,170,220]])
for f in sorted(BASE.glob('train-camera-*.npz')):
 c=f.stem[-2:];q=np.load(f);rgb=q['rgb'];K=q['K'];W=q['w2c'];h,w=rgb.shape[:2]
 cam=verts@W[:3,:3].T+W[:3,3];pp=cam@K.T;uv=pp[...,:2]/pp[...,2:];tri=cam[:,faces];norm=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=-1,keepdims=True),1e-8)
 center=tri.mean(2);direction=-center/np.maximum(np.linalg.norm(center,axis=-1,keepdims=True),1e-8);shade=.35+.65*np.abs((norm*direction).sum(-1));colors=(basecolors[:,None,:]*shade[...,None]).clip(0,255).astype(np.uint8)
 depths=tri[...,2].mean(-1).reshape(-1);order=np.argsort(depths)[::-1];polys=np.rint(uv[:,faces]).astype(np.int32).reshape(-1,3,2);col=colors.reshape(-1,3)
 canvas=np.zeros_like(rgb);mask=np.zeros((h,w),np.uint8);good=(tri[...,2].min(-1)>.1).reshape(-1)
 for j in order:
  if good[j]:cv2.fillConvexPoly(canvas,polys[j],tuple(int(v) for v in col[j]));cv2.fillConvexPoly(mask,polys[j],255)
 overlay=rgb.copy();valid=mask>0;overlay[valid]=np.rint(.7*canvas[valid]+.3*rgb[valid]).astype(np.uint8)
 yy,xx=np.where((q['matte']>.5)|valid);box=[max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(w,int(xx.max())+21),min(h,int(yy.max())+21)]
 x0,y0,x1,y1=box;cw=x1-x0;ch=y1-y0;panel=Image.new('RGB',(2*cw,ch+32),'white');d=ImageDraw.Draw(panel)
 for i,(title,img) in enumerate([('Training photo',rgb),('SMPL prior overlay - no appearance fit',overlay)]):
  panel.paste(Image.fromarray(img[y0:y1,x0:x1]),(i*cw,32));d.text((i*cw+4,10),title,fill='black')
 panel.save(OUT/f'camera{c}-body-prior.png')
 rows.append(dict(camera=c,input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),box=box,render='CPU depth-sorted opaque triangles; diagnostic projection, not differentiable appearance renderer',mesh_pixels=int(valid.sum())))
(OUT/'projection-renders.json').write_text(json.dumps(dict(status='15training_camera_body_prior_overlays_rendered_review_pending',records=rows,goal_complete=False),indent=2)+'\n')
shutil.copy2(__file__,OUT/Path(__file__).name)
print('15 body-prior projection panels rendered')
