"""Visible hand-interior normal-flow hypotheses, avoiding tangential ambiguity."""
from pathlib import Path
import json,hashlib,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';N=O/'normal-flow';N.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (N/'preparation.json').exists()
masks=read(O/'hand-masks/completion.json')['records'];mb={(r['camera'],r['frame']):r for r in masks};pairs=read(O/'raft/preparation.json')['records']
save(O/'hand-masks/visual-review.json',dict(status='all15sheets45handmaskphotos_actually_reviewed',files=[dict(path=f'camera{c}-review.png',sha256=sha(O/'hand-masks'/f'camera{c}-review.png')) for c in sorted({r['camera'] for r in masks})],exclude_cameras=['14','16'],reason='14/16include some green toy throughpalm gap. Othermasks conservativeinterior selectors after1pxerosion;728somewrists only, do not treat missingfingersasbackground. No originalmatte changes.',training_mask_ground_truth=False))
save(N/'protocol.json',dict(status='fixed_before_normal_flow_selection',source_method='RAFTnativeflow; DIS usedonly asnormal-componentconsistencycheck.',gates=dict(inside_both_hand_masks='1px erosion',fb_px=.5,normal_RAFT_DIS_agreement_px=1,mean_pixel_RGB_difference_u8=18,min_source_gradient_u8_per_pixel=5,min_target_gradient=3,min_normal_cosine=.5,max_motion_px=40),
    formula='n=gradientI/|gradientI|; target=n dotRAFTflow. Constrainonlynormalcomponent, leaveedge-tangentmotion unconstrained.',
    exception='Doesnotrequire unique2Dpatchmatch because it makesno tangentialmotionclaim. Fixedbeforemodelcomparison; modelconfidenceandRGBtargetsmustnotbechanged.',excluded_cameras=['14','16'],goal_complete=False))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);cv2.setNumThreads(4)
def sample(a,p):return cv2.remap(a.astype(np.float32),p[:,0,None].astype(np.float32),p[:,1,None].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(len(p),-1)
rows=[]
for c in sorted({r['camera'] for r in pairs}-{'14','16'}):
 data={};maskdata={}
 for t in [728,730,732]:
  m=mb[c,t];p=O/'hand-masks'/m['filename'];assert sha(p)==m['sha256'];q=np.load(p);maskdata[t]=(cv2.erode(q['mask'].astype(np.uint8),np.ones((3,3),np.uint8)),q['crop'])
  p=S/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(p)==m['input_sha256'];z=np.load(p);rgb=z['rgb'];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32);gradient=np.stack([cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8),cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)],-1);data[t]=(rgb,gradient)
 sheet=Image.new('RGB',(690,810),'white');draw=ImageDraw.Draw(sheet);draw.text((8,4),f'Camera{c}: source / source accepted / target accepted / native target',font=font,fill='black')
 for ri,r in enumerate([r for r in pairs if r['camera']==c]):
  a,b=r['source_frame'],r['target_frame'];p=O/'raft'/r['path'];assert sha(p)==r['sha256'];q=np.load(p);dis=np.load(O/r['path']);xy=q['xy'];dest=q['destination'];assert np.array_equal(xy,dis['xy'])
  srcmask,srcbox=maskdata[a];dstmask,dstbox=maskdata[b];g=sample(data[a][1],xy);h=sample(data[b][1],dest);mag=np.linalg.norm(g,axis=1);othermag=np.linalg.norm(h,axis=1);normal=g/np.maximum(mag[:,None],1e-6);cos=(normal*h/np.maximum(othermag[:,None],1e-6)).sum(1)
  delta=np.abs(sample(data[a][0],xy)-sample(data[b][0],dest)).mean(1);agree=np.abs(((q['flow']-dis['flow'])*normal).sum(1))
  accepted=(sample(srcmask,xy-srcbox[:2])[:,0]>.999)&(sample(dstmask,dest-dstbox[:2])[:,0]>.999)&(q['fb']<.5)&(mag>=5)&(othermag>=3)&(cos>=.5)&(delta<=18)&(agree<=1)&(np.linalg.norm(q['flow'],axis=1)<=40)
  out=N/Path(r['path']).name;np.savez_compressed(out,xy=xy[accepted],destination=dest[accepted],flow=q['flow'][accepted],normal=normal[accepted],normal_target=(q['flow'][accepted]*normal[accepted]).sum(1),fb=q['fb'][accepted],color_delta=delta[accepted],normal_dis_agreement=agree[accepted],source_gradient=mag[accepted],target_gradient=othermag[accepted],normal_cosine=cos[accepted],source_crop=srcbox)
  rows.append(dict(camera=c,source_frame=a,target_frame=b,filename=out.name,sha256=sha(out),accepted=int(accepted.sum()),source_sha256=r['source_sha256'],target_sha256=r['target_sha256']))
  raw=Image.fromarray(data[a][0]).crop(srcbox);src=raw.copy();rawtarget=Image.fromarray(data[b][0]).crop(srcbox);dst=rawtarget.copy();sd=ImageDraw.Draw(src);td=ImageDraw.Draw(dst)
  for point,end,n in zip(xy[accepted],dest[accepted],normal[accepted]):
   x,y=point-srcbox[:2];sd.line([tuple([x,y]-1.5*n),tuple([x,y]+1.5*n)],fill=(0,255,65),width=1);x,y=end-srcbox[:2];td.ellipse((x-.7,y-.7,x+.7,y+.7),fill=(0,255,65))
  y=46+ri*190;draw.text((8,y-18),f'{a}->{b}: {accepted.sum()} hand-interior normal constraints; not dense flow GT',font=font,fill='black')
  for k,im in enumerate([raw,src,dst,rawtarget]):sheet.paste(im,(8+170*k,y))
 sheet.save(N/f'camera{c}-review.png');print('NORMAL_FLOW',c,sum(r['accepted'] for r in rows if r['camera']==c),flush=True)
save(N/'preparation.json',dict(status='52pairs_normal_candidates_visualreview_pending',records=rows,total_accepted=sum(r['accepted'] for r in rows),training_approved=False,goal_complete=False));shutil.copy2(__file__,N/Path(__file__).name)
