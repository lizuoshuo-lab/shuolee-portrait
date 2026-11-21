"""Verify104real ROI images, compareconsecutiveimages and encode two review videos."""
from pathlib import Path
import json,hashlib,subprocess,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-trajectories-095';T=O/'temporal';F=L/'outputs/dynamic-human-enerf-open-hand-flow-083';M=F/'model-support'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (T/'summary.json').exists();assert len(read(T/'completion.json')['records'])==104;plan=read(T/'protocol.json');branches=['parent','frozen','nodes','points'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13);labels=['Original','Parent','Fixed motion','Node motion','Point motion'];panels=T/'panels';panels.mkdir(exist_ok=True);rows=[];transitions=[];pngchecks=0;panelrecords=[]
def grad(x):
 grey=(x*np.array([.2126,.7152,.0722])).sum(-1)
 return np.stack([cv2.Sobel(grey,cv2.CV_64F,1,0,ksize=3,scale=1/8),cv2.Sobel(grey,cv2.CV_64F,0,1,ksize=3,scale=1/8)],-1)
for c in plan['cameras']:
 previous=None;contact=[]
 for t in plan['frames']:
  stem=f'camera{c}-frame{t:06d}';z=np.load(T/'source'/f'{stem}.npz');raw=z['rgb'];matte=z['matte'];valid=z['valid'];target=raw.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None]);h,w=raw.shape[:2];m=valid&(matte>.5);gm=m.copy();gm[[0,-1]]=False;gm[:,[0,-1]]=False;preds={};metrics={};arrays=[raw]
  assert np.array_equal(np.array(Image.open(T/'source'/f'{stem}.png')),raw)
  for b in branches:
   p=np.load(T/b/f'{stem}.npz');pred=np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1);assert np.array_equal(np.array(Image.open(T/b/f'{stem}.png')),u8(pred));pngchecks+=1;preds[b]=pred;arrays.append(u8(pred));error=pred.astype(np.float64)[m]-target.astype(np.float64)[m];metrics[b]=dict(psnr_db=float(-10*np.log10(np.mean(error*error))),mae=float(abs(error).mean()),gradient_mae_per_pixel=float(abs(grad(pred)-grad(target))[gm].mean()))
  rows.append(dict(camera=c,frame=t,split='heldout_time' if t%2 else 'train',metrics=metrics,pixels=int(m.sum())))
  im=Image.new('RGB',(w*len(labels),h+32),'white');draw=ImageDraw.Draw(im)
  for k,(arr,label) in enumerate(zip(arrays,labels)):
   im.paste(Image.fromarray(arr),(w*k,32));draw.text((w*k+4,8),f'{label} | {t}',font=font,fill='black')
  path=panels/f'{stem}-panel.png';im.save(path);check=np.array(Image.open(path))
  for k,a in enumerate(arrays):assert np.array_equal(check[32:,k*w:(k+1)*w],a)
  panelrecords.append(dict(camera=c,frame=t,file=str(path.relative_to(T)),sha256=sha(path)))
  if t in [721,727,731]:contact.append(im)
  if previous:
   union=valid&previous['valid']&((matte>.5)|(previous['matte']>.5));dt=(target-previous['target']).astype(np.float64);values={b:float(abs((preds[b]-previous['preds'][b]).astype(np.float64)[union]-dt[union]).mean()) for b in branches};transitions.append(dict(camera=c,frames=[t-1,t],mean_abs_temporal_difference_residual=values,pixels=int(union.sum())))
  previous=dict(valid=valid,matte=matte,target=target,preds=preds)
 sheet=Image.new('RGB',(contact[0].width,sum(im.height for im in contact)),'white');y=0
 for im in contact:sheet.paste(im,(0,y));y+=im.height
 sheet.save(T/f'camera{c}-comparison.png')
assert pngchecks==104 and len(transitions)==24;videos=[];ff=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe');fp=ff.with_name('ffprobe.exe')
for c in plan['cameras']:
 out=T/f'camera{c}-hand-continuity.mp4';assert not out.exists();cmd=[str(ff),'-v','error','-n','-framerate','5','-start_number','720','-i',str(panels/f'camera{c}-frame%06d-panel.png'),'-frames:v','13','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out)];subprocess.run(cmd,check=True,timeout=90);info=json.loads(subprocess.check_output([str(fp),'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(out)]))['streams'][0];assert int(info['nb_read_frames'])==13 and info['r_frame_rate']=='5/1';subprocess.run([str(ff),'-v','error','-i',str(out),'-f','null','-'],check=True,timeout=90);videos.append(dict(file=out.name,sha256=sha(out),bytes=out.stat().st_size,**info,scope='5fpsslowreview,notcaptureFPS;lossyviewingcopy,exactPNGsourcepreserved'))
flow={}
summary={c:{split:{b:{k:float(np.mean([r['metrics'][b][k] for r in rows if r['camera']==c and r['split']==split])) for k in ['psnr_db','mae','gradient_mae_per_pixel']} for b in branches} for split in ['train','heldout_time']} for c in plan['cameras']}
save(T/'records.json',dict(records=rows));save(T/'transition-records.json',dict(records=transitions));save(T/'summary.json',dict(status='104nativePNGs_26comparisonpanels_24consecutive_transitions_2videos_verified_visual_review_pending',metrics=summary,temporal_residuals={c:{b:float(np.mean([r['mean_abs_temporal_difference_residual'][b] for r in transitions if r['camera']==c])) for b in branches} for c in plan['cameras']},normal_flow_checks=flow,PNG_checks=104,videos=videos,panels=panelrecords,scope='Fixedsourceunion handcontext,originalmatte,notexacthandsemanticmask.13consecutiveframesinclude7train6heldout. Recomputed Sobelgradient error refers tooriginalcompositedtarget,notasharpnessscore. Same-scene development.',goal_complete=False));shutil.copy2(__file__,T/Path(__file__).name);print(json.dumps(dict(metrics=summary,flow=flow)),flush=True)
