from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw
from audit_enerf_input import read_cameras
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-heldout-cameras-068';RAW=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');OUT=R/'paired-comparison';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
branches=['head_estimated','graph_estimated','graph_zero'];reports={b:read(R/b/'native-evaluation/metrics.json') for b in branches};assert all(len(x['records'])==18 for x in reports.values());records=[r for r in read(R/'input-audit.json')['records'] if r['frame']%2];cal=read_cameras(RAW/'optimized');regions={}
for r in read(R/'review-protocol.json')['regions']:regions.setdefault((r['camera'],r['frame']),[]).append(r)
checks=[];rows=[];details=[];maximum=0
for i,r in enumerate(records):
 c,t=r['camera'],r['frame'];prefix=f'camera{c}-frame{t:06d}';path=R/'native-input'/f'{prefix}.npz';assert sha(path)==r['input_sha256'];archive=np.load(path);z={k:archive[k] for k in archive.files};archive.close();cam=cal[c];K=np.array(cam['K']);D=np.array(cam['D']);assert np.array_equal(z['K'],K) and np.array_equal(z['w2c'],cam['w2c'])
 rgbfile=RAW/'images'/c/f'{t:06d}.jpg';mattefile=RAW/'bgmtv2'/c/f'{t:06d}.jpg';assert sha(rgbfile)==r['source_rgb_sha256'] and sha(mattefile)==r['source_matte_sha256'];rgb=cv2.undistort(cv2.imread(str(rgbfile)),K.astype(np.float32),D.astype(np.float32))[...,::-1].copy();matte=cv2.undistort(cv2.imread(str(mattefile),0).astype(np.float32)/255,K.astype(np.float32),D.astype(np.float32));assert np.array_equal(rgb,z['rgb']) and np.array_equal(matte,z['matte'])
 mx,my=cv2.initUndistortRectifyMap(K.astype(np.float32),D.astype(np.float32),None,K.astype(np.float32),(1920,1080),cv2.CV_32FC1);valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079);assert np.array_equal(valid,z['valid']);ref=rgb.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None]);pred={};scores={};x0,y0,x1,y1=z['bbox'];bb=np.zeros_like(valid);bb[y0:y1,x0:x1]=True;masks=dict(full_valid=valid,foreground=valid&(matte>.5),foreground_bbox=valid&bb)
 for b in branches:
  reported=reports[b]['records'][i]
  for k in ['camera','frame','split','input_sha256','bbox']:assert reported[k]==r[k]
  E=R/b/'native-evaluation';raw=np.load(E/f'{prefix}-prediction.npz');image=np.clip(raw['rgb']+.5*(1-raw['alpha'][...,None]),0,1);assert np.isfinite(image).all();pred[b]=image;score={}
  for name,m in masks.items():
   delta=image.astype(np.float64)[m]-ref.astype(np.float64)[m];value=float(-10*np.log10(np.mean(delta*delta)));maximum=max(maximum,abs(value-reported['metrics'][name]['psnr_db']));assert abs(value-reported['metrics'][name]['psnr_db'])<1e-9;score[name]=value
  scores[b]=score
  for label,im in [('source',rgb),('reference',u8(ref)),('trained',u8(image)),('alpha',u8(raw['alpha']))]:assert np.array_equal(np.array(Image.open(E/f'{prefix}-{label}.png')),im)
  panel=np.array(Image.open(E/f'{prefix}-person-comparison.png'));w=x1-x0;assert np.array_equal(panel[32:,:w],rgb[y0:y1,x0:x1]) and np.array_equal(panel[32:,w:],u8(image)[y0:y1,x0:x1]);checks.append(dict(branch=b,camera=c,frame=t,prediction_sha256=sha(E/f'{prefix}-prediction.npz')))
 def panel(box,images,labels,path):
  a,b,cx,dy=box;w=cx-a;h=dy-b;cw=max(w,180);im=Image.new('RGB',(len(images)*cw,h+32),'white');d=ImageDraw.Draw(im)
  for j,(arr,label) in enumerate(zip(images,labels)):im.paste(Image.fromarray(arr[b:dy,a:cx]),(j*cw,32));d.text((j*cw+4,10),label,fill='black')
  im.save(path);saved=np.array(Image.open(path))
  for j,arr in enumerate(images):assert np.array_equal(saved[32:,j*cw:j*cw+w],arr[b:dy,a:cx])
 ims=[rgb,u8(pred['head_estimated']),u8(pred['graph_estimated'])];labels=['Original photograph','Head motion baseline','Local residual motion']
 for region in [dict(name='full-person',bbox=r['bbox'])]+regions.get((c,t),[]):
  name=f"{prefix}-{region['name']}.png";panel(region['bbox'],ims,labels,OUT/name)
  if region['name']!='full-person':
   a,b,cx,dy=region['bbox'];mask=valid[b:dy,a:cx];reference=ref[b:dy,a:cx];values={key:float(-10*np.log10(np.mean((pred[branch][b:dy,a:cx][mask].astype(np.float64)-reference[mask].astype(np.float64))**2))) for key,branch in [('before','head_estimated'),('after','graph_estimated')]};details.append(dict(camera=c,frame=t,region=region['name'],bbox=region['bbox'],panel=name,**values,delta_db=values['after']-values['before']))
 if t in [745,751,755]:panel(r['bbox'],[rgb,u8(pred['graph_zero']),u8(pred['graph_estimated'])],['Original photograph','Graph: frame index only','Graph: estimated capture time'],OUT/f'{prefix}-clock-comparison.png')
 rows.append(dict(camera=c,frame=t,metrics=scores,foreground_delta_db=scores['graph_estimated']['foreground']-scores['head_estimated']['foreground'],clock_delta_db=scores['graph_estimated']['foreground']-scores['graph_zero']['foreground']))
 print('HELDOUT_AUDIT',c,t,flush=True)
summary=dict(status='54native_images216PNGs54panels_independently_verified_and18paired_views18details_generated',records=rows,details=details,validation_foreground=dict(before_mean_psnr_db=float(np.mean([r['metrics']['head_estimated']['foreground'] for r in rows])),after_mean_psnr_db=float(np.mean([r['metrics']['graph_estimated']['foreground'] for r in rows])),mean_delta_db=float(np.mean([r['foreground_delta_db'] for r in rows])),improved_images=sum(r['foreground_delta_db']>0 for r in rows),total_images=18),per_camera={c:dict(graph_vs_head_mean_delta_db=float(np.mean([r['foreground_delta_db'] for r in rows if r['camera']==c])),clock_mean_delta_db=float(np.mean([r['clock_delta_db'] for r in rows if r['camera']==c]))) for c in ['00','06','12']},worst_change=min(rows,key=lambda r:r['foreground_delta_db']),clock_mean_delta_db=float(np.mean([r['clock_delta_db'] for r in rows])),goal_complete=False)
save(OUT/'summary.json',summary);save(R/'independent-image-audit.json',dict(status='54native_predictions_original_sources_valid_masks_PNGs_and_metrics_independently_verified',images=54,source_images=18,png_images=216,two_column_panels=54,max_psnr_difference_db=maximum,records=checks,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps({k:v for k,v in summary.items() if k not in ['records','details','worst_change']}))
