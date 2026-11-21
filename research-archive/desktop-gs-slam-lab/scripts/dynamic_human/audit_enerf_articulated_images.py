from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,cv2
from PIL import Image
from audit_enerf_input import read_cameras
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064';RAW=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');ap=argparse.ArgumentParser();ap.add_argument('--branch',required=True);ap.add_argument('--result-root',type=Path,default=OUT);args=ap.parse_args();E=args.result_root/args.branch/'native-evaluation'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8);cams=read_cameras(RAW/'optimized');inputs={(r['camera'],r['frame']):r for r in read(OUT/'input-audit.json')['records']};report=read(E/'metrics.json');assert len(report['records'])==102
checks=[];maximum=0
for row in report['records']:
 c,t=row['camera'],row['frame'];prefix=f'camera{c}-frame{t:06d}';ir=inputs[c,t];path=OUT/'native-input'/f'{prefix}.npz';assert sha(path)==ir['input_sha256'];z=np.load(path);cam=cams[c];K=np.array(cam['K']);D=np.array(cam['D']);rgbfile=RAW/'images'/c/f'{t:06d}.jpg';maskfile=RAW/'bgmtv2'/c/f'{t:06d}.jpg';assert sha(rgbfile)==ir['source_rgb_sha256'] and sha(maskfile)==ir['source_matte_sha256']
 rgb=cv2.undistort(cv2.imread(str(rgbfile)),K.astype(np.float32),D.astype(np.float32))[...,::-1].copy();mask=cv2.undistort(cv2.imread(str(maskfile),0).astype(np.float32)/255,K.astype(np.float32),D.astype(np.float32));assert np.array_equal(rgb,z['rgb']) and np.array_equal(mask,z['matte']) and np.array_equal(z['K'],K) and np.array_equal(z['w2c'],cam['w2c'])
 predraw=np.load(E/f'{prefix}-prediction.npz');pred=np.clip(predraw['rgb']+.5*(1-predraw['alpha'][...,None]),0,1);ref=rgb.astype(np.float32)/255*mask[...,None]+.5*(1-mask[...,None]);assert np.isfinite(pred).all()
 for label,arr in [('source',rgb),('reference',u8(ref)),('trained',u8(pred)),('alpha',u8(predraw['alpha']))]:assert np.array_equal(np.array(Image.open(E/f'{prefix}-{label}.png')),arr)
 masks={'full_valid':z['valid'],'foreground':z['valid']&(mask>.5)};x0,y0,x1,y1=z['bbox'];b=np.zeros_like(z['valid']);b[y0:y1,x0:x1]=True;masks['foreground_bbox']=b&z['valid']
 for name,m in masks.items():
  d=pred.astype(np.float64)[m]-ref.astype(np.float64)[m];score=float(-10*np.log10(np.mean(d*d)));maximum=max(maximum,abs(score-row['metrics'][name]['psnr_db']));assert abs(score-row['metrics'][name]['psnr_db'])<1e-9
 panel=np.array(Image.open(E/f'{prefix}-person-comparison.png'));w=x1-x0;assert np.array_equal(panel[32:,:w],rgb[y0:y1,x0:x1]) and np.array_equal(panel[32:,w:],u8(pred)[y0:y1,x0:x1]);checks.append(dict(camera=c,frame=t,split=row['split'],prediction_sha256=sha(E/f'{prefix}-prediction.npz')))
summary=dict(status='102native_predictions408PNGs102panels_source_bytes_and_metrics_independently_verified',images=102,validation_images=sum(r['split']=='validation_time' for r in checks),training_diagnostic_images=sum(r['split']=='train' for r in checks),max_metric_difference_db=maximum,records=checks,scope='IndependentCPUrecomputation; releasedmatteisnotsemanticGT.',goal_complete=False)
(E/'independent-image-audit.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copy2(__file__,args.result_root/args.branch/'source-snapshot'/Path(__file__).name);print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
