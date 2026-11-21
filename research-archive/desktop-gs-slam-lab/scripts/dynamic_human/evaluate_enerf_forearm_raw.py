"""Additional raw-photograph metric; does not overwrite original matte metrics."""
from pathlib import Path
import json,hashlib,numpy as np,shutil
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';O=R/'raw-forearm-evaluation';I=L/'outputs/dynamic-human-enerf-articulated-control-064';B=L/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph/native-evaluation';E=R/'residual_graph/native-evaluation'
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
protocol=read(O/'protocol.json');assert protocol['status']=='source_only_regions_visually_accepted_before_model_evaluation'
rows=[]
for r in protocol['records']:
 c,t=r['camera'],r['frame'];name=f'camera{c}-frame{t:06d}';source=I/'native-input'/f'{name}.npz';assert sha(source)==r['source_sha256'];z=np.load(source);m=Image.new('L',(1920,1080),0);ImageDraw.Draw(m).polygon([tuple(p) for p in r['polygon_xy']],fill=255);mask=np.array(m)>0
 assert mask.sum()==r['pixels'] and np.all(z['valid'][mask]);rgb=z['rgb'].astype(np.float64)/255
 oldref=rgb*z['matte'][...,None]+.5*(1-z['matte'][...,None]);images=[z['rgb'],u8(oldref)];scores={}
 for label,root in [('before',B),('after',E)]:
  f=root/f'{name}-prediction.npz';p=np.load(f);pred=np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1);assert np.isfinite(pred).all()
  error=pred[mask].astype(np.float64)-rgb[mask];mse=float(np.mean(error**2))
  scores[label]=dict(mse=mse,psnr_db=float(-10*np.log10(mse)),mae=float(np.mean(abs(error))),mean_alpha=float(p['alpha'][mask].mean()),prediction_sha256=sha(f));images.append(u8(pred))
 x0,y0,x1,y1=660,420,870,640;w=x1-x0;h=y1-y0;panel=Image.new('RGB',(4*w,h+32),'white');draw=ImageDraw.Draw(panel)
 for i,(label,im) in enumerate(zip(['Original photograph','Original matte target','Baseline reconstruction','Local matte repair'],images)):
  panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*w,32));draw.text((i*w+4,10),label,fill='black')
 target=O/f'{name}-raw-comparison.png';panel.save(target);actual=np.array(Image.open(target))
 for i,im in enumerate(images):assert np.array_equal(actual[32:,i*w:(i+1)*w],im[y0:y1,x0:x1])
 rows.append(dict(camera=c,frame=t,pixels=r['pixels'],source_matte_mean=float(z['matte'][mask].mean()),**scores,delta_psnr_db=scores['after']['psnr_db']-scores['before']['psnr_db'],panel=target.name))
total=sum(r['pixels'] for r in rows);pooled={label:float(-10*np.log10(sum(r[label]['mse']*r['pixels'] for r in rows)/total)) for label in ['before','after']}
summary=dict(status='four_source_only_visible_forearm_interiors_compared_to_unmodified_RGB',records=rows,total_native_pixels=total,pooled_psnr_db=pooled,pooled_delta_db=pooled['after']-pooled['before'],protocol_sha256=sha(O/'protocol.json'),qualification=protocol['scope'],goal_complete=False)
(O/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8');shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(summary))
