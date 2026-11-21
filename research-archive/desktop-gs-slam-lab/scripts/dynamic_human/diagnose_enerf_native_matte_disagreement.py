"""Read-only audit of supervision conflicts; predicted masks are never treated as GT."""
from pathlib import Path
import json,numpy as np,cv2
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-actor-supervision-070';I=L/'outputs/dynamic-human-enerf-articulated-control-064';O=R/'matte-disagreement';O.mkdir(exist_ok=True)
rows=[]
for path in sorted((R/'masks').glob('camera*-frame000750.npz')):
 c=path.name[6:8];z=np.load(I/'native-input'/path.name);masks=np.load(path)['raw_masks'];rgb=z['rgb'];matte=z['matte'];valid=z['valid']
 core=cv2.erode(masks[:2].any(0).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
 conflict=core&(matte<.5)&valid
 n,labels,stats,_=cv2.connectedComponentsWithStats(conflict.astype(np.uint8),8)
 components=[]
 for j in sorted(range(1,n),key=lambda j:-stats[j,4])[:3]:
  x,y,w,h,count=map(int,stats[j]);
  if count<50:continue
  x0=max(0,x-24);y0=max(0,y-24);x1=min(1920,x+w+24);y1=min(1080,y+h+24)
  ref=np.rint(rgb.astype(float)*matte[...,None]+127.5*(1-matte[...,None])).clip(0,255).astype(np.uint8)
  imgs=[rgb, np.repeat(np.rint(matte[...,None]*255).astype(np.uint8),3,2),ref]
  names=['Original training photograph','Released training matte','Current composited RGB target']
  width=x1-x0;height=y1-y0;panel=Image.new('RGB',(3*max(width,190),height+32),'white');d=ImageDraw.Draw(panel)
  for k,im in enumerate(imgs):panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(k*max(width,190),32));d.text((k*max(width,190)+3,10),names[k],fill='black')
  name=f'camera{c}-component{j:03d}.png';panel.save(O/name);components.append(dict(area=count,bbox=[x0,y0,x1,y1],panel=name))
 rows.append(dict(camera=c,predicted_person_core_pixels=int(core.sum()),disagreement_pixels=int(conflict.sum()),components=components))
report=dict(status='training_only_matte_disagreement_candidates_visual_review_required',frame=750,records=rows,qualification='SAM2 internal foreground versus released matte; disagreement is not automatically matte error. No masks/RGB/models modified. Only training cameras used.',goal_complete=False)
(O/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(sorted(rows,key=lambda x:-x['disagreement_pixels'])[:6]))
