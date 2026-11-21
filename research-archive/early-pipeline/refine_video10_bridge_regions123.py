"""Pre-extraction review correction; old9features never changed."""
from pathlib import Path
from PIL import Image,ImageDraw
from datetime import datetime,timezone
import json,shutil
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-temporal-bridge')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
p=read(O/'regions.json'); assert not (O/'run').exists() and not (O/'regions-proposal-before-review.json').exists()
shutil.copy2(O/'regions.json',O/'regions-proposal-before-review.json')
for j in range(8):shutil.copy2(O/f'regions_{j:02d}.jpg',O/f'regions_proposed_{j:02d}.jpg')
head={27:[1280,300,1510,600],54:[1280,300,1530,630],80:[1280,290,1540,650],134:[1220,370,1490,680],161:[1280,300,1560,650],188:[1250,280,1560,660],242:[1230,130,1550,500],268:[1190,120,1540,530],295:[1220,130,1570,550],349:[1210,100,1560,520],376:[1180,120,1530,550],403:[1190,100,1520,540],457:[1180,100,1540,550],484:[1180,120,1540,550],510:[1190,130,1550,570],564:[1160,130,1560,560],590:[1160,130,1560,560],617:[1160,130,1560,590],671:[1180,150,1520,610],698:[1180,150,1510,590],725:[1180,160,1510,590],779:[1220,140,1500,560],806:[1230,150,1500,590],832:[1210,150,1480,590]}
# Read the already registered per-frame exclusions without executing that script.
import ast
tree=ast.parse((Path(__file__).parent/'register_video10_bridge_regions123.py').read_text(encoding='utf-8'))
extra=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='extra' for t in n.targets))
extra[27].append([280,470,790,1090]);extra[54].append([330,500,990,1520]);extra[80]+=[[450,570,1090,1520],[2120,530,2704,1520]]
extra[188].append([1500,630,1850,1330]);extra[457].append([580,320,1020,730]);extra[484].append([160,300,720,720]);extra[510].append([0,280,390,740]);extra[832]+=[[0,270,400,1100],[250,550,680,1330]]
for r in p['rows']:
 r['head']=head[r['frame']];r['exclude_background']=extra[r['frame']]
p.update(updated=datetime.now(timezone.utc).isoformat(),reviewed=False,method='Body envelope retained; new24head boxes tightened from actual native overview. Background exclusions use only per-frame observed potential TV/glass/window/plant regions instead of unioning neighboring-camera exclusion rectangles.',
 proposal_correction='Viewed all8rawoverview pages and first2proposal pages. Neighbor-exclusion union removed useful fixed door/heater structure, and coarse head envelopes contained too much clothing/ceiling. Correction before any new descriptors. Final8annotationpages still need review.',
 causal_boundary='This is an added-time-observation plus region-allocation control, not a pure temporal-density or K-only causal experiment. Anchor keypoints/descriptors remain frozen. Gain must be decomposed on the common anchor endpoints.')
(O/'regions.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for j in range(8):
 page=Image.new('RGB',(1352,2280),'white')
 for k,r in enumerate(p['rows'][3*j:3*j+3]):
  with Image.open(r['source']) as im:im=im.convert('RGBA')
  layer=Image.new('RGBA',im.size,(0,0,0,0));dr=ImageDraw.Draw(layer)
  for box in r['exclude_background']:dr.rectangle(box,fill=(235,40,40,75),outline=(235,40,40,255),width=4)
  dr.rectangle(r['body'],fill=(120,40,220,45),outline=(120,40,220,255),width=6);dr.rectangle(r['head'],outline=(255,180,0,255),width=6)
  page.paste(Image.alpha_composite(im,layer).convert('RGB').resize((1352,760)),(0,760*k))
 page.save(O/f'regions_{j:02d}.jpg',quality=93)
print('Region correction completed pre-extraction, original proposal retained.')
