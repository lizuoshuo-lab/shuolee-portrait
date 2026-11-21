"""Freeze conservative new-frame regions before any new descriptors."""
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image,ImageDraw
import json,hashlib
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'video10-temporal-bridge'; B=R/'video10-static-material'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'regions.json').exists()
p=read(O/'plan.json');b=read(B/'plan.json');inputs=read(O/'native-inputs.json');anchors=p['anchor_frames']
extra={
27:[[0,250,160,1000],[600,1000,1800,1520]],54:[[0,250,150,1000],[1100,1100,1820,1520],[2520,420,2704,1520]],80:[[1150,1100,1850,1520]],
134:[[900,500,1790,1520],[1740,650,2130,1290],[2300,480,2704,1200],[0,400,620,880]],161:[[1500,600,2500,1520],[0,400,540,900]],188:[[1880,450,2704,1520],[0,420,380,900]],
242:[[1580,440,2280,870],[2260,430,2704,1110],[1510,900,2704,1520]],268:[[1430,450,2020,870],[1940,490,2280,1080],[1500,900,2704,1520]],295:[[1310,460,1800,880],[850,470,1190,1200],[1750,500,2030,1090],[1480,980,2704,1520]],
349:[[910,430,1250,860],[410,440,770,1250],[2130,300,2704,1080],[1510,850,2450,1520],[1560,340,1880,660]],376:[[640,450,1160,930],[0,450,420,1250],[1820,370,2704,1120],[1610,930,2050,1520]],403:[[330,430,920,970],[1510,410,2270,1160],[1570,1020,2300,1520]],
457:[[0,480,550,1100],[2280,280,2704,1020],[1580,890,2110,1520],[1220,440,1780,1100]],484:[[1970,400,2704,1030],[850,450,1330,1120],[1660,960,1960,1520],[0,490,190,1100]],510:[[650,400,1130,1130],[1720,440,2390,1040],[800,980,1170,1520]],
564:[[0,390,800,1020],[1500,440,1830,1040],[650,820,1210,1520]],590:[[0,390,520,1020],[1170,430,1650,1000],[590,840,1240,1520]],617:[[0,390,200,1050],[1000,430,1260,1050],[600,810,1210,1520]],
671:[[650,420,1130,900],[650,860,1170,1520],[2250,390,2704,760],[0,550,450,1000]],698:[[500,430,980,950],[520,950,1170,1520],[1990,350,2380,750],[0,600,190,1080]],725:[[240,330,850,900],[450,1000,1180,1520],[1760,350,2110,750]],
779:[[0,300,430,1000],[350,950,1240,1520],[1400,350,1650,700]],806:[[0,330,340,1000],[300,970,1220,1520]],832:[[330,1050,1400,1520]]}
rows=[]
for e in inputs['entries']:
 f=e['frame'];i=next(i for i in range(8) if anchors[i]<f<anchors[i+1])
 def envelope(key):
  a,c=b['regions'][key][i:i+2]
  return [min(a[0],c[0]),min(a[1],c[1]),max(a[2],c[2]),max(a[3],c[3])]
 body=envelope('body_boxes');head=envelope('head_boxes');exclude=b['regions']['background_exclusions'][i]+b['regions']['background_exclusions'][i+1]+extra[f]
 for box in [body,head]+exclude:assert 0<=box[0]<box[2]<=2704 and 0<=box[1]<box[3]<=1520
 assert sha(Path(e['path']))==e['sha256']
 rows.append(dict(frame=f,body=body,head=head,exclude_background=exclude,source=e['path'],source_sha256=e['sha256']))
save(O/'regions.json',dict(updated=datetime.now(timezone.utc).isoformat(),rows=rows,method='New-frame person/head envelopes of neighboring anchors, plus conservative per-frame background exclusions based on8actuallyreviewedoverviewpages. No source pixels or anchor features changed.',
 semantics='Purple person/headROI is not anatomy/physicalidentity. Red excludes potential reflections/TV/window/plant occlusion from background descriptors. Excluded areas inside personROI may still generate subject candidates and require material review. Remaining glare/repetitive surfaces are unverified, never automatically static.',
 limitation='Masks on newly added frames are new observation preprocessing, so any aggregate gain is not pure evidence about K. Common anchor features remain bitexact.',reviewed=False))
for j in range(8):
 page=Image.new('RGB',(1352,2280),'white')
 for k,r in enumerate(rows[3*j:3*j+3]):
  with Image.open(r['source']) as im:im=im.convert('RGBA')
  layer=Image.new('RGBA',im.size,(0,0,0,0));dr=ImageDraw.Draw(layer)
  for box in r['exclude_background']:dr.rectangle(box,fill=(235,40,40,75),outline=(235,40,40,255),width=4)
  dr.rectangle(r['body'],fill=(120,40,220,45),outline=(120,40,220,255),width=6);dr.rectangle(r['head'],outline=(255,180,0,255),width=6)
  page.paste(Image.alpha_composite(im,layer).convert('RGB').resize((1352,760)),(0,760*k))
 page.save(O/f'regions_{j:02d}.jpg',quality=93)
print('24newregion proposals,8pages; no new features. Inspect before admitting extraction.')
