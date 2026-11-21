"""Render all frozen remaining representatives. No image warp or new observation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,html,collections
from PIL import Image
import numpy as np
import match_video10_temporal_bridge123 as bridge
R=bridge.R;U=R/'video10-remaining-material-review';N=R/'video10-semantic-role'
read=bridge.read;save=bridge.save;sha=bridge.sha
inv=read(U/'inventory.json');plan=read(U/'plan.json');stamp=bridge.now()
assert sha(U/'inventory.json')==plan['inventory_sha256'] and len(inv['rows'])==222
for p,h in plan['source_sha256'].items():assert sha(Path(p))==h,p
assert not (U/'review').exists()
out=U/'review';out.mkdir()
frames=sorted({o['frame'] for r in inv['rows'] for o in r['observations']})
paths={f:bridge.imgpath(f) for f in frames};images={f:Image.open(p).convert('RGB') for f,p in paths.items()}
assert all(im.size==(2704,1520) for im in images.values())
save(U/'execution-admission.json',dict(updated=stamp,plan_sha256=sha(U/'plan.json'),inventory_sha256=sha(U/'inventory.json'),script_sha256=sha(Path(__file__)),
    native_images_sha256={str(p):sha(p) for p in paths.values()},operation='Readonly crop visualization of all696 pre-existing observations in222 frozen families',
    native_crop_px=128,columns=5,rows_per_page=12,burned_text=False,point_center=[64,64],display_rounding_only=True,new_inference=0,new_camera=0,training=False))
rows=[];pages=[];page=None;page_rows=[]
for i,r in enumerate(inv['rows']):
    assert i==r['row'] and len(r['observations'])<=5
    if i%12==0:page=Image.new('RGB',(640,12*132),(235,235,235));page_rows=[]
    strip=Image.new('RGB',(640,128),(235,235,235))
    for j,o in enumerate(r['observations']):
        x,y=np.rint(o['uv']).astype(int);strip.paste(images[o['frame']].crop((x-64,y-64,x+64,y+64)),(128*j,0))
    filename=f'row_{i:03d}_family_{r["family"]}.png';strip.save(out/filename)
    rows.append(dict(**r,file=filename));page.paste(strip,(0,(i%12)*132));page_rows.append(i)
    if i%12==11 or i==len(inv['rows'])-1:
        name=f'page_{len(pages):02d}.png';page.crop((0,0,640,(i%12)*132+128)).save(out/name)
        pages.append(dict(file=name,rows=page_rows.copy(),sha256=sha(out/name)))
save(out/'index.json',dict(updated=stamp,rows=rows,pages=pages,all_observations=True,native_crop_px=128,burned_text=False,display_rounding_only=True))
sections=[]
for p in pages:
    table='<table><tr><th>行</th><th>原族/划分</th><th>原帧顺序</th></tr>'+''.join(f'<tr><td>{i}</td><td>{rows[i]["family"]}/{rows[i]["split"]}</td><td>{rows[i]["frames"]}</td></tr>' for i in p['rows'])+'</table>'
    sections.append(f'<section><h2>行{p["rows"][0]}—{p["rows"][-1]}</h2><img src="{p["file"]}">'+table+'</section>')
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>视频10剩余材料核验</title><style>body{font:17px sans-serif;max-width:900px;margin:32px auto}img{max-width:100%}td,th{padding:6px;border:1px solid #aaa}section{margin:30px 0}</style><h1>222个原背景候选族</h1><p>696个原生观测；每格128像素。每行从左到右按表中原帧顺序，中心为取整后的显示查询位置。原浮点观测、划分和原图未改。未审阅前不能视为有效固定材料；粗图像身份也不等于亚像素真值。</p>'+''.join(sections),encoding='utf-8')
save(U/'progress.json',dict(updated=stamp,state='native_review_prepared_not_reviewed',pid=None,rows=222,observations=696,pages=len(pages),new_inference=0,training=False))
print(json.dumps(dict(rows=len(rows),pages=len(pages),observations=696,physical_review_complete=False)))
