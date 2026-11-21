"""Existing numerical survivors only, same cached fit, overview evidence only."""
from PIL import Image
import audit_gaussianavatars_params123 as a
from prepare_quality_eight124 import OUT,RUN,local

d=a.read(OUT/'grass-shortlist.json')
pool=sorted([r['frame'] for r in d['records'] if r['eligible']])
ev={r['frame']:r for r in a.read(RUN/'review-094/result.json')['records']}
manifest=[];tiles=[]
for f in pool:
    r=ev[f]
    for k in ['body','sides']: assert a.sha(local(r['images'][k]['path']))==r['images'][k]['sha256']
    native=Image.open(local(r['images']['body']['path'])).convert('RGB'); w=native.width//3
    pair=Image.new('RGB',(w*2,native.height))
    pair.paste(native.crop((0,0,w,native.height)),(0,0))
    pair.paste(native.crop((2*w,0,3*w,native.height)),(w,0))
    pair.thumbnail((620,480),Image.Resampling.LANCZOS)
    views=Image.open(local(r['images']['sides']['path'])).convert('RGB')
    tile=Image.new('RGB',(1300,480));tile.paste(pair,(0,0))
    tile.paste(views.crop((340,480,680,960)),(620,0))
    tile.paste(views.crop((340,1440,680,1920)),(960,0))
    tiles.append((f,tile))
for start in range(0,len(tiles),4):
    chunk=tiles[start:start+4];page=Image.new('RGB',(1300,len(chunk)*480))
    for j,(_,im) in enumerate(chunk):page.paste(im,(0,480*j))
    p=a.lp('D:/00/记录/005')/f'分析_质量候选124_原图_同帧拟合_双侧面_页{start//4+1:02d}.png'
    assert not p.exists();page.save(p)
    manifest.append(dict(frames=[f for f,_ in chunk],path=str(p).removeprefix('\\\\?\\'),
        sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest(),overview_only=True,not_native_resolution=True))
a.save(OUT/'grass-pool-review-pages.json',manifest)
print(dict(pool=len(pool),pages=len(manifest),frames=pool))
