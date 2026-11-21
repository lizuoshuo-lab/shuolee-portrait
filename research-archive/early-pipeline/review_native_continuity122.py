"""Prepare deterministic unannotated native crops for the single completed T6 run."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image

R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-continuity';S=R/'native-supervision';DATA=Path('D:/GS-SLAM-Data/robust122-whiteT-native-supervision')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

r=read(O/'result.json');assert r['state']=='complete_pending_material_review'
out=O/'review-inputs';assert not (out/'index.json').exists()
prior_images={p.name:sha(p) for p in out.glob('*.png')} if out.exists() else {}
out.mkdir(exist_ok=True)
st=read(O/'seeds.json');a=np.load(O/'tracks.npz');uv=a['uv'];state=a['state'];reason=a['rejection_bits']
cache={}
def img(f):
    if f not in cache:
        # Keep the cache bounded; each saved crop is at native pixel scale.
        if len(cache)>=10:cache.pop(next(iter(cache)))
        cache[f]=Image.open(DATA/'rgb'/f'frame_{f:06d}.png').convert('RGB')
    return cache[f]
def crop(f,xy,size=128):
    x,y=np.rint(xy).astype(int);box=(x-size//2,y-size//2,x+size//2,y+size//2)
    return img(f).crop(box),list(map(int,box))

rows=[];pages=[];ctx=[]
for j,t in enumerate(st):
    valid=np.flatnonzero(state[:,j]>0);bad=np.flatnonzero(state[:,j]==0)
    selected=[int(valid[0]),int(valid[len(valid)//4]),int(t['seed']['frame']),int(valid[3*len(valid)//4]),int(valid[-1])]
    selected+=list(map(int,bad))
    fs=sorted(set(selected));assert len(fs)<=7
    canvas=Image.new('RGB',(7*128,128),(240,240,240));cells=[]
    for c,f in enumerate(fs):
        if np.isfinite(uv[f,j]).all():
            tile,box=crop(f,uv[f,j]);canvas.paste(tile,(128*c,0))
            cells.append(dict(column=c,frame=f,state=int(state[f,j]),rejection_bits=int(reason[f,j]),uv=uv[f,j].tolist(),box=box))
        else:cells.append(dict(column=c,frame=f,state=int(state[f,j]),rejection_bits=int(reason[f,j]),uv=None,box=None))
    name=f'track_{j:03d}_family_{t["family"]}.png';canvas.save(out/name)
    rows.append(dict(index=j,family=t['family'],track=t['track'],group=t['group'],split=t['split'],material=t['material'],
        seed=t['seed'],valid_count=len(valid),first_valid=int(valid[0]),last_valid=int(valid[-1]),file=name,cells=cells))
    if t['material']=='face_surface_coarse':
        original={x['frame']:x for x in t['observations']}
        # All existing face anchor views, in temporal order. Flow centered and
        # detector centered crops are paired; neither coordinate alters the other.
        c=Image.new('RGB',(256*len(original),128),(240,240,240));cc=[]
        for k,(f,ob) in enumerate(sorted(original.items())):
            left,b1=crop(f,ob['uv']);c.paste(left,(k*256,0));b2=None
            if state[f,j]>0:
                right,b2=crop(f,uv[f,j]);c.paste(right,(k*256+128,0))
            cc.append(dict(frame=f,raw_uv=ob['uv'],flow_uv=uv[f,j].tolist() if state[f,j]>0 else None,
                raw_box=b1,flow_box=b2,numerical_flow_valid=bool(state[f,j]>0)))
        fn=f'face_anchors_family_{t["family"]}.png';c.save(out/fn);ctx.append(dict(family=t['family'],file=fn,cells=cc))
for start in range(0,len(rows),7):
    c=Image.new('RGB',(896,7*128),(240,240,240))
    for i,row in enumerate(rows[start:start+7]):c.paste(Image.open(out/row['file']),(0,i*128))
    n=f'page_{start:03d}_{min(start+6,len(rows)-1):03d}.png';c.save(out/n);pages.append(dict(file=n,rows=list(range(start,min(start+7,len(rows))))))
comparisons=read(O/'anchor-comparison.json')
badanchors=sorted([x for x in comparisons if not x['seed'] and x['disagreement_px'] is not None],key=lambda x:-x['disagreement_px'])[:8]
details=[]
for n,a in enumerate(badanchors):
    j=a['track_index'];t=st[j];f=a['frame'];s=t['seed'];c=Image.new('RGB',(768,256),(240,240,240))
    for k,(ff,xy) in enumerate([(s['frame'],s['uv']),(f,a['raw_sift_uv']),(f,a['flow_uv'])]):
        tile,box=crop(ff,xy,256);c.paste(tile,(k*256,0))
    fn=f'disagreement_{n:02d}_family_{t["family"]}_frame_{f:06d}.png';c.save(out/fn)
    details.append(dict(**a,file=fn,layout='seed native crop, comparison native SIFT crop, propagated native crop;256px each'))
assert all(sha(out/name)==h for name,h in prior_images.items()),'Prior review image changed during index recovery'
save(out/'index.json',dict(rows=rows,pages=pages,face_anchors=ctx,largest_disagreement_details=details,
    recovery_existing_images_unchanged=len(prior_images),
    annotations_in_images=False,geometric_image_warps=False,selected_for_visual_review_not_fitting=True,
    selection='Every family: chronological first/quarter/seed/three-quarter/last numeric-valid and both firstfailed attempts. All6face family raw anchors separately paired with flow. Largest8numeric disagreements include failures for inspection, never selection/removal from fitting.',
    underlying_arrays_sha256=sha(O/'tracks.npz')))
print(json.dumps(dict(rows=len(rows),pages=len(pages),face_anchor_strips=len(ctx),largest_disagreements=details),ensure_ascii=False))
