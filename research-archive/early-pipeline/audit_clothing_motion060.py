"""Read-only image-track check of the body-derived clothing motion field."""
from pathlib import Path
import json,datetime,hashlib,os
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
if os.name!='nt':B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-clothing-pipeline-fresh-060';OUT=R/'motion-audit';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=read(R/'data/inputs.json')['records']
original=read(B/'current/gaussianavatars-smplx-from-scratch-001/data/inputs.json')['records']
z=np.load(R/'clothing-support/surface.npz');motion=np.load(R/'clothing-init/motion.npz')
xyz=z['xyz'];observations=z['observations'];n=len(xyz)
v=motion['vertices_world'];A=motion['vertex_affine'];ref=int(motion['reference_frame']);scale=float(motion['world_scale'])
d,idx=cKDTree(v[ref]).query(xyz,k=4);w=1/(d*d+(.015*scale)**2);w/=w.sum(1,keepdims=True)
linear=(A[:,idx]*w[None,:,:,None,None]).sum(2)
translation=((v[:,idx]-np.einsum('npkij,pkj->npki',A[:,idx],v[ref,idx]))*w[None,:,:,None]).sum(2)
warped=np.einsum('npij,pj->npi',linear,xyz)+translation
assert np.max(np.abs(warped[ref]-xyz))<1e-5
E=np.array([a['w2c'] for a in rows]);K=np.array([a['K'] for a in rows])
def project(points,frames):
    cam=np.einsum('nij,nj->ni',E[frames,:3,:3],points)+E[frames,:3,3]
    q=np.einsum('nij,nj->ni',K[frames],cam)
    return q[:,:2]/q[:,2:]
pid=observations[:,0].astype(int);oldf=observations[:,1].astype(int)
# The preparation manifest records original calibrated rows at raw 1,4,...175.
newf=oldf*3
assert all(rows[i]['camera_kind']=='calibrated' for i in np.unique(newf))
for j in np.unique(oldf):
    assert np.max(np.abs(np.array(rows[j*3]['w2c'])-np.array(original[j]['w2c'])))<1e-8
target=observations[:,2:4];static=project(xyz[pid],newf);driven=project(warped[newf,pid],newf)
es=np.linalg.norm(static-target,axis=-1);ed=np.linalg.norm(driven-target,axis=-1)
def stats(x):return dict(median=float(np.median(x)),p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)))
tracks=[]
for i in range(n):
    used=pid==i
    tracks.append(dict(point_id=int(z['point_ids'][i]),observations=int(used.sum()),static=stats(es[used]),body_motion=stats(ed[used]),
        motion_model_cm=float(np.median(np.linalg.norm(warped[newf[used],i]-xyz[i],axis=-1))/scale*100)))
receipt=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),scope='read_only_motion_field_consistency',
    input_tracks=n,observations=len(pid),static_reprojection_px=stats(es),body_driven_reprojection_px=stats(ed),
    body_worse_by_over5px_fraction=float(((ed-es)>5).mean()),reference_identity_world_error=float(np.max(np.abs(warped[ref]-xyz))),
    training_updates=0,geometry_accepted=False,tracks=tracks,
    limits=['Tracks were preselected for static multiview consistency; comparison is selection biased.',
        'This can test whether the body field contradicts these particular observations, not whether the whole person is stationary.',
        'Shared SfM cameras and used training images are not independent GT.',
        'Correct clothing motion does not uniquely determine hidden anatomical pose.'],
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'clothing-support/surface.npz',R/'clothing-init/motion.npz',R/'data/inputs.json']})
(OUT/'audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
np.savez_compressed(OUT/'residuals.npz',source_indices=oldf,point_indices=pid,observed=target,static=static,body_driven=driven)
for j in [11,23,33,45]:
    row=original[j];src=Image.open(row['image']).convert('RGB');mask=np.array(Image.open(row['mask']))>127
    yy,xx=np.where(mask);box=(max(0,int(xx.min())-50),max(0,int(yy.min())-50),min(src.width,int(xx.max())+51),min(src.height,int(yy.max())+51))
    im=src.crop(box);s=850/im.height;im=im.resize((round(im.width*s),850),Image.Resampling.LANCZOS)
    used=np.flatnonzero(oldf==j);sheet=Image.new('RGB',(im.width*3,880),'white')
    for c,(label,pr) in enumerate([('Observed image features',None),('Static spatial support',static),('Body-derived motion',driven)]):
        tile=im.copy();draw=ImageDraw.Draw(tile)
        for k in used:
            p=(target[k]-np.array(box[:2]))*s
            draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),outline='white',width=1)
            if pr is not None:
                q=(pr[k]-np.array(box[:2]))*s
                draw.line([tuple(p),tuple(q)],fill='#f14133',width=1)
                draw.ellipse((q[0]-2,q[1]-2,q[0]+2,q[1]+2),outline='#00e0ff',width=1)
        sheet.paste(tile,(c*im.width,30));ImageDraw.Draw(sheet).text((c*im.width+6,8),label,fill='black')
    sheet.save(OUT/f'source-{j:06d}.jpg',quality=95)
print(json.dumps({k:v for k,v in receipt.items() if k not in ['tracks','source_hashes']},ensure_ascii=False,indent=2))
