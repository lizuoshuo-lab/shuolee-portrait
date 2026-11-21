"""CPU rasterization of the actual coarse mesh in two training cameras."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import trimesh
from shuolee_audit_diffuman_cameras import OUT
from shuolee_prepare_elevation_benchmark import dump,digest


def raster(mesh,K,V,width,height):
    vertices=np.asarray(mesh.vertices);colors=np.asarray(mesh.visual.vertex_colors)[:,:3].astype('float32')
    p=vertices@V[:3,:3].T+V[:3,3]
    xy=p[:,:2]/p[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    depth=np.full((height,width),np.inf,dtype='float32');rgb=np.zeros((height,width,3),dtype='float32')
    for triangle in np.asarray(mesh.faces):
        q=xy[triangle];z=p[triangle,2]
        if z.min()<=.01:continue
        x0=max(0,int(np.floor(q[:,0].min())));x1=min(width-1,int(np.ceil(q[:,0].max())))
        y0=max(0,int(np.floor(q[:,1].min())));y1=min(height-1,int(np.ceil(q[:,1].max())))
        if x1<x0 or y1<y0:continue
        den=(q[1,1]-q[2,1])*(q[0,0]-q[2,0])+(q[2,0]-q[1,0])*(q[0,1]-q[2,1])
        if abs(den)<1e-8:continue
        xx,yy=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1))
        u=((q[1,1]-q[2,1])*(xx-q[2,0])+(q[2,0]-q[1,0])*(yy-q[2,1]))/den
        v=((q[2,1]-q[0,1])*(xx-q[2,0])+(q[0,0]-q[2,0])*(yy-q[2,1]))/den
        w=1-u-v;inside=(u>=-1e-6)&(v>=-1e-6)&(w>=-1e-6)
        iz=u/z[0]+v/z[1]+w/z[2];zz=1/np.maximum(iz,1e-10)
        old=depth[y0:y1+1,x0:x1+1];accept=inside&(zz<old)
        if not accept.any():continue
        mixed=(u[:,:,None]/z[0]*colors[triangle[0]]+v[:,:,None]/z[1]*colors[triangle[1]]+w[:,:,None]/z[2]*colors[triangle[2]])*zz[:,:,None]
        old[accept]=zz[accept];rgb[y0:y1+1,x0:x1+1][accept]=mixed[accept]
    return np.uint8(np.clip(np.round(rgb),0,255)),depth


def main():
    root=OUT/'elevation-benchmark/training-geometry';dest=root/'visual-hull'
    assert not (dest/'train-view-check.json').exists()
    state=json.loads((root/'manifest.json').read_text(encoding='utf-8'));meta={r['camera']:r for r in state['rows']}
    mesh=trimesh.load(dest/'surface.ply',process=False)
    before=digest(dest/'surface.ply');canvas=Image.new('RGB',(1536,1104),'white');rows=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);start=time.time()
    for row,name in enumerate(['02','26']):
        c=meta[name];K=np.asarray(c['K']).copy();K[:2]*=.5;V=np.asarray(c['w2c'])
        rgb,depth=raster(mesh,K,V,512,512)
        original=np.array(Image.open(root/'images'/(name+'.png')).convert('RGB').resize((512,512),Image.Resampling.LANCZOS))
        target=np.array(Image.open(root/'silhouettes'/(name+'.png')).convert('L').resize((512,512),Image.Resampling.NEAREST))>127
        valid=np.array(Image.open(root/'source-validity'/(name+'.png')).convert('L').resize((512,512),Image.Resampling.NEAREST))>127
        mask=np.isfinite(depth);intersection=(target&mask&valid).sum();union=((target|mask)&valid).sum()
        # Actual mismatch map: dark=background, white=agreement, red/blue=two error types.
        overlap=np.zeros((512,512,3),np.uint8)
        overlap[target&mask]=[210,220,220];overlap[target&~mask]=[230,90,55];overlap[~target&mask]=[40,120,220]
        for col,(label,a) in enumerate([('Training photograph',original),('Coarse silhouette surface',rgb),('Silhouette agreement',overlap)]):
            canvas.paste(Image.fromarray(a),(512*col,552*row+40))
            ImageDraw.Draw(canvas).text((512*col+12,552*row+8),f'{label} / camera {name}',font=font,fill='#18333f')
        rows.append(dict(camera=name,training_mask_iou=float(intersection/max(union,1)),
            missing_foreground_fraction=float((target&~mask&valid).sum()/max((target&valid).sum(),1)),
            excess_foreground_fraction=float((~target&mask&valid).sum()/max((mask&valid).sum(),1))))
        print(rows[-1],flush=True)
    canvas.save(dest/'train-view-check.png')
    assert digest(dest/'surface.ply')==before
    result=dict(status='rendered_pending_visual_inspection',rows=rows,elapsed_seconds=time.time()-start,
        cameras=['02','26'],source='Actual mesh perspective rasterization with training-photo vertex colors, not generated imagery.',
        interpretation='Self consistency against training silhouettes used to build the hull. Not held-out quality or independent geometry.',
        test_or_development_images_used=False,mesh_unchanged=True)
    dump(dest/'train-view-check.json',result)


if __name__=='__main__':main()
