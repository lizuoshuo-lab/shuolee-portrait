"""Read stored reference fits and render unwarped evidence, without refitting."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
import cv2
from run_smplify_reference123 import O,R,read,save,stats,now
rows=read(R/'data/inputs.json')['records'];crops=read(O/'input-crops.json');result=read(O/'result.json')
edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]
coco=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
def points(im,b,xy,indices,color,conf):
    dr=ImageDraw.Draw(im)
    for i in indices:
        if conf[i]>.25:
            p=xy[i]-b[:2];dr.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),fill=color)
def skeleton(im,b,xy,conf,color):
    dr=ImageDraw.Draw(im)
    for a,c in edges:
        i,j=coco[a],coco[c]
        if conf[i]>.25 and conf[j]>.25:dr.line([tuple(xy[i]-b[:2]),tuple(xy[j]-b[:2])],fill=color,width=2)
def mesh(im,b,fit):
    v=fit['vertices_camera'][0];f=fit['faces'];K=fit['K_integer_centers'];q=v@K.T;uv=q[:,:2]/q[:,2:]
    tris=v[f];normal=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=-1)[:,None],1e-10)
    shade=.3+.7*np.abs(normal@np.array([.3,-.2,-.9327379]))
    layer=np.array(im);mask=np.zeros(layer.shape[:2],np.uint8)
    for i in np.argsort(tris[...,2].mean(1))[::-1]:
        if np.any(tris[i,:,2]<=.05):continue
        poly=np.round(uv[f[i]]-b[:2]).astype(np.int32)
        if poly[:,0].max()<0 or poly[:,0].min()>im.width or poly[:,1].max()<0 or poly[:,1].min()>im.height:continue
        col=tuple(int(z*shade[i]) for z in [80,210,220]);cv2.fillConvexPoly(layer,poly,col,lineType=cv2.LINE_AA);cv2.fillConvexPoly(mask,poly,255)
    raw=np.array(im);out=raw.copy();out[mask>0]=(.6*layer[mask>0]+.4*raw[mask>0]).astype(np.uint8)
    return Image.fromarray(out)
files=[]
for cr in crops:
    slot=cr['slot'];im=Image.open(rows[slot]['image']).convert('RGB');fit=dict(np.load(O/f'slot{slot:03d}-fit.npz'));conf=fit['confidence'][0]
    for name,rec in cr['regions'].items():
        b=np.array(rec['crop_integer']);raw=im.crop(tuple(b));ini=raw.copy();end=raw.copy()
        ids=np.arange(118) if name=='body' else np.arange(67,118) if name=='face' else np.arange(25,46) if name=='left_hand' else np.arange(46,67)
        for image,key in [(ini,'initial_projected'),(end,'projected')]:
            points(image,b,fit['targets'][0],ids,(0,255,80),conf);points(image,b,fit[key][0],ids,(255,0,220),conf)
            if name=='body':skeleton(image,b,fit['targets'][0],conf,(0,220,70));skeleton(image,b,fit[key][0],conf,(255,0,220))
        surface=mesh(raw,b,fit)
        scale=rec['uniform_scale'];wh=(round(raw.width*scale),round(raw.height*scale))
        panels=[p.resize(wh,Image.Resampling.NEAREST if scale>1 else Image.Resampling.LANCZOS) for p in [raw,ini,end,surface]]
        sheet=Image.new('RGB',(wh[0]*4,wh[1]));
        for j,p in enumerate(panels):sheet.paste(p,(j*wh[0],0))
        path=O/f'result_slot{slot:03d}_{name}_raw_initial_final_mesh.png';sheet.save(path);files.append(str(path))
save(O/'rendered-evidence.json',{'created':now(),'files':files,'same_integer_crop_all_panels':True,'no_warp':True,'no_text_on_images':True,'mesh_preview':'CPU depth-sorted triangle overlay for inspection, not a Gaussian rendering or an exact rasterizer acceptance test','panels':['raw','initial VPoser reconstruction projection','final direct SMPL-X projection','final mesh overlay'],'observations_green':True,'projection_magenta':True})
print(json.dumps({'images':len(files),'files':files}))
