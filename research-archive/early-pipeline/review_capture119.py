"""Same-camera inspection sheets; no labels burned into pixels, no warping."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from plyfile import PlyData
from capture119_clothed import ROOT,BASE,REC,rt

def point_image(ply,row,size):
    v=PlyData.read(ply)['vertex'].data
    xyz=np.column_stack([v[x] for x in ['x','y','z']]).astype(float)
    rgb=np.clip(.5+.28209479177387814*np.column_stack([v[f'f_dc_{j}'] for j in range(3)]),0,1)
    K=np.diag([size[0]/row['wh'][0],size[1]/row['wh'][1],1])@np.array(row['K'])
    E=np.array(row['w2c']);p=xyz@E[:3,:3].T+E[:3,3];q=p@K.T
    uv=np.rint(q[:,:2]/np.maximum(q[:,2:],1e-9)-.5).astype(int)
    valid=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<size[0])&(uv[:,1]>=0)&(uv[:,1]<size[1])
    ids=np.flatnonzero(valid);ids=ids[np.argsort(p[ids,2])[::-1]]
    pixels=np.zeros((size[1],size[0],3),np.uint8)
    pixels[uv[ids,1],uv[ids,0]]=np.rint(rgb[ids]*255).astype(np.uint8)
    return Image.fromarray(pixels)

def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--phase',choices=['fit','surface','train','final'],required=True)
    p.add_argument('--step',type=int,default=1000);a=p.parse_args();root=ROOT/a.subject
    rows=rt.read(root/'data/training-inputs.json')['records'];chosen=np.unique(np.linspace(0,len(rows)-1,6).astype(int))
    out=REC/f'{a.subject}-{a.phase}-{a.step:06d}-review';out.mkdir(exist_ok=True)
    index=[]
    if a.phase=='fit':source=root/'reviews/fit'
    elif a.phase=='surface':source=root/'clothing-init'
    elif a.phase=='final':source=root/'export/review'
    else:source=root/'gaussian-run'/f'step-{a.step:06d}'
    for i in chosen:
        r=rows[i];name=r['frame']
        raw=Image.open(source/f'{name}_input.png').convert('RGB');size=raw.size
        if a.phase=='fit':
            ims=[raw,Image.open(source/f'{name}_mesh_overlay.png').convert('RGB')];columns=['raw','fresh_mesh_overlay']
        elif a.phase=='surface':
            ims=[raw,Image.open(source/f'{name}_initialization.png').convert('RGB'),point_image(source/'initialization.ply',r,size)]
            columns=['raw','untrained_clothed_envelope_gaussian','untrained_centers_one_pixel']
        else:
            snap=root/'gaussian-run'/f'step-{a.step:06d}'
            new=Image.open(source/(name+('_native.png' if a.phase=='final' else '_render.png'))).convert('RGB')
            old=BASE/a.subject/'gaussian-run'/f'step-{a.step:06d}'/f'{name}_render.png'
            if not old.exists():old=BASE/a.subject/'gaussian-run/step-001000'/f'{name}_render.png'
            ims=[raw,Image.open(old).convert('RGB').resize(size,Image.Resampling.LANCZOS),new,point_image(snap/'world.ply',r,size)]
            columns=['raw','mesh_bound_baseline_'+old.parent.name,'new_clothed_world','new_centers_one_pixel']
        for region,b in [('body',r['initial_box_xyxy']),('face',r['face_box_xyxy'])]:
            if b is None:continue
            box=np.array(b,dtype=float)*np.array([size[0]/r['wh'][0],size[1]/r['wh'][1]]*2)
            margin=max(6,(box[3]-box[1])*.1);box+=np.array([-margin,-margin,margin,margin]);box=np.rint(np.clip(box,[0,0,0,0],[*size,*size])).astype(int)
            target_h=480 if region=='body' else 260;scale=target_h/(box[3]-box[1]);tile_size=(round((box[2]-box[0])*scale),target_h)
            tiles=[im.crop(tuple(box)).resize(tile_size,Image.Resampling.LANCZOS) for im in ims]
            panel=Image.new('RGB',(tile_size[0]*len(tiles),target_h),'black')
            for k,im in enumerate(tiles):panel.paste(im,(k*tile_size[0],0))
            file=f'{name}_{region}_comparison.png';panel.save(out/file)
            index.append(dict(file=file,frame=name,region=region,columns=columns,same_crop=True,crop_xyxy=box.tolist(),aspect_preserved=True))
    for region in ['body','face']:
        ims=[Image.open(out/x['file']).convert('RGB') for x in index if x['region']==region]
        width=1000;ims=[im.resize((width,round(im.height*width/im.width))) for im in ims]
        sheet=Image.new('RGB',(width,sum(im.height for im in ims)),'white');y=0
        for im in ims:sheet.paste(im,(0,y));y+=im.height
        sheet.save(out/(region+'-contact.png'))
    rt.save(out/'index.json',dict(created=rt.now(),subject=a.subject,phase=a.phase,step=a.step,images=index,actual_review_pending=True))
    print(out,flush=True)

if __name__=='__main__':main()
