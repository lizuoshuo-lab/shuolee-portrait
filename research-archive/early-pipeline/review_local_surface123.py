"""Unwarped, unlabelled same-camera review images; never auto-approves."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
from plyfile import PlyData

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nativepath(p):
    if p.startswith('/opt/'):return Path('//wsl.localhost/Ubuntu-20.04'+p)
    if p.startswith('/mnt/'):return Path(p[5].upper()+':/'+p[7:])
    return Path(p)

def points(ply,row,size):
    v=PlyData.read(ply)['vertex'].data
    xyz=np.column_stack([v[x] for x in ['x','y','z']]).astype(float)
    color=np.rint(np.clip(.5+.28209479177387814*np.column_stack([v[f'f_dc_{i}'] for i in range(3)]),0,1)*255).astype('uint8')
    E=np.array(row['w2c']);K=np.diag([size[0]/row['wh'][0],size[1]/row['wh'][1],1])@np.array(row['K'])
    q=xyz@E[:3,:3].T+E[:3,3];imq=q@K.T
    uv=np.rint(imq[:,:2]/np.maximum(imq[:,2:],1e-9)-.5).astype(int)
    good=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<size[0])&(uv[:,1]>=0)&(uv[:,1]<size[1])
    ids=np.flatnonzero(good);ids=ids[np.argsort(q[ids,2])[::-1]]
    image=np.zeros((size[1],size[0],3),np.uint8);image[uv[ids,1],uv[ids,0]]=color[ids]
    return Image.fromarray(image)

def main(subject,phase,step):
    root=ROOT/subject;out=REC/f'{subject}-{phase}-{step:06d}-review';out.mkdir(exist_ok=True)
    source=root/'data/training-inputs.json'
    rows=read(source)['records'] if source.exists() else read(root/'decoded.json')['records']
    ids=np.unique(np.linspace(0,len(rows)-1,6).astype(int));index=[]
    for i in ids:
        r=rows[i];f=r['frame']
        if phase=='input':
            ims=[Image.open(nativepath(r['image']) if 'image' in r else root/'raw'/r['name']).convert('RGB')]
            if r.get('mask'):ims.append(Image.open(nativepath(r['mask'])).convert('RGB'))
            columns=['native_input']+(['person_mask'] if len(ims)>1 else [])
        elif phase=='fit':
            p=root/'reviews/fit';ims=[Image.open(p/f'{f}_input.png').convert('RGB'),Image.open(p/f'{f}_mesh_overlay.png').convert('RGB')]
            columns=['input','fresh_mesh_overlay']
        elif phase=='surface':
            p=root/'clothing-init';raw=Image.open(p/f'{f}_input.png').convert('RGB')
            ims=[raw,Image.open(p/f'{f}_initialization.png').convert('RGB'),points(p/'initialization.ply',r,raw.size)]
            columns=['input','untrained_surface_gaussians','actual_center_points']
        else:
            p=root/'gaussian-run'/f'step-{step:06d}'
            raw=Image.open(p/f'{f}_input.png').convert('RGB')
            ims=[raw,Image.open(p/f'{f}_render.png').convert('RGB'),points(p/'world.ply',r,raw.size)]
            columns=['input',f'free_surface_step_{step}','actual_center_points']
        size=ims[0].size
        regions=[('body',r.get('initial_box_xyxy',[0,0,*r['wh']]))]
        if r.get('face_box_xyxy') is not None:regions.append(('face',r['face_box_xyxy']))
        for region,b in regions:
            box=np.array(b,float)*np.array([size[0]/r['wh'][0],size[1]/r['wh'][1]]*2)
            margin=(box[3]-box[1])*.06
            box=np.rint(np.clip(box+[-margin,-margin,margin,margin],[0,0,0,0],[*size,*size])).astype(int)
            if min(box[2:]-box[:2])<=0:continue
            targeth=480 if region=='body' else 240
            scale=targeth/(box[3]-box[1]);tile=(max(1,round((box[2]-box[0])*scale)),targeth)
            panel=Image.new('RGB',(tile[0]*len(ims),targeth),'black')
            for j,im in enumerate(ims):panel.paste(im.crop(tuple(box)).resize(tile,Image.Resampling.LANCZOS),(j*tile[0],0))
            name=f'{f}_{region}.png';panel.save(out/name)
            index.append(dict(frame=f,source_frame_index=r['source_frame_index'],region=region,file=name,
                columns=columns,same_crop=True,aspect_preserved=True,source_pixel_crop=box.tolist(),sha256=sha(out/name)))
    for region in ['body','face']:
        imgs=[Image.open(out/r['file']) for r in index if r['region']==region]
        if not imgs:continue
        width=900
        imgs=[im.resize((width,max(1,round(im.height*width/im.width))),Image.Resampling.LANCZOS) for im in imgs]
        sheet=Image.new('RGB',(width,sum(im.height for im in imgs)),'white');y=0
        for im in imgs:sheet.paste(im,(0,y));y+=im.height
        sheet.save(out/f'{region}-contact.png')
    (out/'index.json').write_text(json.dumps(dict(subject=subject,phase=phase,step=step,images=index,
        image_labels_burned_in=False,actual_review_pending=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--phase',required=True,choices=['input','fit','surface','train']);p.add_argument('--step',type=int,default=0)
    a=p.parse_args();main(a.subject,a.phase,a.step)
