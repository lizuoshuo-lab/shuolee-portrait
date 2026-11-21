"""Untouched-pixel observation review; no rendered or fitted geometry implied."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from preprocess_demo_fps3_123 import ROOT,DATA,REPORT
from run_official_gaussians123 import read,save,now,sha


def join(images):
    out=Image.new('RGB',(sum(i.width for i in images),max(i.height for i in images)),(255,255,255))
    x=0
    for im in images:out.paste(im,(x,0));x+=im.width
    return out


def bounded(box,size,pad):
    return tuple(int(x) for x in [max(0,np.floor(box[0]-pad)),max(0,np.floor(box[1]-pad)),
                                min(size[0],np.ceil(box[2]+pad)),min(size[1],np.ceil(box[3]+pad))])


def main():
    result=read(REPORT/'result.json');assert result['state']=='complete_pending_visual_review'
    out=REPORT/'review';out.mkdir(exist_ok=False)
    record=Path('/mnt/d/00/记录/demo');record.mkdir(exist_ok=True)
    rows=read(DATA/'data/inputs.json')['records']
    with np.load(DATA/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    slots=np.linspace(0,len(rows)-1,6).round().astype(int).tolist();images=[]
    contacts={k:[] for k in ('mask','body','face','hands')}
    for i in slots:
        r=rows[i];raw=Image.open(r['image']).convert('RGB');mask=Image.open(r['mask']).convert('L')
        foreground=Image.composite(raw,Image.new('RGB',raw.size,'white'),mask)
        box=bounded(obs['boxes'][i],raw.size,16)
        panels=join([raw.crop(box),mask.convert('RGB').crop(box),foreground.crop(box)])
        name=f'20261030_3fps_源帧{r["source_frame_index"]:06d}_mask_左原图_中遮罩_右前景.png'
        panels.save(record/name);panels.save(out/f'{i:03d}-mask.png');contacts['mask'].append(panels)
        marked=raw.copy();draw=ImageDraw.Draw(marked);kp=obs['keypoints'][i]
        for start,end,color in [(0,23,'red'),(23,91,'cyan'),(91,112,'yellow'),(112,133,'magenta')]:
            for x,y in kp[start:end]: draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        regions=dict(body=box,face=bounded(np.r_[kp[23:91].min(0),kp[23:91].max(0)],raw.size,24),
            hands=bounded(np.r_[kp[91:133].min(0),kp[91:133].max(0)],raw.size,24))
        for region,crop in regions.items():
            panel=join([raw.crop(crop),marked.crop(crop)]);p=out/f'{i:03d}-{region}.png';panel.save(p)
            panel.save(record/f'20261030_3fps_源帧{r["source_frame_index"]:06d}_{region}_左原图_右2D观测_尚未3D拟合.png')
            contacts[region].append(panel);images.append(dict(slot=i,source_index=r['source_frame_index'],region=region,path=str(p),crop=crop))
    for region,panels in contacts.items():
        thumbs=[]
        for panel in panels:
            thumb=panel.copy();thumb.thumbnail((600,430));thumbs.append(thumb)
        canvas=Image.new('RGB',(1200,430*3),'white')
        for i,im in enumerate(thumbs):canvas.paste(im,((i%2)*600,(i//2)*430))
        canvas.save(out/f'contact-{region}.jpg',quality=95)
    save(out/'manifest.json',dict(updated=now(),state='saved_pending_actual_review',images=images,slots=slots,
        record_path=str(record),image_text=False,body_face_hands_observed=True,physical_visibility_established=False,
        fitted_mesh_available=False,quality_accepted=False,observations_sha256=sha(DATA/'data/observations.npz')))
    print(str(out))


if __name__=='__main__':main()
