"""Render every completed head5 contrast in the original image and common views."""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import control_head_absence124 as c
from review_full_sequence123 import mesh_renderer,crop_box

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (c.OUT/'visual-review.json').exists()
p=c.verify();result=c.read(c.OUT/'result.json')
assert result['state']=='complete_pending_visual_review'
rows={r['frame']:r for r in c.read(c.base.SRC/'data/inputs.json')['records']}
render,_=mesh_renderer();record=Path('/mnt/d/00/记录/008');evidence={}

def join(images):
    w,h=images[0].size;out=Image.new('RGB',(w*len(images),h))
    for i,im in enumerate(images):assert im.size==(w,h);out.paste(im,(i*w,0))
    return out
def saved(frame,label,im,**metadata):
    path=record/f'smpl-x_124_头部完整缺席对照_{frame}_{label}.png'
    assert not path.exists();im.save(path)
    return dict(path=str(path),sha256=c.sha(path),**metadata)

for name in result['completed']:
    row=rows[name];raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
    arms=['original','face_absence','head_absence'];meshes=[]
    for arm in arms:
        root=c.OUT/name if arm=='head_absence' else c.base.OUT/arm/name
        with np.load(root/'winner-camera-space.npz',allow_pickle=False) as z:meshes.append(dict(z))
    target=meshes[0]['original_native_observations'][0]
    person=np.asarray(row['initial_box_xyxy'],float);head=person.copy();head[3]=head[1]+.3*(head[3]-head[1])
    boxes=dict(body=crop_box(mask.getbbox(),raw.size,30),head=crop_box(head,raw.size,20))
    hands=target[25:67];hands=hands[hands[:,2]>0,:2]
    if len(hands):boxes['arms']=crop_box(np.r_[hands.min(0),hands.max(0)],raw.size,100)
    images={}
    for label,box in boxes.items():
        crop=raw.crop(box);panels=[crop]+[render(crop,np.asarray(box,int),a) for a in meshes]
        images[label]=saved(name,label+'_原图_原输入_内脸停用_头部停用_原生像素',join(panels),
            columns=['raw']+arms,native_crop=list(map(int,box)),resized=False)
    vv=np.concatenate([a['vertices_camera'][0] for a in meshes]);center=(vv.min(0)+vv.max(0))/2;extent=float(np.max(np.ptp(vv,axis=0)))
    K=np.array([[1150.,0,170],[0,1150.,240],[0,0,1.]])
    canvas=Image.new('RGB',(340*3,480*4))
    for j,degree in enumerate([0,90,180,-90]):
        angle=np.deg2rad(degree);co,si=np.cos(angle),np.sin(angle);R=np.array([[co,0,si],[0,1,0],[-si,0,co]])
        for k,a in enumerate(meshes):
            v=(a['vertices_camera'][0]-center)@R.T;v[:,2]+=3*extent
            im=render(Image.new('RGB',(340,480)),np.array([0,0,340,480]),dict(vertices_camera=v[None],faces=a['faces'],K_integer_centers=K))
            canvas.paste(im,(340*k,480*j))
    images['sides']=saved(name,'同相机四方向_原输入_内脸停用_头部停用',canvas,
        camera_center=center.tolist(),extent=extent,rows_degrees=[0,90,180,-90],columns=arms,synthetic_not_observed=True)
    evidence[name]=dict(images=images,arms=arms)
c.save(c.OUT/'visual-review.json',dict(updated=c.now(),state='generated_pending_actual_review',frames=evidence,
    actual_visual_review=False,raw_pixels_unchanged=True,text_on_images=False,
    renderer='Existing CPU mesh preview; synthetic side views have no physical GT.',quality_accepted=False,default_applied=False))
print('HEAD_ABSENCE_REVIEW_READY',len(evidence),flush=True)
