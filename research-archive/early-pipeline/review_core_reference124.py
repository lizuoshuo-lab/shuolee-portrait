"""Read-only native-pixel review of the registered original reference fits."""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import mesh_renderer, joined, crop_box


def main(subject):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import cv2
    cv2.setNumThreads(1)
    root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')/(subject+'-input')
    selected=read(root/'selection.json')['reference_frame']
    row=next(r for r in read(root/'data/inputs.json')['records'] if r['frame']==selected)
    result=read(root/'reference-fit/result.json')
    assert result['state']=='complete' and result['frame_id']==selected
    out=root/'reference-review';out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists()
    with np.load(root/'reference-fit/winner-camera-space.npz',allow_pickle=False) as z:f=dict(z)
    assert str(f['frame_id'].item())==selected
    assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
    raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
    box=crop_box(mask.getbbox(),raw.size,30);face=crop_box(row['face_box_xyxy'],raw.size,35)
    feet=crop_box([box[0],box[1]+.72*(box[3]-box[1]),box[2],box[3]],raw.size)
    record=Path('/mnt/d/00/记录')/('006' if subject=='sofa' else '008')
    render,_=mesh_renderer();images={}
    for region,bb in [('body',box),('face',face),('feet',feet)]:
        original=raw.crop(bb)
        overlay=render(original,np.array(bb),f)
        path=record/f'smpl-x_原版参考124_{selected}_{region}_原图_拟合投影.png'
        assert not path.exists();joined([original,overlay]).save(path);images[region]=str(path)
    solid=render(Image.new('RGB',(box[2]-box[0],box[3]-box[1]),'black'),np.array(box),f)
    path=record/f'smpl-x_原版参考124_{selected}_原图_同相机实体.png'
    assert not path.exists();joined([raw.crop(box),solid]).save(path);images['solid']=str(path)
    points=[];observed=f['observed_native_keypoints'][0];projected=f['joints_projected_native'][0]
    for xy,color in [(observed[:,:2],(30,235,70)),(projected,(240,65,220))]:
        pic=raw.crop(box);draw=ImageDraw.Draw(pic)
        for k,p in enumerate(xy):
            if observed[k,2]<=0 or k in (1,9,12):continue
            x,y=p-np.array(box[:2]);draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        points.append(pic)
    path=record/f'特征点_原版参考124_{selected}_原图_二维观测_拟合三维关节投影.png'
    assert not path.exists();joined([raw.crop(box),*points]).save(path);images['keypoints']=str(path)
    v=f['vertices_camera'][0];center=(v.min(0)+v.max(0))/2;extent=float(np.ptp(v,axis=0).max());sides=[]
    for deg in (90,-90,180):
        a=np.deg2rad(deg);rot=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        vv=(v-center)@rot.T;vv[:,2]+=3*extent
        sides.append(render(Image.new('RGB',(340,480),'black'),np.array([0,0,340,480]),dict(vertices_camera=vv[None],faces=f['faces'],K_integer_centers=np.array([[1150,0,170],[0,1150,240],[0,0,1]]))))
    path=record/f'分析_原版参考124_{selected}_同相机实体_两侧与背面.png'
    original=Image.open(images['body']);original.thumbnail((500,480));solid.thumbnail((340,480))
    page=Image.new('RGB',(1860,480),'black');page.paste(original,(0,0));page.paste(solid,(500,0))
    for i,pic in enumerate(sides):page.paste(pic,(840+340*i,0))
    assert not path.exists();page.save(path);images['side_review']=str(path)
    err=np.linalg.norm(projected-observed[:,:2],axis=-1)
    save(out/'result.json',dict(updated=now(),subject=subject,reference_frame=selected,images=images,
        original_fit_result=str(root/'reference-fit/result.json'),source_fit_sha256=sha(root/'reference-fit/winner-camera-space.npz'),
        error_px=err.tolist(),confidence=observed[:,2].tolist(),disabled_original_joints=[1,9,12],
        synthetic_views_not_ground_truth=True,keypoints3d_are_fitted_not_independent=True,
        all_reference_native_images_generated=True,actual_visual_review=False,quality_accepted=False,
        full_sequence_admitted=False,gaussian_admitted=False))
    print('CORE_REFERENCE_REVIEW_READY',subject,selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('subject',choices=['sofa','indoor']);a=p.parse_args();main(a.subject)
