"""Native image and four-direction review, no optimization or pose selection."""
from pathlib import Path
import os,sys
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from review_full_sequence123 import mesh_renderer,crop_box,joined
from run_official_gaussians123 import read,save,sha,now
import run_sofa_half8_fit124 as fit

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=fit.OUT;assert read(out/'process-exit.json')['exit_code']==0
    assert not (out/'review/result.json').exists()
    (out/'review').mkdir(exist_ok=True)
    p=read(out/'plan.json');r=read(out/'result.json')
    assert sha(out/'selected-fit.npz')==r['fit_sha256']
    renderer,source=mesh_renderer();records=[];tiles=[]
    for row in p['rows']:
        name=row['frame'];path=out/'frames'/name/'winner-camera-space.npz'
        with np.load(path,allow_pickle=False) as z:a=dict(z)
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB');m=Image.open(row['mask']).convert('L')
        bb=crop_box(m.getbbox(),raw.size,30)
        h=a['observed_native_keypoints'][0,25:67];h=h[h[:,2]>0,:2]
        boxes=dict(body=bb,face=crop_box(row['face_box_xyxy'],raw.size,35),
            hands=crop_box(np.r_[h.min(0),h.max(0)],raw.size,55),
            feet=crop_box([bb[0],bb[1]+.72*(bb[3]-bb[1]),bb[2],bb[3]],raw.size))
        images={}
        for region,b in boxes.items():
            im=raw.crop(b);panel=joined([im,renderer(im,np.array(b),a)])
            dest=Path('/mnt/d/00/记录/006')/f'smpl-x_沙发半圈8帧124_{name}_{region}_原图_当前mesh.png'
            assert not dest.exists();panel.save(dest)
            images[region]=dict(path=str(dest),sha256=sha(dest),native_crop=b,resized=False)
        v=a['vertices_camera'][0];center=(v.min(0)+v.max(0))/2;extent=float(np.ptp(v,axis=0).max())
        K=np.array([[1150.,0,170],[0,1150.,240],[0,0,1.]])
        sides=Image.new('RGB',(1360,480))
        for j,angle in enumerate([0,90,180,-90]):
            R=Rotation.from_rotvec([0,np.deg2rad(angle),0]).as_matrix();vv=(v-center)@R.T;vv[:,2]+=3*extent
            im=renderer(Image.new('RGB',(340,480)),np.array([0,0,340,480]),dict(vertices_camera=vv[None],faces=a['faces'],K_integer_centers=K))
            sides.paste(im,(j*340,0))
        dest=Path('/mnt/d/00/记录/006')/f'分析_沙发半圈8帧124_{name}_当前mesh四方向.png'
        assert not dest.exists();sides.save(dest);images['sides']=dict(path=str(dest),sha256=sha(dest),synthetic_not_truth=True)
        tile=Image.new('RGB',(1480,600));im=Image.open(images['body']['path']);im.thumbnail((720,600));tile.paste(im,(0,0))
        si=sides.copy();si.thumbnail((760,600));tile.paste(si,(720,0));tiles.append(tile)
        records.append(dict(frame=name,images=images,fit_sha256=sha(path),actual_visual_review=False))
    pages=[]
    for first in range(0,len(tiles),4):
        im=Image.new('RGB',(1480,600*min(4,len(tiles)-first)))
        for i,tile in enumerate(tiles[first:first+4]):im.paste(tile,(0,600*i))
        dest=out/'review'/f'page-{first//4+1:02d}.png';im.save(dest);pages.append(str(dest))
    with np.load(out/'selected-fit.npz') as z:a=dict(z)
    R=Rotation.from_rotvec(a['global_orient']).as_matrix();ray=np.einsum('nji,nj->ni',R,-a['joints_camera'][:,8])
    yaw=np.degrees(np.arctan2(ray[:,0],ray[:,2]))%360;ordered=np.sort(yaw);gaps=np.diff(np.r_[ordered,ordered[0]+360])
    save(out/'review/result.json',dict(updated=now(),records=records,pages=pages,actual_visual_review=False,
        direction_proxy_degrees=dict(zip(a['frame_ids'].astype(str),yaw.tolist())),estimated_arc_degrees=float(360-gaps.max()),
        physical_calibration_claimed=False,optimizer_updates=0,quality_accepted=False,
        renderer_sha256=sha(source),reviewer_sha256=sha(__file__)))
    print('SOFA_HALF8_NATIVE_REVIEW_READY',flush=True)
if __name__=='__main__':main()
