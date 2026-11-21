"""Apply the original SIMPLE_RADIAL calibration at twice the training resolution.

The raw source-resolution staging experiment is retained but explicitly rejected
as a training dataset. This output has the same rectified camera field of view
as the previous data; all 3D points and camera extrinsics remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001/hover-sfm-001'


def psnr(a,b):
    return float(-10*np.log10(max(float(np.mean(((a.astype('float32')-b.astype('float32'))/255)**2)),1e-12)))


def main():
    cv2.setNumThreads(3)
    staged=ROOT/'source-resolution-input-001';original_manifest=json.loads((staged/'manifest.json').read_text())
    assert original_manifest['used_for_training'] is False
    out=ROOT/'source-resolution-rectified-002';out.mkdir(exist_ok=False)
    dataset=out/'dataset';dest=dataset/'images';dest.mkdir(parents=True);sparse=dataset/'sparse/0';sparse.mkdir(parents=True)
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'));import read_write_model as rw
    raw=rw.read_cameras_binary(str(BASE/'raw_sparse/0/cameras.bin'))[1]
    cameras,images,points=rw.read_model(str(BASE/'dataset/sparse/0'),ext='.bin');u=cameras[1]
    assert raw.model=='SIMPLE_RADIAL' and u.model=='PINHOLE'
    f,cx,cy,k=raw.params;fx,fy,ux,uy=u.params
    K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
    newK=np.array([[fx,0,ux],[0,fy,uy],[0,0,1.]])
    distortion=np.array([k,0.,0.,0.,0.]);scale=2
    fullK=K.copy();fullK[:2]*=scale;fullNewK=newK.copy();fullNewK[:2]*=scale
    width,height=u.width*scale,u.height*scale
    mx,my=cv2.initUndistortRectifyMap(fullK,distortion,None,fullNewK,(width,height),cv2.CV_32FC1)
    lx,ly=cv2.initUndistortRectifyMap(K,distortion,None,newK,(u.width,u.height),cv2.CV_32FC1)
    valid=(mx>=0)&(my>=0)&(mx<1520-1)&(my<2704-1)
    assert float(valid.mean())>.995
    rows=[];panels=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for item in original_manifest['rows']:
        path=staged/'dataset/images'/item['name'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['source_sha256']
        source=cv2.imread(str(path));assert source.shape[:2]==(2704,1520)
        rectified=cv2.remap(source,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        assert cv2.imwrite(str(dest/item['name']),rectified,[cv2.IMWRITE_PNG_COMPRESSION,3])
        name=str(Path(item['name']).with_suffix('.jpg'))
        old_raw=cv2.imread(str(BASE/'raw_images'/name));old=cv2.imread(str(BASE/'dataset/images'/name))
        source_small=cv2.resize(source,(raw.width,raw.height),interpolation=cv2.INTER_AREA)
        corrected_small=cv2.resize(rectified,(u.width,u.height),interpolation=cv2.INTER_AREA)
        reference_remap=cv2.remap(old_raw,lx,ly,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        rows.append(dict(name=item['name'],source_frame_index=item['source_frame_index'],split=item['split'],
            source_sha256=item['source_sha256'],rectified_sha256=hashlib.sha256((dest/item['name']).read_bytes()).hexdigest(),
            raw_downsample_vs_old_raw_psnr=psnr(source_small,old_raw),
            rectified_downsample_vs_old_rectified_psnr=psnr(corrected_small,old),
            old_raw_cv_remap_vs_old_colmap_psnr=psnr(reference_remap,old)))
        if int(Path(name).stem) in [1,21,49]:
            panel=Image.new('RGB',(600,394),'white');draw=ImageDraw.Draw(panel)
            for j,(label,arr) in enumerate([('Previous rectified',old),('Raw, wrong camera',cv2.resize(source,(u.width,u.height),interpolation=cv2.INTER_AREA)),('Full-res rectified',corrected_small)]):
                thumb=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB));thumb.thumbnail((196,350));panel.paste(thumb,(j*200,36));draw.text((j*200+2,6),label,font=font,fill='black')
            panels.append(panel)
    new_cameras={1:u._replace(width=width,height=height,params=u.params*scale)}
    new_images={key:im._replace(name=str(Path(im.name).with_suffix('.png')),xys=im.xys*scale) for key,im in images.items()}
    rw.write_model(new_cameras,new_images,points,str(sparse),ext='.bin')
    assert (sparse/'points3D.bin').read_bytes()==(BASE/'dataset/sparse/0/points3D.bin').read_bytes()
    _,check_images,check_points=rw.read_model(str(sparse),ext='.bin')
    assert all(np.array_equal(im.qvec,check_images[key].qvec) and np.array_equal(im.tvec,check_images[key].tvec) for key,im in images.items())
    sheet=Image.new('RGB',(1800,394),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,(i*600,0))
    sheet.save(out/'rectification-review.jpg',quality=96)
    comparisons={metric:dict(mean=float(np.mean([r[metric] for r in rows])),minimum=min(r[metric] for r in rows))
        for metric in ['raw_downsample_vs_old_raw_psnr','rectified_downsample_vs_old_rectified_psnr','old_raw_cv_remap_vs_old_colmap_psnr']}
    manifest=dict(status='prepared_pending_visual_and_memory_validation',created=time.time(),dataset=str(dataset),rows=rows,
        selected_images=59,training_images=51,development_images=8,
        raw_video_resolution=[1520,2704],rectified_resolution=[width,height],original_rectified_resolution=[u.width,u.height],
        original_raw_camera=raw._asdict(),new_pinhole_camera=dict(width=width,height=height,parameters=(u.params*scale).tolist()),
        scale=2,pose_and_structure_unchanged=True,points3D_binary_identical=True,rectification_valid_fraction=float(valid.mean()),
        comparisons=comparisons,used_for_training=False,
        correction='Restore the prior SIMPLE_RADIAL undistortion at twice the image dimensions and intrinsics. Raw stage001 is not suitable for training.',
        caveats=['No new cameras or calibration were optimized','OpenCV and COLMAP interpolation plus JPEG introduce pixel differences',
            'Full-resolution GPU memory has not yet been validated','Same video and prior train/development assignment'])
    manifest['original_raw_camera']['params']=raw.params.tolist()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(dataset=str(dataset),comparisons=comparisons,valid_fraction=float(valid.mean())),indent=2))


if __name__=='__main__':main()
