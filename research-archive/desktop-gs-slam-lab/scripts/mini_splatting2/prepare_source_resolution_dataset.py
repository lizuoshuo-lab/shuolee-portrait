"""Recover exact selected video frames and rescale fixed COLMAP intrinsics.

No new calibration or Gaussian optimization. Geometry and train/test assignment
remain unchanged. The source-resolution dataset is staged for a separate later
resolution ablation, never silently substituted into an active run.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001'


def main():
    cv2.setNumThreads(3)
    out=ROOT/'source-resolution-input-001';out.mkdir(exist_ok=False)
    dataset=out/'dataset';image_dir=dataset/'images';image_dir.mkdir(parents=True)
    sparse=dataset/'sparse/0';sparse.mkdir(parents=True)
    video=LAB/'inputs/computer00-local-pipeline/HOVER_VERSA_0102.mp4'
    video_hash=hashlib.sha256(video.read_bytes()).hexdigest()
    assert video_hash=='5d4b6c5cb38edfb17c90faa8352e0edb93370a6d230076a463d3e3ec9c70714f'
    selection=json.loads((ROOT/'additional-frames-001/selection.json').read_text())['original']
    by_index={index:name for name,index in selection.items()};assert len(by_index)==59
    cap=cv2.VideoCapture(str(video));assert cap.isOpened();width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert (width,height)==(1520,2704)
    rows=[];index=0;panels=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            if index in by_index:
                name=by_index[index];original=BASE/'hover-sfm-001/dataset/images'/name
                old=cv2.imread(str(original));h,w=old.shape[:2]
                assert (w,h)==(757,1347)
                resized=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
                mse=float(np.mean(((resized.astype('float32')-old.astype('float32'))/255)**2))
                psnr=-10*np.log10(max(mse,1e-12))
                filename=Path(name).with_suffix('.png').name
                assert cv2.imwrite(str(image_dir/filename),image,[cv2.IMWRITE_PNG_COMPRESSION,3])
                row=dict(name=filename,source_frame_index=index,split='development_holdout' if int(Path(name).stem)%8==0 else 'train',
                    width=width,height=height,source_sha256=hashlib.sha256((image_dir/filename).read_bytes()).hexdigest(),
                    previous_image_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    downsample_area_vs_previous_jpeg_psnr=float(psnr))
                rows.append(row)
                if int(Path(name).stem) in [1,21,49]:
                    tile=Image.new('RGB',(400,394),'white');draw=ImageDraw.Draw(tile)
                    for j,(label,arr) in enumerate([('Previous training image',old),('Source frame, area resized',resized)]):
                        thumb=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB));thumb.thumbnail((196,350))
                        tile.paste(thumb,(j*200,36));draw.text((j*200+2,6),label,font=font,fill='black')
                    panels.append(tile)
            index+=1
    finally:cap.release()
    assert len(rows)==59 and sum(r['split']=='train' for r in rows)==51
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
    import read_write_model as rw
    source_sparse=BASE/'hover-sfm-001/dataset/sparse/0'
    cameras,images,points=rw.read_model(str(source_sparse),ext='.bin')
    assert len(cameras)==1 and cameras[1].model=='PINHOLE'
    old=cameras[1];sx=width/old.width;sy=height/old.height
    params=old.params*np.array([sx,sy,sx,sy])
    new_cameras={1:old._replace(width=width,height=height,params=params)}
    new_images={key:im._replace(name=str(Path(im.name).with_suffix('.png')),xys=im.xys*np.array([sx,sy])) for key,im in images.items()}
    normalized_errors=[]
    for key,im in images.items():
        before=(im.xys-old.params[2:])/old.params[:2]
        after=(new_images[key].xys-params[2:])/params[:2]
        normalized_errors.append(float(np.max(np.abs(before-after))))
    assert max(normalized_errors)<1e-12
    rw.write_model(new_cameras,new_images,points,str(sparse),ext='.bin')
    check_c,check_i,check_p=rw.read_model(str(sparse),ext='.bin')
    assert all(np.array_equal(im.qvec,check_i[key].qvec) and np.array_equal(im.tvec,check_i[key].tvec) for key,im in images.items())
    assert all(np.array_equal(point.xyz,check_p[key].xyz) for key,point in points.items())
    assert (sparse/'points3D.bin').read_bytes()==(source_sparse/'points3D.bin').read_bytes()
    if panels:
        sheet=Image.new('RGB',(1200,394),'white')
        for j,panel in enumerate(panels):sheet.paste(panel,(j*400,0))
        sheet.save(out/'source-alignment-review.jpg',quality=95)
    psnrs=[r['downsample_area_vs_previous_jpeg_psnr'] for r in rows]
    protocol=dict(status='prepared_pending_visual_and_memory_validation',created=time.time(),source_video=str(video),source_video_sha256=video_hash,
        decoded_frame_count=index,selected_images=len(rows),training_images=51,development_images=8,rows=rows,
        old_size=[757,1347],new_size=[width,height],scale_factors=[sx,sy],
        old_pinhole_parameters=old.params.tolist(),new_pinhole_parameters=params.tolist(),
        normalized_image_coordinates_max_difference=max(normalized_errors),
        old_camera_extrinsics_preserved=True,points3D_binary_identical=True,split_unchanged=True,
        downsample_comparison=dict(mean_psnr=float(np.mean(psnrs)),min_psnr=min(psnrs),max_psnr=max(psnrs)),
        used_for_training=False,
        caveats=['Lossless frames from the original work video; original training images were resized JPEGs',
            'Downsample comparison includes resampling and JPEG differences; this is a separate data-resolution ablation',
            'Fixed camera intrinsics and sparse observations are scaled in pixel coordinates, without bundle adjustment',
            'No independent scene or newly captured viewpoint was introduced',
            'Before training, check peak CUDA memory with representative Gaussian count; existing jobs and models remain unchanged'])
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2));print(json.dumps({key:value for key,value in protocol.items() if key!='rows'},indent=2))


if __name__=='__main__':main()
