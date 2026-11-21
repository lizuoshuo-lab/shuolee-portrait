"""Read-only source-to-current-training pixel audit; no fitted state is changed."""
from pathlib import Path
import json, hashlib, time, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
B=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
O=B/'current/camera-feature-root-cause-115'
S=B/'current/full-scene-localization-074'
P=B/'current/avatar-repository-pair-114'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(q,p)
def main():
    cv2.setNumThreads(4);O.mkdir(exist_ok=True)
    assert not (O/'pixel-result.json').exists(), 'completed evidence is immutable'
    paths=[P/'smplx/data/training-inputs.json',P/'gaussianavatar/data/training-inputs.json',B/'current/person-static-native-align-fresh-113/data/training-inputs.json']
    rows=read(paths[0])['records'];others=[read(p)['records'] for p in paths[1:]]
    assert len(rows)==175
    for rr in others:
        assert len(rr)==len(rows)
        for a,b in zip(rows,rr):
            for k in ['frame','source_frame_index','image','mask','source_sha256','mask_sha256','wh','K','w2c']:assert a[k]==b[k],k
    protocol=dict(question='Can current113/114 portrait pixels and masks be mapped back to the original video without crop/scale/frame errors?',
        phase='read_only_diagnosis',new_training_started=False,source_files={str(p):sha(p) for p in paths},
        tests=['Decode all175 actual original video frames, independently rectify and compare every RGB byte.',
               'Reconstruct the inherited mask rectification through the original sensor.',
               'Compare native training K with resize and integer-pixel-center convention.',
               'Inspect full-person and face contours on original image pixels.'],
        limits='Exact RGB and mask transform consistency cannot prove the segmentation boundary or the camera estimate is physically accurate.',
        training_gate='Do not train until projection/mapping or competing geometric cause has a discriminating result.',
        quality_accepted=False)
    save(O/'protocol.json',protocol)
    sfm=read(S/'cameras.json');f,cx,cy,k=sfm['intrinsics']
    Kn=np.array([[f,0,cx-.5],[0,f,cy-.5],[0,0,1.]])
    dn=np.array([k,0,0,0,0.]);K=np.array(rows[0]['K']);Kcv=K.copy();Kcv[:2,2]-=.5
    W,H=rows[0]['wh'];mx,my=cv2.initUndistortRectifyMap(Kn,dn,None,Kcv,(W,H),cv2.CV_32FC1)
    oldrows=read(B/'current/person-static-registered-camera-fresh-072/data/training-inputs.json')['records']
    oldraw=np.array(read(B.parent/'source-resolution-rectified-002/manifest.json')['original_raw_camera']['params']);oldraw[:3]*=2
    Ko=np.array([[oldraw[0],0,oldraw[1]],[0,oldraw[0],oldraw[2]],[0,0,1.]])
    do=np.array([oldraw[3],0,0,0,0.]);Kold=np.array(oldrows[0]['K'])
    oldmap=cv2.undistortPoints(np.stack([mx,my],-1).reshape(-1,1,2).astype(float),Ko,do,P=Kold).reshape(H,W,2).astype('float32')
    train=read(P/'smplx/gaussian-run/training-cameras.json')['records']
    video=LAB/'inputs/computer00-local-pipeline/HOVER_VERSA_0102.mp4'
    cap=cv2.VideoCapture(str(video));assert cap.isOpened();records=[];start=time.monotonic()
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',16)
    selected={34,70,100,136};obs=np.load(P/'smplx/data/observations.npz')['keypoints']
    for raw in range(176):
        ok,bgr=cap.read();assert ok
        if raw==0:continue
        r=rows[raw-1];assert r['source_frame_index']==raw
        actual=cv2.imread(r['image']);mask=cv2.imread(r['mask'],0)
        assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
        rect=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        previous=cv2.imread(oldrows[raw-1]['mask'],0)
        recreated_mask=cv2.remap(previous,oldmap[:,:,0],oldmap[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        pixel_error=int(np.max(np.abs(rect.astype('int16')-actual.astype('int16'))))
        mask_error=int(np.max(np.abs(mask.astype('int16')-recreated_mask.astype('int16'))))
        t=train[raw-1];tw,th=t['wh'];scaled=np.diag([tw/W,th/H,1.])@K
        ke=float(np.max(abs(scaled-np.array(t['K']))));ee=float(np.max(abs(np.array(t['w2c'])-r['w2c'])))
        assert pixel_error==mask_error==0 and ke<1e-10 and ee<1e-12
        records.append(dict(raw_frame=raw,rgb_max_8bit_error=pixel_error,mask_transform_max_8bit_error=mask_error,K_max_error=ke,E_max_error=ee,
            full_canvas_wh=[W,H],training_wh=[tw,th],crop=t.get('crop'),foreground_pixels=int((mask>254).sum())))
        if raw in selected:
            rgb=cv2.cvtColor(actual,cv2.COLOR_BGR2RGB);lines=rgb.copy()
            contours,_=cv2.findContours((mask>127).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(lines,contours,-1,(0,220,90),2)
            yy,xx=np.mgrid[:H,:W];check=np.where(((xx//24+yy//24)%2)[...,None],.20,.32)
            composite=np.rint(rgb*(mask[...,None]/255.)+check*(1-mask[...,None]/255.)*255).astype('uint8')
            lm=obs[raw-1,23:91];lo=np.floor(lm.min(0)-[55,75]).astype(int);hi=np.ceil(lm.max(0)+[55,45]).astype(int)
            facebox=(int(max(lo[0],0)),int(max(lo[1],0)),int(min(hi[0],W)),int(min(hi[1],H)))
            panels=[rgb,lines,composite];canvas=Image.new('RGB',(1320,1230),'#f7f8fa');draw=ImageDraw.Draw(canvas)
            titles=['原视频按训练标定去畸变','同一张图叠加 mask 轮廓','训练人像：仅遮去背景']
            for j,(arr,title) in enumerate(zip(panels,titles)):
                im=Image.fromarray(arr);thumb=im.copy();thumb.thumbnail((426,760))
                canvas.paste(thumb,(j*440+(440-thumb.width)//2,54));draw.text((j*440+12,14),title,font=font,fill='#132a3a')
                crop=im.crop(facebox);crop.thumbnail((426,350));canvas.paste(crop,(j*440+(440-crop.width)//2,840))
            draw.text((12,803),f'帧 {raw:06d}  |  三列保持相同像素坐标和等比例显示，无独立平移或拉伸',font=font,fill='#132a3a')
            draw.text((12,1200),'RGB 逐像素差 = 0；绿色线只表示分割边界，不代表人体几何或三维投影已正确。',font=small,fill='#334b5e')
            canvas.save(O/f'portrait-source-{raw:06d}.jpg',quality=94)
        if raw%35==0:print(json.dumps(dict(stage='pixel_mapping',checked=raw,seconds=time.monotonic()-start)),flush=True)
    cap.release()
    grid=np.stack(np.meshgrid(np.linspace(20,W-21,30),np.linspace(20,H-21,40)),-1).reshape(-1,1,2)
    directions=cv2.undistortPoints(grid,Kn,dn)
    rectcoords=cv2.convertPointsToHomogeneous(directions)[:,0]@Kcv.T
    rays=(rectcoords@np.linalg.inv(Kcv).T)
    back=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),Kn,dn)[0]
    inverse_err=float(abs(back-grid).max())
    result=dict(state='current_input_mapping_passed',checked_frames=175,rounds_checked=['113','114a','114b'],
        all_current_input_records_identical=True,full_frame_training=True,person_bbox_cropping=False,
        maximum_RGB_8bit_difference=max(r['rgb_max_8bit_error'] for r in records),
        maximum_mask_remap_8bit_difference=max(r['mask_transform_max_8bit_error'] for r in records),
        source_wh=[W,H],training_wh=train[0]['wh'],raw_to_rectified_roundtrip_max_px=inverse_err,
        expected_mapping='Original video -> camera undistortion (recentered principal point) -> full-image resize. Mask shares that mapping; no person-only recentering.',
        segmentation_accuracy_proven=False,physical_camera_accuracy_proven=False,
        new_training_started=False,source_modified=False,seconds=time.monotonic()-start)
    save(O/'pixel-records.json',records);save(O/'pixel-result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
