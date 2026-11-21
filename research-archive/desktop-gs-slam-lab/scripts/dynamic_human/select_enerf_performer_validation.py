"""Training-camera-only motion scan for future cross-performer validation.

No heldout appearance camera is opened, and no candidate model is evaluated.
Optical flow is a selection statistic, not verified human motion ground truth.
"""
from pathlib import Path
import json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_enerf_input import read_cameras,digest

LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-performer-selection-031';out.mkdir(exist_ok=True);root=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted');assert not (out/'selection.json').exists();camera='08';font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
protocol=dict(status='frozen_before_loading_candidate_images',datasets=['actor2_3','actor5_6'],camera=camera,appearance_heldout_cameras_never_opened=['00','06','12'],candidate_centers='100,150,... ending100frames before sequence end',source_times='center-10,center,center+10',selection='Mean Farneback flow magnitude in union of foreground masks between endpoint images; quarter-resolution flow multiplied by4 for native-pixel displacement. Choose3centers by descending score, at least150indices apart.',mask='Released background-matting product, may include objects and artifacts',display='Whole two-person foreground inspection, display scaled to640wide; not native evaluation renders',scope='Training-camera-only screening for a future protocol, not validation scores or proof of walking/occlusion coverage')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8');results=[]
for name in protocol['datasets']:
    data=root/name;cameras=read_cameras(data/'optimized');K=np.array(cameras[camera]['K']);D=np.array(cameras[camera]['D']);files={int(p.stem):p for p in (data/'images'/camera).glob('*.jpg')};available=sorted(files);candidate=range(100,max(available)-99,50);rows=[];maps=None;cache={};sources={}
    def load(t):
        global maps
        if t in cache:return cache[t]
        path=files[t];mask=data/'bgmtv2'/camera/path.name;im=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);m=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE);assert im.shape[:2]==m.shape
        if maps is None:maps=cv2.initUndistortRectifyMap(K,D,None,K,(im.shape[1],im.shape[0]),cv2.CV_32FC1)
        im=cv2.remap(im,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);m=cv2.remap(m,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);cache[t]=(im,m);sources[t]=dict(frame=t,camera=camera,image_path=path.relative_to(data).as_posix(),sha256=digest(path),mask_path=mask.relative_to(data).as_posix(),mask_sha256=digest(mask));return cache[t]
    for t in candidate:
        if not all(k in files for k in [t-10,t,t+10]):continue
        a,ma=load(t-10);b,mb=load(t+10);load(t)
        ga=cv2.resize(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA);gb=cv2.resize(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA);mask=cv2.resize(((ma>128)|(mb>128)).astype('uint8'),(ga.shape[1],ga.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
        flow=cv2.calcOpticalFlowFarneback(ga,gb,None,.5,4,25,5,7,1.5,0);mag=np.linalg.norm(flow,axis=2)[mask]*4;rows.append(dict(center=t,mean_foreground_displacement_px=float(mag.mean()),p90_displacement_px=float(np.quantile(mag,.9)),foreground_quarter_pixels=int(mask.sum())))
    selected=[]
    for row in sorted(rows,key=lambda x:x['mean_foreground_displacement_px'],reverse=True):
        if all(abs(row['center']-r['center'])>=150 for r in selected):selected.append(row)
        if len(selected)==3:break
    assert len(selected)==3
    for row in selected:
        t=row['center'];canvas=Image.new('RGB',(1920,404),'white');draw=ImageDraw.Draw(canvas)
        for j,frame in enumerate([t-10,t,t+10]):
            im,m=load(frame);composite=np.rint(im.astype('float64')*(m[...,None]/255)+.5*(255-m[...,None])).clip(0,255).astype('uint8');native=Image.fromarray(composite);native.save(out/f'{name}-camera08-frame{frame:06d}-foreground.png');canvas.paste(native.resize((640,360),Image.Resampling.LANCZOS),(j*640,44));draw.text((j*640+8,12),f'{name} | training camera08 | frame{frame}',font=font,fill='black')
        row['contact_sheet']=f'{name}-center{t:06d}-motion.png';canvas.save(out/row['contact_sheet'])
    results.append(dict(dataset=name,available_frames=[min(available),max(available)],selected=selected,candidates=rows,source_files=list(sources.values()),calibration_sha256={p.name:digest(p) for p in [data/'optimized/intri.yml',data/'optimized/extri.yml']}));print(name,json.dumps(selected),flush=True)
    del cache
result=dict(status='training_camera_motion_candidates_selected_visual_review_pending',protocol='protocol.json',results=results,heldout_appearance_photos_read=0,models_trained=0)
(out/'selection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
