"""Select a separate timing-calibration segment by observed vertical motion.

The fixed-camera rig has mostly horizontal epipolar lines; horizontal image
motion alone cannot constrain clock offset through epipolar distance.
Only camera08 and frames>=80 are used for selecting anchors.
"""
from pathlib import Path
import json,time
import cv2,numpy as np
from audit_enerf_camera_time import flow
from audit_enerf_input import digest
LAB=Path(__file__).resolve().parents[2];spec=json.loads((LAB/'outputs/dynamic-human-enerf-full-023/actor1_4-native-input-audit.json').read_text());root=Path(spec['root']);out=LAB/'outputs/dynamic-human-enerf-time-calibration-028';dest=out/'motion-selection.json';assert not dest.exists();cv2.setNumThreads(8)
c=spec['cameras']['08'];K=np.array(c['K']);D=np.array(c['D']);mx,my=cv2.initUndistortRectifyMap(K,D,np.eye(3),K,(1920,1080),cv2.CV_32FC1)
def gray(t):
    f=root/'images/08'/f'{t:06d}.jpg';im=cv2.imread(str(f),cv2.IMREAD_GRAYSCALE);return cv2.remap(im,mx,my,cv2.INTER_LINEAR),digest(f)
rows=[];began=time.time()
for t in range(80,1181,10):
    im,sha=gray(t);before,bsha=gray(t-1);after,asha=gray(t+1);mf=root/'bgmtv2/08'/f'{t:06d}.jpg';m=cv2.remap(cv2.imread(str(mf),0),mx,my,cv2.INTER_LINEAR);xy=cv2.goodFeaturesToTrack(im,maxCorners=1000,qualityLevel=.02,minDistance=8,mask=(m>230).astype('uint8')*255)[:,0]
    pre,pok=flow(im,before,xy);nex,nok=flow(im,after,xy);ok=pok&nok;v=(nex-pre)*.5;vertical=np.abs(v[:,1]);norm=np.linalg.norm(v,axis=1)
    r=dict(frame=t,tracks=int(ok.sum()),vertical_over2=int((ok&(vertical>2)).sum()),vertical_over4=int((ok&(vertical>4)).sum()),moving_over2=int((ok&(norm>2)).sum()),vertical_p90=float(np.percentile(vertical[ok],90)),rgb_sha256=sha,before_sha256=bsha,after_sha256=asha,matte_sha256=digest(mf));rows.append(r)
selected=[]
for r in sorted(rows,key=lambda r:(r['vertical_over2'],r['vertical_p90']),reverse=True):
    if r['vertical_over2']<40:continue
    if all(abs(r['frame']-q['frame'])>=90 for q in selected):selected.append(r)
    if len(selected)==3:break
assert len(selected)==3,'Insufficient timing-observable motion; do not invent anchors'
result=dict(status='separate_anchor_selection_frozen_before_cross_camera_offset_fit',selection='Camera08 only, native rectified feature flow. Scan80..1180 step10; rank number of FB-consistent corners with abs(vertical flow)>2px/frame, require>=40, min90-frame separation. First2 selected anchors fit, third independent check.',selected_anchors=[r['frame'] for r in selected],selected=selected,rows=rows,elapsed_seconds=time.time()-began,limitations=['Vertical flow is an approximate observability proxy for this rig; exact epipolar-normal sensitivity must still be examined','No0..59 appearance train/test images accessed','No offsets inferred by this selection script'])
dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
