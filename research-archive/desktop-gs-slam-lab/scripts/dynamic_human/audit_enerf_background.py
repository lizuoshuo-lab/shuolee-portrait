"""Independently reproject candidate background geometry against source features."""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_enerf_input import read_cameras,digest
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-background-032';root=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');r=json.loads((out/'triangulation.json').read_text());cameras=read_cameras(root/'optimized');v=np.load(out/'background-init.npz');points=v['points'];colors=v['colors'];assert len(points)==6480 and len(points)==len(r['tracks']);images={};max_error=0.;error_delta=0.;color_delta=0.;coverage={c:0 for c in r['train_cameras']}
for n,h in r['calibration_sha256'].items():assert digest(root/'optimized'/n)==h
for cam,s in r['sources'].items():
    assert cam not in ['00','06','12'];assert digest(root/s['path'])==s['sha256'];rgb=cv2.cvtColor(cv2.imread(str(root/s['path'])),cv2.COLOR_BGR2RGB);K=np.array(cameras[cam]['K']);D=np.array(cameras[cam]['D']);maps=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1);images[cam]=cv2.remap(rgb,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
for i,track in enumerate(r['tracks']):
    cols=[];assert len(track['observations'])>=3 and len({o['camera'] for o in track['observations']})==len(track['observations'])
    for j,o in enumerate(track['observations']):
        cam=o['camera'];assert cam in coverage;coverage[cam]+=1;camera=cameras[cam];w=np.array(camera['w2c']);K=np.array(camera['K']);rot=cv2.Rodrigues(w[:3,:3])[0];uv,_=cv2.projectPoints(points[i].astype('float64')[None],rot,w[:3,3],K,None);e=float(np.linalg.norm(uv.reshape(2)-o['xy']));assert e<1.501;max_error=max(max_error,e);error_delta=max(error_delta,abs(e-track['errors_px'][j]));depth=float((w[:3,:3]@points[i]+w[:3,3])[2]);assert .1<depth<20;xy=np.rint(o['xy']).astype(int);cols.append(images[cam][xy[1],xy[0]])
    delta=float(np.abs(np.median(cols,axis=0)/255-colors[i]).max());color_delta=max(color_delta,delta);assert delta<1e-7
assert error_delta<1e-3
sort=sorted(range(len(points)),key=lambda i:max(r['tracks'][i]['errors_px']));chosen=[sort[int(q*(len(sort)-1))] for q in [0,.25,.5,.75,1]];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15);sheet=Image.new('RGB',(480,5*150),'white');draw=ImageDraw.Draw(sheet)
for row,i in enumerate(chosen):
    track=r['tracks'][i];obs=track['observations'];selected=[obs[0],obs[len(obs)//2],obs[-1]]
    for col,o in enumerate(selected):
        xy=np.rint(o['xy']).astype(int);patch=Image.fromarray(images[o['camera']]).crop((xy[0]-48,xy[1]-48,xy[0]+48,xy[1]+48));sheet.paste(patch,(col*160+20,row*150+35));draw.text((col*160+4,row*150+5),f'track{i} / cam{o["camera"]}',font=font,fill='black')
sheet.save(out/'matched-background-source-patches.png')
report=dict(status='all6480candidate_points_reprojected_and_colors_source_checked_visual_review_pending',points=len(points),source_images=len(images),heldout_images_read=0,coverage_observations=coverage,maximum_reprojection_px=max_error,maximum_difference_from_saved_double_DLT_errors_px=error_delta,maximum_color_readback_difference=color_delta,projection_method='Independent OpenCV Rodrigues/projectPoints against saved native feature positions',patch_sheet='matched-background-source-patches.png',patch_selection='Five tracks across max-reprojection error quantiles0,.25,.5,.75,1; first/middle/last camera of each track; native96x96 RGB, no warp or enhancement',source_geometry_used=False,limits='This validates provenance and projection consistency; wrong repeating-texture matches can still satisfy geometry. No learned background or novel-view quality claimed.')
(out/'independent-geometry-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='coverage_observations'},indent=2))
