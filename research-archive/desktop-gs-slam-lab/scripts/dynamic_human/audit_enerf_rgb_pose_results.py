"""Independent statistics and same-target trajectory checks, plus native hand panels."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];H=L/'outputs/dynamic-human-enerf-hand-observability-073';R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074';S=L/'outputs/dynamic-human-enerf-pose-sequence-062'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
old=read(S/'observations.json')['records'];new=read(R/'observations.json')['records'];assert len(old)==len(new)==195
unchanged=[k for k in old[0] if k!='detections']
for a,b in zip(old,new):
 assert all(a[k]==b[k] for k in unchanged)
 assert [x['box'] for x in a['detections']]==[x['box'] for x in b['detections']]
scores=np.array([d['scores'][91:] for r in old for d in r['detections']]);assert scores.shape==(390,42)
stored=read(H/'hand-statistics.json')['records'];assert len(stored)==780
assert all(x['points_ge04']==21 for x in stored) and (scores>=.4).all()
handmins=np.min(scores,axis=1)
summary=dict(status='all195metadata_boxes_and16380hand_scores_rechecked',old_hand_landmark_count=int(scores.size),old_hand_score_min=float(scores.min()),old_hand_score_max=float(scores.max()),old_hand_fraction_passing04=float((scores>=.4).mean()),all_image_and_camera_metadata_unchanged=True,goal_complete=False)
save(H/'independent-audit.json',summary)

# Evaluate both already-fitted cubic models against exactly the same new and
# old image measurements. The score is agreement with predictions, not GT.
a=dict(np.load(S/'estimated_offsets-fit.npz'));b=dict(np.load(R/'estimated_offsets-fit.npz'))
for k in ['camera_index','frame','joint_index','fitmask','P','cameras']:assert np.array_equal(a[k],b[k]),k
reports={}
for model,fit in [('historical',a),('RGB_corrected',b)]:
 tau=(b['frame']-750+fit['offsets'][b['camera_index']])/6;basis=np.stack([tau**k for k in range(4)],axis=1)
 xyz=np.einsum('nk,nkd->nd',basis,fit['coefficients'][b['joint_index']]);q=np.einsum('nij,nj->ni',b['P'][:,:,:3],xyz)+b['P'][:,:,3];assert (q[:,2]>0).all()
 xy=q[:,:2]/q[:,2,None]
 reports[model]={}
 for targets,truth in [('historical_predictions',a['xy']),('RGB_corrected_predictions',b['xy'])]:
  err=np.linalg.norm(xy-truth,axis=1);r={}
  for split,m in [('fit',b['fitmask']),('development_time',~b['fitmask'])]:
   e=err[m];r[split]=dict(count=len(e),median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)),rmse_px=float(np.sqrt(np.mean(e**2))),fraction_below8_px=float(np.mean(e<8)))
  reports[model][targets]=r
save(R/'same-target-trajectory-audit.json',dict(status='old_and_RGB_models_cross_evaluated_on_identical_camera_time_joint_indices',reports=reports,
 offset_change_frames={str(c):float(y-x) for c,x,y in zip(a['cameras'],a['offsets'],b['offsets'])},
 limits='Both targets are neural predictions; this is model-data agreement. Same059camera/jointsupport retained. Does not establish geometrictruth, RGBquality or causalblurreduction.',goal_complete=False))

pilot=read(H/'hand-pilot-rgb/observations.json')['records'];byold={r['camera']:r for r in old if r['frame']==750};bynew={r['camera']:r for r in new if r['frame']==750}
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',11);colors=[(237,90,40),(200,55,145),(45,140,220),(85,175,55),(175,90,210)]
def overlay(im,points,crop):
 im=im.copy();d=ImageDraw.Draw(im);pts=np.asarray(points)-np.asarray(crop[:2])
 for f in range(5):
  chain=[0]+list(range(1+4*f,5+4*f))
  for i,j in zip(chain[:-1],chain[1:]):d.line([tuple(pts[i]),tuple(pts[j])],fill=colors[f],width=1)
 for x,y in pts:d.ellipse([x-1,y-1,x+1,y+1],fill=(255,50,30))
 return im
for c in sorted({r['camera'] for r in pilot}):
 rows=[r for r in pilot if r['camera']==c];rgb=np.load(S/'native-input'/f'camera{c}-frame000750.npz')['rgb'];im=Image.fromarray(rgb)
 sheet=Image.new('RGB',(690,40+len(rows)*194),'white');d=ImageDraw.Draw(sheet);d.text((10,2),f'Camera{c} frame750; ALL21 predicted points, NOT visibility labels; native pixels',font=font,fill='black')
 for col,label in enumerate(['Native photograph','Whole-body, old BGR','Whole-body, correct RGB','Hand-specific, correct RGB']):d.text((10+col*170,18),label,font=font,fill='black')
 for i,row in enumerate(rows):
  person=row['person'];crop=row['panel_crop'];raw=im.crop(crop);o=byold[c]['detections'][person];n=bynew[c]['detections'][person];y=55+i*194
  d.text((10,y-14),('dark' if person==0 else 'light')+' right',font=font,fill='black')
  images=[raw,overlay(raw,o['keypoints'][112:133],crop),overlay(raw,n['keypoints'][112:133],crop),overlay(raw,row['keypoints'],crop)]
  for j,img in enumerate(images):sheet.paste(img,(10+j*170,y))
 sheet.save(H/'hand-pilot-rgb'/f'camera{c}-all-points.png')
print(json.dumps(dict(independent_input_audit=summary,matched_target_reports=reports,pilot_native_panels=14)))
