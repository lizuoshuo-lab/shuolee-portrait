"""Treat disagreement between SAM and filled foreground as unlabelled supervision."""
from pathlib import Path
import json,hashlib,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];S=LAB/'scripts/dynamic_human'
BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
MASK=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15'
OUT=LAB/'outputs/dynamic-human-enerf-mask-uncertainty-053';OUT.mkdir(exist_ok=True)
assert not (OUT/'input-audit.json').exists()
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save(OUT/'protocol.json',dict(status='fixed_before_input_review_and_training',
    intervention='Only loss-valid pixels change: unknown=(hybridforeground AND NOT SAMunion). Keep original RGB, hybrid target matte, bbox, matrices and init.',
    rationale='Hybrid mask restores missing limbs but can reintroduce wall patches. Disagreement is neither certifiedforeground nor certifiedbackground; omit supervision there instead of forcing a class.',
    loss='Existing masked-image L1/DSSIM and maskedalphaL1 unchanged; both target/pred zeroed at invalid pixels. SSIM windows overlapping invalid pixels still change local moments; not a pure per-pixel loss.',
    fixed='Same500baseline initial parameters and6000target/background draws,6000updates,original750observations,15traincams,allunchangedheldoutreferences.',
    evaluation='Same six development views and17fixed050details; no threshold tuning to their images.',
    limitations='SAM foreground is not ground truth; this removes only one source of uncertain supervision, not all mask errors.',goal_complete=False))
panels=OUT/'input-review';panels.mkdir(exist_ok=True);records=[];patches=[]
for source in sorted(BASE.glob('train-camera-*.npz')):
 c=source.stem[-2:];z=np.load(source);data={k:z[k] for k in z.files}
 hybrid=data['matte']>.5;sam=np.array(Image.open(MASK/f'camera{c}-candidate.png'))>0
 assert np.all(~sam|hybrid)
 unknown=hybrid&~sam;oldvalid=data['valid'].copy();data['valid']=oldvalid&~unknown
 assert np.array_equal(hybrid,np.array(Image.open(MASK/f'hybrid-closed-core/camera{c}-mask.png'))>0)
 np.savez_compressed(OUT/source.name,**data)
 y,x=np.where(hybrid);box=[max(0,int(x.min())-16),max(0,int(y.min())-16),min(1920,int(x.max())+17),min(1080,int(y.max())+17)]
 x0,y0,x1,y1=box;w=x1-x0;h=y1-y0;rgb=data['rgb']
 overlay=rgb.copy();overlay[unknown]=[255,0,160]
 hybridcomp=np.rint(rgb.astype(float)*hybrid[...,None]+127.5*(~hybrid[...,None])).astype(np.uint8)
 samcomp=np.rint(rgb.astype(float)*sam[...,None]+127.5*(~sam[...,None])).astype(np.uint8)
 panel=Image.new('RGB',(3*w,h+32),'white');draw=ImageDraw.Draw(panel)
 for j,(name,im) in enumerate([('Training photo / ignored pixels magenta',overlay),('Existing hybrid mask',hybridcomp),('SAM mask (not new target)',samcomp)]):
  panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(j*w,32));draw.text((j*w+4,10),name,fill='black')
 panel.save(panels/f'camera{c}-input.png')
 n,labels,stats,_=cv2.connectedComponentsWithStats(unknown.astype(np.uint8),8)
 for k in np.argsort(stats[1:,4])[-3:]+1:
  a,b,ww,hh,area=map(int,stats[k]);
  if area<20:continue
  a=max(0,a-12);b=max(0,b-12);xx=min(1920,a+ww+24);yy=min(1080,b+hh+24)
  patches.append(dict(camera=c,bbox=[a,b,xx,yy],area=area))
 records.append(dict(camera=c,unknown_pixels=int((oldvalid&unknown).sum()),hybrid_foreground_pixels=int(hybrid.sum()),
  ignored_fraction_of_foreground=float(unknown.sum()/hybrid.sum()),source_sha256=sha(source),candidate_sha256=sha(OUT/source.name),
  changed_arrays=[k for k in z.files if not np.array_equal(z[k],data[k])]))
 assert records[-1]['changed_arrays'] in [[],['valid']]
for i,row in enumerate(sorted(patches,key=lambda r:r['area'],reverse=True)[:12]):
 z=np.load(BASE/f"train-camera-{row['camera']}.npz");sam=np.array(Image.open(MASK/f"camera{row['camera']}-candidate.png"))>0;hybrid=z['matte']>.5
 unknown=hybrid&~sam;im=z['rgb'].copy();im[unknown]=[255,0,160];x0,y0,x1,y1=row['bbox'];w=x1-x0;h=y1-y0
 panel=Image.new('RGB',(2*max(180,w),h+32),'white');d=ImageDraw.Draw(panel)
 for j,(label,a) in enumerate([('Photo',z['rgb']),('Ignored supervision',im)]):
  panel.paste(Image.fromarray(a[y0:y1,x0:x1]),(j*max(180,w),32));d.text((j*max(180,w)+3,10),label,fill='black')
 row['panel']=f"detail{i:02d}-camera{row['camera']}.png";panel.save(panels/row['panel'])
save(OUT/'input-audit.json',dict(status='only_validity_mask_changes_all15RGB_matte_K_T_identical',records=records,
    ignored_pixels=sum(r['unknown_pixels'] for r in records),top12_components=sorted(patches,key=lambda r:r['area'],reverse=True)[:12]))
for n in ['geometry-audit.json','initialization.json','train-visual-hull-init.npz']:shutil.copy2(BASE/n,OUT/n)
t=(S/'train_enerf_camera10_time.py').read_text().replace("method='gsplat1.3DefaultStrategy paired camera10 observation control with fixed target/background schedules'", "method='gsplat1.3loss validity excludes hybrid-minus-SAM mask disagreement; fixed6000schedule control'")
(S/'train_enerf_mask_uncertainty.py').write_text(t)
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
for n in ['prepare_enerf_mask_uncertainty.py','train_enerf_mask_uncertainty.py','evaluate_enerf_camera10_time.py']:shutil.copy2(S/n,snap/n)
print(json.dumps(records,indent=2))
