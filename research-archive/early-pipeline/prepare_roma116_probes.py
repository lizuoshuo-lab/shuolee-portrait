"""Freeze image-only P2 probes and the smoke protocol before any neural inference."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import cv2
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/full-scene-localization-074';P=B/'current/avatar-repository-pair-114/smplx/data'
O=B/'current/portrait-alignment-open-source-116';R=O/'roma';D=O/'pixsfm/independent-validation'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def loc(p):
    s=str(p).replace('\\','/');return Path('/mnt/'+s[0].lower()+'/'+s[3:]) if len(s)>1 and s[1]==':' else Path(s)
assert not (R/'probes.npz').exists(),'Do not regenerate frozen probes after inspecting inference'
assert read(R/'coordinate-check.json')['imports_passed']
source=read(S/'cameras.json');intr=np.array(source['intrinsics']);rows=read(P/'training-inputs.json')['records'];K=np.array(rows[0]['K'])
assert [r['source_frame_index'] for r in rows]==list(range(1,176))
with np.load(D/'tracks.npz') as z:validation=z['observations'].copy()
face=np.load(P/'observations.npz')['keypoints'][:,23:91]
yy,xx=np.mgrid[:2704,:1520];n=(np.stack([xx,yy],-1).astype(np.float64)+.5-intr[1:3])/intr[0];u=n.copy()
for _ in range(12):u=n/(1+intr[3]*(u*u).sum(-1,keepdims=True))
rect=u*[K[0,0],K[1,1]]+K[:2,2]-.5
mx=rect[:,:,0].astype(np.float32);my=rect[:,:,1].astype(np.float32)
del n,u,rect,xx,yy
triples=[[13,25,38],[38,50,63],[63,75,88],[88,100,113],[113,125,138],[138,150,163]]
labels=['background','face_lower','face_upper','torso','legs_shoes'];all_probes=[];coverage=[];hashes={};frames=sorted(set(sum(triples,[])));bbox={}
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',17)
for frame in frames:
    path=S/'images'/f'raw_{frame:06d}.png';raw=cv2.imread(str(path));assert raw.shape==(2704,1520,3);hashes[str(path)]=sha(path)
    f=frame-1;mask=cv2.imread(str(loc(rows[f]['mask'])),0);assert mask.shape==(2694,1514)
    fg=cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>127
    inside=cv2.erode(fg.astype(np.uint8),np.ones((13,13),np.uint8))>0
    outside=cv2.erode((~fg).astype(np.uint8),np.ones((21,21),np.uint8))>0
    ys=np.where(mask>127)[0];top,ht=ys.min(),np.ptp(ys)+1
    region=np.full(fg.shape,-1,np.int8);region[outside]=0
    region[inside&(my>=top+.23*ht)&(my<top+.64*ht)]=3;region[inside&(my>=top+.64*ht)]=4
    lo=face[f].min(0);hi=face[f].max(0);split=np.median(face[f,27:36,1])
    facial=inside&(mx>=lo[0]-5)&(mx<=hi[0]+5)&(my>=lo[1]-5)&(my<=hi[1]+5)
    region[facial&(my>=split)]=1;region[facial&(my<split)]=2
    fy,fx=np.where(facial);bbox[frame]=[int(fx.min()),int(fy.min()),int(fx.max()),int(fy.max())]
    if frame not in [t[0] for t in triples]:continue
    gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
    reserved=validation[validation[:,0]==f,1:]
    tree=cKDTree(reserved) if len(reserved) else None
    for group in range(5):
        pts=cv2.goodFeaturesToTrack(gray,maxCorners=80,qualityLevel=.01,minDistance=12,mask=(region==group).astype(np.uint8)*255,blockSize=5)
        pts=np.empty((0,2)) if pts is None else pts[:,0]
        if tree is not None:pts=pts[tree.query(pts)[0]>12]
        # Selection is fixed before RoMa confidence, camera errors or rendered outputs exist.
        # Hash rank avoids hand-picking the easiest corners; no geometry is involved.
        order=sorted(range(len(pts)),key=lambda j:hashlib.sha256(f'{frame}:{group}:{pts[j,0]:.4f}:{pts[j,1]:.4f}'.encode()).digest())
        pts=pts[order[:24 if group==0 else 16]]
        all_probes.extend([[frame,group,*p] for p in pts]);coverage.append(dict(raw_frame=frame,region=labels[group],probes=len(pts)))
probes=np.asarray(all_probes,dtype=np.float64);np.savez_compressed(R/'probes.npz',probes=probes,triples=np.array(triples),frame_ids=np.array(frames))
reg=dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),candidate='P2',stage='fixed_camera_material_correspondence_smoke',main_variable='RoMa learned correspondence on original full images; cameras unchanged',
    triples_raw_frame_ids=triples,all_frame_identity_contract='175 unchanged source074 cameras; this smoke uses13 raw images and18 pair inferences, not a175-frame result',
    probes=dict(count=len(probes),coordinate_origin='raw raster centers',selection='GFTT texture corners inside eroded image-only regions, deterministic hash rank; exclude12px around frozen P1 validation observations at seed',coverage=coverage,sha256=sha(R/'probes.npz')),
    official_model=dict(commit='77f8d68803526dcddfd9b7a46bc76125bdc25f15',entry='roma_outdoor',weights='roma_outdoor.pth',backbone='dinov2_vitl14_pretrain.pth',coarse_resolution_hw=[798,448],upsample_resolution_hw=[1344,756],resolution_reason='Retain full portrait image aspect ratio approximately; modest coarse token count and <=1.02M upsampled pixels for16GB GPU',symmetric=True,use_custom_corr=False,do_compile=False,amp_dtype='float16',matmul_precision='highest'),
    confidence_threshold=.95,forward_backward_max_raw_px=4.,three_view_cycle_max_raw_px=6.,threshold_scope='Only a material-match screening test, not acceptance of camera or reconstruction accuracy',
    required_support=dict(lower_face_total=20,lower_face_distinct_triples=4,upper_face_total=20,torso_total=40,legs_total=40,background_total=50),
    visual_review='Fixed first/middle/last hash-ranked probe in each region of every triple, including failures; inspect original same-size unwarped crops for material identity and occlusion. Confidence/cycle consistency alone never proves physical identity.',
    camera_check='With frozen074, triangulate from A and C only, predict B without optimizing toB; separately report image-only cycle, raw epipolar distance, positive depth, held projection for all valid observations. Do not drop high camera residuals.',
    outcome='If coverage/visual identity fails or RoMa precision is inadequate, do not build a camera candidate; review limitations and move toP3 under bounded protocol. If passes, preregister full175 camera BA candidate with immutable frozen validation before running.',
    limitations=['Sequential video cannot by itself establish a synchronous static subject','Named face boxes only define approximate regions, not material-point truth','Higher learned matching confidence is not geometric GT','Unknown capture metadata remains unresolved'],
    frozen_validation_sha256=sha(D/'tracks.npz'),input_hashes=hashes,budget_seconds=1800,max_gpu_jobs=1,seed=116,
    stop_on=['Changed original or bank hash','NaN, missing weight, invalid image identity or output shape','GPU out of memory: record failure, do not auto sweep resolution','Budget exceeded'],
    training_started=False,camera_optimization_started=False)
save(R/'smoke-preregistration.json',reg)
# Real-image contact sheets selected by the fixed temporal triples, not an inferred outcome.
for page,subset in enumerate([frames[:7],frames[7:]],1):
    sheet=Image.new('RGB',(210*len(subset),620),'white');draw=ImageDraw.Draw(sheet)
    for col,frame in enumerate(subset):
        im=Image.open(S/'images'/f'raw_{frame:06d}.png').convert('RGB');thumb=im.copy();thumb.thumbnail((200,356));sheet.paste(thumb,(col*210+5,25))
        draw.text((col*210+5,4),f'Raw {frame:06d}',fill='#173342',font=font)
        lo=np.array(bbox[frame]);cx=(lo[0]+lo[2])/2;cy=(lo[1]+lo[3])/2;size=max(lo[2]-lo[0],lo[3]-lo[1])+40
        crop=im.crop((int(cx-size/2),int(cy-size/2),int(cx+size/2),int(cy+size/2)));crop.thumbnail((200,200));sheet.paste(crop,(col*210+5,390))
        draw.text((col*210+5,595),'Original face crop',fill='#173342',font=font)
    sheet.save(R/f'planned-original-views-{page}.jpg',quality=92)
print(json.dumps(dict(state='fixed_inputs_registered_no_inference',frames=len(frames),pairs=18,probes=len(probes),coverage=coverage)))
