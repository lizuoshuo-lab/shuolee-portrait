from pathlib import Path
import json,os,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
B=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
if os.name!='nt':B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/fullscene-person-constraint-audit-076';R=B/'current/person-static-fullscene-camera-fresh-075'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
rows=read(R/'data/training-inputs.json')['records'];old=read(B/'current/person-static-registered-camera-fresh-072/data/training-inputs.json')['records']
orig=read(B/'current/gaussianavatars-smplx-from-scratch-001/data/inputs.json')['records']
index={r['source_sha256']:j for j,r in enumerate(old)};idx=np.array([index[r['source_sha256']] for r in orig])
oldK=np.array(old[0]['K']);intr=np.array(read(B.parent/'source-resolution-rectified-002/manifest.json')['original_raw_camera']['params']);intr[:3]*=2
new=read(R/'camera-source.json');K=np.array(new['native_K']);cvK=np.array(new['opencv_output_K']);rawK=np.array(new['opencv_raw_K']);radial=new['raw_intrinsics'][3]
a=np.load(B/'current/pixel-mapping-audit-073/independent-tracks/tracks.npz')
allrows=read(O/'correspondence-pairs.json')['rows']
chosen=[]
for track in [94,100,143]:
    rr=[r for r in allrows if r['track']==track and r['admitted_patch'] and r['errors_rectified_px']['triangulated']<3]
    chosen.append(max(rr,key=lambda r:r['warp_center_error_training_px']))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf' if os.name=='nt' else '/mnt/c/Windows/Fonts/arial.ttf',17)
board=Image.new('RGB',(1200,360*len(chosen)),(248,248,248));d=ImageDraw.Draw(board)
for rownum,r in enumerate(chosen):
    obs=a['observations'][a['offsets'][r['track']]:a['offsets'][r['track']+1]]
    loc=np.flatnonzero(idx[obs[:,0].astype(int)]==r['source'])[0];uv=obs[loc,1:]
    q=(uv-oldK[:2,2])/np.diag(oldK)[:2];raw=q*(1+intr[3]*(q*q).sum())*intr[0]+intr[1:3]
    src=cv2.undistortPoints(raw[None,None].astype(float),rawK,np.array([radial,0,0,0,0.]),P=cvK)[0,0]
    target=np.array(r['observed']);W=round(1536*rows[r['target']]['wh'][0]/rows[r['target']]['wh'][1]);scale=np.array([W,1536])/rows[r['target']]['wh']
    prediction=(np.array(r['warp_center'])+.5)/scale-.5
    for col,(fi,center,label) in enumerate([(r['source'],src,f'Source raw{r["raw_source"]}'),(r['target'],target,f'Target raw{r["raw_target"]}')]):
        origin=np.maximum(0,np.rint(center-[180,140])).astype(int)
        image=Image.open(rows[fi]['image']).convert('RGB').crop((*origin,*(origin+[360,280])));draw=ImageDraw.Draw(image)
        p=center-origin;draw.ellipse((*list(p-4),*list(p+4)),outline=(0,240,255),width=2)
        if col:
            p1=prediction-origin;draw.line((*p,*p1),fill=(255,40,20),width=2)
            draw.line((p1[0]-5,p1[1],p1[0]+5,p1[1]),fill=(255,40,20),width=2)
            draw.line((p1[0],p1[1]-5,p1[0],p1[1]+5),fill=(255,40,20),width=2)
        board.paste(image,(col*380,rownum*360+50));d.text((col*380+6,rownum*360+10),label,font=font,fill='black')
    x=775;y=rownum*360+50
    for k,text in enumerate([f'Track {r["track"]}',f'Mismatch {r["warp_center_error_training_px"]:.2f} training px',f'Patch correlation {r["signed_patch_correlation"]:.3f}',f'NCC loss {r["ncc_loss"]:.3f} < 0.9',f'Round trip {r["roundtrip_training_px"]:.3f} px < 1', 'Geometry + patch gate: PASSED', 'Cyan: tracked texture', 'Red: frozen plane warp']):
        d.text((x,y+k*29),text,font=font,fill='black')
board.save(O/'patch-witnesses.jpg',quality=97)
(O/'patch-witnesses.json').write_text(json.dumps(dict(rows=chosen,description='Image-only track observations and native frozen075 warp centers; a tracked location is not3D ground truth'),indent=2),encoding='utf8')
print(O/'patch-witnesses.jpg')
