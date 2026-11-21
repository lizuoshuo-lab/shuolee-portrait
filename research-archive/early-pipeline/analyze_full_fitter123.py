"""CPU inspection of the completed synthetic integration, without another fit."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/opt/gs-slam-lab/shuolee-adapter-a0c-synthetic')
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/generic-2d3d-research/adapter-a0c')
ASSET=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
result=read(O/'full-fit-result.json');assert result['state']=='synthetic_full_fit_passed'
rows=read(ROOT/'data/inputs.json')['records']
with np.load(ROOT/'fit/initialization.npz') as z:initial=z['vertices_world'].astype(np.float64)
with np.load(ROOT/'fit/fitted.npz') as z:fitted={k:z[k].copy() for k in z.files}
with np.load(ROOT/'data/observations.npz') as z:target=z['keypoints'][:,40:91]
with np.load(ASSET,allow_pickle=True) as z:
    faces=z['f'];indices=z['lmk_faces_idx'];weights=z['lmk_bary_coords']
tri=faces[indices]
def anchors(v):return np.einsum('nlkc,lk->nlc',v[:,tri],weights)
def project(v):
    result=[]
    for i,row in enumerate(rows):
        E=np.asarray(row['w2c']);K=np.asarray(row['K_integer_centers'])
        cam=np.asarray(v[i],dtype=np.float64)@E[:3,:3].T+E[:3,3]
        pix=cam@K.T;result.append(pix[:,:2]/pix[:,2:])
    return np.stack(result)
before=project(anchors(initial));after=project(anchors(fitted['vertices_world']))
parity=dict(face=float(np.abs(after-fitted['projected_face51']).max()),
    body=float(np.abs(project(fitted['coco17_world'])-fitted['projected_coco17']).max()),
    joints=float(np.abs(project(fitted['joints_world'])-fitted['projected_joints']).max()))
assert max(parity.values())<.001,parity
np.savez_compressed(O/'synthetic-projection-evidence.npz',frame_ids=[r['frame'] for r in rows],target_face=target,
    initial_face=before,fitted_face=after,K_integer_centers=[r['K_integer_centers'] for r in rows],w2c=[r['w2c'] for r in rows])
canvas=Image.new('RGB',(576,384),(245,245,245));crops=[]
for col,i in enumerate([0,2,4]):
    c=target[i].mean(0);x,y=int(round(c[0]))-48,int(round(c[1]))-48
    crops.append(dict(frame=rows[i]['frame'],native_crop_xywh=[x,y,96,96]))
    source=Image.open(rows[i]['image']).convert('RGB').crop((x,y,x+96,y+96)).resize((192,192),Image.Resampling.NEAREST)
    for r,(points,color) in enumerate([(before[i],(220,65,70)),(after[i],(40,130,255))]):
        panel=source.copy();draw=ImageDraw.Draw(panel)
        for u,v in target[i]:
            a,b=(u-x+.5)*2-.5,(v-y+.5)*2-.5;draw.ellipse((a-1.3,b-1.3,a+1.3,b+1.3),fill=(20,220,110))
        for u,v in points:
            a,b=(u-x+.5)*2-.5,(v-y+.5)*2-.5;draw.ellipse((a-2,b-2,a+2,b+2),outline=color,width=1)
        canvas.paste(panel,(col*192,r*192))
canvas.save(O/'synthetic_3views_initial_red_final_blue_target_green_2x.png')
for name in ['protocol.json','history.json','result.json']:
    shutil.copy2(ROOT/'fit'/name,O/('actual-fitter-'+name))
save(O/'independent-projection-check.json',dict(state='passed',method='Independent NumPy float64 matrix projection from saved world geometry and cameras',
    parity_max_px=parity,tolerance_px=.001,crops=crops,display_uniform_scale=2,
    no_real_images=True,measurement_coordinates_native=True,no_extra_fit_or_GPU=True))
print(json.dumps(dict(parity_max_px=parity,figure=str(O/'synthetic_3views_initial_red_final_blue_target_green_2x.png'))))
