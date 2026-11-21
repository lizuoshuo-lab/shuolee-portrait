from pathlib import Path
import json, hashlib, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-shared-face-fresh-098';O=B/'current/shared-point-ray-consistency-099';J=O/'joint-camera-points'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def path(p):return Path(re.sub(r'^/mnt/c/','C:/',str(p)))
rows=read(R/'data/training-inputs.json')['records'];a=np.load(J/'input.npz')
arms={name:np.load(J/(name+'.npz')) for name in ['scene_bundle','scene_named_bundle']}
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
def marks(im,uv,pr,origin=(0,0),scale=1):
    d=ImageDraw.Draw(im)
    for x,y in zip(uv,pr):
        p=(x-np.asarray(origin))*scale;q=(y-np.asarray(origin))*scale
        d.line((*p,*q),fill='#eaba24',width=2)
        d.ellipse((*list(p-2),*list(p+2)),outline='#00a06b',width=2)
        d.line((q[0]-3,q[1],q[0]+3,q[1]),fill='#d600ad',width=2)
        d.line((q[0],q[1]-3,q[0],q[1]+3),fill='#d600ad',width=2)
    return im
records=[]
for frame in [12,37,62,87,112,137,162]:
    ix=(a['face_observations'][:,0]==frame-1)&a['face_held'];uv=a['face_observations'][ix,1:]
    assert len(uv)>0
    origin=np.rint(np.median(uv,axis=0)-[210,210]).astype(int)
    photo=Image.open(path(rows[frame-1]['image'])).convert('RGB').crop((*origin,*(origin+[420,440])))
    board=Image.new('RGB',(1260,525),'white');d=ImageDraw.Draw(board)
    titles=['实拍 '+f'{frame:06d}','整图特征优化相机','加入同名五官共享三维点']
    panels=[photo,marks(photo.copy(),uv,arms['scene_bundle']['face_prediction'][ix],origin),
            marks(photo.copy(),uv,arms['scene_named_bundle']['face_prediction'][ix],origin)]
    for col,(title,panel) in enumerate(zip(titles,panels)):
        board.paste(panel,(col*420,64));d.text((col*420+10,8),title,font=font,fill='#132b3c')
    d.text((10,35),'绿圈：检测位置   紫叉：唯一三维点的相机投影   黄线：偏差；本帧五官观测未参与拟合',font=small,fill='#334455')
    dest=O/f'held-face-{frame:06d}.jpg';board.save(dest,quality=96)
    records.append(dict(frame=frame,held=True,observations=int(ix.sum()),image=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                        median_px={name:float(np.median(np.linalg.norm(data['face_prediction'][ix]-uv,axis=1))) for name,data in arms.items()}))
# Independent body texture observations at the same fixed example frame.
frame=137;indices=np.flatnonzero((a['observations'][:,0]==frame-1)&a['held']&(a['groups']>=3))
# Deterministic thinning for legibility only, never used in quantitative metrics.
indices=indices[::max(1,int(np.ceil(len(indices)/80)))];uv=a['observations'][indices,1:]
raw=Image.open(path(rows[frame-1]['image'])).convert('RGB');factor=800/raw.height;W=round(raw.width*factor)
board=Image.new('RGB',(W*3,872),'white');d=ImageDraw.Draw(board)
titles=['实拍 000137','整图特征优化相机','加入同名五官共享三维点']
for col,(name,title) in enumerate(zip([None,'scene_bundle','scene_named_bundle'],titles)):
    im=raw.resize((W,800),Image.Resampling.LANCZOS)
    if name:marks(im,uv,arms[name]['scene_prediction'][indices],scale=factor)
    board.paste(im,(col*W,65));d.text((col*W+8,8),title,font=font,fill='#132b3c')
d.text((8,36),'同一张照片，另一组人物纹理观测；显示抽样不改变全部观测的误差统计',font=small,fill='#334455')
dest=O/'held-body-000137.jpg';board.save(dest,quality=96)
result=read(J/'result.json');result['overlay_files']=records;result['body_overlay']=str(dest)
(J/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(face=records,body=str(dest)),ensure_ascii=False))
