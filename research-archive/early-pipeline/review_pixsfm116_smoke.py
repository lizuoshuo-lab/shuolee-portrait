"""Read-only source-image correspondence inspection after the bounded KA smoke."""
from pathlib import Path
import collections
import datetime as dt
import json
import sqlite3
import struct
import itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/full-scene-localization-074'
P=B/'current/portrait-alignment-open-source-116/pixsfm'
R=P/'ka-vgg-smoke'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
assert read(R/'result.json')['runtime_smoke_pass']
db=sqlite3.connect((P/'smoke-input.db').as_uri()+'?mode=ro',uri=True)
names=dict(db.execute('SELECT image_id,name FROM images'))
kps={iid:np.frombuffer(blob,np.float32).reshape(n,c)[:,:2].astype(float)
     for iid,n,c,blob in db.execute('SELECT * FROM keypoints')}
delta=np.load(R/'keypoint-displacements.npz')
regions=np.load(P/'database-regions.npz')
parents={}
def root(x):
    parents.setdefault(x,x)
    while parents[x]!=x:
        parents[x]=parents[parents[x]];x=parents[x]
    return x
for pair_id,n,c,blob in db.execute('SELECT * FROM matches'):
    if not n:continue
    a,b=divmod(pair_id,2147483647)
    for i,j in np.frombuffer(blob,np.uint32).reshape(n,c):
        ra,rb=root((a,int(i))),root((b,int(j)))
        if ra!=rb:parents[rb]=ra
components=collections.defaultdict(list)
for node in parents:components[root(node)].append(node)
components=sorted((sorted(v) for v in components.values()),key=lambda v:v[0])
tracks=[]
for index,members in enumerate(components):
    if len(members)<3 or len({n[0] for n in members})!=len(members):continue
    if not any(regions[str(i)][j]==1 for i,j in members):continue
    observations=[]
    for iid,j in members:
        observations.append({'image_id':iid,'name':names[iid],'keypoint_index':j,
            'original_colmap_xy':kps[iid][j].tolist(),
            'refined_colmap_xy':(kps[iid][j]+delta[names[iid]][j]).tolist(),
            'approx_region_index':int(regions[str(iid)][j])})
    tracks.append({'component_id':index,'observations':observations})

source=read(S/'cameras.json')
inputdata=B/'current/avatar-repository-pair-114/smplx/data'
records=read(inputdata/'training-inputs.json')['records']
det=np.load(inputdata/'observations.npz')['keypoints']
f,cx,cy,k=source['intrinsics']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
thumbs=[]
for track in tracks:
    for ob in track['observations']:
        iid=ob['image_id'];frame=int(Path(ob['name']).stem.split('_')[-1])
        K=np.array(records[frame-1]['K'])
        face=det[frame-1,23:91]+.5
        normal=(face-[K[0,2],K[1,2]])/[K[0,0],K[1,1]]
        raw=normal*(1+k*(normal*normal).sum(1,keepdims=True))*f+[cx,cy]-.5
        center=(raw.min(0)+raw.max(0))/2
        width=max(300,float(np.ptp(raw[:,0]))*1.6)
        height=max(330,float(np.ptp(raw[:,1]))*1.5)
        box=np.array([center[0]-width/2,center[1]-height/2,
                      center[0]+width/2,center[1]+height/2]).round().astype(int)
        with Image.open(S/'images'/ob['name']) as img:
            crop=img.convert('RGB').crop(tuple(box))
        # Both panels share precisely the same crop and one uniform scale.
        scale=min(420/crop.width,420/crop.height)
        w,h=round(crop.width*scale),round(crop.height*scale)
        crop=crop.resize((w,h),Image.Resampling.LANCZOS)
        panel=Image.new('RGB',(860,480),'white');d=ImageDraw.Draw(panel)
        panel.paste(crop,((420-w)//2,45));panel.paste(crop,(440+(420-w)//2,45))
        d.text((12,10),f'Track {track["component_id"]} | {ob["name"]} | original',fill='black',font=font)
        d.text((452,10),'Original / refined feature location',fill='black',font=font)
        for xy,color in [(ob['original_colmap_xy'],'#006cb6'),(ob['refined_colmap_xy'],'#ed6b00')]:
            xy=(np.array(xy)-.5-box[:2])*scale+[440+(420-w)//2,45]
            x,y=xy;d.ellipse((x-7,y-7,x+7,y+7),outline=color,width=2)
            d.line((x-11,y,x+11,y),fill=color,width=1)
            d.line((x,y-11,x,y+11),fill=color,width=1)
        d.text((12,460),'Same source pixels; no warp. Blue: SIFT. Orange: VGG KA. Circle radius is for visibility.',fill='#444444',font=small)
        thumbs.append(panel)
        ob['review_crop_raw_xyxy']=box.tolist();ob['review_uniform_scale']=scale
sheet=Image.new('RGB',(860,480*len(thumbs)),'white')
for n,panel in enumerate(thumbs):sheet.paste(panel,(0,480*n))
sheet.save(R/'lower-face-three-view-track.jpg',quality=95)
point_ids={}
with (S/'refined_sparse/images.bin').open('rb') as fbin:
    for _ in range(struct.unpack('<Q',fbin.read(8))[0]):
        iid,*_=struct.unpack('<i7di',fbin.read(64))
        while fbin.read(1)!=b'\0':pass
        count=struct.unpack('<Q',fbin.read(8))[0]
        rawpoints=np.frombuffer(fbin.read(count*24),dtype=[('xy','<f8',(2,)),('pid','<i8')])
        if iid in names:point_ids[iid]=rawpoints['pid'].copy()
context_panels=[]
for track in tracks:
    track['source_geometric_edges']=[]
    for ob in track['observations']:
        ob['source074_point3D_id']=int(point_ids[ob['image_id']][ob['keypoint_index']])
        original=Image.open(S/'images'/ob['name']).convert('RGB')
        panel=Image.new('RGB',(1040,740),'white');d=ImageDraw.Draw(panel)
        d.text((12,12),f'Track {track["component_id"]} | {ob["name"]}',fill='black',font=font)
        base=original.resize((380,676),Image.Resampling.LANCZOS)
        panel.paste(base,(12,40))
        xy=np.array(ob['original_colmap_xy'])-.5
        x,y=xy*.25+[12,40]
        d.ellipse((x-9,y-9,x+9,y+9),outline='#006cb6',width=3)
        box=np.round([xy[0]-100,xy[1]-100,xy[0]+100,xy[1]+100]).astype(int)
        crop=original.crop(tuple(box)).resize((500,500),Image.Resampling.LANCZOS)
        panel.paste(crop,(470,100))
        for pos,color in [(ob['original_colmap_xy'],'#006cb6'),(ob['refined_colmap_xy'],'#ed6b00')]:
            x,y=(np.array(pos)-.5-box[:2])*2.5+[470,100]
            d.ellipse((x-10,y-10,x+10,y+10),outline=color,width=2)
            d.line((x-14,y,x+14,y),fill=color,width=1)
            d.line((x,y-14,x,y+14),fill=color,width=1)
        d.text((470,55),'200 x 200 source-pixel crop; uniform 2.5x scale',fill='black',font=font)
        d.text((470,635),'Blue: original. Orange: feature refinement.',fill='#444444',font=font)
        d.text((470,670),f'074 reconstructed point ID: {ob["source074_point3D_id"]}',fill='#444444',font=font)
        context_panels.append(panel)
    for left,right in itertools.combinations(track['observations'],2):
        ia,ib=left['image_id'],right['image_id'];ka,kb=left['keypoint_index'],right['keypoint_index']
        pair_id=ia*2147483647+ib
        counts={}
        for table in ('matches','two_view_geometries'):
            row=db.execute(f'SELECT rows,cols,data FROM {table} WHERE pair_id=?',(pair_id,)).fetchone()
            arr=np.frombuffer(row[2],np.uint32).reshape(row[0],row[1]) if row and row[2] else np.empty((0,2),np.uint32)
            counts[table]=bool(np.any(np.all(arr==[ka,kb],axis=1)))
        track['source_geometric_edges'].append({'left_frame':ia,'right_frame':ib,**counts})
    ids=[ob['source074_point3D_id'] for ob in track['observations']]
    track['all_observations_share_one_original074_point3D']=len(set(ids))==1 and ids[0]>=0
context=Image.new('RGB',(1040,740*len(context_panels)),'white')
for i,panel in enumerate(context_panels):context.paste(panel,(0,740*i))
context.save(R/'lower-face-track-full-context.jpg',quality=95)
result={'created':dt.datetime.now(dt.timezone.utc).isoformat(),
        'selection':'All unambiguous >=3-view raw graph components touching approximate lower-face region; no filtering by optimized error.',
        'tracks':tracks,'preview':'lower-face-track-full-context.jpg',
        'coordinate_convention':'COLMAP edge coordinates minus0.5 for source-image pixel-center overlay.',
        'human_material_identity_review':'pending',
        'warning':'Approximate region labels and same graph component are not a proof of physical feature identity.'}
(R/'material-track-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
