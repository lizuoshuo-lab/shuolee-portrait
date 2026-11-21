"""Read-only 074 source graph audit and a copied six-view PixSfM smoke database."""
from pathlib import Path
import json, sqlite3, hashlib, collections, itertools, datetime, sys
import numpy as np
from PIL import Image

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/full-scene-localization-074';P=B/'current/avatar-repository-pair-114/smplx/data'
O=B/'current/portrait-alignment-open-source-116/pixsfm';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(name,d):
    p=O/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def loc(p):
    s=str(p)
    if s.startswith('/mnt/c/'):s='C:/'+s[7:]
    return Path(s)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
srcdb=S/'database.db';before=sha(srcdb)
db=sqlite3.connect(srcdb.as_uri()+'?mode=ro',uri=True)
source=read(S/'cameras.json');records=read(P/'training-inputs.json')['records'];obs=np.load(P/'observations.npz')['keypoints']
assert len(records)==len(source['records'])==175
smoke=read(O/'protocol.json')['initial_smoke_frame_indices']
f,cx,cy,k=source['intrinsics'];K=np.array(records[0]['K']);Kinv=np.linalg.inv(K)
imrows=db.execute('SELECT image_id,name,camera_id FROM images ORDER BY image_id').fetchall()
byid={r[0]:r[1] for r in imrows};ids={r[1]:r[0] for r in imrows}
id2raw={r[0]:int(Path(r[1]).stem.split('_')[-1]) for r in imrows}
kpmap={};regmap={};results=[];max_roundtrip=0
labels=['background','face_lower','face_upper','torso','legs_shoes','other_foreground']
for iid,name,_ in imrows:
    raw=id2raw[iid];row=records[raw-1];assert row['source_frame_index']==raw
    n,d,blob=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(iid,)).fetchone()
    kp=np.frombuffer(blob,np.float32).reshape(n,d)[:,:2].astype(np.float64);kpmap[iid]=kp
    nd=(kp-[cx,cy])/f;nu=nd.copy()
    for _ in range(12):nu=nd/(1+k*(nu*nu).sum(1,keepdims=True))
    reconstructed=f*nu*(1+k*(nu*nu).sum(1,keepdims=True))+[cx,cy]
    max_roundtrip=max(max_roundtrip,float(np.max(abs(reconstructed-kp))))
    rect=nu*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]-.5
    mask=np.array(Image.open(loc(row['mask'])).convert('L'))>127
    px=np.rint(rect).astype(int);valid=(px[:,0]>=0)&(px[:,0]<mask.shape[1])&(px[:,1]>=0)&(px[:,1]<mask.shape[0])
    fg=np.zeros(n,bool);fg[valid]=mask[px[valid,1],px[valid,0]]
    face=obs[raw-1,23:91];low=face.min(0);high=face.max(0)
    fr=(rect[:,0]>=low[0]-5)&(rect[:,0]<=high[0]+5)&(rect[:,1]>=low[1]-5)&(rect[:,1]<=high[1]+5)&fg
    split=float(np.median(face[27:36,1]))
    yy=np.flatnonzero(mask.any(1));top=float(yy[0]);height=float(yy[-1]-yy[0]+1)
    reg=np.full(n,5,np.int8);reg[~fg]=0
    reg[fg&(rect[:,1]>=top+.23*height)&(rect[:,1]<top+.64*height)]=3
    reg[fg&(rect[:,1]>=top+.64*height)]=4
    reg[fr&(rect[:,1]>=split)]=1;reg[fr&(rect[:,1]<split)]=2;regmap[iid]=reg
    results.append({'raw_frame':raw,'keypoints':n,'regions':{l:int((reg==j).sum()) for j,l in enumerate(labels)}})
assert max_roundtrip<1e-7
if '--regions-only' in sys.argv:
    np.savez_compressed(O/'database-regions.npz',**{str(k):v for k,v in regmap.items()})
    assert sha(srcdb)==before
    print(json.dumps({'region_labels_saved':True,'frames':175,'source_unchanged':True}));sys.exit(0)
chosen=[ids[f'raw_{r:06d}.png'] for r in smoke]
smoke_db=O/'smoke-input.db'
assert not smoke_db.exists(),'Never replace a prepared experiment input.'
dst=sqlite3.connect(smoke_db)
for typ,name,sql in db.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL AND type='table'"):
    if name.startswith('sqlite_'):continue
    dst.execute(sql)
def copy_where(table,where,vals):
    rows=db.execute(f'SELECT * FROM {table} WHERE {where}',vals).fetchall()
    if rows:dst.executemany(f'INSERT INTO {table} VALUES('+','.join('?' for _ in rows[0])+')',rows)
    return rows
copy_where('cameras','camera_id IN ('+','.join('?' for _ in set(r[2] for r in imrows))+')',tuple(set(r[2] for r in imrows)))
for iid in chosen:
    for table in ['images','keypoints','descriptors']:copy_where(table,'image_id=?',(iid,))
pair_records=[];maxid=2147483647
parent={};members={}
def root(n):
    parent.setdefault(n,n)
    while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
    return n
def union(a,b):
    a=root(a);b=root(b)
    if a!=b:parent[b]=a
for a,b in itertools.combinations(sorted(chosen),2):
    pair=a*maxid+b
    rows=copy_where('matches','pair_id=?',(pair,));geo=copy_where('two_view_geometries','pair_id=?',(pair,))
    count={}
    for label,rows2 in [('tentative',rows),('geometrically_verified',geo)]:
        if not rows2 or rows2[0][3] is None:count[label]={'all':0};continue
        n,cols,blob=rows2[0][1:4];m=np.frombuffer(blob,np.uint32).reshape(n,cols)
        rr1,rr2=regmap[a][m[:,0]],regmap[b][m[:,1]]
        face1=np.isin(rr1,[1,2]);face2=np.isin(rr2,[1,2])
        count[label]={'all':int(n),'both_same_region':{l:int(((rr1==j)&(rr2==j)).sum()) for j,l in enumerate(labels)},
                      'face_region_to_different_region':int(np.count_nonzero(face1!=face2))}
        if label=='tentative':
            for x,y in m:union((a,int(x)),(b,int(y)))
    pair_records.append({'raw_frames':[id2raw[a],id2raw[b]],'counts':count})
dst.commit();dst.close();db.close()
components=collections.defaultdict(list)
for n in parent:components[root(n)].append(n)
ambiguous=sum(len({n[0] for n in nodes})!=len(nodes) for nodes in components.values())
support=collections.Counter();views=collections.Counter()
for nodes in components.values():
    v=len({i for i,_ in nodes});views[v]+=1
    if v>=3 and v==len(nodes):
        regs={int(regmap[i][j]) for i,j in nodes}
        for j in regs:support[labels[j]]+=1
after=sha(srcdb);assert after==before
save('graph-preflight.json',{'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_database_sha256':before,'source_database_unchanged':True,
 'source_frames':175,'smoke_frames':smoke,'smoke_database':str(smoke_db),'smoke_database_bytes':smoke_db.stat().st_size,'smoke_database_sha256':sha(smoke_db),
 'raw_to_rectified_inverse_roundtrip_max_pixels':max_roundtrip,'coordinate_scope':'Region classification only: COLMAP keypoints use edge coordinates; inverse SIMPLE_RADIAL then rectified K then minus0.5 for pixel-center observations. PixSfM inputs retain original DB coordinates.',
 'region_scope':'Face box from existing2D detections, foreground from existing mask, torso/legs from mask height bands; approximate coverage labels, NOT correspondence identity or geometry truth.',
 'all175_keypoint_counts':{l:sum(x['regions'][l] for x in results) for l in labels},
 'smoke_components':len(components),'components_with_same_image_conflicts':ambiguous,
 'smoke_component_view_histogram':dict(views),'unambiguous_3plus_view_components_touching_region':dict(support),
 'pair_records':pair_records,'training_started':False,'refinement_started':False,'acceptance_observations_changed':False})
save('graph-frame-counts.json',results)
print(json.dumps({'source_unchanged':True,'smoke_database_bytes':smoke_db.stat().st_size,'smoke_components':len(components),'conflicting_components':ambiguous,'three_plus_view_support':dict(support),'roundtrip':max_roundtrip},ensure_ascii=False))
