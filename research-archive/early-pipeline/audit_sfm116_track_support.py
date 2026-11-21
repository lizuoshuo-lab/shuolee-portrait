from pathlib import Path
import json,sqlite3,struct,hashlib,collections,datetime
import numpy as np
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/full-scene-localization-074';O=B/'current/portrait-alignment-open-source-116/pixsfm'
labels=['background','face_lower','face_upper','torso','legs_shoes','other_foreground']
with np.load(O/'database-regions.npz') as arrays:regs={k:arrays[k] for k in arrays.files}
frames={};max_diff=0
db=sqlite3.connect((S/'database.db').as_uri()+'?mode=ro',uri=True)
with (S/'refined_sparse/images.bin').open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    for _ in range(n):
        iid,*_=struct.unpack('<i7di',f.read(64));name=b''
        while True:
            b=f.read(1)
            if b==b'\x00':break
            if not b:raise ValueError('truncated name')
            name+=b
        count=struct.unpack('<Q',f.read(8))[0]
        xyid=np.frombuffer(f.read(count*24),dtype=[('xy','<f8',(2,)),('id','<i8')])
        nr,nc,blob=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(iid,)).fetchone()
        kps=np.frombuffer(blob,np.float32).reshape(nr,nc)[:,:2]
        assert nr==count and count==len(regs[str(iid)])
        max_diff=max(max_diff,float(abs(xyid['xy']-kps).max()))
        frames[iid]=int(Path(name.decode()).stem.split('_')[-1])
db.close();assert len(frames)==175 and max_diff<1e-6
points=[];summaries={l:[] for l in labels};cross=0
with (S/'refined_sparse/points3D.bin').open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    for _ in range(n):
        pid,*_=struct.unpack('<Q3d3Bd',f.read(43));nt=struct.unpack('<Q',f.read(8))[0]
        track=np.frombuffer(f.read(nt*8),dtype='<i4').reshape(nt,2)
        counts=collections.Counter(int(regs[str(i)][k]) for i,k in track)
        group,num=counts.most_common(1)[0];fraction=num/nt
        if fraction<.8:cross+=1;continue
        matching=[frames[i] for i,k in track if int(regs[str(i)][k])==group]
        rec={'id':int(pid),'total_observations':int(nt),'consistent_region_observations':num,'raw_frame_span':max(matching)-min(matching),'region_fraction':fraction}
        summaries[labels[group]].append(rec)
output={}
for label,rows in summaries.items():
    lengths=np.array([r['consistent_region_observations'] for r in rows]);spans=np.array([r['raw_frame_span'] for r in rows])
    output[label]={'tracks_majority_region_at_least_80pct':len(rows),
       'tracks_with_observations_at_least':{str(t):int((lengths>=t).sum()) for t in [2,3,5,10,20,50]},
       'median_consistent_view_count':float(np.median(lengths)) if len(rows) else None,
       'median_raw_frame_span':float(np.median(spans)) if len(rows) else None}
result={'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_model_frames':175,'source_model_points':int(n),
    'native_image_feature_index_vs_db_max_px_difference':max_diff,'region_scope':'Approximate existing mask/detection regions, not physical feature identity. An80% majority is used ONLY to summarize coverage, not to alter reconstruction or accept/reject candidate data.',
    'tracks_without_80pct_majority_region':cross,'regions':output,'optimization_started':False,'source_updated':False,
    'limitations':['Many SIFT tracks in a region do not establish that nose, mouth or glasses are each correctly supported.','Camera/geometry-dependent SfM tracks are a diagnostic description, not independent evaluation observations.','Six-view smoke graph coverage is not175-view coverage.']}
(O/'sfm-track-support.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
