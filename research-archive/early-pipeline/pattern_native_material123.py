"""One preregistered native SIFT material-support inventory; no cameras or fitting."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,time,sys,traceback,signal,fcntl
import numpy as np

T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PRE=REC/'pattern-preflight';O=PRE/'native-material'
ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/preflight/pattern')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def identity(nodes):return hashlib.sha256(json.dumps(nodes,separators=(',',':')).encode()).hexdigest()

class Union:
    def __init__(self,n):self.parent=np.arange(n);self.size=np.ones(n,dtype=np.int64)
    def find(self,a):
        while a!=self.parent[a]:self.parent[a]=self.parent[self.parent[a]];a=int(self.parent[a])
        return int(a)
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.size[a]<self.size[b] or (self.size[a]==self.size[b] and a>b):a,b=b,a
        self.parent[b]=a;self.size[a]+=self.size[b]
    def labels(self):return np.array([self.find(i) for i in range(len(self.parent))],np.int64)

def controls():
    # Duplicate family must be split as one unit even when its appearance tracks differ.
    tracks=Union(8)
    for edge in [(0,2),(2,4),(1,3),(3,5),(6,7)]:tracks.join(*edge)
    a=tracks.labels();assert a[0]==a[4] and a[1]==a[5] and a[0]!=a[1]
    families=Union(8)
    for i,r in enumerate(a):families.join(i,int(r))
    families.join(0,1);b=families.labels();assert b[0]==b[5] and b[0]!=b[6]
    nodes=sorted([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)])
    assert identity(nodes)==identity(sorted(reversed(nodes)))
    # A bad link placing two distinct observations in one frame is ambiguous, not repaired.
    assert len({i for i,j in [(0,0),(1,0),(0,3)]})!=3
    return dict(whole_family_split_control=True,deterministic_identity=True,ambiguous_component_rejection=True,
                physical_material_truth=False,synthetic_only=True)

def main():
    import cv2
    from scipy.spatial import cKDTree
    assert not O.exists(),'Existing inventory: inspect result/progress; do not repeat.'
    plan=read(PRE/'native-material-plan.json');obs=read(PRE/'coverage/observations.json')
    assert sha(PRE/'decoded.json')==plan['decoded_manifest_sha256']
    assert sha(ROOT/'decoded.json')==plan['decoded_manifest_sha256']
    assert sha(PRE/'coverage/observations.json')==plan['ROI_observations_sha256']
    decoded=read(ROOT/'decoded.json');rows=obs['rows'];n=len(rows)
    assert n==99==len(decoded['records']) and decoded['video_sha256']==plan['source_sha256']
    lock=open('/tmp/shuolee-pattern-material123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    O.mkdir();(O/'features').mkdir();(O/'pairs').mkdir()
    start=time.monotonic()
    def limit(signum,frame):raise TimeoutError('Preregistered 1200second material inventory budget exhausted')
    signal.signal(signal.SIGALRM,limit);signal.alarm(1200)
    cv2.setNumThreads(4);cv2.setRNGSeed(12304)
    save(O/'execution-plan.json',dict(registered=now(),parent_sha256=sha(PRE/'native-material-plan.json'),
        source_sha256=sha(Path(__file__)),opencv_version=cv2.__version__,stage='native_material_only',
        feature_extraction='Each head/person integer ROI and fullframe is detected once. nfeatures=0 then deterministic1.5pxdedup and reserved top budgets; no unused quota redistribution.',
        graph='Mutual ratio0.75 edges on all464temporalpairs. All kNN distances and accepted edges saved. Components with repeatedframe are quarantined;3pxsameframe duplicate-family closure before20percentHELD assignment.',
        evaluation='No K, extrinsic, keypoint semantics or residual involved. Physical identity remains unverified. No subpixel claim.',
        budget=plan['budget']))
    save(O/'controls.json',controls())
    def progress(stage,**kwargs):
        save(O/'progress.json',dict(updated=now(),state=stage,pid=os.getpid(),command=sys.argv,seconds=time.monotonic()-start,training=False,**kwargs))
    progress('extracting',done=0,total=n)
    sift=cv2.SIFT_create(nfeatures=0,**plan['method']['detector_parameters'])
    all_xy=[];all_desc=[];all_region=[];offset=[0];manifest=[]
    for slot,(row,dec) in enumerate(zip(rows,decoded['records'])):
        assert row['chosen'] is not None and row['source_sha256']==dec['source_sha256']
        p=Path(row['image']);assert sha(p)==dec['source_sha256']
        gray=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE);assert gray.shape==(1520,2704)
        bounds=[]
        for box in [row['face_box'],row['body_box'],[0,0,2704,1520]]:
            box=np.array(box,float);assert box.shape==(4,) and np.isfinite(box).all()
            x0,y0=np.maximum(np.floor(box[:2]).astype(int),0);x1,y1=np.minimum(np.ceil(box[2:]).astype(int),[2704,1520])
            assert x1>x0 and y1>y0
            bounds.append([int(x0),int(y0),int(x1),int(y1)])
        selected=[];descs=[];cells={};duplicates=[];per_region=[]
        for region,(box,budget) in enumerate(zip(bounds,[1000,1500,2500])):
            x0,y0,x1,y1=box;keys,desc=sift.detectAndCompute(gray[y0:y1,x0:x1],None)
            order=sorted(range(len(keys)),key=lambda j:(-keys[j].response,keys[j].pt[0],keys[j].pt[1],keys[j].size,keys[j].angle,keys[j].octave,j))
            accepted=0
            for j in order:
                key=keys[j];xy=np.array(key.pt,dtype=np.float64)+[x0,y0]
                assert np.isfinite(xy).all() and 0<=xy[0]<2704 and 0<=xy[1]<1520
                cell=tuple(np.floor(xy/1.5).astype(int));neighbours=[i for dx in [-1,0,1] for dy in [-1,0,1] for i in cells.get((cell[0]+dx,cell[1]+dy),[])]
                near=[i for i in neighbours if np.linalg.norm(xy-selected[i]['xy'])<1.5]
                if near:
                    duplicates.append([region,j,float(xy[0]),float(xy[1]),min(near)])
                    continue
                selected.append(dict(xy=xy.tolist(),region=region,local_xy=list(key.pt),size=key.size,angle=key.angle,response=key.response,octave=key.octave))
                descs.append(desc[j]);cells.setdefault(cell,[]).append(len(selected)-1);accepted+=1
                if accepted==budget:break
            per_region.append(dict(region=['headROI','personROI','global'][region],bounds=box,detected=len(keys),kept=accepted,budget=budget))
        xy=np.array([x['xy'] for x in selected],np.float64);ds=np.asarray(descs,np.float32)
        region=np.array([x['region'] for x in selected],np.int8)
        assert 0<len(xy)<=5000 and ds.shape==(len(xy),128)
        f=O/'features'/f'{slot:03d}.npz'
        np.savez_compressed(f,xy=xy,descriptors=ds,region=region,local_xy=np.array([x['local_xy'] for x in selected]),
            size=np.array([x['size'] for x in selected]),angle=np.array([x['angle'] for x in selected]),response=np.array([x['response'] for x in selected]),
            octave=np.array([x['octave'] for x in selected]),crop_bounds=np.array(bounds),duplicate_candidates=np.array(duplicates).reshape(-1,5))
        all_xy.append(xy);all_desc.append(ds);all_region.append(region);offset.append(offset[-1]+len(xy))
        manifest.append(dict(slot=slot,source_frame=row['source_frame_index'],image=str(p),source_sha256=row['source_sha256'],feature_file=f.name,sha256=sha(f),regions=per_region,total=len(xy)))
        save(O/'feature-manifest.json',dict(frames=manifest,native_coordinates=True,physical_labels=False))
        progress('extracting',done=slot+1,total=n)
        print('features',slot+1,'of',n,[x['kept'] for x in per_region],flush=True)
    if any(x['regions'][0]['kept']==0 for x in manifest):raise RuntimeError('At least one headROI lacks native texture support; no second extraction or threshold retry.')
    pairs=[(i,i+hop) for i in range(n) for hop in [1,2,4,8,16] if i+hop<n]
    assert len(pairs)==464
    union=Union(offset[-1]);edge_counts=[];bf=cv2.BFMatcher(cv2.NORM_L2)
    for pair_idx,(i,j) in enumerate(pairs):
        fw=bf.knnMatch(all_desc[i],all_desc[j],k=2);bw=bf.knnMatch(all_desc[j],all_desc[i],k=2)
        def arrays(m):
            indices=np.full((len(m),2),-1,np.int64);dist=np.full((len(m),2),np.inf,np.float64)
            for rr,nn in enumerate(m):
                for kk,match in enumerate(nn):indices[rr,kk]=match.trainIdx;dist[rr,kk]=match.distance
            return indices,dist
        fi,fd=arrays(fw);bi,bd=arrays(bw)
        good=fd[:,0]<.75*fd[:,1];back=bd[:,0]<.75*bd[:,1]
        js=fi[:,0];good &= (js>=0)&back[np.maximum(js,0)]&(bi[np.maximum(js,0),0]==np.arange(len(fi)))
        ii=np.flatnonzero(good);matches=np.c_[ii,js[ii]]
        np.savez_compressed(O/'pairs'/f'{i:03d}_{j:03d}.npz',forward_indices=fi,forward_distances=fd,reverse_indices=bi,reverse_distances=bd,accepted_pairs=matches)
        for aa,bb in matches:union.join(offset[i]+int(aa),offset[j]+int(bb))
        edge_counts.append(dict(i=i,j=j,matches=len(matches)))
        if pair_idx%10==0 or pair_idx==len(pairs)-1:
            progress('matching',done=pair_idx+1,total=len(pairs));print('pairs',pair_idx+1,'of',len(pairs),'matches',len(matches),flush=True)
    progress('duplicate_family_audit',pairs_completed=len(pairs))
    labels=union.labels();families=Union(offset[-1]);near_rows=[]
    for node,label in enumerate(labels):families.join(node,int(label))
    for slot,xy in enumerate(all_xy):
        near=sorted(cKDTree(xy).query_pairs(3.0))
        for aa,bb in near:families.join(offset[slot]+aa,offset[slot]+bb)
        near_rows.extend((slot,aa,bb) for aa,bb in near)
    famlabels=families.labels()
    np.savez_compressed(O/'component-index.npz',offsets=np.array(offset),appearance_component=labels,duplicate_family=famlabels,within_frame_near_pairs=np.array(near_rows).reshape(-1,3))
    groups={};fgroups={}
    for slot,xy in enumerate(all_xy):
        for j in range(len(xy)):
            node=offset[slot]+j;groups.setdefault(int(labels[node]),[]).append((slot,j));fgroups.setdefault(int(famlabels[node]),[]).append((slot,j))
    ambiguous={cid for cid,nodes in groups.items() if len({i for i,j in nodes})!=len(nodes)}
    tracks=[];frows=[];support=[dict(slot=i,FIT=dict(head=0,person=0,background=0),HELD=dict(head=0,person=0,background=0),quarantined=0) for i in range(n)]
    family_lookup={}
    for fid,nodes in sorted(fgroups.items()):
        if len({i for i,j in nodes})<3:continue
        components=sorted({int(labels[offset[i]+j]) for i,j in nodes});ident=identity(nodes);split='HELD' if int(ident[:8],16)%5==0 else 'FIT'
        issues=[]
        if any(c in ambiguous for c in components):issues.append('ambiguous_appearance_component')
        # Prevent a chain of neighbouring detections from joining different physical locations.
        byframe={}
        for i,j in nodes:byframe.setdefault(i,[]).append(all_xy[i][j])
        for i,pts in byframe.items():
            if len(pts)>1:
                pts=np.array(pts)
                if np.ptp(pts,axis=0).max()>3.+1e-9 or np.linalg.norm(pts[:,None]-pts[None,:],axis=2).max()>3.+1e-9:
                    issues.append('duplicate_family_spans_distinct_points');break
        item=dict(id=ident,split=split,quarantined=bool(issues),reasons=issues,nodes=nodes,components=components,physical_identity_verified=False)
        frows.append(item);family_lookup[fid]=item
    for cid,nodes in sorted(groups.items()):
        if len({i for i,j in nodes})<3:continue
        fid=int(famlabels[offset[nodes[0][0]]+nodes[0][1]]);f=family_lookup[fid]
        tr=dict(id=identity(nodes),family_id=f['id'],split=f['split'],nodes=nodes,quarantined=f['quarantined'],reasons=f['reasons'],physical_identity_verified=False)
        tracks.append(tr)
        for i,j in nodes:
            if tr['quarantined']:support[i]['quarantined']+=1;continue
            x,y=all_xy[i][j];head,body=manifest[i]['regions'][0]['bounds'],manifest[i]['regions'][1]['bounds']
            key='head' if head[0]<=x<head[2] and head[1]<=y<head[3] else ('person' if body[0]<=x<body[2] and body[1]<=y<body[3] else 'background')
            support[i][tr['split']][key]+=1
    save(O/'appearance-tracks.json',dict(records=tracks,identity_verified=False,pregeometry_split=True))
    save(O/'duplicate-families.json',dict(records=frows,pregeometry_split=True,all_original_nodes_kept=True))
    save(O/'pair-counts.json',dict(records=edge_counts));save(O/'region-support.json',dict(frames=support,regions_are_ROI_not_physical_labels=True))
    result=dict(updated=now(),state='complete_physical_material_review_pending',frames=n,features=int(offset[-1]),pairs=len(pairs),accepted_pair_edges=sum(x['matches'] for x in edge_counts),
        appearance_components=len(groups),ambiguous_components=len(ambiguous),tracks_with_at_least3frames=len(tracks),families_with_at_least3frames=len(frows),
        unquarantined_tracks=sum(not t['quarantined'] for t in tracks),quarantined_tracks=sum(t['quarantined'] for t in tracks),
        fit_tracks=sum(not t['quarantined'] and t['split']=='FIT' for t in tracks),held_tracks=sum(not t['quarantined'] and t['split']=='HELD' for t in tracks),
        raw_coordinate_domain='native_MP4_integer_pixels',K_used=False,extrinsics_estimated=False,physical_face_HELD=0,SMPLX_fitted=False,training_started=False,quality_accepted=False,seconds=time.monotonic()-start,
        preserved_files=dict(feature_manifest_sha256=sha(O/'feature-manifest.json'),tracks_sha256=sha(O/'appearance-tracks.json'),families_sha256=sha(O/'duplicate-families.json')))
    save(O/'result.json',result);signal.alarm(0);save(O/'progress.json',dict(updated=now(),state='complete',pid=None,done=n,total=n,seconds=time.monotonic()-start,training=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if O.exists():
            save(O/'failure.json',dict(updated=now(),error=repr(exc),traceback=traceback.format_exc(),observations_preserved=True,training_started=False))
            save(O/'progress.json',dict(updated=now(),state='failed',pid=None,error=repr(exc),training=False))
        raise
