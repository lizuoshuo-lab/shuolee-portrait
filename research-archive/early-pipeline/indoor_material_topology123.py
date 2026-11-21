"""Apply known topology protection once to indoor candidates; no geometric solver."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, time, os, signal, sys, traceback
import numpy as np
import networkx as nx
from PIL import Image
from pattern_bridge_control123 import split_graph

T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
F=T/'shuolee/current/local-clothed-surface-123/indoor-preflight'
O=F/'native-material'; B=O/'topology-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def identity(nodes):return hashlib.sha256(json.dumps(nodes,separators=(',',':')).encode()).hexdigest()

def main():
    if B.exists():raise RuntimeError('Existing indoor topology review: inspect instead of rerunning.')
    plan=read(F/'native-material-plan.json');reg=read(F/'topology-review-plan.json')
    assert sha(Path(__file__))==reg['helper_sha256']
    assert sha(Path(__file__).with_name('pattern_bridge_control123.py'))==reg['reference_sha256']
    result=read(O/'result.json');assert result['frames']==177 and not result['training_started']
    hashes={n:sha(O/n) for n in ['appearance-tracks.json','duplicate-families.json','feature-manifest.json','component-index.npz']}
    assert hashes['appearance-tracks.json']==result['preserved_files']['tracks_sha256']
    assert hashes['feature-manifest.json']==result['preserved_files']['feature_manifest_sha256']
    B.mkdir();(B/'native-review').mkdir();started=time.monotonic()
    def timeout(*args):raise TimeoutError('Single registered 600s graph/review preparation budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    def progress(stage,**kw):save(B/'progress.json',dict(updated=now(),stage=stage,pid=os.getpid(),seconds=time.monotonic()-started,command=sys.argv,training=False,**kw))
    progress('reading_immutable_tracks')
    manifest=read(O/'feature-manifest.json')['frames'];assert len(manifest)==177
    tracks=[x for x in read(O/'appearance-tracks.json')['records'] if not x['quarantined']]
    with np.load(O/'component-index.npz',allow_pickle=False) as z:offsets=z['offsets'].copy()
    owner=np.full(int(offsets[-1]),-1,np.int32);globalnodes=[]
    for i,t in enumerate(tracks):
        nodes=[int(offsets[f])+j for f,j in t['nodes']]
        assert (owner[nodes]==-1).all() and len({f for f,j in t['nodes']})==len(nodes)
        owner[nodes]=i;globalnodes.append(nodes)
    graph_edges=[[] for _ in tracks];pairhash=[]
    for p in sorted((O/'pairs').glob('*.npz')):
        pairhash.append(dict(file=p.name,sha256=sha(p)))
        i,j=map(int,p.stem.split('_'))
        with np.load(p,allow_pickle=False) as z:matches=z['accepted_pairs']
        left=matches[:,0]+offsets[i];right=matches[:,1]+offsets[j]
        idx=np.flatnonzero(owner[left]>=0)
        assert np.array_equal(owner[left[idx]],owner[right[idx]])
        for k in idx:graph_edges[int(owner[left[k]])].append((int(left[k]),int(right[k])))
    assert len(pairhash)==854
    xy=[]
    for row in manifest:
        p=O/'features'/row['feature_file'];assert sha(p)==row['sha256']
        with np.load(p,allow_pickle=False) as z:xy.append(z['xy'].copy())
    def native(node):
        i=int(np.searchsorted(offsets,node,side='right')-1)
        return [i,int(node-offsets[i])]
    def region(node):
        i,j=node;x,y=xy[i][j];h,b=manifest[i]['regions'][0]['bounds'],manifest[i]['regions'][1]['bounds']
        return 'head' if h[0]<=x<h[2] and h[1]<=y<h[3] else ('person' if b[0]<=x<b[2] and b[1]<=y<b[3] else 'background')
    segments=[];bridges=[];summaries=[]
    for i,(tr,nodes,edges) in enumerate(zip(tracks,globalnodes,graph_edges)):
        cuts,parts=split_graph(nodes,edges)
        graph=nx.Graph();graph.add_nodes_from(nodes);graph.add_edges_from(edges)
        assert cuts=={tuple(sorted(x)) for x in nx.bridges(graph)}
        graph.remove_edges_from(cuts)
        assert sorted(sorted(c) for c in nx.connected_components(graph))==parts
        for a,b in sorted(cuts):bridges.append(dict(parent_track_id=tr['id'],family_id=tr['family_id'],split=tr['split'],a=native(a),b=native(b)))
        for part in parts:
            original=sorted(native(n) for n in part)
            segments.append(dict(id=identity(original),parent_track_id=tr['id'],family_id=tr['family_id'],split=tr['split'],
                nodes=original,topologically_eligible=len(part)>=3,physical_identity_verified=False,development_only=False))
        summaries.append(dict(track_id=tr['id'],nodes=len(nodes),edges=len(edges),bridges=len(cuts),segments=len(parts)))
        if i%2000==0:progress('topology',done=i,total=len(tracks))
    allnodes=[tuple(n) for s in segments for n in s['nodes']]
    assert len(allnodes)==len(set(allnodes))==sum(len(t['nodes']) for t in tracks)
    eligible=sorted([s for s in segments if s['topologically_eligible']],key=lambda s:(s['family_id'],s['id']))
    selected=set();review=[];missing=[]
    for slot in plan['review']['source_slots']:
        for area in plan['review']['regions']:
            for split in plan['review']['splits']:
                options=[s for s in eligible if s['split']==split and s['family_id'] not in selected and
                    any(n[0]==slot and region(n)==area for n in s['nodes'])]
                if not options:
                    missing.append(dict(slot=slot,region=area,split=split));continue
                seg=options[0];selected.add(seg['family_id'])
                query=next(n for n in seg['nodes'] if n[0]==slot)
                views=[seg['nodes'][0],query,seg['nodes'][-1]]
                if len({tuple(n) for n in views})<3:views=[seg['nodes'][0],seg['nodes'][len(seg['nodes'])//2],seg['nodes'][-1]]
                assert len({n[0] for n in views})==3
                panels=[];viewrows=[]
                for f,j in views:
                    row=manifest[f];p=Path(row['image']);assert sha(p)==row['source_sha256']
                    im=Image.open(p).convert('RGB');assert im.size==(2704,1520)
                    u,v=xy[f][j];cx,cy=int(np.floor(u)),int(np.floor(v));box=[cx-96,cy-96,cx+96,cy+96]
                    panels.append(im.crop(box));viewrows.append(dict(slot=f,keypoint=j,source_frame=row['source_frame'],
                        source_sha256=row['source_sha256'],feature_sha256=row['sha256'],uv=[float(u),float(v)],crop_xyxy=box,region=region([f,j])))
                canvas=Image.new('RGB',(576,192))
                for k,im in enumerate(panels):canvas.paste(im,(k*192,0))
                name=f'row_{len(review):03d}_native.png';canvas.save(B/'native-review'/name)
                review.append(dict(row=len(review),query_slot=slot,query_region=area,split=split,
                    family_id=seg['family_id'],segment_id=seg['id'],segment_nodes=len(seg['nodes']),
                    all_segment_views_reviewed=len(seg['nodes'])==3,views=viewrows,file=name,sha256=sha(B/'native-review'/name)))
    for seg in segments:seg['development_only']=seg['family_id'] in selected
    support=[]
    for i in range(177):support.append(dict(slot=i,source_frame=manifest[i]['source_frame'],
        FIT={r:0 for r in ['head','person','background']},HELD={r:0 for r in ['head','person','background']},
        unopened_HELD={r:0 for r in ['head','person','background']}))
    for seg in eligible:
        for node in seg['nodes']:
            i,j=node;r=region(node);support[i][seg['split']][r]+=1
            if seg['split']=='HELD' and seg['family_id'] not in selected:support[i]['unopened_HELD'][r]+=1
    for n,h in hashes.items():assert sha(O/n)==h
    for row in pairhash:assert sha(O/'pairs'/row['file'])==row['sha256']
    save(B/'segments.json',dict(records=segments,original_splits_preserved=True,physical_truth=False))
    save(B/'quarantined-bridges.json',dict(records=bridges,original_edges_not_deleted=True))
    save(B/'pair-hashes.json',dict(records=pairhash));save(B/'track-summary.json',dict(records=summaries))
    save(B/'region-support.json',dict(frames=support,regions_are_ROI_not_physical_labels=True))
    save(B/'native-review/index.json',dict(rows=review,missing_queries=missing,development_family_ids=sorted(selected),
        physical_labels_pending=True,no_pixel_resampling=True,annotations_burned_in=False,selection_plan_sha256=sha(F/'native-material-plan.json')))
    def counts(split,area):
        vals=[r[split][area] for r in support]
        return dict(observations=sum(vals),minimum=min(vals),maximum=max(vals),zero_slots=[i for i,v in enumerate(vals) if v==0])
    out=dict(updated=now(),state='complete_native_review_pending',frames=177,original_tracks=len(tracks),
        original_observations=len(allnodes),quarantined_bridges=len(bridges),all_segments=len(segments),eligible_segments=len(eligible),
        fit_segments=sum(s['split']=='FIT' for s in eligible),held_segments=sum(s['split']=='HELD' for s in eligible),
        retained_observations=sum(len(s['nodes']) for s in eligible),review_queries=len(review),missing_queries=missing,
        support={s:{r:counts(s,r) for r in ['head','person','background']} for s in ['FIT','HELD','unopened_HELD']},
        all_original_nodes_preserved=True,all_input_hashes_unchanged=True,family_and_split_unchanged=True,
        independent_graph_implementation_checked=len(tracks),physical_face_HELD=0,material_identity_verified=False,
        cameras_estimated=False,SMPLX_fitted=False,training_started=False,quality_accepted=False,
        source_sha256=sha(Path(__file__)),registered_plan_sha256=sha(F/'topology-review-plan.json'),seconds=time.monotonic()-started)
    save(B/'result.json',out);signal.alarm(0);progress('complete',done=len(tracks),total=len(tracks))
    p=read(B/'progress.json');p['pid']=None;save(B/'progress.json',p)
    print(json.dumps({k:v for k,v in out.items() if k!='support'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if B.exists():
            save(B/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
            save(B/'progress.json',dict(updated=now(),stage='failed',pid=None,error=repr(e)))
        raise
