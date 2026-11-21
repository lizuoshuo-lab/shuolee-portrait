"""One topology-only identity-fusion control on immutable native correspondences."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, os, time, signal, traceback, fcntl
import numpy as np
import networkx as nx

T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/pattern-preflight/native-material';B=O/'bridge-control'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ident(nodes):return hashlib.sha256(json.dumps(nodes,separators=(',',':')).encode()).hexdigest()

def split_graph(nodes,edges):
    """Tarjan bridges; component depth <=99 for the eligible unique-frame tracks."""
    adj={n:set() for n in nodes}
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    disc={};low={};bridges=set();clock=0
    def dfs(u,parent):
        nonlocal clock
        disc[u]=low[u]=clock;clock+=1
        for v in sorted(adj[u]):
            if v==parent:continue
            if v not in disc:
                dfs(v,u);low[u]=min(low[u],low[v])
                if low[v]>disc[u]:bridges.add(tuple(sorted((u,v))))
            else:low[u]=min(low[u],disc[v])
    for u in sorted(nodes):
        if u not in disc:dfs(u,None)
    seen=set();groups=[]
    for start in sorted(nodes):
        if start in seen:continue
        stack=[start];seen.add(start);group=[]
        while stack:
            u=stack.pop();group.append(u)
            for v in adj[u]:
                if v not in seen and tuple(sorted((u,v))) not in bridges:seen.add(v);stack.append(v)
        groups.append(sorted(group))
    return bridges,sorted(groups)

def controls():
    cases=[(list(range(6)),[(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)],{(2,3)},[[0,1,2],[3,4,5]]),
           (list(range(3)),[(0,1),(1,2),(2,0)],set(),[[0,1,2]]),
           (list(range(4)),[(0,1),(1,2),(2,3)],{(0,1),(1,2),(2,3)},[[0],[1],[2],[3]])]
    for nodes,edges,want,parts in cases:
        assert split_graph(nodes,edges)==(want,parts)
        assert split_graph(list(reversed(nodes)),list(reversed(edges)))==(want,parts)
        g=nx.Graph();g.add_nodes_from(nodes);g.add_edges_from(edges)
        assert {tuple(sorted(e)) for e in nx.bridges(g)}==want
    return dict(cases=3,order_invariant=True,independent_networkx=True,synthetic_only=True)

def main():
    assert not B.exists(),'Existing result: review, never repeat this trial.'
    plan=read(O/'bridge-plan.json');lock=open('/tmp/shuolee-pattern-bridge123.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in plan['inputs'].items():assert sha(O/name)==h,name
    B.mkdir();start=time.monotonic()
    def alarm(*args):raise TimeoutError('Registered600secondCPUwallbudget exceeded')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    save(B/'execution-plan.json',dict(started=now(),plan_sha256=sha(O/'bridge-plan.json'),source_sha256=sha(Path(__file__)),pid=os.getpid(),networkx_version=nx.__version__))
    save(B/'controls.json',controls())
    def progress(stage,**kw):save(B/'progress.json',dict(updated=now(),stage=stage,pid=os.getpid(),seconds=time.monotonic()-start,training=False,**kw))
    progress('loading_immutable_edges')
    tracks=read(O/'appearance-tracks.json')['records'];ts=[t for t in tracks if not t['quarantined']]
    development=set(plan['development_family_ids']);index=np.load(O/'component-index.npz',allow_pickle=False)
    offsets=index['offsets'];count=int(offsets[-1]);owner=np.full(count,-1,np.int32)
    globalnodes=[]
    for ti,tr in enumerate(ts):
        nodes=[int(offsets[i])+j for i,j in tr['nodes']]
        assert (owner[nodes]==-1).all();owner[nodes]=ti;globalnodes.append(nodes)
        assert len({i for i,j in tr['nodes']})==len(nodes)
    graph_edges=[[] for t in ts]
    pair_records=read(O/'pair-hashes.json')['records']
    for pi,item in enumerate(pair_records):
        p=O/'pairs'/item['file'];assert sha(p)==item['sha256']
        i,j=map(int,p.stem.split('_'));pairs=np.load(p,allow_pickle=False)['accepted_pairs']
        left=pairs[:,0]+offsets[i];right=pairs[:,1]+offsets[j];keepers=np.flatnonzero(owner[left]>=0)
        assert np.array_equal(owner[left[keepers]],owner[right[keepers]])
        for k in keepers:graph_edges[int(owner[left[k]])].append((int(left[k]),int(right[k])))
        if pi%100==0:progress('loading_immutable_edges',done=pi,total=len(pair_records))
    manifest=read(O/'feature-manifest.json')['frames']
    xy=[np.load(O/'features'/r['feature_file'],allow_pickle=False)['xy'] for r in manifest]
    def native(n):
        i=int(np.searchsorted(offsets,n,side='right')-1);return [i,int(n-offsets[i])]
    def region(node):
        i,j=node;x,y=xy[i][j];h,b=manifest[i]['regions'][0]['bounds'],manifest[i]['regions'][1]['bounds']
        return 'head' if h[0]<=x<h[2] and h[1]<=y<h[3] else ('person' if b[0]<=x<b[2] and b[1]<=y<b[3] else 'background')
    support=[dict(slot=i,before={s:dict(head=0,person=0,background=0) for s in ['FIT','HELD']},
                  after={s:dict(head=0,person=0,background=0) for s in ['FIT','HELD']},
                  unopened_HELD=dict(head=0,person=0,background=0)) for i in range(99)]
    segments=[];bridge_records=[];all_summaries=[];node_segment={};valid_segments=set()
    for ti,(tr,nodes,edges) in enumerate(zip(ts,globalnodes,graph_edges)):
        assert len(edges)==len(set(tuple(sorted(e)) for e in edges))
        bridges,parts=split_graph(nodes,edges)
        g=nx.Graph();g.add_nodes_from(nodes);g.add_edges_from(edges)
        reference={tuple(sorted(e)) for e in nx.bridges(g)}
        assert bridges==reference,(ti,'independent_bridge_disagreement')
        g.remove_edges_from(reference)
        assert sorted(sorted(c) for c in nx.connected_components(g))==parts
        for a,b in sorted(bridges):bridge_records.append(dict(parent_track_id=tr['id'],family_id=tr['family_id'],split=tr['split'],a=native(a),b=native(b),reason='bridge_without_alternative_identity_path'))
        for n in tr['nodes']:support[n[0]]['before'][tr['split']][region(n)]+=1
        for part in parts:
            original=sorted(native(n) for n in part);sid=ident(original);eligible=len(original)>=3
            for n in part:node_segment[n]=sid
            if eligible:valid_segments.add(sid)
            seg=dict(id=sid,parent_track_id=tr['id'],family_id=tr['family_id'],split=tr['split'],nodes=original,
                     development_only=tr['family_id'] in development,topologically_eligible=eligible,
                     physical_identity_verified=False,reason='cycle_connected_candidate' if eligible else 'insufficient_cycle_supported_views')
            segments.append(seg)
            if eligible:
                for n in original:
                    r=region(n);support[n[0]]['after'][tr['split']][r]+=1
                    if tr['split']=='HELD' and not seg['development_only']:support[n[0]]['unopened_HELD'][r]+=1
        all_summaries.append(dict(track_id=tr['id'],family_id=tr['family_id'],split=tr['split'],original_nodes=len(nodes),edges=len(edges),bridges=len(bridges),segments=len(parts),eligible_segments=sum(len(p)>=3 for p in parts)))
        if ti%1000==0:progress('independent_graph_checks',done=ti,total=len(ts))
    before=read(O/'region-support.json')['frames']
    for i in range(99):
        for split in ['FIT','HELD']:assert support[i]['before'][split]==before[i][split]
    all_retained_nodes=[tuple(n) for s in segments for n in s['nodes']]
    assert len(all_retained_nodes)==len(set(all_retained_nodes))==sum(len(t['nodes']) for t in ts)
    parent={t['id']:t for t in ts}
    for s in segments:
        p=parent[s['parent_track_id']];assert s['family_id']==p['family_id'] and s['split']==p['split']
    frozen=read(O/'visual-review/index.json');review_rows=[]
    for row in frozen['rows']:
        ids=[node_segment[int(offsets[v['slot']])+v['keypoint']] for v in row['views']]
        review_rows.append(dict(row=row['row'],family_id=row['family_id'],split=row['split'],development_only=True,
            three_views_same_segment=len(set(ids))==1,all_three_topologically_eligible=all(i in valid_segments for i in ids),segment_ids=ids))
    false=next(t for t in ts if t['id']=='63807ef2776a069409fc908c22b313b6e544728a60c43a7404e5e44a19f3e425')
    cloth=int(offsets[1])+421;eye=int(offsets[17])+9
    assert node_segment[cloth]!=node_segment[eye]
    false_segments=[s for s in segments if s['parent_track_id']==false['id']]
    for name,h in plan['inputs'].items():assert sha(O/name)==h
    save(B/'segments.json',dict(records=segments,original_splits_preserved=True,physical_truth=False))
    save(B/'quarantined-bridges.json',dict(records=bridge_records,original_edges_not_deleted=True))
    save(B/'track-summary.json',dict(records=all_summaries));save(B/'region-support.json',dict(frames=support,regions_not_semantic_labels=True))
    save(B/'frozen-review-comparison.json',dict(rows=review_rows,no_new_query_selection=True,independent_validation=False))
    def summary(stage,split,r):
        vals=[f[stage][split][r] for f in support]
        return dict(min=min(vals),max=max(vals),observations=sum(vals),zero_frames=[i for i,v in enumerate(vals) if v==0])
    eligible=[s for s in segments if s['topologically_eligible']]
    result=dict(updated=now(),state='complete_graph_control_physical_identity_pending',frames=99,
        original_tracks=len(ts),original_observations=len(all_retained_nodes),processed_edges=sum(len(e) for e in graph_edges),
        quarantined_bridges=len(bridge_records),all_segments=len(segments),eligible_segments=len(eligible),
        eligible_FIT_segments=sum(s['split']=='FIT' for s in eligible),eligible_HELD_segments=sum(s['split']=='HELD' for s in eligible),
        eligible_unopened_HELD_segments=sum(s['split']=='HELD' and not s['development_only'] for s in eligible),
        retained_observations=sum(len(s['nodes']) for s in eligible),quarantined_short_observations=sum(len(s['nodes']) for s in segments if not s['topologically_eligible']),
        independent_networkx_checked_graphs=len(ts),all_inputs_unchanged=True,all_original_nodes_preserved=True,FIT_HELD_membership_unchanged=True,
        frozen30_same_valid_segment=sum(r['three_views_same_segment'] and r['all_three_topologically_eligible'] for r in review_rows),
        known_false_cloth_eye_connection_separated=True,known_false_track_segments=false_segments,
        region_counts={stage:{s:{r:summary(stage,s,r) for r in ['head','person','background']} for s in ['FIT','HELD']} for stage in ['before','after']},
        physical_face_HELD=0,physical_identity_not_established_by_cycles=True,camera_estimated=False,SMPLX_fitted=False,training_started=False,quality_accepted=False,
        seconds=time.monotonic()-start)
    save(B/'result.json',result);signal.alarm(0)
    save(B/'progress.json',dict(updated=now(),stage='complete',pid=None,seconds=time.monotonic()-start,training=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['region_counts','known_false_track_segments']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if B.exists():
            save(B/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
            save(B/'progress.json',dict(updated=now(),stage='failed',pid=None,error=repr(e)))
        raise
