"""Train-only neighborhood controls; preserve nodes/points/colors exactly."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra,connected_components
L=Path(__file__).resolve().parents[2];I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';O.mkdir(exist_ok=True);assert not (O/'protocol.json').exists();read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();z=np.load(I/'model/initial.npz');base=dict(np.load(I/'model/graph.npz'));pts=z['points'].astype(float);group=z['component'];spacing=.012
pairs=cKDTree(pts).query_pairs(1.8*spacing,output_type='ndarray');length=np.linalg.norm(pts[pairs[:,0]]-pts[pairs[:,1]],axis=1)
def adjacency(p,d):return csr_matrix((np.r_[d,d],(np.r_[p[:,0],p[:,1]],np.r_[p[:,1],p[:,0]])),shape=(len(pts),len(pts)))
n,cc=connected_components(adjacency(pairs,length),directed=False);counts=np.bincount(cc);bridges=[]
for comp in np.where(counts<128)[0]:
    ids=np.where(cc==comp)[0];assert len(np.unique(group[ids]))==1;g=group[ids[0]];major=np.where((group==g)&(counts[cc]>=128))[0];dist,j=cKDTree(pts[major]).query(pts[ids]);k=int(np.argmin(dist));bridges.append(dict(from_point=int(ids[k]),to_point=int(major[j[k]]),length_scene_units=float(dist[k]),fragment_points=int(len(ids)),assigned_component=int(g)))
if bridges:
    pairs=np.concatenate([pairs,np.array([[r['from_point'],r['to_point']] for r in bridges])]);length=np.r_[length,[r['length_scene_units'] for r in bridges]]
adj=adjacency(pairs,length);n,cc=connected_components(adj,directed=False);assert n==len(np.unique(group))
for c in np.unique(group):assert len(np.unique(cc[group==c]))==1
sources=base['source_point_indices'];dist=dijkstra(adj,directed=False,indices=sources);node_group=group[sources];edges=[];ew=[];new_ids=np.empty_like(base['point_node_indices']);new_weights=np.empty_like(base['point_node_weights'])
for c in np.unique(group):
    ni=np.where(node_group==c)[0];ii=np.where(group==c)[0];d=dist[np.ix_(ni,ii)].T;assert np.isfinite(d).all();jj=np.argsort(d,axis=1,kind='stable')[:,:4];dd=np.take_along_axis(d,jj,axis=1);new_ids[ii]=ni[jj];w=np.exp(-.5*(dd/np.maximum(dd[:,-1:]*.5,1e-5))**2);w[dd[:,0]<1e-7]=[1,0,0,0];w/=w.sum(1,keepdims=True);new_weights[ii]=w
    dn=dist[np.ix_(ni,sources[ni])].copy();np.fill_diagonal(dn,np.inf);neighbors=np.argsort(dn,axis=1,kind='stable')[:,:8];nd=np.take_along_axis(dn,neighbors,axis=1)
    for k,i in enumerate(ni):
        for distance,j in zip(nd[k],neighbors[k]):edges.append([i,int(ni[j])]);ew.append(np.exp(-distance*distance/(2*max(nd[k,-1],1e-5)**2)))
edges=np.array(edges,dtype=base['edges'].dtype);ew=np.array(ew,dtype=base['edge_weights'].dtype);assert edges.shape==base['edges'].shape and len(edges)==3072;assert np.all(node_group[edges[:,0]]==node_group[edges[:,1]]) and np.all(node_group[new_ids]==group[:,None]);assert abs(new_weights.sum(-1)-1).max()<2e-7
variants={}
for name,binding in [('path_rigidity',False),('path_rigidity_binding',True)]:
    root=O/name;root.mkdir(exist_ok=True);out=dict(base);out.update(edges=edges,edge_weights=ew)
    if binding:out.update(point_node_indices=new_ids,point_node_weights=new_weights)
    assert np.array_equal(out['node_points'],base['node_points']) and np.array_equal(out['source_point_indices'],sources);np.savez_compressed(root/'graph.npz',**out);variants[name]=dict(graph_sha256=sha(root/'graph.npz'),changes=['graph_edges_and_weights']+(['point_binding_indices_and_weights'] if binding else []))
edge_old={tuple(e) for e in base['edges']};edge_new={tuple(e) for e in edges};report=dict(status='two_train_only_graph_controls_prepared_not_trained',recorded_utc=datetime.now(timezone.utc).isoformat(),base_stage=I.name,base_checkpoint_sha256=read(I/'dynamic_clock/completion.json')['checkpoint_sha256'],initial_sha256=sha(I/'model/initial.npz'),baseline_graph_sha256=sha(I/'model/graph.npz'),variants=variants,shared_nodes=384,shared_initial_points=len(pts),shared_directed_edges_per_model=3072,graph_edge_pairs_replaced=len(edge_new-edge_old),points_with_changed_neighbor_set=int(np.any(np.sort(new_ids,1)!=np.sort(base['point_node_indices'],1),1).sum()),minor_fragment_bridge_policy='One shortest Euclidean bridge from each original<128pointfragment to its alreadyassigned majorcomponent. No crossmajorcomponentlink. All originalpoints retained; these bridges can still carry geometry errors.',minor_fragment_bridges=bridges,training_contract='Reuse106native195photos,clockoffsets,initialGaussianattributes,samplingseed,schedules,15000updates,andallmotionbounds. path_rigidity changes only rigidityedges/weights;path_rigidity_binding also changes materialbinding. Baseline106dynamic_clock retained. Exactsourceandinitialsamplingauditsrequired when launching.',evaluation_contract='106validation becomes development afterinspection. Keep its fixedfullnative/detail/continuity windows; use an additional untouchedtimewindow for eventual confirmation, never claim these are newblindtestresults.',rationale='Read-only106diagnosticfound surface-path/direct ratios above3 on some Euclideangraph edges, includinglowerprojectioncontext. This motivates a test; doesnotestablishanatomicalgroundtruth orcausality.',training_started=False,goal_complete=False)
(O/'protocol.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps({k:report[k] for k in ['status','graph_edge_pairs_replaced','points_with_changed_neighbor_set','minor_fragment_bridges']}),flush=True)
