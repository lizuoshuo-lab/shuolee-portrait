"""Train-initialization-only graph-distance diagnostic; no model edits."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra,connected_components
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';D=O/'motion-diagnostic';D.mkdir(exist_ok=True);start=time.monotonic();read=lambda p:json.loads(p.read_text(encoding='utf8'));z=np.load(O/'model/initial.npz');g=np.load(O/'model/graph.npz');pts=z['points'].astype(np.float64);spacing=.012
grid=(pts-pts.min(0))/spacing;grid_error=float(abs(grid-np.rint(grid)).max());assert grid_error<.002
pairs=cKDTree(pts).query_pairs(np.sqrt(3)*spacing*1.001,output_type='ndarray');length=np.linalg.norm(pts[pairs[:,0]]-pts[pairs[:,1]],axis=1);assert length.min()>.011
adj=csr_matrix((np.r_[length,length],(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(pts),len(pts)));ncomp,cc=connected_components(adj,directed=False)
sources=g['source_point_indices'];dist=dijkstra(adj,directed=False,indices=sources);assert dist.shape==(384,len(pts));nodes=pts[sources];edges=g['edges'];direct=np.linalg.norm(nodes[edges[:,0]]-nodes[edges[:,1]],axis=1);path=dist[edges[:,0],sources[edges[:,1]]];ratio=path/direct
cal=read(O/'cameras.json')['04'];W=np.array(cal['w2c']);K=np.array(cal['K']);q=(pts@W[:3,:3].T+W[:3,3])@K.T;uv=q[:,:2]/q[:,2:];major=np.argsort(-np.bincount(z['component']))[:2];tall=max(major,key=lambda c:np.ptp(uv[z['component']==c,1]));y=uv[z['component']==tall,1];lower=(z['component']==tall)&(uv[:,1]>y.min()+.6*np.ptp(y));node_lower=lower[sources];edge_lower=node_lower[edges[:,0]]|node_lower[edges[:,1]]
def stats(r):
    finite=np.isfinite(r);x=r[finite]
    return dict(count=int(len(r)),unreachable=int((~finite).sum()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(x.max()),fraction_ratio_above2=float((x>2).mean()),fraction_ratio_above3=float((x>3).mean()))
nn=g['point_node_indices'];dd=dist[nn,np.arange(len(pts))[:,None]];straight=np.linalg.norm(pts[:,None,:]-nodes[nn],axis=-1);safe=straight>1e-7;binding_ratio=np.divide(dd,straight,out=np.ones_like(dd),where=safe)
order=np.argsort(-np.nan_to_num(ratio,nan=0,posinf=1e9));worst=[dict(edge_index=int(i),node_ids=edges[i].tolist(),direct_scene_units=float(direct[i]),lattice_path_scene_units=float(path[i]) if np.isfinite(path[i]) else None,ratio=float(ratio[i]) if np.isfinite(ratio[i]) else None,lower_projection_context=bool(edge_lower[i])) for i in order[:30]]
np.savez_compressed(D/'graph-path-diagnostic.npz',edge_ratio=ratio,edge_path=path,edge_direct=direct,edge_lower=edge_lower,binding_ratio=binding_ratio,lower_points=lower)
out=dict(status='train_only_lattice_geodesic_detour_diagnostic_no_graph_change',initial_sha256=hashlib.sha256((O/'model/initial.npz').read_bytes()).hexdigest(),graph_sha256=hashlib.sha256((O/'model/graph.npz').read_bytes()).hexdigest(),voxel_spacing_scene_units=spacing,maximum_grid_rounding_error=grid_error,lattice_connected_components=ncomp,local_lattice_edges=int(len(pairs)),node_edges=dict(all=stats(ratio),lower_projection_context=stats(ratio[edge_lower])),point_bindings=dict(all=stats(binding_ratio[safe]),lower_projection_context=stats(binding_ratio[safe&lower[:,None]])),worst_edges=worst,elapsed_seconds=time.monotonic()-start,interpretation='Large path/direct ratios show Euclidean shortcuts across the sampled initialization surface. They do not prove anatomical misbinding or causally explain a rendered failure; visual-hull topology may itself be wrong. These values use training-only initial geometry, no validation RGB. A graph-topology control would need retraining and new native evaluation.',goal_complete=False)
(D/'graph-path-summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({k:out[k] for k in ['node_edges','point_bindings','elapsed_seconds']}),flush=True)
