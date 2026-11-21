"""Training-only graph topology for a future articulated residual-motion control."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from scipy.spatial import cKDTree
LAB=Path(__file__).resolve().parents[2];SRC=LAB/'outputs/dynamic-human-enerf-articulated-control-064';OUT=LAB/'outputs/dynamic-human-enerf-fine-graph-071';OUT.mkdir(exist_ok=True)
assert not (OUT/'graph.npz').exists()
z=dict(np.load(SRC/'initial.npz'));points=z['points'];person=z['anchor']//6890;chosen=[]
for p in [0,1]:
 ids=np.where(person==p)[0];x=points[ids].astype(np.float64);selected=int(np.argmin(np.linalg.norm(x-x.mean(0),axis=1)));best=np.full(len(x),np.inf)
 for _ in range(768):
  chosen.append(int(ids[selected]));best=np.minimum(best,((x-x[selected])**2).sum(-1));selected=int(np.argmax(best))
chosen=np.array(chosen);nodes=points[chosen];node_person=person[chosen];neighbors=np.empty((len(points),4),np.int64);weights=np.empty((len(points),4),np.float32);edges=[];edge_weights=[]
for p in [0,1]:
 index=np.where(person==p)[0];ni=np.where(node_person==p)[0];tree=cKDTree(nodes[ni]);distance,near=tree.query(points[index],k=4);neighbors[index]=ni[near];sigma=np.maximum(distance[:,-1:]*.5,1e-5);w=np.exp(-.5*(distance/sigma)**2);atnode=distance[:,0]<1e-7;w[atnode]=[1,0,0,0];w/=w.sum(1,keepdims=True);weights[index]=w
 dd,jj=tree.query(nodes[ni],k=9)
 for k,i in enumerate(ni):
  for dist,j in zip(dd[k,1:],jj[k,1:]):edges.append([int(i),int(ni[j])]);edge_weights.append(float(np.exp(-dist**2/(2*max(dd[k,-1],1e-5)**2))))
edges=np.array(edges,dtype=np.int64);edge_weights=np.array(edge_weights,np.float32)
assert len(np.unique(chosen))==1536 and np.all(node_person[neighbors]==person[:,None]) and np.all(node_person[edges[:,0]]==node_person[edges[:,1]])
assert np.max(abs(weights.sum(-1)-1))<2e-7 and np.all(weights>=0) and np.all(neighbors[chosen,0]==np.arange(1536)) and np.all(weights[chosen,0]==1)
np.savez_compressed(OUT/'graph.npz',node_points=nodes,node_anchor=z['anchor'][chosen],node_person=node_person,source_point_indices=chosen,point_node_indices=neighbors,point_node_weights=weights,edges=edges,edge_weights=edge_weights)
report=dict(status='1536nodes_training_initialization_only_material_bindings_and_sameperson_edges_verified',source_sha256=hashlib.sha256((SRC/'initial.npz').read_bytes()).hexdigest(),graph_sha256=hashlib.sha256((OUT/'graph.npz').read_bytes()).hexdigest(),initial_points=len(points),nodes=1536,nodes_per_person=768,neighbors_per_gaussian=4,directed_edges=len(edges),max_weight_sum_error=float(np.max(abs(weights.sum(-1)-1))),
 source='Same064train-only visual-hull initialization, no heldout or odd-time images read. Person assignment inherited from nearestSMPLvertex; it is approximate, not instanceGT.',
 method='Deterministic farthest point sampling per person; four nearest node material weights with adaptive Gaussian distance falloff; exactnodepositions onehot; eight sameperson neighbors pernode.',
 next='Integrate bounded time-polynomial local rigid residuals after a verified body prior; propagate covariance and canonical color directions, then actualCUDA forward/backward tests and matched-budget training.',
 qualification='Preparation only, not a rendered model or evidence of improved clarity. Licensed derived bindings remain local.',goal_complete=False)
(OUT/'graph-preparation.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(report))
