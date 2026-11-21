"""Train-only visual-hull and component graph initialization, no body templates."""
from pathlib import Path
import json,hashlib,time
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from train_enerf_camera10_time import visual_hull,tensor

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');O=L/'outputs/dynamic-human-actor23-independent-106';Q=O/'geometry-time';M=O/'model';M.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (M/'initialization-and-graph.json').exists()
plan=read(O/'validation-reservation.json');clocks=read(Q/'candidate-clock-offsets.json')['offsets'];audit={(r['camera'],r['frame']):r for r in read(O/'training-input-audit.json')['records']};summary=read(Q/'summary.json');assert summary['clock_fit']['validation_after']['median']<summary['clock_fit']['validation_before']['median'] and not summary['clock_fit']['bound_hits']
torch.set_num_threads(6);cv2.setNumThreads(2);points=[]
for row in audit.values():row['input']=row['input'].replace('\\','/')
def interp(a,t):return a[:,1]+t/4*(a[:,2]-a[:,0])+t*t/8*(a[:,2]-2*a[:,1]+a[:,0])
for r in summary['clock_fit']['validation_pairs']:
    z=np.load(Q/r['file']);ca,cb=r['cameras'];a=np.load(O/audit[ca,612]['input']);b=np.load(O/audit[cb,612]['input']);x=interp(z['a'],-clocks[ca]);y=interp(z['b'],-clocks[cb]);P1=a['K']@a['w2c'][:3];P2=b['K']@b['w2c'][:3];h=cv2.triangulatePoints(P1,P2,x.T,y.T).T;X=h[:,:3]/h[:,3:];Xh=np.c_[X,np.ones(len(X))];q1=Xh@P1.T;q2=Xh@P2.T;re=np.maximum(np.linalg.norm(q1[:,:2]/q1[:,2:]-x,axis=1),np.linalg.norm(q2[:,:2]/q2[:,2:]-y,axis=1));C1=-a['w2c'][:3,:3].T@a['w2c'][:3,3];C2=-b['w2c'][:3,:3].T@b['w2c'][:3,3];d1=X-C1;d2=X-C2;cos=(d1*d2).sum(1)/np.linalg.norm(d1,axis=1)/np.linalg.norm(d2,axis=1);ok=np.isfinite(X).all(1)&(q1[:,2]>.1)&(q2[:,2]>.1)&(re<2)&(cos<np.cos(np.deg2rad(2)));points.append(X[ok])
points=np.concatenate(points);assert len(points)>500
save(M/'geometry-audit.json',dict(frame=612,point_quantiles=np.quantile(points,[.005,.5,.995],axis=0).tolist(),source='Training-feature coordinates interpolated to common index612 using candidate clocks; no RGB warped. Bounds only, not groundtruth geometry.',points=len(points)))
obs=[];selected=[]
for c in plan['train_cameras']:
    f=min(plan['train_frames'],key=lambda f:abs(f+clocks[c]-612));p=O/audit[c,f]['input'];assert sha(p)==audit[c,f]['input_sha256'];z=np.load(p);m=z['matte'];obs.append(dict(id=c,rgb=tensor(z['rgb']/255),matte=tensor(m),valid=tensor(z['valid']),K=tensor(z['K']),w2c=tensor(z['w2c']),dilated=tensor(cv2.dilate(m,np.ones((7,7),np.uint8)))));selected.append(dict(camera=c,frame=f,estimated_common_index=f+clocks[c],deviation_from_reference=f+clocks[c]-612,input_sha256=sha(p)))
init=visual_hull(M,obs);pts=init['points'].astype(np.float32);colors=init['colors'].astype(np.float32);del obs;torch.cuda.empty_cache()
tree=cKDTree(pts);pairs=tree.query_pairs(float(init['voxel_size'])*1.8,output_type='ndarray');adj=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(pts),len(pts))).tocsr();nc,labels=connected_components(adj,directed=False);counts=np.bincount(labels);major=np.flatnonzero(counts>=128);assert 1<=len(major)<=12
major_indices=np.where(np.isin(labels,major))[0];minor_indices=np.where(~np.isin(labels,major))[0]
if len(minor_indices):labels[minor_indices]=labels[major_indices[cKDTree(pts[major_indices]).query(pts[minor_indices])[1]]]
labels=np.searchsorted(major,labels);counts=np.bincount(labels);weights=np.sqrt(counts);budgets=np.full(len(counts),16)+np.floor((384-16*len(counts))*weights/weights.sum()).astype(int)
for i in np.argsort(-counts)[:384-budgets.sum()]:budgets[i]+=1
assert budgets.sum()==384 and np.all(budgets<=counts)
chosen=[]
for c,n in enumerate(budgets):
    ids=np.where(labels==c)[0];x=pts[ids].astype(float);sel=int(np.argmin(np.linalg.norm(x-x.mean(0),axis=1)));best=np.full(len(x),np.inf)
    for _ in range(int(n)):
        chosen.append(int(ids[sel]));best=np.minimum(best,((x-x[sel])**2).sum(1));sel=int(np.argmax(best))
chosen=np.array(chosen);nodes=pts[chosen];node_class=labels[chosen];near=np.empty((len(pts),4),np.int64);w=np.empty((len(pts),4),np.float32);edges=[];ew=[]
for c in range(len(counts)):
    ii=np.where(labels==c)[0];ni=np.where(node_class==c)[0];tree=cKDTree(nodes[ni]);dd,jj=tree.query(pts[ii],k=4);near[ii]=ni[jj];ww=np.exp(-.5*(dd/np.maximum(dd[:,-1:]*.5,1e-5))**2);ww[dd[:,0]<1e-7]=[1,0,0,0];ww/=ww.sum(1,keepdims=True);w[ii]=ww
    dd,jj=tree.query(nodes[ni],k=9)
    for k,i in enumerate(ni):
        for dist,j in zip(dd[k,1:],jj[k,1:]):edges.append([i,ni[j]]);ew.append(np.exp(-dist*dist/(2*max(dd[k,-1],1e-5)**2)))
edges=np.array(edges,dtype=np.int64);assert np.all(node_class[near]==labels[:,None]) and np.max(abs(w.sum(1)-1))<2e-7
np.savez_compressed(M/'initial.npz',points=pts,colors=colors,component=labels)
np.savez_compressed(M/'graph.npz',node_points=nodes,node_anchor=node_class,edges=edges,edge_weights=np.array(ew,np.float32),source_point_indices=chosen,point_node_indices=near,point_node_weights=w,point_component=labels)
save(M/'initialization-and-graph.json',dict(status='train_only_hull_and384_node_graph_prepared_runtime_probe_required',points=len(pts),scene_scale=float(np.linalg.norm(pts.max(0)-pts.min(0))*.5),initial_sha256=sha(M/'initial.npz'),graph_sha256=sha(M/'graph.npz'),nodes=384,component_point_counts=counts.tolist(),nodes_per_component=budgets.tolist(),minor_fragment_points_associated_with_nearest_major_component=len(minor_indices),selected_training_frames=selected,clock_sha256=sha(Q/'candidate-clock-offsets.json'),hull='Same original native observations at nearest available even source indices to common612; residual timing remains.14/15 silhouettes with3pixel dilation,0.012 dataset world units (metric scale unverified).',component_semantics='Connected visual-hull components, not person/hand/object semantic groundtruth. Small fragments assigned nearest major component; no points removed. All graph neighbors and edges remain within these components.',author_surface_used=False,heldout_RGB_opened=False,body_template_used=False,goal_complete=False));print('ACTOR23_INITIALIZED',len(pts),counts.tolist(),flush=True)
