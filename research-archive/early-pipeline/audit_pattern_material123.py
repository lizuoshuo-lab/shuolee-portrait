"""Independent coordinate, graph and split audit of the one saved material pass."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/pattern-preflight/native-material'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'independent-audit.json').exists()
result=read(O/'result.json');manifest=read(O/'feature-manifest.json')['frames'];index=np.load(O/'component-index.npz',allow_pickle=False)
offset=index['offsets'];features=[];max_map=0.;out_of_bounds=0
for slot,f in enumerate(manifest):
    p=O/'features'/f['feature_file'];assert sha(p)==f['sha256']
    d=np.load(p,allow_pickle=False);xy=d['xy'];origin=d['crop_bounds'][d['region'],:2]
    max_map=max(max_map,float(np.abs((d['local_xy']+origin)-xy).max()))
    out_of_bounds+=int((~((xy[:,0]>=0)&(xy[:,0]<2704)&(xy[:,1]>=0)&(xy[:,1]<1520))).sum())
    assert len(xy)==offset[slot+1]-offset[slot]==f['total'];features.append(d)
assert max_map==0 and out_of_bounds==0
edges=[];dist_errors=[];pair_hashes=[]
for row in read(O/'pair-counts.json')['records']:
    i,j=row['i'],row['j'];p=O/'pairs'/f'{i:03d}_{j:03d}.npz';d=np.load(p,allow_pickle=False)
    fi,fd,bi,bd=d['forward_indices'],d['forward_distances'],d['reverse_indices'],d['reverse_distances']
    k=fi[:,0];good=(fd[:,0]<.75*fd[:,1])&(bd[k,0]<.75*bd[k,1])&(bi[k,0]==np.arange(len(fi)))
    aa=np.flatnonzero(good);matches=np.c_[aa,k[aa]]
    assert np.array_equal(matches,d['accepted_pairs']) and len(matches)==row['matches']
    edges.append(matches+np.array([offset[i],offset[j]]))
    if len(matches):
        sample=matches[np.unique([0,len(matches)//2,len(matches)-1])]
        actual=np.linalg.norm(features[i]['descriptors'][sample[:,0]].astype(float)-features[j]['descriptors'][sample[:,1]].astype(float),axis=1)
        dist_errors.extend(np.abs(actual-fd[sample[:,0],0]).tolist())
    pair_hashes.append(dict(file=p.name,sha256=sha(p)))
assert max(dist_errors)<.001
edge=np.concatenate(edges);N=int(offset[-1])
def independent_labels(edge):
    g=coo_matrix((np.ones(len(edge),np.int8),(edge[:,0],edge[:,1])),shape=(N,N)).tocsr()
    return connected_components(g,directed=False,return_labels=True)[1]
def equal_partition(a,b):
    pairs=np.unique(np.stack([a,b],1),axis=0)
    return len(pairs)==len(np.unique(a))==len(np.unique(b))
lab=independent_labels(edge);assert equal_partition(lab,index['appearance_component'])
near=index['within_frame_near_pairs'];ne=[];near_dist_max=0.
for slot,aa,bb in near:
    distance=float(np.linalg.norm(features[slot]['xy'][aa]-features[slot]['xy'][bb]));assert distance<=3.+1e-10;near_dist_max=max(near_dist_max,distance)
    ne.append([offset[slot]+aa,offset[slot]+bb])
flab=independent_labels(np.concatenate([edge,np.array(ne,dtype=np.int64).reshape(-1,2)]));assert equal_partition(flab,index['duplicate_family'])
families=read(O/'duplicate-families.json')['records'];tracks=read(O/'appearance-tracks.json')['records']
fdict={f['id']:f for f in families}
def identity(nodes):return hashlib.sha256(json.dumps(nodes,separators=(',',':')).encode()).hexdigest()
for f in families:
    assert identity(f['nodes'])==f['id']
    assert f['split']==('HELD' if int(f['id'][:8],16)%5==0 else 'FIT')
for tr in tracks:
    assert identity(tr['nodes'])==tr['id'] and tr['split']==fdict[tr['family_id']]['split']
    assert tr['quarantined']==fdict[tr['family_id']]['quarantined']
    if not tr['quarantined']:assert len({i for i,j in tr['nodes']})==len(tr['nodes'])
save(O/'pair-hashes.json',dict(records=pair_hashes))
summary=dict(updated=datetime.now(timezone.utc).isoformat(),state='implementation_checks_passed',frames=len(manifest),native_points=N,pairs=len(pair_hashes),
    native_crop_offset_parity_max_px=max_map,out_of_image_features=out_of_bounds,accepted_mutual_ratio_edges_recomputed=int(len(edge)),
    sampled_descriptor_distances=len(dist_errors),descriptor_L2_max_difference=max(dist_errors),
    independent_sparse_graph_matches=True,independent_duplicate_family_graph_matches=True,near_pair_max_px=near_dist_max,
    complete_family_split_consistent=True,unquarantined_tracks_have_unique_frame=True,
    source_manifest_sha256=sha(O/'feature-manifest.json'),source_tracks_sha256=sha(O/'appearance-tracks.json'),source_families_sha256=sha(O/'duplicate-families.json'),
    physical_identity_verified=False,physical_face_HELD=0,calibration_accepted=False,training_started=False,
    limitation='Coordinate arithmetic, descriptor indexing and graph/split implementation only. Similar texture, independence of physical surface, true correspondence and camera accuracy remain unverified.')
save(O/'independent-audit.json',summary);print(json.dumps(summary,ensure_ascii=False))
