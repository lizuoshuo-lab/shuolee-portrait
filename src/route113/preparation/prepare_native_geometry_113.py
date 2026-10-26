"""Fresh 175-view silhouette directions and point-only shared targets.

No previous subject geometry is used. Input 101 is read for its original 2-D
observations, camera matrices and split flags only; its old XYZ is not read.
"""
from pathlib import Path
import json, time
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from PIL import Image
import shuolee_common as c

R=c.LAB/'outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-native-align-fresh-113' if hasattr(c,'LAB') else Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-native-align-fresh-113')
B=R.parent.parent

def qstat(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=int(x.size),min=float(x.min()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max())) if x.size else dict(n=0)

def normals():
    out=R/'surface-init';out.mkdir(exist_ok=True);assert not (out/'normals.npz').exists()
    c.save(out/'protocol.json',dict(source='fresh113 silhouette occupancy and fresh neutral head only',spacing=.012,threshold=.97,
        smoothing_voxels=1.,confidence='SDF gradient norm >=0.1; nearest-boundary distance reported, not a body-distance filter',
        orientation='outward negative gradient of signed inside-positive distance; normal sign does not affect plane intersection',
        centers_changed=False,normal_std_ratio=.25,training_updates=0))
    torch=c.runtime();g=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
    fresh=np.load(R/'fit/fitted.npz');rows=c.read(R/'data/training-inputs.json')['records'];ref=69
    scale=float(fresh['world_scale']);rot=fresh['root_rotation'][ref];translation=fresh['world_translation'][ref]
    local=(fresh['vertices_world'][ref]-translation)@rot/scale;spacing=.012
    axes=[np.arange(local[:,j].min()-.18,local[:,j].max()+.18,spacing) for j in range(3)]
    shape=tuple(map(len,axes));grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    world=grid@rot.T*scale+translation;pts=g(world);votes=torch.zeros(len(pts),device='cuda');total=0.
    with torch.no_grad():
        for row in rows:
            assert c.sha(row['mask'])==row['mask_sha256']
            mask=g(np.asarray(Image.open(row['mask']).convert('L').resize((288,512),Image.Resampling.BILINEAR))/255.)[None,None]
            mask=torch.nn.functional.max_pool2d(mask,3,1,1);E=g(row['w2c']);K=g(row['K']);cam=pts@E[:3,:3].T+E[:3,3]
            q=cam@K.T;uv=q[:,:2]/q[:,2:].clamp_min(.01)-.5;uv=2*uv/(g(row['wh'])-1)-1
            inside=torch.nn.functional.grid_sample(mask,uv[None,:,None],align_corners=True,padding_mode='zeros')[0,0,:,0]>.5
            weight=row.get('observation_weight',1. if row['camera_kind']=='calibrated' else .5)
            votes+=weight*(inside&(cam[:,2]>0));total+=weight
    occ=(votes/total>.97).cpu().numpy().reshape(shape);boundary=occ&~binary_erosion(occ)
    sdf=distance_transform_edt(occ)*spacing-distance_transform_edt(~occ)*spacing
    grad=-np.stack(np.gradient(gaussian_filter(sdf,1.),spacing),-1)
    unit=grad/(np.linalg.norm(grad,axis=-1,keepdims=True)+1e-30)
    surface=grid[np.flatnonzero(boundary)];npz=np.load(R/'clothing-init/points.npz');xyz=npz['xyz'];before=c.sha(R/'clothing-init/points.npz')
    loc=(xyz-translation)@rot/scale;distance,near=cKDTree(surface).query(loc)
    n=unit[boundary][near]@rot.T;confidence=np.linalg.norm(grad,axis=-1)[boundary][near]>=.1
    source=np.zeros(len(xyz),np.uint8);mp=np.load(R/'clothing-init/head-sample-map.npz');head=np.load(R/'head-fit/rigid_neutral.npz')
    tri=head['vertices'][head['faces'][mp['triangle'][mp['supported']]]];hn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);hn/=np.linalg.norm(hn,axis=-1,keepdims=True)
    ids=mp['changed_ids'];n[ids]=hn;confidence[ids]=True;source[ids]=1
    n/=np.linalg.norm(n,axis=1,keepdims=True)
    # Unit quaternion (w,x,y,z), rotating local +Z onto the world normal.
    quat=np.c_[1+n[:,2],-n[:,1],n[:,0],np.zeros(len(n))];flip=quat[:,0]<1e-8;quat[flip]=[0,1,0,0];quat/=np.linalg.norm(quat,axis=1,keepdims=True)
    assert np.isfinite(n).all() and np.isfinite(quat).all() and len(xyz)==200000
    np.savez_compressed(out/'normals.npz',normal=n.astype(np.float32),quaternion=quat.astype(np.float32),confidence=confidence,source=source,nearest_surface_distance_model=distance.astype(np.float32),points_sha256=before)
    assert c.sha(R/'clothing-init/points.npz')==before
    c.save(out/'result.json',dict(created=c.stamp(),points=len(xyz),surface_voxels=len(surface),world_scale=scale,unique_normal_count=int(confidence.sum()),
        uncertain_count=int((~confidence).sum()),head_patch_count=len(ids),nearest_surface_distance_model=qstat(distance),norm_error_max=float(abs(np.linalg.norm(n,axis=1)-1).max()),
        centers_bitwise_unchanged=True,points_sha256=before,normals_sha256=c.sha(out/'normals.npz'),old_geometry_loaded=False,optimizer_updates=0))
    print('113 fresh normals completed',flush=True)

def project(x,E,K):
    v=np.einsum('fij,j->fi',E[:,:3,:3],x)+E[:,:3,3];p=v@K.T
    return p[:,:2]/p[:,2:]-.5,v[:,2]

def fit(uv,E,K):
    origin=-E[0,:3,:3].T@E[0,:3,3];P=E[:,:3].copy();P[:,:,3]+=P[:,:,:3]@origin
    q=np.c_[uv+.5,np.ones(len(uv))]@np.linalg.inv(K).T
    A=np.concatenate([q[:,0,None]*P[:,2]-P[:,0],q[:,1,None]*P[:,2]-P[:,1]])
    _,_,vt=np.linalg.svd(A,full_matrices=False);x=vt[-1,:3]/vt[-1,3]+origin
    fun=lambda y:(project(y,E,K)[0]-uv).ravel()
    rr=least_squares(fun,x,loss='soft_l1',f_scale=2,max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    pred,z=project(rr.x,E,K);err=np.linalg.norm(pred-uv,axis=-1);pixel_sigma=max(2.,float(np.median(err)))
    jac=np.stack([(fun(rr.x+np.eye(3)[i]*1e-5)-fun(rr.x-np.eye(3)[i]*1e-5))/2e-5 for i in range(3)],1)
    covariance=np.linalg.pinv(jac.T@jac,rcond=1e-10)*pixel_sigma**2
    sigma=max(float(np.sqrt(max(np.linalg.eigvalsh(covariance).max(),0))),pixel_sigma*float(np.median(z))/np.sqrt(K[0,0]*K[1,1]))
    return rr.x, bool(rr.success), sigma, err

def targets():
    out=R/'shared-targets';out.mkdir(exist_ok=True);assert not (out/'native-targets.npz').exists()
    src=B/'current/shared-point-bundle-convergence-101/input.npz'
    c.save(out/'protocol.json',dict(created=c.stamp(),source=str(src),source_sha256=c.sha(src),old_xyz_read=False,
        association='Use actual full-resolution foreground masks at each original observation. At least90percent of a material track must be confidently foreground; only foreground observations participate. This is a geometric prerequisite, not a 3D residual filter.',
        split='Material trackID modulo5==0 is disjoint validation only. Existing temporal held flags remain excluded from XYZfit and auxiliary training. Face detector held flags also excluded. All photos remain ordinary photometric training, so this is not novel-image GT.',
        admission='At least4fit observations and2held observations for material; >=4fit face observations. Positive depth and finite converged DLT+robust XYZ required. All failures reported; no residual-based filtering.',
        uncertainty='Max of conditioned triangulation marginal sigma and training-view median reprojection pixel sigma converted to world scale; >=2source px floor. Held residuals never select sigma.',
        regions='Face landmark bbox in the image; otherwise relative vertical location inside foreground bbox: upperbody<=.52, lowerbody<=.86, feet below. These are image regions, not anatomical ground truth.',
        optimization='Only new shared XYZ from original2D. Fixed cameras, no old subject state, no dynamic warp or camera refinement.'))
    with np.load(src) as z:
        keys=['observations','offsets','held','original_track_ids','K','w2c','face_ids','face_observations','face_point_indices','face_held']
        a={k:z[k] for k in keys}
    rows=c.read(R/'data/training-inputs.json')['records'];K=a['K'];E=a['w2c'];assert np.array_equal(E,np.asarray([r['w2c'] for r in rows]))
    assert all(np.array_equal(K,np.asarray(r['K'])) for r in rows)
    allob=a['observations'];fo=a['face_observations'];fg=np.zeros(len(allob),bool);regions=np.zeros(len(allob),np.int16);bboxes=[]
    for i,row in enumerate(rows):
        mask=np.asarray(Image.open(row['mask']).convert('L'));ys,xs=np.where(mask>250);bb=[xs.min(),ys.min(),xs.max(),ys.max()];bboxes.append(bb)
        ii=np.flatnonzero(allob[:,0].astype(int)==i);uv=np.rint(allob[ii,1:]).astype(int)
        inbounds=(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0]);okay=np.zeros(len(ii),bool);okay[inbounds]=mask[uv[inbounds,1],uv[inbounds,0]]>250;fg[ii]=okay
        yn=(allob[ii,2]-bb[1])/max(bb[3]-bb[1],1);reg=np.where(yn<=.52,1,np.where(yn<=.86,2,3))
        ff=fo[fo[:,0].astype(int)==i,1:]
        if len(ff)>=4:
            low=np.min(ff,axis=0)-10;high=np.max(ff,axis=0)+10;reg[((allob[ii,1:]>=low)&(allob[ii,1:]<=high)).all(1)]=0
        regions[ii]=reg
    observations=[];pointids=[];xyzs=[];sigmas=[];records=[];failures=[]
    def add(pid,ob,held,region,disjoint,kind):
        frame=ob[:,0].astype(int);fitids=~held
        if fitids.sum()<4 or (kind=='material' and held.sum()<2):return
        x,ok,sigma,err=fit(ob[fitids,1:],E[frame[fitids]],K);pred,depth=project(x,E[frame],K);residual=np.linalg.norm(pred-ob[:,1:],axis=-1)
        valid=ok and np.isfinite(x).all() and np.isfinite(sigma) and sigma>0 and (depth>.2).all()
        rec=dict(id=int(pid),kind=kind,disjoint=bool(disjoint),fit=qstat(residual[fitids]),held=qstat(residual[held]),valid=bool(valid),views=len(ob),sigma_world=float(sigma))
        records.append(rec)
        if not valid:failures.append(rec);return
        j=len(xyzs);pointids.append(pid);xyzs.append(x);sigmas.append(sigma)
        for f,uv,h,reg in zip(frame,ob[:,1:],held,region):observations.append([f,j,*uv,int(h),int(disjoint),int(reg)])
    for j,(lo,hi) in enumerate(zip(a['offsets'][:-1],a['offsets'][1:])):
        if hi-lo<6 or fg[lo:hi].mean()<.9:continue
        good=np.flatnonzero(fg[lo:hi])+lo;pid=int(a['original_track_ids'][j])
        add(pid,allob[good],a['held'][good].astype(bool),regions[good],pid%5==0,'material')
    for j,pid in enumerate(a['face_ids']):
        ii=np.flatnonzero(a['face_point_indices']==j)
        add(10000000+int(pid),fo[ii],a['face_held'][ii].astype(bool),np.zeros(len(ii),int),False,'named_face')
    ob=np.asarray(observations);xyz=np.asarray(xyzs);sigma=np.asarray(sigmas);assert len(xyz)>50
    np.savez_compressed(out/'native-targets.npz',xyz=xyz,sigma_world=sigma,point_ids=np.asarray(pointids),observations=ob,K=K,w2c=E,wh=np.asarray([r['wh'] for r in rows]))
    use=(ob[:,4]==0)&(ob[:,5]==0);coverage=[]
    for i in range(175):
        q=ob[use&(ob[:,0]==i)];coverage.append(dict(raw=i+1,count=len(q),regions=np.bincount(q[:,6].astype(int),minlength=4).tolist()))
    c.save(out/'result.json',dict(created=c.stamp(),shared_points=len(xyz),observations=len(ob),auxiliary_observations=int(use.sum()),
        frames_with_auxiliary=int(sum(r['count']>0 for r in coverage)),minimum_per_frame=min(r['count'] for r in coverage),
        named_face_points=sum(r['kind']=='named_face' and r['valid'] for r in records),failures=failures,records=records,coverage=coverage,
        old_xyz_read=False,source_sha256=c.sha(src),target_sha256=c.sha(out/'native-targets.npz'),not_ground_truth=True,optimizer_updates=0))
    print(json.dumps(dict(points=len(xyz),observations=len(ob),frames_with_auxiliary=sum(r['count']>0 for r in coverage),failures=len(failures))),flush=True)

if __name__=='__main__':
    import sys
    (normals if sys.argv[1]=='normals' else targets)()
