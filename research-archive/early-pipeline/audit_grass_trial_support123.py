"""Read-only region/temporal support of the actual fixed-K candidate."""
from trial_grass_surface123 import ROOT, OUT, read, save, sha, now
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

rows=read(ROOT/'data/training-inputs.json')['records'];d=dict(np.load(ROOT/'camera/solver-output.npz'))
ci=d['obs_cam'];pi=d['obs_point'];uv=d['observations'];hits=np.zeros(len(ci),bool)
region=[]
for i,row in enumerate(rows):
    ids=np.flatnonzero(ci==i);xy=np.rint(uv[ids]).astype(int)
    mask=np.array(Image.open(row['mask']).convert('L'))>127
    good=(xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0])
    hits[ids[good]]=mask[xy[good,1],xy[good,0]]
    region.append(dict(frame=row['frame'],source_frame=row['source_frame_index'],
                       foreground=int(hits[ids].sum()),outside_mask=int((~hits[ids]).sum())))
inside=np.bincount(pi,weights=hits,minlength=len(d['points']));count=np.bincount(pi,minlength=len(d['points']))
groups={}
for name,test in [('all_inside_mask',(inside==count)&(count>0)),('mixed',(inside>0)&(inside<count)),('all_outside_mask',inside==0)]:
    point_ids=np.flatnonzero(test);spans=[]
    for j in point_ids:
        frames=ci[pi==j];spans.append(int(frames.max()-frames.min()))
    groups[name]=dict(points=len(point_ids),observations=int(count[point_ids].sum()),
                     median_frame_span=float(np.median(spans)) if spans else None,max_frame_span=max(spans,default=None))
E=d['w2c'];C=-np.einsum('nji,nj->ni',E[:,:3,:3],E[:,:3,3])
delta=np.linalg.norm(np.diff(C,axis=0),axis=1)
angles=Rotation.from_matrix(E[1:,:3,:3]@np.transpose(E[:-1,:3,:3],(0,2,1))).magnitude()*180/np.pi
result=dict(updated=now(),readonly=True,solver_sha256=sha(ROOT/'camera/solver-output.npz'),
    observations=len(ci),foreground_observations=int(hits.sum()),groups=groups,per_frame=region,
    adjacent_camera_rotation_degrees=dict(median=float(np.median(angles)),max=float(angles.max())),
    adjacent_center_distance_world=dict(median=float(np.median(delta)),max=float(delta.max())),
    masks_classify_image_support_not_physical_identity=True,background_stationarity_verified=False,
    no_new_camera_or_threshold_or_training=True)
save(OUT/'camera-support-audit.json',result)
print({k:v for k,v in result.items() if k!='per_frame'})
