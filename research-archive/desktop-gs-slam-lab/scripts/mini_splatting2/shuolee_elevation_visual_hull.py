"""Coarse shape-from-silhouette initialization from the16horizontal training cameras.

This is a CPU visual-hull implementation, not a GaussianObject reproduction.
The hull encodes silhouette compatibility, not measured or complete geometry.
"""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linprog
from scipy.ndimage import map_coordinates, minimum_filter, label
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from shuolee_audit_diffuman_cameras import LAB, OUT
from shuolee_prepare_elevation_benchmark import digest,dump


def ply(path,xyz,normals,colors,faces=None):
    header=['ply','format binary_little_endian 1.0',f'element vertex {len(xyz)}',
        'property float x','property float y','property float z','property float nx','property float ny','property float nz',
        'property uchar red','property uchar green','property uchar blue']
    if faces is not None:header += [f'element face {len(faces)}','property list uchar int vertex_indices']
    header += ['end_header']
    dt=np.dtype([(n,'<f4') for n in ['x','y','z','nx','ny','nz']]+[(n,'u1') for n in ['red','green','blue']])
    data=np.empty(len(xyz),dtype=dt)
    for i,n in enumerate(['x','y','z']):data[n]=xyz[:,i]
    for i,n in enumerate(['nx','ny','nz']):data[n]=normals[:,i]
    for i,n in enumerate(['red','green','blue']):data[n]=colors[:,i]
    with path.open('xb') as f:
        f.write(('\n'.join(header)+'\n').encode('ascii'));f.write(data.tobytes())
        if faces is not None:
            d=np.empty(len(faces),dtype=[('count','u1'),('indices','<i4',(3,))]);d['count']=3;d['indices']=faces;f.write(d.tobytes())


def main():
    cv2.setNumThreads(2)
    root=OUT/'elevation-benchmark/training-geometry'
    state=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert state['fixed_poses_verified'] and not state['reads_development_or_test_pixels']
    dest=root/'visual-hull'
    if dest.exists():
        assert {p.name for p in dest.iterdir()} <= {'protocol.json'},'Do not overwrite a completed or partially exported hull'
    dest.mkdir(exist_ok=True)
    protocol=dict(owner='shuolee',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        source='16horizontal training silhouettes only; no test/development images, no learned generation, no author point cloud.',
        method='CPU voxel visual hull plus marching cubes, with image-validity-aware carving.',
        reference='https://doi.org/10.1109/34.273735',gaussian_application='https://github.com/chensjtu/GaussianObject',
        not_official_reproduction=True,foreground_threshold=.25,silhouette_dilation_pixels=3,
        maximum_long_axis_cells=192,maximum_total_cells=4000000,minimum_observed_cameras=12,
        maximum_inconsistent_known_silhouettes=0,maximum_initialization_points=30000,
        fixed_random_seed=0,units='Author calibrated coordinate units; metric scale not independently measured.',
        color='Training-view visibility approximation for initial color only; fallback to training foreground median.',
        limits=['Silhouettes do not constrain invisible concavities, texture, or complete fine geometry.',
            'Dilation accommodates mask/calibration errors and makes this a relaxed, coarse hull.',
            'Clipped image pixels are unknown, not background. At least12observing cameras are required.',
            'This artifact is initialization; a visually pleasing mesh is not final Gaussian quality acceptance.'])
    if not (dest/'protocol.json').exists():dump(dest/'protocol.json',protocol)
    cameras=[];A=[];B=[];median_colors=[]
    for c in state['rows']:
        name=c['camera'];K=np.asarray(c['K']);V=np.asarray(c['w2c']);R=V[:3,:3];t=V[:3,3]
        mask=np.asarray(Image.open(root/'silhouettes'/(name+'.png')).convert('L'))
        valid=np.asarray(Image.open(root/'source-validity'/(name+'.png')).convert('L'))
        rgb=np.asarray(Image.open(root/'images'/(name+'.png')).convert('RGB'))
        assert digest(root/'images'/(name+'.png'))==c['image_sha256']
        soft=mask.astype('float32')/255
        silhouette=cv2.dilate((soft>.25).astype('uint8'),np.ones((7,7),np.uint8))
        ys,xs=np.nonzero(silhouette)
        x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
        # Viewing cones of bounding rectangles give a finite conservative box.
        for axis,lo,hi,size in [(0,x0,x1,c['width']),(1,y0,y1,c['height'])]:
            if lo>1:
                edge=(lo-K[axis,2])/K[axis,axis]
                A.append(edge*R[2]-R[axis]);B.append(t[axis]-edge*t[2])
            if hi<size-2:
                edge=(hi-K[axis,2])/K[axis,axis]
                A.append(R[axis]-edge*R[2]);B.append(edge*t[2]-t[axis])
        A.append(-R[2]);B.append(t[2]-.1)
        median_colors.append(np.median(rgb[soft>.99],axis=0))
        cameras.append(dict(**c,silhouette=silhouette,valid=valid,Karray=K,Varray=V,rgb=rgb,soft=soft))
    A=np.asarray(A);B=np.asarray(B);bounds=[]
    for axis in range(3):
        e=np.eye(3)[axis];lo=linprog(e,A_ub=A,b_ub=B,bounds=[(None,None)]*3,method='highs')
        hi=linprog(-e,A_ub=A,b_ub=B,bounds=[(None,None)]*3,method='highs')
        assert lo.success and hi.success,(lo.message,hi.message)
        bounds.append((lo.fun,-hi.fun))
    box=np.asarray(bounds);extent=box[:,1]-box[:,0]
    step=float(extent.max()/190);box[:,0]-=step*3;box[:,1]+=step*3
    shape=np.ceil((box[:,1]-box[:,0])/step).astype(int)+1
    if np.prod(shape)>4000000:
        step*=float(np.cbrt(np.prod(shape)/4000000));shape=np.ceil((box[:,1]-box[:,0])/step).astype(int)+1
    assert np.prod(shape)<=4100000 and (extent>0).all()
    total=int(np.prod(shape));flat=np.zeros(total,dtype=bool);start=time.time()
    origin=box[:,0];minimum_known=16
    # OpenCV remap destination dimensions must be below SHRT_MAX.
    for begin in range(0,total,16384):
        idx=np.arange(begin,min(begin+16384,total))
        ijk=np.stack(np.unravel_index(idx,shape),-1)
        xyz=origin+ijk*step;accepted=np.ones(len(xyz),bool);known_count=np.zeros(len(xyz),np.uint8)
        for c in cameras:
            V=c['Varray'];K=c['Karray'];q=xyz@V[:3,:3].T+V[:3,3]
            uv=q[:,:2]/q[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
            x=uv[:,0,None].astype('float32');y=uv[:,1,None].astype('float32')
            known=(q[:,2]>.1)&(cv2.remap(c['valid'],x,y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).ravel()>0)
            fg=cv2.remap(c['silhouette'],x,y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).ravel()>0
            accepted &= (~known)|fg;known_count+=known
        accepted &= known_count>=12;flat[idx]=accepted
        if accepted.any():minimum_known=min(minimum_known,int(known_count[accepted].min()))
    volume=flat.reshape(tuple(shape));assert volume.sum()>100
    assert not any(x.any() for x in [volume[0],volume[-1],volume[:,0],volume[:,-1],volume[:,:,0],volume[:,:,-1]])
    vertices,faces,normals,_=marching_cubes(volume.astype('float32'),.5,spacing=(step,step,step))
    vertices+=origin
    # Orient each vertex normal towards the empty side of the voxel surface.
    plus=map_coordinates(volume.astype('float32'),((vertices+normals*step-origin)/step).T,order=1,mode='constant')
    minus=map_coordinates(volume.astype('float32'),((vertices-normals*step-origin)/step).T,order=1,mode='constant')
    normals[plus>minus]*=-1
    colors=np.tile(np.uint8(np.median(median_colors,axis=0)),(len(vertices),1))
    scores=np.zeros(len(vertices),dtype='float32')
    for c in cameras:
        V=c['Varray'];K=c['Karray'];q=vertices@V[:3,:3].T+V[:3,3]
        center=-V[:3,:3].T@V[:3,3];direction=center-vertices
        cosine=np.sum(normals*direction/np.linalg.norm(direction,axis=-1,keepdims=True),axis=-1)
        uv=q[:,:2]/q[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
        ix=np.round(uv[:,0]).astype(int);iy=np.round(uv[:,1]).astype(int)
        ok=(q[:,2]>.1)&(ix>=0)&(iy>=0)&(ix<c['width'])&(iy<c['height'])
        zbuf=np.full((c['height'],c['width']),np.inf,dtype='float32')
        np.minimum.at(zbuf,(iy[ok],ix[ok]),q[ok,2])
        zbuf=minimum_filter(zbuf,size=5)
        visible=np.zeros(len(vertices),bool);sel=np.flatnonzero(ok)
        visible[sel]=(q[sel,2]<=zbuf[iy[sel],ix[sel]]+3*step)&(c['soft'][iy[sel],ix[sel]]>.95)
        score=np.maximum(cosine,0)**4
        choose=visible&(score>scores)
        colors[choose]=c['rgb'][iy[choose],ix[choose]];scores[choose]=score[choose]
    ply(dest/'surface.ply',vertices,normals,colors,faces)
    rng=np.random.default_rng(0);selected=np.arange(len(vertices))
    if len(selected)>30000:selected=np.sort(rng.choice(selected,30000,replace=False))
    ply(dest/'initialization.ply',vertices[selected],normals[selected],colors[selected])
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
    import read_write_model as rw
    points=rw.read_points3D_binary(str(root/'sparse/0/points3D.bin'))
    distance,_=cKDTree(vertices).query(np.array([p.xyz for p in points.values()]))
    components,ncomponents=label(volume);sizes=np.bincount(components.ravel())[1:]
    result=dict(status='coarse_surface_initialized_pending_review',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        elapsed_seconds=time.time()-start,grid_shape=shape.tolist(),voxel_step=step,bounds=box.tolist(),
        occupied_voxels=int(volume.sum()),volume_in_calibration_units=float(volume.sum()*step**3),
        surface_vertices=len(vertices),surface_faces=len(faces),initialization_points=len(selected),
        minimum_observing_cameras=minimum_known,components=ncomponents,largest_components=sorted(sizes.tolist(),reverse=True)[:5],
        color_observation_fraction=float((scores>0).mean()),sparse_point_count=len(points),
        sparse_to_surface_median=float(np.median(distance)),sparse_to_surface_p90=float(np.quantile(distance,.9)),
        sparse_to_surface_scope='Shared training evidence only; visual hull cannot recover all concavities.',
        test_or_development_rgb_used=False,training_started=False,accepted_as_best=False,
        initialization_sha256=digest(dest/'initialization.ply'))
    dump(dest/'manifest.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
