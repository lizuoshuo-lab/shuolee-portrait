"""Free clothing-envelope seeds from fixed native cameras and observed pixels.

This is an outer visual hull, not measured skin, concavity or unseen geometry.
Occluded/out-of-image pixels abstain; they are not background votes. A surface
touching an unknown-support region is refused rather than closed artificially.
"""
from pathlib import Path
from native_residual_camera122 import project_world,validate_camera
from native_residual_domain122 import sampled_domain_check
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
from skimage.measure import marching_cubes
from native_training_admission122_v2 import checked_asset,read,sha,save

def evaluate_support(points,views,cameras,base,*,tolerance_px,min_views,vote_fraction):
    if type(tolerance_px)is not int or tolerance_px<0:raise ValueError('Explicit integer silhouette tolerance required')
    if type(min_views)is not int or not 3<=min_views<=len(views):raise ValueError('At least three observed views required')
    if not 0<vote_fraction<=1:raise ValueError('Invalid fixed vote fraction')
    points=np.asarray(points,np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError("Finite world point matrix required")
    if len({r["frame_id"] for r in views})!=len(views):raise ValueError("Distinct observed views required")
    positive=np.zeros(len(points),np.int32);observed=np.zeros(len(points),np.int32)
    for row in views:
        camera=cameras[row['frame_id']];validate_camera(camera);sampled_domain_check(camera)
        if camera['id']!=str(row['frame_id']):raise ValueError('Surface camera/frame identity mismatch')
        E=np.asarray(camera['w2c'],np.float64)
        with Image.open(checked_asset(base,row['alpha'])) as im:alpha=np.array(im)
        with Image.open(checked_asset(base,row['observation_valid'])) as im:valid=np.array(im)
        if alpha.ndim!=2 or valid.shape!=alpha.shape or not np.isin(valid,[0,255]).all():raise ValueError('Native alpha and Boolean observation support required')
        foreground=alpha>=128
        if tolerance_px:foreground=binary_dilation(foreground,structure=np.ones((2*tolerance_px+1,)*2,bool))
        foreground &= valid==255
        height,width=alpha.shape
        if [width,height]!=[camera["width"],camera["height"]]:raise ValueError("Native mask/camera raster mismatch")
        for start in range(0,len(points),250000):
            p=points[start:start+250000];cam=p@E[:3,:3].T+E[:3,3]
            uv=np.full((len(p),2),np.nan);positive_depth=cam[:,2]>0
            if positive_depth.any():uv[positive_depth]=project_world(p[positive_depth],camera)-.5
            # CameraK uses corners; raw masks use integer centers. Convert once.
            ok=(cam[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=-.5)&(uv[:,0]<width-.5)&(uv[:,1]>=-.5)&(uv[:,1]<height-.5)
            ids=np.flatnonzero(ok);xy=np.floor(uv[ids]+.5).astype(np.int64)
            seen=valid[xy[:,1],xy[:,0]]==255;ids=ids[seen];xy=xy[seen]
            observed[start+ids]+=1;positive[start+ids]+=foreground[xy[:,1],xy[:,0]].astype(np.int32)
    known=observed>=min_views
    values=np.divide(positive,observed,out=np.zeros(len(points),np.float64),where=observed>0)
    inside=known&(values>vote_fraction)
    return dict(positive=positive,observed=observed,known=known,values=values,inside=inside,
                legacy_all_view_fraction=positive/len(views))

def fit_proposal(fit,*,spacing_m,padding_m,max_voxels):
    for name in ['vertices_world','root_rotation','world_translation','world_scale']:
        if name not in fit or not np.isfinite(fit[name]).all():raise ValueError('Finite newly fitted world geometry required')
    v=np.asarray(fit['vertices_world']);rot=np.asarray(fit['root_rotation']);translation=np.asarray(fit['world_translation'])
    if v.ndim!=3 or v.shape[2]!=3 or not len(v) or rot.shape!=(len(v),3,3) or translation.shape!=(len(v),3):raise ValueError('Shared fit dimensions invalid')
    if not np.array_equal(v,np.broadcast_to(v[0],v.shape)) or not np.array_equal(rot,np.broadcast_to(rot[0],rot.shape)) or not np.array_equal(translation,np.broadcast_to(translation[0],translation.shape)):raise ValueError('Static one-pose fit required; no per-frame deformation')
    R=rot[0];t=translation[0];scale=float(fit['world_scale'])
    if not np.isfinite(scale) or scale<=0 or not np.allclose(R@R.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1,atol=1e-5):raise ValueError('Proper world rotation and positive scale required')
    if not (0<spacing_m<=.05 and padding_m>0):raise ValueError('Explicit model-metre spacing and padding required')
    local=(v[0]-t)@R/scale
    axes=[np.arange(local[:,j].min()-padding_m,local[:,j].max()+padding_m+spacing_m*.5,spacing_m) for j in range(3)]
    shape=tuple(len(x) for x in axes)
    if np.prod(shape)>max_voxels or min(shape)<3:raise ValueError('Registered volume budget exceeded or insufficient volume')
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    return grid@R.T*scale+t,shape,axes,R,t,scale

def initialize_surface(fit,views,cameras,base,output,config):
    out=Path(output)
    if out.exists():raise FileExistsError('Fresh surface output only')
    out.mkdir(parents=True)
    if config['initial_points']!=200000 or config['max_voxels']>5000000:raise ValueError('Registered200k/5M budget required')
    world,shape,axes,R,t,scale=fit_proposal(fit,spacing_m=config['spacing_m'],padding_m=config['padding_m'],max_voxels=config['max_voxels'])
    support=evaluate_support(world,views,cameras,base,tolerance_px=config['mask_tolerance_px'],min_views=config['min_observed_views'],vote_fraction=config['vote_fraction'])
    inside=support['inside'].reshape(shape);known=support['known'].reshape(shape)
    diagnostics=dict(grid_shape=shape,voxels=len(world),occupied=int(inside.sum()),unknown=int((~known).sum()),
        unknown_is_background=False,normalization='foreground votes / observed-valid votes',minimum_observed_views=config['min_observed_views'],
        legacy_all_view_disagreement=int(np.count_nonzero(support['inside']!=(support['legacy_all_view_fraction']>config['vote_fraction']))))
    save(out/'support.json',diagnostics)
    if inside.sum()<1000:raise ValueError('Insufficient supported silhouette volume; no automatic relaxation')
    if any(np.any(np.take(inside,[0,-1],axis=j)) for j in range(3)):raise ValueError('Surface touches fresh proposal boundary')
    if np.any(binary_dilation(inside,structure=np.ones((3,3,3),bool))&~known):raise ValueError('Proposed surface reaches unknown support; do not close unseen geometry')
    # Known empty is determined by actual background votes. The support guard
    # above prevents assigning an artificial zero surface against unknown cells.
    values=support['values'].reshape(shape).copy();values[~known]=0
    vertices,faces,normals,_=marching_cubes(values,level=config['vote_fraction'],spacing=(config['spacing_m'],)*3)
    vertices+=np.array([x[0] for x in axes]);tri=vertices[faces]
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('Empty surface area')
    rng=np.random.default_rng(config['seed']);selected=rng.choice(len(faces),size=200000,p=area/area.sum())
    uv=rng.random((200000,2));u=np.sqrt(uv[:,0]);bary=np.stack([1-u,u*(1-uv[:,1]),u*uv[:,1]],1)
    means=(tri[selected]*bary[:,:,None]).sum(1)@R.T*scale+t
    distances=cKDTree(means).query(means,k=4,workers=2)[0][:,1:]
    std=np.clip(np.sqrt((distances**2).mean(1)),1e-5*scale,.012*scale)
    arrays=dict(means=means.astype(np.float32),quats=np.tile(np.array([1,0,0,0],np.float32),(200000,1)),
        scales=np.repeat(std[:,None],3,1).astype(np.float32),opacities=np.full(200000,.1,np.float32),sh=np.zeros((200000,16,3),np.float32))
    np.savez_compressed(out/'initial.npz',**arrays)
    np.savez_compressed(out/'surface.npz',vertices_world=(vertices@R.T*scale+t).astype(np.float32),faces=faces)
    local_means=(means-t)@R/scale;local_bare=(np.asarray(fit['vertices_world'])[0]-t)@R/scale
    lo=local_bare.min(0);hi=local_bare.max(0)
    outside=np.maximum(np.maximum(lo-local_means,local_means-hi),0)
    result=dict(**diagnostics,points=200000,faces=len(faces),surface_vertices=len(vertices),training_steps=0,
        representation='free_world_clothing_surface',body_fit_use='proposal bounds and model-to-world scale only',
        per_triangle_quota=False,body_attraction=False,dynamic_model=False,
        centers_beyond_bare_proposal_box=int(np.count_nonzero(np.linalg.norm(outside,axis=1)>.001)),
        max_distance_outside_bare_box_model_m=float(np.max(np.linalg.norm(outside,axis=1))),
        world_frame_unchanged=True,input_pixel_convention='integer_centers',camera_projection='opencv_brown12_output',complete_cameras_used=True,arrays_sha256=sha(out/'initial.npz'),
        initializer_source_sha256=sha(__file__),config=config,
        limitations=['Visual hull is not true clothing depth,concavity or skin accuracy.','A fully unseen region remains unsupported.','Native nearest mask sampling has pixel discretization, not subpixel surface accuracy.'],quality_accepted=False)
    save(out/'result.json',result);return result
