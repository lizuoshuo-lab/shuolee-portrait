"""Fresh shared static silhouette surface; no SMPL point binding or image warp.

Adapted from the earlier local initialize_static_clothing_cloud route. This is
an observed silhouette envelope, not metric clothing/skin ground truth. It is
deliberately separate from the previous mesh-bound training implementation.

Historical 119 deliberately retains its all-view vote denominator. Points
outside a camera are counted as negative silhouette evidence in that version;
this known field-of-view limitation is recorded in every new audit. A corrected
visibility policy must use a separately versioned representation contract.
"""
from types import SimpleNamespace
import time
import numpy as np
from PIL import Image
from . import runtime as rt


def make_model(root):
    import torch
    from scene.static_world_gaussian_model import StaticWorldGaussianModel
    from utils.graphics_utils import BasicPointCloud
    model=StaticWorldGaussianModel(3)
    model.load_motion(root/'clothing-init/motion.npz')
    points=np.load(root/'clothing-init/points.npz')['xyz']
    model.create_from_pcd(BasicPointCloud(points,np.full_like(points,.5),np.zeros_like(points)),model.world_scale)
    with torch.no_grad():
        model._scaling.clamp_(max=np.log(.012*model.world_scale))
        model._features_dc.zero_();model._features_rest.zero_()
    assert model.binding is None
    return model


def center_statistics(xyz,fit,rows):
    """Geometry-only diagnostics, independent of Gaussian kernel magnification."""
    import cv2
    from scipy.spatial import cKDTree
    scale=float(fit['world_scale'])
    distance=cKDTree(fit['vertices_world'][0]).query(xyz,workers=4)[0]/scale
    stats=dict(nearest_bare_mesh_VERTEX_distance_m={str(q):float(np.quantile(distance,q)) for q in [.5,.9,.95,.99]},
        centers_over_3cm_from_bare_vertices=int((distance>.03).sum()),
        centers_over_8cm_from_bare_vertices=int((distance>.08).sum()),
        distance_is_not_true_surface_error=True,projected_center_coverage=[])
    for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
        row=rows[i];height=768;width=round(row['wh'][0]*height/row['wh'][1])
        K=np.diag([width/row['wh'][0],height/row['wh'][1],1.])@np.asarray(row['K'])
        E=np.asarray(row['w2c']);p=xyz@E[:3,:3].T+E[:3,3];q=p@K.T
        uv=np.rint(q[:,:2]/np.maximum(q[:,2:],1e-9)-.5).astype(int)
        good=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        hits=np.zeros((height,width),np.uint8);hits[uv[good,1],uv[good,0]]=1
        distance2d=cv2.distanceTransform(1-hits,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        mask=np.array(Image.open(row['mask']).convert('L').resize((width,height),Image.Resampling.NEAREST))>127
        stats['projected_center_coverage'].append(dict(frame=row['frame'],width=width,height=height,
            mask_pixels=int(mask.sum()),fraction_within_2px_of_center=float((distance2d[mask]<=2).mean()),
            center_occlusion_not_resolved=True))
    return stats


def initialize(root,cfg):
    import torch
    import cv2
    from skimage.measure import marching_cubes
    from gaussian_renderer import render
    from .training import camera,image
    if cfg.get('surface_vote_policy','legacy_all_views_denominator')!='legacy_all_views_denominator':
        raise ValueError('This 119 initializer only implements legacy_all_views_denominator; no silent FOV-policy change.')
    gate=rt.read(root/'fit/geometry-gate.json')
    assert gate['coarse_initialization_gate_passed'] and gate['fit_sha256']==rt.sha(root/'fit/fitted.npz')
    out=root/'clothing-init';out.mkdir()
    fit=dict(np.load(root/'fit/fitted.npz'));rows=rt.read(root/'data/training-inputs.json')['records']
    scale=float(fit['world_scale']);R=fit['root_rotation'][0].astype(float);t=fit['world_translation'][0].astype(float)
    assert np.max(abs(fit['vertices_world']-fit['vertices_world'][0]))==0
    local=(fit['vertices_world'][0]-t)@R/scale
    spacing=cfg['surface_spacing_m'];padding=cfg['surface_padding_m']
    axes=[np.arange(local[:,j].min()-padding,local[:,j].max()+padding,spacing) for j in range(3)]
    shape=tuple(len(a) for a in axes)
    if np.prod(shape)>5000000:raise RuntimeError(f'Surface volume exceeds registered5Mvoxel budget:{shape}')
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    points=torch.tensor(grid@R.T*scale+t,device='cuda',dtype=torch.float32)
    votes=torch.zeros(len(points),device='cuda',dtype=torch.int16);start=time.monotonic()
    tolerance=int(cfg['surface_mask_tolerance_px']);kernel=np.ones((2*tolerance+1,)*2,np.uint8)
    with torch.no_grad():
        for i,row in enumerate(rows):
            assert rt.sha(row['mask'])==row['mask_sha256']
            mask=np.array(Image.open(row['mask']).convert('L'))
            assert list(mask.shape[::-1])==row['wh']
            mask=cv2.dilate((mask>127).astype(np.uint8),kernel)
            im=torch.tensor(mask,device='cuda',dtype=torch.float32)[None,None]
            E=torch.tensor(row['w2c'],device='cuda',dtype=torch.float32)
            K=torch.tensor(row['K'],device='cuda',dtype=torch.float32)
            wh=torch.tensor(row['wh'],device='cuda',dtype=torch.float32)
            for begin in range(0,len(points),250000):
                chunk=points[begin:begin+250000];cam=chunk@E[:3,:3].T+E[:3,3];q=cam@K.T
                uv=q[:,:2]/q[:,2:].clamp_min(.001)-.5
                normalized=2*uv/(wh-1)-1
                inside=torch.nn.functional.grid_sample(im,normalized[None,:,None],align_corners=True,padding_mode='zeros')[0,0,:,0]>.5
                votes[begin:begin+len(chunk)]+=(inside&(cam[:,2]>.001)).to(torch.int16)
            if i%16==0 or i==len(rows)-1:
                progress=dict(stage='fresh_clothed_surface',views=i+1,total=len(rows),seconds=time.monotonic()-start)
                rt.save(root/'progress.json',progress);print(progress,flush=True)
    values=(votes.float()/len(rows)).cpu().numpy().reshape(shape)
    threshold=cfg['surface_vote_fraction'];occupied=values>threshold
    if occupied.sum()<1000:raise RuntimeError('Insufficient common silhouette volume; do not relax threshold automatically.')
    if any(np.any(np.take(occupied,[0,-1],axis=j)) for j in range(3)):
        raise RuntimeError('Silhouette volume reaches proposal boundary; do not publish clipped initialization.')
    verts,faces,normals,_=marching_cubes(values,level=threshold,spacing=(spacing,)*3)
    verts+=np.array([a[0] for a in axes]);tri=verts[faces]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
    assert np.isfinite(area).all() and area.sum()>0
    rng=np.random.default_rng(cfg['seed']);count=cfg['initial_points']
    selected=rng.choice(len(faces),size=count,p=area/area.sum());u=rng.random((count,2));s=np.sqrt(u[:,0])
    bary=np.stack([1-s,s*(1-u[:,1]),s*u[:,1]],axis=1)
    seed=np.sum(tri[selected]*bary[:,:,None],axis=1)@R.T*scale+t
    np.savez_compressed(out/'surface.npz',vertices_world=(verts@R.T*scale+t).astype('float32'),faces=faces,
        voxel_spacing_m=spacing,grid_shape=shape,threshold=threshold)
    np.savez_compressed(out/'points.npz',xyz=seed.astype('float32'),reference_frame=0)
    np.savez_compressed(out/'motion.npz',representation='shared_static_world',world_scale=scale,num_timesteps=len(rows),reference_frame=0)
    # This carries metadata only, no affine/pose fields or previous learned data.
    del points,votes,im;torch.cuda.empty_cache()
    model=make_model(root);model.export_world_ply(out/'initialization.ply',0)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    with torch.no_grad():
        for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
            cam,_=camera(rows[i],int(i),768);size=(cam.image_width,cam.image_height)
            Image.open(rows[i]['image']).convert('RGB').resize(size,Image.Resampling.LANCZOS).save(out/f'{rows[i]["frame"]}_input.png')
            image(render(cam,model,pipe,bg)['render']).save(out/f'{rows[i]["frame"]}_initialization.png')
    audit=dict(created=rt.now(),representation='shared_static_world',from_scratch=True,
        old_subject_fit_loaded=False,old_gaussians_loaded=False,old_optimizer_loaded=False,
        body_fit_usage='Fresh broad proposal box and global scale only; no point attraction or triangle membership.',
        points=count,frames=len(rows),occupied_voxels=int(occupied.sum()),surface_vertices=len(verts),
        surface_faces=len(faces),surface_area_model_m2=float(area.sum()),volume_grid_shape=shape,
        surface_spacing_m=spacing,padding_m=padding,mask_tolerance_native_px=tolerance,mask_vote_fraction=threshold,
        surface_vote_policy='legacy_all_views_denominator',
        out_of_fov_is_negative_evidence=True,
        source_code_sha256=rt.sha(__file__),fit_sha256=rt.sha(root/'fit/fitted.npz'),
        camera_sha256=rt.sha(root/'data/training-inputs.json'),point_sha256=rt.sha(out/'points.npz'),
        per_triangle_quota=False,body_attraction=False,training_steps=0,quality_accepted=False,
        initialization_statistics=center_statistics(seed,fit,rows),
        limitations=['Outer silhouette envelope, not independently measured surface depth.',
            'Historical 119 counts out-of-image points as negative silhouette evidence; not a corrected visibility model.',
            'Unseen concavities/occluders/micro-motion or camera error may invalidate this envelope.'])
    rt.save(out/'audit.json',audit);print('SURFACE_READY_FOR_ACTUAL_REVIEW',flush=True)
