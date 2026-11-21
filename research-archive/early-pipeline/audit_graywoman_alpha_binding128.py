"""CPU-only point-alpha/binding/mask proxies, never a front-surface metric."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

import numpy as np
from PIL import Image
from plyfile import PlyData

T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
ROOT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/graywoman-half12-paired18k128'
SNAP=ROOT/'gaussian-run/step-001000'
DELIVERY=ROOT/'delivery/step-001000'
MODEL=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
NAMES=Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/lib/python3.10/site-packages/smplx/joint_names.py')
PROBS=[0.,.1,.5,.9,1.]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def positions(ply):return np.stack([ply[x] for x in ('x','y','z')],axis=1).astype(np.float64)
def project(points,K,E):
    cv=points@E[:3,:3].T+E[:3,3]
    homogeneous=cv@K.T
    return homogeneous[:,:2]/homogeneous[:,2,None],cv[:,2]


def main():
    result=read(SNAP/'result.json');validation=read(DELIVERY/'validation.json')
    assert result['step']==1000 and result['optimizer_updates']==1000 and result['points']==34027
    assert result['reference_frame']=='frame_000023'
    rows=read(ROOT/'data/training-inputs.json')['records']
    row=next(r for r in rows if r['frame']==result['reference_frame'])
    mapping=next(r for r in read(ROOT/'pixel-mapping.json')['records'] if r['frame']==row['frame'])
    source=mapping['source'];camera=read(DELIVERY/'cameras.json')[0]
    viewer=T/'viewer/viewer.js'
    files=[SNAP/'world.ply',SNAP/'bound-local.ply',SNAP/'binding.npz',ROOT/'fit/surface-cache.npz',
        ROOT/'data/training-inputs.json',ROOT/'pixel-mapping.json',Path(source['mask']),Path(row['mask']),
        DELIVERY/'point_cloud.ply',DELIVERY/'cameras.json',DELIVERY/'validation.json',MODEL,NAMES,viewer]
    before={str(p):sha(p) for p in files}
    assert sha(SNAP/'world.ply')==result['world_ply_sha256']
    assert sha(Path(source['mask']))==source['mask_sha256'] and sha(Path(row['mask']))==row['mask_sha256']
    assert sha(DELIVERY/'point_cloud.ply')==validation['sha256']
    text=viewer.read_text(encoding='utf-8')
    assert 'mesh.forEachSplat((_index,p,_scale,_quat,_opacity,c)=>{' in text
    assert "transparent:false,toneMapped:false" in text
    native=PlyData.read(SNAP/'world.ply')['vertex'].data
    portable=PlyData.read(DELIVERY/'point_cloud.ply')['vertex'].data
    bound=PlyData.read(SNAP/'bound-local.ply')['vertex'].data
    xyz=positions(native);portable_xyz=positions(portable)
    assert len(xyz)==34027 and np.array_equal(native['opacity'],portable['opacity'])
    assert np.array_equal(xyz,portable_xyz@np.asarray(validation['rotation']))
    with np.load(SNAP/'binding.npz',allow_pickle=False) as archive: binding=archive['triangle_ids']
    assert np.array_equal(binding,bound['binding_0']) and len(binding)==len(xyz)
    with np.load(ROOT/'fit/surface-cache.npz',allow_pickle=False) as archive:
        faces=archive['faces'];protected=archive['protected_face_vertices']
    with np.load(MODEL,allow_pickle=False) as archive:
        weights=archive['weights'];assert np.array_equal(faces,archive['f'])
    assert weights.shape==(10475,55) and faces.shape==(20908,3)
    tree=ast.parse(NAMES.read_text())
    names=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='JOINT_NAMES' for t in n.targets)))[:55]
    dominant=weights[faces].mean(axis=1).argmax(axis=1)[binding]
    hand_names=[name for name in names if any(part in name for part in ('wrist','index','middle','pinky','ring','thumb'))]
    hand_joints=[i for i,name in enumerate(names) if name in hand_names]
    elbow_joints=[i for i,name in enumerate(names) if name in ('left_elbow','right_elbow')]
    hand=np.isin(dominant,hand_joints);forearm=np.isin(dominant,elbow_joints)
    protected_face=protected[faces].any(axis=1)[binding]
    K=np.asarray(source['K'],dtype=np.float64);Ki=np.asarray(source['K_integer_centers'],dtype=np.float64)
    assert np.array_equal(K[:2,2]-.5,Ki[:2,2])
    uv,z=project(xyz,K,np.asarray(source['w2c'],dtype=np.float64))
    duv,dz=project(xyz,np.asarray(row['K']),np.asarray(row['w2c']))
    suv_integer=uv-.5
    predicted=(np.c_[suv_integer,np.ones(len(uv))]@np.asarray(mapping['integer_center_transform']).T)[:,:2]+.5
    affine_error=float(np.abs(predicted-duv).max());assert affine_error<1e-8
    cameraK=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1.]])
    assert camera['pixelConvention']=='pixel_corners' and np.array_equal(cameraK,np.asarray(row['K']))
    puv,pz=project(portable_xyz,cameraK,np.asarray(validation['actual_cuda_projection']['w2c']))
    portable_error=float(np.abs(puv-duv).max());assert portable_error<1e-8
    mask=np.asarray(Image.open(source['mask']).convert('L'))
    assert list(mask.shape[::-1])==source['wh'] and set(np.unique(mask))=={0,255}
    valid=np.isfinite(uv).all(axis=1)&(z>0)
    in_raster=valid&(uv[:,0]>=0)&(uv[:,0]<source['wh'][0])&(uv[:,1]>=0)&(uv[:,1]<source['wh'][1])
    inside=np.zeros(len(xyz),dtype=bool)
    ij=np.floor(uv[in_raster]).astype(int)
    inside[in_raster]=mask[ij[:,1],ij[:,0]]>0
    outside=in_raster&~inside;unknown=~in_raster
    logits=native['opacity'].astype(np.float64)
    alpha=np.empty_like(logits);positive=logits>=0
    alpha[positive]=1./(1.+np.exp(-logits[positive]))
    ex=np.exp(logits[~positive]);alpha[~positive]=ex/(1.+ex)
    rgb=.28209479177387814*np.stack([native[f'f_dc_{i}'] for i in range(3)],axis=1).astype(np.float64)+.5
    assert np.isfinite(alpha).all() and np.isfinite(xyz).all()
    def stats(selection):
        n=int(selection.sum())
        if not n:return dict(points=0)
        return dict(points=n,alpha_quantiles=np.quantile(alpha[selection],PROBS).tolist(),
            fraction_alpha_below_1_over_255=float((alpha[selection]<1./255).mean()),
            SH0_RGB_median_unclamped=np.median(rgb[selection],axis=0).tolist(),
            xyz_min_model_gauge=xyz[selection].min(axis=0).tolist(),
            xyz_max_model_gauge=xyz[selection].max(axis=0).tolist())
    groups={}
    for name,selection in [('all_centers',np.ones(len(xyz),dtype=bool)),('hand_wrist_fingers_binding_proxy',hand),
        ('forearm_elbow_dominant_binding_proxy',forearm),('hand_plus_forearm_binding_proxy',hand|forearm),
        ('existing_protected_face_triangle_proxy',protected_face)]:
        groups[name]={label:stats(selection&domain) for label,domain in
            [('all',np.ones(len(xyz),dtype=bool)),('source_mask_inside',inside),('source_mask_outside',outside),('unknown_outside_raster_or_nonpositive_depth',unknown)]}
    output=dict(schema='graywoman128-reference23-point-alpha-mask-binding-proxy-v1',
        updated=datetime.now(timezone.utc).isoformat(),CPU_only=True,GPU_started=False,
        step=1000,actual_optimizer_updates=1000,points=len(xyz),reference_frame=row['frame'],
        source_hashes=before,viewer_points=dict(source=str(viewer),lines=[237,243,255,260],
            per_point_opacity_explicitly_discarded=True,opaque_center_sprites=True,
            Gaussian_covariance_not_used=True,points_mode_is_not_surface_opacity=True),
        source_projection=dict(wh=source['wh'],K_pixel_corners=K.tolist(),w2c=source['w2c'],
            pixel_lookup='Positive-depth centers with corner coordinates0<=u<W and0<=v<H; mask index=floor(u),floor(v). Integer-center coordinate is corner-.5.',
            derived_wh=row['wh'],derived_K_pixel_corners=row['K'],
            native_to_derived_affine_max_abs_px=affine_error,portable_reference_projection_max_abs_px=portable_error,
            conditional_camera_not_independent_physical_calibration=True),
        alpha='Float64 sigmoid of unchanged standard PLY float32 opacity logits, not Spark packed opacity and not cumulative ray alpha.',
        quantile_probabilities=PROBS,groups=groups,
        binding_proxy=dict(method='Mean original skinning weights over each bound triangle, then dominant joint; no optimized3D/2D landmark or manual anatomical truth.',
            hand_joint_names=[names[i] for i in hand_joints],forearm_joint_names=[names[i] for i in elbow_joints],
            existing_face_proxy='Any bound triangle vertex in the existing protected_face_vertices mask.',
            topology_and_point_order_verified=True),
        limitations=['Mask membership uses point centers in one reference view, not ellipse coverage, depth ordering or a physically identified hand.',
            'A low base alpha is one reason an opaque center sprite may appear while the Gaussian render suppresses it; covariance, depth and accumulated transmittance also matter.',
            'Setting Gaussian base alpha to1 would remove this learned suppression and can expose wrong gray geometry, but no forced-alpha Gaussian render was run; it is not equivalent to points mode.',
            'No cutoff scan, point filtering, model, training, loss or viewer change was made. The1/255 readout is descriptive, not an acceptance rule.'],
        quality_accepted=False,known_pose_failure=True,planned18000_completed=False)
    assert before=={str(p):sha(p) for p in files}
    output['input_hashes_unchanged']=True
    destination=SNAP/'point-alpha-mask-binding-proxy128.json'
    assert not destination.exists()
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(destination),sha256=sha(destination),
        mask_inside=int(inside.sum()),mask_outside=int(outside.sum()),unknown=int(unknown.sum()),
        all=groups['all_centers'],hands=groups['hand_wrist_fingers_binding_proxy'],
        forearms=groups['forearm_elbow_dominant_binding_proxy']),allow_nan=False))


if __name__=='__main__':main()
