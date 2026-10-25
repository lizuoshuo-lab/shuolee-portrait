"""Mandatory standard 3DGS world-PLY export for the Shuolee viewer on PC00.

Bake the selected pose and all local binding/deformation. PLY contains world
means, log standard deviations, wxyz rotations, logit alpha and standard SH.
An intact PLY is distinct from an anatomically accepted reconstruction.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from plyfile import PlyData
import shuolee_common as c
from scene.gaussian_model import GaussianModel
from scene.cameras import CameraProj2
from gaussian_renderer import render
from floor_reference_cameras import make_cameras

@torch.no_grad()
def export_and_verify(model,root,step,timestep,existing_only=False):
    root=Path(root);run=root/'gaussian-run';destination=run/'point_cloud.ply'
    model.select_mesh_by_timestep(timestep)
    tmp=destination if existing_only else run/'point_cloud.checking.ply'
    if not existing_only:model.export_world_ply(tmp,timestep)
    raw=PlyData.read(str(tmp),mmap=False);names=raw['vertex'].data.dtype.names
    required=['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    assert raw.byte_order=='<' and not raw.text
    assert all(k in names for k in required)
    assert not any('binding' in k or 'face' in k for k in names)
    assert all(np.isfinite(raw['vertex'][k]).all() for k in names)
    rotation=np.stack([raw['vertex'][f'rot_{j}'] for j in range(4)],-1)
    # Ordinary 3DGS optimizes unconstrained quaternion parameters and normalizes
    # inside the renderer. Serialize their equivalent unit rotations for a
    # portable PLY; leave the trained parameters and covariance unchanged.
    if not existing_only:
        norms=np.linalg.norm(rotation.astype(np.float64),axis=-1,keepdims=True)
        assert np.isfinite(norms).all() and (norms>1e-12).all()
        if np.abs(norms-1).max()>=1e-4:
            rotation=(rotation/norms).astype(np.float32)
            for j in range(4):raw['vertex'][f'rot_{j}']=rotation[:,j]
            raw.write(str(tmp))
    assert np.abs(np.linalg.norm(rotation,axis=-1)-1).max()<1e-4
    file=GaussianModel(model.max_sh_degree);file.load_ply(str(tmp))
    assert len(file._xyz)==len(model._xyz) and file.binding is None
    checks={k:float((a-b).abs().max()) for k,a,b in [
        ('world_position_max_abs',model.get_xyz,file.get_xyz),('opacity_max_abs',model.get_opacity,file.get_opacity),
        ('sh_max_abs',model.get_features,file.get_features),('world_covariance_max_abs',model.get_covariance(),file.get_covariance())]}
    assert checks['world_position_max_abs']==0 and checks['sh_max_abs']==0 and checks['opacity_max_abs']==0
    assert checks['world_covariance_max_abs']<1e-5,checks
    cams,_,_=make_cameras();rows=c.read(root/'data/inputs.json')['records'];row=rows[timestep]
    K=np.array(row['K'])*.35;K[2,2]=1
    cams=[dict(view='exported_pose_source_camera',w2c=row['w2c'],K=K.tolist(),width=round(row['wh'][0]*.35),height=round(row['wh'][1]*.35))]+cams
    nativepipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=True,debug=False)
    plainpipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.ones(3,device='cuda');renders=[]
    for i,rowcam in enumerate(cams):
        E=np.array(rowcam['w2c']);K=np.array(rowcam['K'])
        cam=CameraProj2(i,E[:3,:3].T,E[:3,3],K[0,0],K[1,1],'white',rowcam['width'],None,rowcam['height'],'',rowcam['view'],i,cx=K[0,2],cy=K[1,2],timestep=timestep)
        a=render(cam,model,nativepipe,bg)['render'];b=render(cam,file,plainpipe,bg)['render'];d=(a-b).abs()
        codes=((a.clamp(0,1)*255+.5).to(torch.int16)-(b.clamp(0,1)*255+.5).to(torch.int16)).abs()
        item=dict(view=rowcam['view'],max_abs=float(d.max()),mean_abs=float(d.mean()),max_8bit=int(codes.max()),channels_over_one_level=int((codes>1).sum()))
        renders.append(item)
    passed=all(x['max_abs']<1/255 and x['mean_abs']<1e-6 and x['channels_over_one_level']==0 for x in renders)
    receipt=dict(created=c.stamp(),experiment=root.name,step=step,pose_index=timestep,raw_source_frame=row['source_frame_index'],
        representation=getattr(model,'representation','body_affine'),
        reference_frame_is_pose_snapshot=getattr(model,'representation','')!='shared_static_world',
        points=len(model._xyz),format='binary_little_endian standard 3DGS PLY',coordinate_space='shared static world; no temporal deformation' if getattr(model,'representation','')=='shared_static_world' else 'world; selected pose baked',
        fields=dict(scale='log standard deviation',rotation='wxyz',opacity='logit',color='SH coefficients in standard 3DGS order'),
        required_external_template=False,required_external_transform=False,parameter_checks=checks,render_checks=renders,
        compatibility_passed=passed,geometry_accepted=False,quality_accepted=False,
        compatible_viewer='Shuolee standard Gaussian PLY loader / Spark',pc00_opened_this_round=False,
        local_path=str(destination),sha256=c.sha(tmp))
    c.save(run/'delivery.json',receipt)
    if not passed:raise RuntimeError('PLY export parity failed; checking artifact is not a deliverable')
    if not existing_only:
        tmp.replace(destination)
        from export_viewer_upright import main as export_upright
        export_upright(root)
    return receipt
