"""Read-only saved18k replay. Preserve the failed save and never resume training."""
import fcntl
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from run_official_gaussians123 import read, save, sha, now, ASSETS, PROJECT
from train_demo_body_upstream123 import module, UP

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/indoor-circle16-paired18k124')

def main():
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snap=ROOT/'gaussian-run/step-018000';out=ROOT/'export-failure-diagnostic'
    out.mkdir(exist_ok=False)
    hashes={str(p):sha(p) for p in (snap/'checkpoint.pt',snap/'world.ply',snap/'bound-local.ply',snap/'binding.npz')}
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    import torch
    from shuolee_portrait.training import camera,image
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from smplify_direct_surface123 import enable,vertices
    from gaussian_renderer import render
    official=module(UP/'scene/gaussian_model.py','circle16_readonly_model')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    rows=read(ROOT/'data/training-inputs.json')['records']
    assert len(rows)==16 and rows[0]['frame']=='frame_000105'
    class Body(official.GaussianModel):
        select_mesh_by_timestep=SMPLXGaussianModel.select_mesh_by_timestep
        get_covariance=SMPLXGaussianModel.get_covariance
        shared_offset_vertices=vertices
    cp=torch.load(snap/'checkpoint.pt',map_location='cuda',weights_only=False)
    assert cp['step']==18000
    model=Body(0);model.static_offset=None;model.num_timesteps=16
    model.vertices_sequence=torch.as_tensor(fit['vertices_world'],device='cuda')
    model.faces=torch.as_tensor(fit['faces'],device='cuda',dtype=torch.long)
    fields=('active_sh_degree','_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','binding','binding_counter','max_radii2D','xyz_gradient_accum','denom')
    for key,value in zip(fields,cp['capture']):setattr(model,key,value)
    assert model.optimizer is None
    enable(model,ROOT/'fit/fitted.npz',ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    with torch.no_grad():model.static_offset.copy_(cp['static_offset'])
    plain=official.GaussianModel(0);plain.load_ply(str(snap/'world.ply'))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');started=time.monotonic()
    with torch.no_grad():
        model.select_mesh_by_timestep(0);cam,_=camera(rows[0],0,pose_index=0)
        native=render(cam,model,pipe,bg)['render']
        portable=render(cam,plain,pipe,bg)['render']
        delta=(native-portable).abs()
        pixel=delta.max(0).values
        comparison=dict(mean_abs=float(delta.mean()),max_abs=float(delta.max()),
            pixels_above_005=int((pixel>=.005).sum()),pixels_above_001=int((pixel>=.001).sum()),
            total_pixels=pixel.numel(),passed=bool(delta.mean()<2e-6 and delta.max()<.005))
        properties={key:float((getattr(model,key)-getattr(plain,key)).abs().max()) for key in ('get_xyz','get_scaling','get_rotation','get_opacity','get_features')}
        # Exact floating-point world parameters isolate serialization/activation
        # from a binding/coordinate error. This is diagnostic only, not a PLY.
        class ExactWorld(official.GaussianModel):
            @property
            def get_scaling(self):return self.exact_scale
            @property
            def get_rotation(self):return self.exact_rotation
        exact=ExactWorld(0)
        for key in ('_xyz','_features_dc','_features_rest','_opacity'):
            setattr(exact,key,model.get_xyz.detach() if key=='_xyz' else getattr(model,key).detach())
        exact.exact_scale=model.get_scaling.detach();exact.exact_rotation=model.get_rotation.detach()
        er=render(cam,exact,pipe,bg)['render']
        control=dict(mean_abs=float((native-er).abs().mean()),max_abs=float((native-er).abs().max()))
        # Two predetermined single-factor controls. No numerical threshold search.
        controls={}
        for name,scale,rotation in (
            ('only_scale_activation',plain.get_scaling,model.get_rotation),
            ('only_quaternion_activation',model.get_scaling,plain.get_rotation)):
            exact.exact_scale=scale;exact.exact_rotation=rotation
            rendered=render(cam,exact,pipe,bg)['render'];d=(native-rendered).abs()
            controls[name]=dict(mean_abs=float(d.mean()),max_abs=float(d.max()))
        image(native).save(out/'reference-checkpoint.png')
        image(portable).save(out/'reference-ply.png')
        image(delta*100).save(out/'difference-times100.png')
        meshes=[]
        for i,row in enumerate(rows):
            model.select_mesh_by_timestep(i)
            ca,_=camera(row,i,pose_index=i)
            image(render(ca,model,pipe,bg)['render']).save(snap/(row['frame']+'_gaussian.png'))
            vw=model.shared_offset_vertices(i).cpu().numpy();E=np.asarray(row['w2c'])
            meshes.append(vw@E[:3,:3].T+E[:3,3])
        np.savez_compressed(snap/'current-meshes.npz',vertices_camera=np.asarray(meshes),faces=fit['faces'],frame_ids=np.asarray([r['frame'] for r in rows]))
        offset=model.static_offset.norm(dim=-1)
        save(snap/'offset-update.json',dict(lr=1e-6,median_m=float(offset.median()),p90_m=float(torch.quantile(offset,.9)),max_m=float(offset.max()),shared_across_frames=True,replayed_read_only=True))
        result=dict(updated=now(),step=18000,points=len(model._xyz),reference_frame='frame_000105',reference_slot=0,
            active_sh_degree=0,world_ply_sha256=sha(snap/'world.ply'),native_ply_parity=comparison,frames_rendered=16,
            read_only_checkpoint_replay=True,optimizer_updates_added=0,quality_accepted=False,export_accepted=comparison['passed'])
        save(snap/'result.json',result)
        save(out/'result.json',dict(updated=now(),snapshot=hashes,parity=comparison,property_max_abs=properties,
            exact_world_control=control,single_factor_controls=controls,seconds=time.monotonic()-started,
            training_resumed=False,optimizer_updates_added=0,quality_accepted=False))
    for path,digest in hashes.items():assert sha(path)==digest
    print('READ_ONLY_EXPORT_DIAGNOSTIC',comparison,controls,flush=True)

if __name__=='__main__':main()
