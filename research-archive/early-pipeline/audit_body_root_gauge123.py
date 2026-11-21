"""Zero-update root-gauge contract probe through the existing SMPL-X/GS inputs.

Framewise paired rigid transforms preserve pixels and Gaussian covariance.
They change the shared world-SH direction convention, so old SH coefficients
cannot be called retrained/valid in that new convention. No live code is edited.
"""
import argparse
import copy
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from export_demo_body_snapshot123 import ROOT, process
from audit_body_sh_directions123 import ply, unit, angle, stats
from shuolee_portrait import runtime as rt

OUT = ROOT/'root-gauge-audit'
STEP = 60000
SNAP = ROOT/'gaussian-run'/f'step-{STEP:06d}'
ASSET = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')


def plan():
    OUT.mkdir(exist_ok=False)
    live = rt.read(ROOT/'plan.json')
    inputs = [ROOT/'fit/fitted.npz', ROOT/'data/training-inputs.json', SNAP/'current-meshes.npz',
              SNAP/'bound-local.ply', SNAP/'binding.npz', SNAP/'result.json', ASSET]
    inputs += [Path(p) for p in live['source_hashes']]
    rt.save(OUT/'plan.json', dict(created=rt.now(), step=STEP,
        hypothesis='Identity extrinsics and camera-relative root poses constrain shared world SH to a narrow cone. Normalize only the frozen fitted root frame, moving cameras and meshes together.',
        formula='G_i=[R_ref R_i^T, pelvis_ref - R_ref R_i^T pelvis_i]; v_new=G_i v_old; E_new=E_old inverse(G_i).',
        input_hashes={str(p):rt.sha(p) for p in inputs}, frames=59, reference='frame_000033',
        source_training_parameters_unchanged=True, fitting_updates=0, gaussian_updates=0,
        camera_status='Derived body-relative coordinate gauge, not physically calibrated scene camera motion.',
        risks=['Incorrect fitted root motion transfers to derived camera motion.',
               'Body-relative SH can absorb moving illumination; invariance of pixels does not validate lighting or pose.',
               'This does not repair the known shoulder/elbow/hand fit.',
               'Old world-SH cannot be reused as trained radiance under a new framewise gauge.'],
        criteria=dict(saved_vertex_reprojection_max_px=.002, camera_covariance_relative_max=2e-5,
            native_alpha_mean_abs=2e-6, native_alpha_max_abs=.005, initial_lbs_vertex_max_m=2e-6,
            projection_gradient_relative_l2=1e-4),
        cpu_budget_seconds=120, gpu_budget_seconds=180,
        stopped_if_any_contract_fails=True, training_launch_authorized_by_this_probe=False))
    print('ROOT_GAUGE_PLAN_REGISTERED', flush=True)


def checked():
    p=rt.read(OUT/'plan.json')
    for name,h in p['input_hashes'].items(): assert rt.sha(name)==h,name
    return p


def face_geometry(v, faces, binding, local):
    t=v[faces]; x=unit(t[:,1]-t[:,0]); y=unit(np.cross(x,t[:,2]-t[:,0])); z=-unit(np.cross(y,x))
    Q=np.stack([x,y,z],-1)[binding]
    scale=((np.linalg.norm(t[:,1]-t[:,0],axis=-1)+abs(np.sum(z*(t[:,2]-t[:,0]),axis=-1)))*.5)[binding]
    xyz=np.einsum('nij,nj->ni',Q,local)*scale[:,None]+t.mean(1)[binding]
    return xyz,Q,scale


def project(v,E,K):
    c=v@E[:3,:3].T+E[:3,3]; h=c@K.T
    assert np.min(c[:,2])>0
    return h[:,:2]/h[:,2:3]


def prepare():
    p=checked(); started=time.monotonic(); dest=OUT/'candidate-inputs';dest.mkdir(exist_ok=False)
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as z: fit={k:z[k].copy() for k in z.files}
    rows=rt.read(ROOT/'data/training-inputs.json'); new_rows=copy.deepcopy(rows)
    assert list(fit['frame_ids'])==[r['frame'] for r in rows['records']] and len(fit['frame_ids'])==59
    ref=list(fit['frame_ids']).index(p['reference']); assert ref==0
    poses=np.asarray(fit['root_rotation'],float); pelvis=np.asarray(fit['world_translation'],float)
    assert np.max(abs(poses@poses.transpose(0,2,1)-np.eye(3)))<1e-6
    G=np.tile(np.eye(4),(59,1,1)); G[:,:3,:3]=poses[ref]@poses.transpose(0,2,1)
    G[:,:3,3]=pelvis[ref]-np.einsum('nij,nj->ni',G[:,:3,:3],pelvis)
    G[ref]=np.eye(4)  # Exact identity for the named export pose, not frame selection.
    old_fit={k:v.copy() for k,v in fit.items()}
    for i,(r,nr) in enumerate(zip(rows['records'],new_rows['records'])):
        E=np.asarray(r['w2c']); assert np.allclose(E,fit['w2c'][i],atol=1e-8)
        fit['vertices_world'][i]=fit['vertices_world'][i]@G[i,:3,:3].T+G[i,:3,3]
        fit['w2c'][i]=E@np.linalg.inv(G[i]);nr['w2c']=fit['w2c'][i].tolist()
        fit['root_rotation'][i]=G[i,:3,:3]@poses[i]
        fit['world_translation'][i]=G[i,:3,:3]@pelvis[i]+G[i,:3,3]
        nr['camera_gauge']='Frozen fitted pelvis/root aligned to named reference; paired mesh transform, not measured extrinsics'
    contract=dict(reference_frame=p['reference'],coordinate_representation='Shared reference-root frame',
        paired_framewise_rigid_transforms=True,projection_unchanged=True,
        body_parameters_remain_in_original_camera_coordinates=True,
        K_masks_images_pose_shape_expression_offsets_unchanged=True,
        G_fixed_from_initial_fit=True,camera_is_physical_calibration=False,
        shared_world_SH_convention_changed=True,training_requires_fresh_Gaussians=True)
    fit['camera_gauge_contract_json']=np.asarray(__import__('json').dumps(contract))
    new_rows['camera_gauge_contract']=contract
    np.savez_compressed(dest/'fitted.npz',**fit)
    rt.save(dest/'training-inputs.json',new_rows)
    np.savez_compressed(dest/'transforms.npz',G=G,frame_ids=fit['frame_ids'])
    # Verify actual serialized values rather than only in-memory algebra.
    fit=dict(np.load(dest/'fitted.npz',allow_pickle=False));new_rows=rt.read(dest/'training-inputs.json')
    unchanged=[]
    for key in old_fit:
        if key not in ('vertices_world','w2c','root_rotation','world_translation'):
            assert np.array_equal(fit[key],old_fit[key]),key
            unchanged.append(key)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z: meshes=dict(z)
    local,_=ply(SNAP/'bound-local.ply'); bound=PlyVertex(SNAP/'bound-local.ply')
    binding=np.load(SNAP/'binding.npz',allow_pickle=False)['triangle_ids']
    L=Rotation.from_quat(np.column_stack([bound[f'rot_{i}'] for i in (1,2,3,0)])).as_matrix()
    sig=np.exp(np.column_stack([bound[f'scale_{i}'] for i in range(3)]).astype(float))**2
    rays=[];new_rays=[];checks=[];mapped_mesh=[]
    for i,(r,nr) in enumerate(zip(rows['records'],new_rows['records'])):
        E=np.array(r['w2c']);En=np.array(nr['w2c']);K=np.array(r['K']);assert r['K']==nr['K']
        old=(meshes['vertices_camera'][i]-E[:3,3])@E[:3,:3]
        new=(old@G[i,:3,:3].T+G[i,:3,3]).astype(np.float32).astype(float);mapped_mesh.append(new)
        xo,Qo,so=face_geometry(old,meshes['faces'],binding,local)
        xn,Qn,sn=face_geometry(new,meshes['faces'],binding,local)
        pixel=float(abs(project(xo,E,K)-project(xn,En,K)).max())
        fitted_pixel=float(abs(project(old_fit['vertices_world'][i],E,K)-project(fit['vertices_world'][i],En,K)).max())
        co=E[:3,:3]@Qo@L;cn=En[:3,:3]@Qn@L
        cov=lambda q,s:np.einsum('nij,nj,nkj->nik',q,sig*s[:,None]**2,q)
        ac,bc=cov(co,so),cov(cn,sn)
        relative=float(np.max(np.linalg.norm(ac-bc,axis=(1,2))/np.maximum(np.linalg.norm(ac,axis=(1,2)),1e-12)))
        ro=unit(xo-(-E[:3,:3].T@E[:3,3]));rn=unit(xn-(-En[:3,:3].T@En[:3,3]))
        rays.append(ro);new_rays.append(rn)
        checks.append(dict(frame=r['frame'],point_projection_max_px=pixel,fit_projection_max_px=fitted_pixel,camera_covariance_relative_max=relative))
        assert max(pixel,fitted_pixel)<p['criteria']['saved_vertex_reprojection_max_px'],checks[-1]
        assert relative<p['criteria']['camera_covariance_relative_max'],checks[-1]
    np.savez_compressed(dest/'snapshot-meshes.npz',vertices_world=np.asarray(mapped_mesh),faces=meshes['faces'],frame_ids=meshes['frame_ids'])
    rt.save(OUT/'cpu-result.json',dict(updated=rt.now(),passed=True,frames=59,checks=checks,unchanged_fields=unchanged,
        reference_mesh_bitwise_equal=bool(np.array_equal(old_fit['vertices_world'][ref],fit['vertices_world'][ref])),
        reference_camera_bitwise_equal=bool(np.array_equal(old_fit['w2c'][ref],fit['w2c'][ref])),
        original_direction_change_degrees=stats(angle(np.array(rays),rays[ref])),
        root_frame_direction_change_degrees=stats(angle(np.array(new_rays),new_rays[ref])),
        candidate_fit_sha256=rt.sha(dest/'fitted.npz'),candidate_inputs_sha256=rt.sha(dest/'training-inputs.json'),
        transforms_sha256=rt.sha(dest/'transforms.npz'),snapshot_meshes_sha256=rt.sha(dest/'snapshot-meshes.npz'),
        seconds=time.monotonic()-started,optimization_steps=0,quality_accepted=False))
    checked();print('ROOT_GAUGE_CPU_CONTRACT_PASSED',flush=True)


def PlyVertex(path):
    from plyfile import PlyData
    return PlyData.read(path)['vertex'].data


def diagnose():
    """Locate a failed contract; do not change thresholds or candidate inputs."""
    checked();dest=OUT/'candidate-inputs';G=np.load(dest/'transforms.npz')['G']
    rows=rt.read(ROOT/'data/training-inputs.json')['records'];nrows=rt.read(dest/'training-inputs.json')['records']
    mesh=dict(np.load(SNAP/'current-meshes.npz',allow_pickle=False));local,_=ply(SNAP/'bound-local.ply')
    b=PlyVertex(SNAP/'bound-local.ply');binding=np.load(SNAP/'binding.npz')['triangle_ids']
    L=Rotation.from_quat(np.column_stack([b[f'rot_{i}'] for i in (1,2,3,0)])).as_matrix()
    sig=np.exp(np.column_stack([b[f'scale_{i}'] for i in range(3)]).astype(float))**2
    records=[]; extrema=[]
    for i,(r,nr) in enumerate(zip(rows,nrows)):
        E=np.array(r['w2c']);En=np.array(nr['w2c']);K=np.array(r['K'])
        old=(mesh['vertices_camera'][i]-E[:3,3])@E[:3,:3]
        exact=old@G[i,:3,:3].T+G[i,:3,3];rounded=exact.astype(np.float32).astype(float)
        xo,Qo,so=face_geometry(old,mesh['faces'],binding,local)
        cov=lambda q,s:np.einsum('nij,nj,nkj->nik',q,sig*s[:,None]**2,q)
        ac=cov(E[:3,:3]@Qo@L,so);parts=[]
        for label,nv in [('float64',exact),('float32_serialized',rounded)]:
            xn,Qn,sn=face_geometry(nv,mesh['faces'],binding,local);bc=cov(En[:3,:3]@Qn@L,sn)
            relative=np.linalg.norm(ac-bc,axis=(1,2))/np.maximum(np.linalg.norm(ac,axis=(1,2)),1e-12)
            point_error=np.linalg.norm(project(xo,E,K)-project(xn,En,K),axis=1)
            parts.append(dict(dtype=label,covariance_relative=stats(relative),point_euclidean_error_px=stats(point_error),
                covariance_failure_points=int(np.sum(relative>2e-5)),projection_failure_points=int(np.sum(point_error>.002))))
            if label=='float32_serialized':
                top=np.argsort(relative)[-5:][::-1]
                for j in top:
                    face=int(binding[j]);tri=old[mesh['faces'][face]];e0=tri[1]-tri[0];e1=tri[2]-tri[0]
                    cross=np.linalg.norm(np.cross(e0,e1));length=max(np.linalg.norm(e0),np.linalg.norm(e1),np.linalg.norm(tri[2]-tri[1]))
                    extrema.append(dict(frame=r['frame'],point=int(j),face=face,covariance_relative=float(relative[j]),
                        triangle_max_edge_m=float(length),twice_triangle_area_m2=float(cross),
                        normalized_area=float(cross/max(length**2,1e-30)),
                        world_center_m=xo[j].tolist(),local_center=local[j].tolist()))
        records.append(dict(frame=r['frame'],controls=parts,max_vertex_rounding_m=float(abs(exact-rounded).max())))
    extrema=sorted(extrema,key=lambda r:r['covariance_relative'],reverse=True)
    rt.save(OUT/'conditioning-diagnosis.json',dict(updated=rt.now(),frames=59,records=records,worst_points=extrema[:20],
        thresholds_unchanged=True,inputs_unchanged=True,training_changed=False,quality_accepted=False))
    print('ROOT_GAUGE_CONDITIONING_DIAGNOSED',extrema[:2],flush=True)


def transport():
    """Keep the failed vertex-remesh candidate; test exact frame transport."""
    p=checked();assert (OUT/'cpu-failure.json').exists() and (OUT/'conditioning-diagnosis.json').exists()
    helper=Path(__file__).parent/'body_root_gauge_transport123.py'
    rt.save(OUT/'frame-transport-plan.json',dict(updated=rt.now(),implementation='Apply paired rigid G to already-computed original triangle frame; never recompute axes from rounded transformed vertices.',
        causal_variable_unchanged='Frozen fitted-root coordinate gauge only',criteria=p['criteria'],
        helper_sha256=rt.sha(helper),failed_vertex_serialization_candidate_preserved=True,
        old_input_fit_retained=True,renderer_losses_optimizer_unchanged=True,training_steps=0))
    rows=rt.read(ROOT/'data/training-inputs.json')['records'];nrows=rt.read(OUT/'candidate-inputs/training-inputs.json')['records']
    G=np.load(OUT/'candidate-inputs/transforms.npz')['G'];mesh=dict(np.load(SNAP/'current-meshes.npz'))
    local,_=ply(SNAP/'bound-local.ply');b=PlyVertex(SNAP/'bound-local.ply');binding=np.load(SNAP/'binding.npz')['triangle_ids']
    L=Rotation.from_quat(np.column_stack([b[f'rot_{i}'] for i in (1,2,3,0)])).as_matrix()
    sig=np.exp(np.column_stack([b[f'scale_{i}'] for i in range(3)]).astype(float))**2
    records=[];rays=[];new_rays=[]
    for i,(r,nr) in enumerate(zip(rows,nrows)):
        E=np.array(r['w2c']);En=np.array(nr['w2c']);K=np.array(r['K'])
        old=(mesh['vertices_camera'][i]-E[:3,3])@E[:3,:3]
        xo,Qo,so=face_geometry(old,mesh['faces'],binding,local)
        xn=xo@G[i,:3,:3].T+G[i,:3,3];Qn=G[i,:3,:3]@Qo
        cov=lambda q:np.einsum('nij,nj,nkj->nik',q,sig*so[:,None]**2,q)
        ac,bc=cov(E[:3,:3]@Qo@L),cov(En[:3,:3]@Qn@L)
        pixel=float(abs(project(xo,E,K)-project(xn,En,K)).max())
        rel=float(np.max(np.linalg.norm(ac-bc,axis=(1,2))/np.maximum(np.linalg.norm(ac,axis=(1,2)),1e-12)))
        assert pixel<p['criteria']['saved_vertex_reprojection_max_px'] and rel<p['criteria']['camera_covariance_relative_max']
        records.append(dict(frame=r['frame'],point_projection_max_px=pixel,camera_covariance_relative_max=rel))
        rays.append(unit(xo-(-E[:3,:3].T@E[:3,3])));new_rays.append(unit(xn-(-En[:3,:3].T@En[:3,3])))
    rt.save(OUT/'transport-cpu-result.json',dict(updated=rt.now(),passed=True,frames=59,checks=records,
        original_direction_change_degrees=stats(angle(np.array(rays),rays[0])),
        root_frame_direction_change_degrees=stats(angle(np.array(new_rays),new_rays[0])),
        camera_inputs_sha256=rt.sha(OUT/'candidate-inputs/training-inputs.json'),transforms_sha256=rt.sha(OUT/'candidate-inputs/transforms.npz'),
        helper_sha256=rt.sha(helper),geometry_unchanged=True,thresholds_not_relaxed=True,quality_accepted=False))
    print('ROOT_GAUGE_FRAME_TRANSPORT_CPU_PASSED',flush=True)


def replay(pid):
    p=checked();c=rt.read(OUT/'transport-cpu-result.json'); assert c['passed'] and process(pid)[1]=='T'
    assert not (OUT/'native-result.json').exists()
    import torch
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera, image
    from body_root_gauge_transport123 import transform_selected_triangle_frames, transform_vertices
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    dest=OUT/'candidate-inputs'
    assert rt.sha(dest/'training-inputs.json')==c['camera_inputs_sha256'] and rt.sha(dest/'transforms.npz')==c['transforms_sha256']
    assert rt.sha(Path(__file__).parent/'body_root_gauge_transport123.py')==c['helper_sha256']
    rows=rt.read(ROOT/'data/training-inputs.json')['records'];nrows=rt.read(dest/'training-inputs.json')['records']
    G=torch.tensor(np.load(dest/'transforms.npz')['G'],dtype=torch.float32,device='cuda')
    class Transported(SMPLXGaussianModel):
        def select_mesh_by_timestep(self,timestep,original=False):
            super().select_mesh_by_timestep(timestep,original=original)
            transform_selected_triangle_frames(self,G[int(timestep)])
        def posed_gauge_vertices(self,timestep):
            return transform_vertices(self.shared_offset_vertices(timestep),G[int(timestep)])
    old=SMPLXGaussianModel(3);new=Transported(3)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z: mesh=dict(z)
    assert all(np.array_equal(np.array(r['w2c']),np.eye(4)) for r in rows)
    for model,verts in [(old,mesh['vertices_camera']),(new,mesh['vertices_camera'])]:
        model.load_ply(str(SNAP/'bound-local.ply'));model.active_sh_degree=3
        model.vertices_sequence=torch.tensor(verts,dtype=torch.float32,device='cuda')
        model.faces=torch.tensor(mesh['faces'],dtype=torch.long,device='cuda');model.num_timesteps=59
        model.static_offset=None
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');white=torch.ones_like(old._xyz);records=[]
    with torch.no_grad():
        for i,(r,nr) in enumerate(zip(rows,nrows)):
            old.select_mesh_by_timestep(i);new.select_mesh_by_timestep(i)
            a,_=camera(r,i,pose_index=i);b,_=camera(nr,i,pose_index=i)
            ra=render(a,old,pipe,bg,override_color=white)['render'];rb=render(b,new,pipe,bg,override_color=white)['render']
            diff=(ra-rb).abs();v=dict(frame=r['frame'],alpha_mean_abs=float(diff.mean()),alpha_max_abs=float(diff.max()))
            records.append(v)
            assert v['alpha_mean_abs']<p['criteria']['native_alpha_mean_abs'] and v['alpha_max_abs']<p['criteria']['native_alpha_max_abs'],v
            if r['frame'] in ('frame_000018','frame_000033'):
                from PIL import Image
                ims=[image(ra),image(rb)];page=Image.new('RGB',(ims[0].width*2,ims[0].height))
                for j,im in enumerate(ims):page.paste(im,(j*im.width,0))
                page.save(OUT/(r['frame']+'_old_new_alpha.png'))
    # The existing shared-offset interface must still reproduce the initial fit.
    nf=dict(np.load(ROOT/'fit/fitted.npz',allow_pickle=False))
    new.vertices_sequence=torch.tensor(nf['vertices_world'],dtype=torch.float32,device='cuda')
    receipt=new.enable_shared_static_offset(ROOT/'fit/fitted.npz',ASSET)
    # Projection gradient of the existing original SMPL-X LBS through both gauges.
    gradient=[]; sample=torch.arange(0,10475,127,device='cuda')
    target_E=torch.tensor(np.stack([r['w2c'] for r in nrows]),dtype=torch.float32,device='cuda')
    for i in (0,17,58):
        new.static_offset.grad=None
        v=new.posed_gauge_vertices(i);E=target_E[i]
        cam=v@E[:3,:3].T+E[:3,3];uv=cam[sample,:2]/cam[sample,2:3]
        loss=uv.square().mean();loss.backward();g1=new.static_offset.grad.detach().clone()
        new.static_offset.grad=None
        v=new.shared_offset_vertices(i);E=new.offset_w2c[i]
        cam=v@E[:3,:3].T+E[:3,3];uv=cam[sample,:2]/cam[sample,2:3]
        uv.square().mean().backward();g0=new.static_offset.grad.detach().clone()
        err=float((g1-g0).norm()/g0.norm().clamp_min(1e-12)); assert err<p['criteria']['projection_gradient_relative_l2'],err
        gradient.append(dict(frame=rows[i]['frame'],relative_l2=err,finite=bool(torch.isfinite(g1).all()),original_norm=float(g0.norm())))
    assert new.optimizer is None and process(pid)[1]=='T'
    checked();rt.save(OUT/'native-result.json',dict(updated=rt.now(),passed=True,frames=59,alpha_checks=records,
        existing_shared_offset_interface=receipt,projection_gradients=gradient,optimizer_created=False,
        gaussian_steps=0,quality_accepted=False,training_sources_changed=False))
    print('ROOT_GAUGE_NATIVE_INTERFACE_PASSED',flush=True)


def run(pid):
    checked();lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid); assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    start=time.monotonic();rt.save(OUT/'gpu-handoff.json',dict(started=rt.now(),pid=pid,command=cmd,identity=identity,state='suspending'))
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        subprocess.run([sys.executable,'-B',__file__,'replay','--pid',str(pid)],check=True,timeout=180)
    finally:
        if process(pid)[2]==identity:os.kill(pid,signal.SIGCONT)
        r=rt.read(OUT/'gpu-handoff.json');r.update(finished=rt.now(),seconds=time.monotonic()-start,state='resumed',trainer_after_state=process(pid)[1])
        rt.save(OUT/'gpu-handoff.json',r);print('TRAINER_RESUMED',pid,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['plan','prepare','diagnose','transport','run','replay']);a.add_argument('--pid',type=int);v=a.parse_args()
    if v.mode in ('plan','prepare','diagnose','transport'):globals()[v.mode]()
    else:globals()[v.mode](v.pid)
