"""CPU-only59-frame teacher association on the unchanged VHAP import mesh.

Canonical target descriptors diagnose articulation consistency; they are not
fitted mesh parameters, new Gaussian centers, or independent physical truth.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import sys
import time
import numpy as np
import torch
import roma
from plyfile import PlyData
from PIL import Image
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,SNAP,OUT as AUDIT,verify
from frozen_skeleton_surface123 import FrozenSkeletonSurface
from gaussian_surface_guidance123 import closest_bound_barycentrics
from audit_teacher_surface_definition123 import methods,fields,MODEL,GRAPHICS
from audit_teacher_base_mesh123 import quality

OUT=AUDIT/'original-import-teacher-contract'
NATIVE=AUDIT/'native-alpha-contribution'
PARENT=AUDIT/'mesh-initialization-boundary/next-contract-plan.json'
JOINT_IDS=[0,1,2,3,4,5,6,9,13,14,16,17,18,19]


def weighted_quantile(values,weights):
    x=np.asarray(values).reshape(-1);w=np.asarray(weights).reshape(-1)
    use=(w>0)&np.isfinite(x);x=x[use];w=w[use]
    if not len(x):return None
    order=np.argsort(x);c=np.cumsum(w[order],dtype=np.float64);c/=c[-1]
    return dict(zip(('median','p90','p99','max'),map(float,[*np.interp([.5,.9,.99],c,x[order]),x.max()])))


def prepare():
    verify();parent=io.read(PARENT);assert parent['state']=='registered_not_run'
    assert io.sha(ROOT/'fit/fitted.npz')==parent['source_fit_sha256']
    assert io.read(NATIVE/'result.json')['state']=='native_contribution_audit_complete_no_optimization'
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),PARENT,ROOT/'fit/fitted.npz',ROOT/'data/training-inputs.json',
        SNAP/'bound-local.ply',SNAP/'world.ply',SNAP/'current-meshes.npz',SNAP/'binding.npz',
        NATIVE/'native-contribution-weights.npz',NATIVE/'result.json',MODEL,GRAPHICS,
        Path(__file__).with_name('frozen_skeleton_surface123.py'),Path(__file__).with_name('gaussian_surface_guidance123.py')]
    io.save(OUT/'plan.json',dict(created=io.now(),cpu_only=True,max_seconds=180,frames=59,
        hashes={str(p):io.sha(p) for p in paths},optimizer_updates=0,
        original_LBS_parity_m=2e-6,teacher_world_PLY_parity_m=3e-6,
        numerical_inverse_condition_max=1e6,anatomical_joint_ids=JOINT_IDS,
        teacher_policy='Same saved Gaussian identity, bound triangle, native center-mask and measured foreground contribution. Recompute only bound-triangle footpoint against original import. No nearest-body rebinding or residual-based point removal.',
        descriptor='Contribution-weighted fixed barycentric coordinate perGaussian; independently invert original frozen LBS at that coordinate for each frame, then measure dispersion of the canonical displacement descriptors. Weighted mean is diagnostic only; do not write it into geometry. This is neither a vertex solver nor a physical material ground-truth check.',
        visibility_threshold='Strictly positive measured contribution; no new confidence or error threshold.',
        failure_policy='Stop on identity/nonfinite/parity/invertibility implementation failure. Empty per-frame clothing support is not training-admitted. Low descriptor dispersion alone is insufficient to admit training.',
        no_mesh_or_PLY_export=True,no_pose_or_camera_change=True,no_SH_change=True,quality_accepted=False))
    print('IMPORT_TEACHER_CONTRACT_REGISTERED',flush=True)


def checked():
    verify();p=io.read(OUT/'plan.json')
    for f,h in p['hashes'].items():assert io.sha(f)==h,f
    return p


def run():
    p=checked();assert not (OUT/'result.json').exists();started=time.monotonic()
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    sys.path.insert(0,str(io.PROJECT/'src/avatar_core'))
    from utils.graphics_utils import compute_face_orientation
    from smplx.joint_names import JOINT_NAMES
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:saved=dict(z)
    with np.load(NATIVE/'native-contribution-weights.npz',allow_pickle=False) as z:contrib=dict(z)
    with np.load(io.ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=True) as z:dominant=np.asarray(z['weights']).argmax(1)
    initial_offset=np.asarray(fit['static_offset']).reshape(10475,3)
    bridge=FrozenSkeletonSurface(fit,initial_offset,io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    faces=bridge.faces;learn=torch.tensor(np.isin(dominant,JOINT_IDS));perface=learn[faces].all(-1)
    assert np.array_equal(fit['faces'],saved['faces'])
    rows=io.read(ROOT/'data/training-inputs.json')['records'];assert len(rows)==59
    sslot={str(f):i for i,f in enumerate(saved['frame_ids'])};cslot={str(f):i for i,f in enumerate(contrib['frame_ids'])}
    ply=PlyData.read(str(SNAP/'bound-local.ply'))['vertex'];world=PlyData.read(str(SNAP/'world.ply'))['vertex']
    binding=torch.tensor(contrib['binding'],dtype=torch.long)
    bn=[x for x in ply.data.dtype.names if x.startswith('binding')];assert len(bn)==1
    assert np.array_equal(ply[bn[0]],binding.numpy())
    obj=SimpleNamespace(binding=binding,_xyz=fields(ply,['x','y','z']),_scaling=fields(ply,[f'scale_{j}' for j in range(3)]),
        _rotation=fields(ply,[f'rot_{j}' for j in range(4)]),scaling_activation=torch.exp,rotation_activation=lambda x:torch.nn.functional.normalize(x))
    fn=methods();records=[];target_list=[];bary_list=[];weight_list=[];order=[]
    original_parity=[];oracle=[];zero=torch.zeros((10475,3))
    with torch.no_grad():
        for i,row in enumerate(rows):
            assert time.monotonic()-started<p['max_seconds']
            assert io.sha(row['image'])==row['source_sha256'] and io.sha(row['mask'])==row['mask_sha256']
            slot=bridge.slots[row['frame']];order.append(slot);base=bridge.base_vertices[slot]
            err=float((base-torch.tensor(fit['vertices_world'][slot])).abs().max());original_parity.append(err)
            assert err<p['original_LBS_parity_m'],(row['frame'],err)
            if row['frame'] in ('frame_000033','frame_000015','frame_000054'):
                # Fixed, small perturbation; independent original forward only.
                d=torch.sin(torch.arange(10475*3).reshape(10475,3).float())*.0001*learn[:,None]
                expected,joints=bridge.independent_original_forward(d,slot)
                e=float((bridge.vertices(d,slot)-expected).abs().max())
                je=float((bridge.joints(slot)-joints).abs().max())
                assert e<p['original_LBS_parity_m'] and je<p['original_LBS_parity_m']
                oracle.append(dict(frame=row['frame'],vertices_max_abs_m=e,skeletal_joints_max_abs_m=je))
            literal=torch.tensor(saved['vertices_camera'][sslot[row['frame']]])
            R,scale=compute_face_orientation(literal,faces,return_scale=True)
            obj.face_orien_mat=R;obj.face_scaling=scale.clamp_min(1e-8);obj.face_center=literal[faces].mean(1)
            obj.face_orien_quat=roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(R))
            target=fn['get_xyz'](obj)
            if row['frame']=='frame_000033':
                teacher_error=float((target-fields(world,['x','y','z'])).abs().max())
                assert teacher_error<p['teacher_world_PLY_parity_m']
            tri=base[faces[binding]];bary,valid,inside=closest_bound_barycentrics(tri,target)
            E=torch.tensor(row['w2c'],dtype=torch.float32);K=torch.tensor(row['K'],dtype=torch.float32)
            cam=target@E[:3,:3].T+E[:3,3];q=cam@K.T;uv=q[:,:2]/q[:,2:].clamp_min(1e-9)
            ix,iy=uv[:,0].floor().long(),uv[:,1].floor().long();w,h=row['wh']
            raw=Image.open(row['image']);mask=np.asarray(Image.open(row['mask']).convert('L')).copy()
            assert raw.size==(w,h) and mask.shape==(h,w)
            inimage=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(cam[:,2]>0)
            native=torch.tensor(contrib['foreground_contribution'][cslot[row['frame']]])
            previous=torch.tensor(contrib['normalized_teacher_weights'][cslot[row['frame']]])
            admission=valid&inside&inimage&perface[binding]
            weight=native*admission*torch.tensor(mask,dtype=torch.float32)[iy.clamp(0,h-1),ix.clamp(0,w-1)]/255
            sums=torch.zeros(len(faces));sums.scatter_add_(0,binding,weight)
            weight=weight/sums[binding].clamp_min(1e-20)
            if weight.sum()>0:weight/=weight.sum()
            n=torch.nn.functional.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
            signed=((target-(tri*bary[:,:,None]).sum(1))*n).sum(-1)
            regions=[]
            for jid in JOINT_IDS:
                region=torch.tensor(dominant[fit['faces'][:,0]]==jid)&perface
                regions.append(dict(joint=JOINT_NAMES[jid],supported_faces=int(((sums>0)&region).sum()),
                    learnable_faces=int(region.sum()),teacher_weight=float(weight[region[binding]].sum())))
            records.append(dict(frame=row['frame'],points=int((weight>0).sum()),faces=int((sums>0).sum()),
                original_import_replay_max_abs_m=err,actual_contributing_points=int((native>0).sum()),
                previous_native_teacher_support=int((previous>0).sum()),
                previous_teacher_weight_lost_by_import_association=float(previous[weight==0].sum()),
                signed_normal_distance_mm=weighted_quantile(signed.numpy()*1000,weight.numpy()),
                positive_normal_weight=float(weight[signed>0].sum()),regions=regions))
            target_list.append(target);bary_list.append(bary);weight_list.append(weight)
            io.save(OUT/'progress.json',dict(state='original_import_CPU_support',frames=i+1,pid=None,elapsed_seconds=time.monotonic()-started))
    target=torch.stack(target_list);bary=torch.stack(bary_list);weights=torch.stack(weight_list)
    base=bridge.base_vertices[order];A=bridge.linear[order];tri_vid=faces[binding]
    sums=weights.sum(0);bary_anchor=(bary*weights[:,:,None]).sum(0)/sums[:,None].clamp_min(1e-20)
    # No support -> zero descriptor. Never invent a valid anchor for these IDs.
    contributing=sums>0
    counts=(weights>0).sum(0);canonical=[];valids=[];condition_records=[]
    for i,row in enumerate(rows):
        assert time.monotonic()-started<p['max_seconds']
        Bi=(A[i][tri_vid]*bary_anchor[:,:,None,None]).sum(1)
        vi=(base[i][tri_vid]*bary_anchor[:,:,None]).sum(1)
        singular=torch.linalg.svdvals(Bi)
        cond=singular[:,0]/singular[:,-1].clamp_min(1e-20)
        ok=contributing&(singular[:,-1]>0)&(cond<p['numerical_inverse_condition_max'])
        d=torch.zeros_like(target[i]);d[ok]=torch.linalg.solve(Bi[ok],(target[i]-vi)[ok])
        assert torch.isfinite(d).all()
        canonical.append(d);valids.append(ok)
        condition_records.append(dict(frame=row['frame'],supported_noninvertible=int(((weights[i]>0)&~ok).sum()),
            max_supported_condition=float(cond[weights[i]>0].max()) if (weights[i]>0).any() else None))
    descriptors=torch.stack(canonical);valids=torch.stack(valids)
    dw=weights*valids;den=dw.sum(0)
    mean=(descriptors*dw[:,:,None]).sum(0)/den[:,None].clamp_min(1e-20)
    residual=(descriptors-mean[None]).norm(dim=-1)
    dispersion=((residual.square()*dw).sum(0)/den.clamp_min(1e-20)).sqrt()
    multi=counts>=2;multi_weight=den*multi
    coverage=dict(gaussians=len(binding),at_least_one_frame=int(contributing.sum()),at_least_two_frames=int(multi.sum()),
        at_least_ten_frames=int((counts>=10).sum()),all59_frames=int((counts==59).sum()),
        unique_supported_triangles=int(torch.unique(binding[contributing]).numel()),
        learned_clothing_triangles=int(perface.sum()),all59_nonempty_support=all(x['points']>0 for x in records),
        supported_clothing_vertices=int(torch.unique(tri_vid[contributing]).numel()),learnable_vertices=int(learn.sum()))
    q=quality(base.numpy(),fit['faces']);clothq=q[:,perface.numpy()]
    numerical_pass=max(original_parity)<p['original_LBS_parity_m'] and coverage['all59_nonempty_support'] and all(x['supported_noninvertible']==0 for x in condition_records)
    assert not torch.cuda.is_initialized();checked()
    np.savez_compressed(OUT/'original-import-lbs-cache.npz',frame_ids=np.array([r['frame'] for r in rows]),
        base_vertices=base.numpy(),linear=A.numpy(),skeletal_joints=bridge.base_skeleton[order].numpy(),faces=fit['faces'],initial_offset=initial_offset,
        learnable_vertices=learn.numpy())
    np.savez_compressed(OUT/'teacher-support-diagnostic-only.npz',frame_ids=np.array([r['frame'] for r in rows]),
        binding=binding.numpy(),targets=target.numpy(),barycentric_per_frame=bary.numpy(),weights=weights.numpy(),
        diagnostic_fixed_barycentric_anchor=bary_anchor.numpy(),canonical_descriptor_rms_m=dispersion.numpy(),
        support_count=counts.numpy())
    result=dict(state='completed_import_teacher_numerical_contract' if numerical_pass else 'import_teacher_contract_not_admitted',
        updated=io.now(),frames=59,source_step=120000,source_mesh='Original immutable VHAP import',
        original_import_replay_max_abs_m=max(original_parity),independent_original_LBS_oracle=oracle,
        teacher_reference_world_PLY_parity_m=teacher_error,coverage=coverage,frames_result=records,
        canonical_descriptor_dispersion_mm=weighted_quantile(dispersion.numpy()*1000,multi_weight.numpy()),
        descriptor_frame_residual_mm=weighted_quantile(residual.numpy()*1000,(dw*multi).numpy()),
        numerical_inverse_records=condition_records,
        import_clothing_triangle_shape_quality=dict(min=float(clothq.min()),p01=float(np.percentile(clothq,1)),median=float(np.median(clothq)),
            below_descriptive_001=int((clothq<.01).sum()),diagnostic_only=True),
        skeletal_joints_frozen=True,numerical_contract_passed=numerical_pass,
        physical_teacher_correspondence_accepted=False,training_admitted=False,quality_accepted=False,
        caveat='Same learnedGaussian identity is not independent material identity. Canonical dispersion uses diagnostic anchor aggregation, not a lower bound on fitting error or a trained mesh. Nonempty support and stable LBS are implementation checks only.',
        optimizer_updates=0,cuda_initialized=False,new_mesh_or_PLY_exported=False,
        elapsed_seconds=time.monotonic()-started)
    assert result['elapsed_seconds']<180
    io.save(OUT/'result.json',result);io.save(OUT/'progress.json',dict(state=result['state'],frames=59,pid=None))
    print('IMPORT_TEACHER_CONTRACT',result['state'],coverage,result['canonical_descriptor_dispersion_mm'],result['import_clothing_triangle_shape_quality'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();globals()[a.mode]()
