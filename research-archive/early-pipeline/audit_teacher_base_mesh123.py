"""No-update check of the already fitted source mesh and regularizer reference."""
from pathlib import Path
import time
import numpy as np
import torch
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import OUT as AUDIT,verify

OUT=AUDIT/'teacher-base-mesh-audit'


def normals(v,f):
    t=v[:,f];return np.cross(t[:,:,1]-t[:,:,0],t[:,:,2]-t[:,:,0])


def quality(v,f):
    t=v[:,f];a,b,c=t[:,:,1]-t[:,:,0],t[:,:,2]-t[:,:,1],t[:,:,0]-t[:,:,2]
    twice=np.linalg.norm(np.cross(a,-c),axis=-1)
    return 2*np.sqrt(3)*twice/np.maximum((a*a+b*b+c*c).sum(-1),1e-30)


def main():
    verify();started=time.monotonic();OUT.mkdir(exist_ok=False)
    cp=AUDIT/'frozen-skeleton-contract/frozen-lbs-cache.npz';candidate=AUDIT/'teacher-visible-surface-100/candidate-diagnostic-only.npz'
    lfile=AUDIT/'upstream-laplacian-contract/original-laplacian.npz'
    files=[Path(__file__),cp,candidate,lfile]
    io.save(OUT/'plan.json',dict(created=io.now(),cpu_only=True,max_seconds=90,optimizer_updates=0,
        hashes={str(p):io.sha(p) for p in files},
        scope='Check whether new failed faces were already fragile, and distinguish original total-offset regularization from local additional-delta smoothing.',
        no_new_mesh_or_training=True,no_geometry_gate_change=True,
        caveat='Reference removes canonical surface offsets using frozen original LBS. It is an anatomical comparison, not measured clothing geometry or a replacement model.'))
    with np.load(cp,allow_pickle=False) as z:cache=dict(z)
    with np.load(candidate,allow_pickle=False) as z:data=dict(z)
    with np.load(lfile,allow_pickle=False) as z:lap=dict(z)
    slots={str(x):i for i,x in enumerate(cache['frame_ids'])};order=np.array([slots[str(x)] for x in data['frame_ids']])
    base=data['vertices_before'];after=data['vertices_after'];faces=data['faces'];A=cache['linear'][order];initial=cache['initial_offset']
    assert np.max(np.abs(base-cache['base_vertices'][order]))<1e-6
    unexpanded=base-(A@initial[None,:,:,None])[...,0]
    n0,n1,nref=normals(base,faces),normals(after,faces),normals(unexpanded,faces)
    valid=np.linalg.norm(n0,axis=-1)>1e-12
    reversed=((n0*n1).sum(-1)<=0)&valid;locations=np.argwhere(reversed)
    q0,q1,qr=quality(base,faces),quality(after,faces),quality(unexpanded,faces)
    with np.load(io.ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=True) as z:dominant=z['weights'].argmax(1)
    from smplx.joint_names import JOINT_NAMES
    details=[]
    for i,j in locations:
        ids=faces[j]
        details.append(dict(frame=str(data['frame_ids'][i]),triangle=int(j),vertices=ids.tolist(),
            dominant_joints=[dict(id=int(dominant[x]),name=JOINT_NAMES[int(dominant[x])]) for x in ids],
            learnable=data['learnable_vertices'][ids].tolist(),source_quality=float(q0[i,j]),candidate_quality=float(q1[i,j]),unexpanded_reference_quality=float(qr[i,j]),
            source_normal_dot_reference=float(np.dot(n0[i,j],nref[i,j])),candidate_normal_dot_reference=float(np.dot(n1[i,j],nref[i,j])),
            source_normal_opposed_to_reference=bool(np.dot(n0[i,j],nref[i,j])<0),
            candidate_normal_opposed_to_reference=bool(np.dot(n1[i,j],nref[i,j])<0)))
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    L=torch.sparse_coo_tensor(torch.tensor(lap['indices']),torch.tensor(lap['values']),tuple(lap['shape'])).coalesce()
    learn=torch.tensor(data['learnable_vertices']);d=torch.zeros_like(torch.tensor(initial),requires_grad=True)
    loss_local=torch.sparse.mm(L,d*learn[:,None]).square().sum(-1).mean()
    loss_total=torch.sparse.mm(L,torch.tensor(initial)+d*learn[:,None]).square().sum(-1).mean()
    g=torch.autograd.grad(loss_total,d)[0]
    source_reversed=((n0*nref).sum(-1)<0)&valid
    result=dict(state='source_mesh_and_regularizer_reference_audited',updated=io.now(),frames=59,
        failed_face_details=details,
        relative_normal_changes_before_new_fit=dict(observations=int(source_reversed.sum()),unique_triangles=int(source_reversed.any(0).sum()),
            caveat='Relative to same skeleton with offsets removed; not a complete collision or physical clothing assessment.'),
        source_triangle_shape_quality=dict(min=float(q0.min()),p01=float(np.percentile(q0,1)),median=float(np.median(q0))),
        unexpanded_triangle_shape_quality=dict(min=float(qr.min()),p01=float(np.percentile(qr,1)),median=float(np.median(qr))),
        regularizer_reference=dict(local_added_delta_loss_at_zero=float(loss_local),total_offset_loss_at_zero=float(loss_total),
            total_offset_gradient_L2_on_clothing=float(g.norm()),source_offset_median_m=float(np.median(np.linalg.norm(initial,axis=-1))),
            distinction='L(delta) preserves all preexisting source roughness. Original VHAP regularizes total static/dynamic offset relative to the offset-free mesh. Formula reuse alone did not reproduce that reference.',
            production_change=False),
        model_promoted=False,optimizer_updates=0,cuda_initialized=False,quality_accepted=False,
        next='Use this evidence to preregister a reference-correct original total-offset regularizer control, preserving the fixed-skeleton interface, actual alpha contribution and all existing protection. Do not silently relax normal gates or add arbitrary learning-rate trials.',
        elapsed_cpu_seconds=time.monotonic()-started)
    assert result['elapsed_cpu_seconds']<90
    io.save(OUT/'result.json',result);print('SOURCE_MESH_AUDIT',details,result['regularizer_reference'],flush=True)


if __name__=='__main__':main()
