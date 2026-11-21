"""CPU-only check of exact upstream Laplacian formulas at the saved mesh scale.

This does not optimize, alter the old experiment, or initialize CUDA.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import time
import numpy as np
import torch
from pytorch3d.structures import Meshes
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import OUT as AUDIT, UP

VHAP=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx')
OUT=AUDIT/'upstream-laplacian-contract'
PILOT=AUDIT/'teacher-surface-100'


def original_method(path,name):
    tree=ast.parse(path.read_text())
    found=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(found)==1
    scope={'torch':torch}
    exec(compile(ast.Module(body=found,type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


def main():
    started=time.monotonic();torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=False)
    tracker=VHAP/'vhap/model/tracker.py';ga=UP/'scene/flame_gaussian_model.py'
    files=[Path(__file__),tracker,ga,VHAP/'vhap/model/flame.py',VHAP/'vhap/config/base.py',
           PILOT/'candidate-diagnostic-only.npz',PILOT/'result.json']
    io.save(OUT/'plan.json',dict(created=io.now(),optimizer_updates=0,max_seconds=120,
        files={str(p):io.sha(p) for p in files},hypothesis='Sparse original PyTorch3D uniform Laplacian has the same loss and gradients as exact original VHAP/GA methods.',
        controls=['Exact original method AST, no rewritten loss oracle.','Full10475 vertices and original20908 faces; no head/hand/foot weight relaxation.',
                  'Zero delta, actual rejected20-step delta, gradient equality and finite differences.','Meters and mean over all10475 vertices, not per-frame accumulation.'],
        trial_weight=1e6,trial_weight_source='VHAP base.py reg_offset_lap',no_new_mesh_or_Gaussian=True))
    with np.load(PILOT/'candidate-diagnostic-only.npz',allow_pickle=False) as z:data=dict(z)
    delta=torch.tensor(data['additional_canonical_delta']);base=torch.tensor(data['vertices_before'][0]);faces=torch.tensor(data['faces'],dtype=torch.long)
    learn=torch.tensor(data['learnable_vertices']);assert base.shape==(10475,3) and faces.shape==(20908,3)
    sparse=Meshes(verts=[base],faces=[faces]).laplacian_packed().coalesce()
    dense=sparse.to_dense()
    row_sums=torch.sparse.mm(sparse,torch.ones((10475,1)))
    vhap=original_method(tracker,'compute_laplacian_smoothing_loss')
    ga_loss=original_method(ga,'compute_laplacian_loss')
    # Both methods regularize the change of Laplacian coordinates. Original GA
    # applies it to dynamic_offset; using it for a new shared surface delta is
    # explicitly a local adapter, not a claim that original GA already did so.
    proxy=SimpleNamespace(flame=SimpleNamespace(laplacian_matrix=dense),
        flame_model=SimpleNamespace(laplacian_matrix=dense),timestep=0)
    d=delta.clone().requires_grad_()
    proxy.flame_param={'dynamic_offset':d[None]};proxy.verts_cano=(base+d)[None]
    actual_vhap=vhap(proxy,base[None],(base+d)[None]).mean()
    actual_ga=ga_loss(proxy)
    compact=torch.sparse.mm(sparse,d).square().sum(-1).mean()
    grads=[torch.autograd.grad(x,d,retain_graph=True)[0] for x in (actual_vhap,actual_ga,compact)]
    scale=max(float(grads[0].norm()),1e-20)
    rel=[float((g-grads[0]).norm()/scale) for g in grads[1:]]
    assert max(rel)<1e-3,rel
    assert abs(float(compact-actual_vhap))<1e-9
    assert float(torch.sparse.mm(sparse,torch.zeros_like(d)).square().sum())==0
    # Float64 independent finite differences on the same full mesh connectivity.
    L64=sparse.double();d64=delta.double().requires_grad_()
    energy64=torch.sparse.mm(L64,d64).square().sum(-1).mean()
    g64=torch.autograd.grad(energy64,d64)[0]
    ids=torch.topk(g64.abs().flatten(),3).indices
    fd=[]
    for flat in ids:
        i,j=divmod(int(flat),3);plus=d64.detach().clone();minus=plus.clone()
        plus[i,j]+=1e-7;minus[i,j]-=1e-7
        f=lambda x:torch.sparse.mm(L64,x).square().sum(-1).mean()
        numeric=float((f(plus)-f(minus))/2e-7);analytic=float(g64[i,j])
        assert abs(numeric-analytic)<1e-12
        fd.append(dict(vertex=i,coordinate=j,analytic=analytic,finite_difference=numeric))
    edges=torch.cat((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]));edges=torch.unique(edges.sort(1).values,dim=0)
    edge_diff=d[edges[:,0]]-d[edges[:,1]]
    old_smooth=.1*torch.nn.functional.smooth_l1_loss(edge_diff,torch.zeros_like(edge_diff),beta=.005,reduction='none').sum(-1).mean()
    old_grad=torch.autograd.grad(old_smooth,d)[0];old=io.read(PILOT/'result.json')
    # Persist original sparse connectivity once, preserving its normalized values.
    np.savez_compressed(OUT/'original-laplacian.npz',indices=sparse.indices().numpy(),values=sparse.values().numpy(),shape=np.array(sparse.shape),faces=data['faces'])
    assert not torch.cuda.is_initialized();assert time.monotonic()-started<120
    result=dict(state='passed_original_laplacian_cpu_contract',updated=io.now(),vertices=10475,faces=20908,
        sparse_nonzeros=sparse._nnz(),row_sum_max_abs=float(row_sums.abs().max()),
        original_vhap_loss=float(actual_vhap),original_ga_loss=float(actual_ga),sparse_loss=float(compact),
        sparse_vs_original_relative_gradient_L2=rel,finite_difference=fd,
        zero_offset_loss=0.,units='square meters, sum XYZ then mean over all10475 vertices; no59-frame multiplier',
        candidate_delta_source='Rejected20-step geometry only; diagnostic object, NOT new initialization.',
        trial_weight=1e6,weighted_laplacian_value_at_rejected_delta=float(compact)*1e6,
        guide_value_at_rejected_delta=old['final_guide_loss'],
        weighted_laplacian_to_guide_value_ratio=float(compact)*1e6/old['final_guide_loss'],
        previous_weighted_edge_value=float(old_smooth),
        laplacian_gradient_norm_at_weight1e6=float(grads[2].norm())*1e6,
        previous_weighted_edge_gradient_norm=float(old_grad.norm()),
        learnable_vertices=int(learn.sum()),sparse_cache_sha256=io.sha(OUT/'original-laplacian.npz'),
        exact_original_methods_executed=True,source_hashes={str(p):io.sha(p) for p in files},
        scope='Formula reuse and unit verification only. Original GA default regularizes dynamic_offset; this proposed shared-delta adaptation is local.',
        next='A single fresh comparison may replace the weak local edge penalty with original normalized Laplacian at VHAP default1e6. Keep teacher, bone freeze, LR, budget and guard unchanged; do not scan coefficients.',
        optimizer_updates=0,cuda_initialized=False,new_mesh_or_Gaussian=False,quality_accepted=False,
        elapsed_cpu_seconds=time.monotonic()-started)
    io.save(OUT/'result.json',result)
    print('UPSTREAM_LAPLACIAN_CONTRACT',result['state'],'weighted/value',result['weighted_laplacian_to_guide_value_ratio'],'grad parity',rel,flush=True)


if __name__=='__main__':main()
