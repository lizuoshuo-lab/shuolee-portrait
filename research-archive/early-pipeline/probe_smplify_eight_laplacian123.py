"""Read-only units/gradient audit for the failed eight-frame shared offset.

Uses the previously verified original normalized Laplacian, without a new
optimizer or a coefficient sweep. The saved failed delta is diagnostic only.
"""
from pathlib import Path
import numpy as np
import torch
import train_smplify_eight123 as eight
from facial_geometry_boundary123 import clothing_only_offset
from run_official_gaussians123 import read, save, sha, now

ROOT=eight.OUT.parent/'body-gaussian-smplify-eight-laplacian'
LAP=eight.direct.AUDIT/'upstream-laplacian-contract/original-laplacian.npz'
FAILED=eight.OUT.parent/'body-gaussian-smplify-eight-failure-capture/gaussian-run/step-000900/checkpoint.pt'
WEIGHT=1e6  # Existing VHAP normalized-offset Laplacian default, not a GA default.

def main():
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    ROOT.mkdir(exist_ok=False)
    save(ROOT/'precheck-plan.json',dict(updated=now(),purpose='Original formula, units and facial exclusion only; no training',
        weight=WEIGHT,weight_source='Existing VHAP reg_offset_lap default1e6; original GA default is0 and acts on dynamic_offset.',
        coefficient_sweep=False,original_training_loss_unchanged=True,optimizer_updates=0))
    with np.load(LAP,allow_pickle=False) as z: lap=dict(z)
    with np.load(eight.OUT/'fit/surface-cache.npz',allow_pickle=False) as z: cache=dict(z)
    assert np.array_equal(lap['faces'],cache['faces'])
    L=torch.sparse_coo_tensor(torch.tensor(lap['indices']),torch.tensor(lap['values']),tuple(lap['shape'])).coalesce().double()
    cp=torch.load(FAILED,map_location='cpu',weights_only=False)
    d=cp['static_offset'].double().requires_grad_()
    learn=torch.tensor(cache['clothing_vertices']);protected=torch.tensor(cache['protected_face_vertices'])
    def loss(x):
        safe=clothing_only_offset(x,learn,protected)
        return torch.sparse.mm(L,safe).square().sum(-1).mean()
    value=loss(d);grad=torch.autograd.grad(value,d)[0]
    assert float(grad[protected].abs().max())==0 and float(grad[~learn].abs().max())==0
    assert loss(torch.zeros_like(d))==0
    checks=[]
    for f in torch.topk(grad.abs().flatten(),3).indices:
        i,j=divmod(int(f),3);a=d.detach().clone();b=a.clone();a[i,j]+=1e-7;b[i,j]-=1e-7
        fd=float((loss(a)-loss(b))/2e-7);analytic=float(grad[i,j]);assert abs(fd-analytic)<1e-12
        checks.append(dict(vertex=i,axis=j,analytic=analytic,finite_difference=fd))
    tail=read(eight.OUT.parent/'body-gaussian-smplify-eight-failure-capture/gaussian-run/preflight.json')
    result=dict(updated=now(),state='cpu_contract_passed',weight=WEIGHT,sparse_loss=float(value),weighted_loss=float(value)*WEIGHT,
        gradient_norm=float(grad.norm()),weighted_gradient_norm=float(grad.norm())*WEIGHT,
        zero_offset_loss=0,protected_face_gradient=0,unlearnable_gradient=0,finite_difference=checks,
        units='meters squared: mean over10475 vertices after sum XYZ; canonical shared delta, not per-frame accumulation',
        weight_source='VHAP original1e6; not original GA default. Same sparse formula already compared to both exact upstream methods.',
        old_first_rgb_gradient_norm=tail['static_offset_gradient_norm'],
        gradient_comparison_scope='Different saved state; values describe scale only, not an equal-state objective comparison.',
        source_hashes={str(p):sha(p) for p in [Path(__file__),LAP,FAILED,eight.OUT/'fit/surface-cache.npz']},
        optimizer_updates=0,cuda_initialized=False,quality_accepted=False)
    save(ROOT/'precheck-result.json',result)
    print(result,flush=True)

if __name__=='__main__':main()
