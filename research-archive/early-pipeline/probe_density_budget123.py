"""Bounded structural GPU checks for the original ADC with a point budget."""
from pathlib import Path
import argparse, fcntl, json, sys
import numpy as np
import run_official_gaussians123 as io


def main():
    source=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle/shared-vertex-offset-adc-1000')
    out=source.parent/'official-gaussianavatars-parameters/density-budget-gpu-check.json'
    assert not out.exists()
    sys.path.insert(0,str(io.PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(io.ASSETS)
    import torch
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from arguments import OptimizationParams
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();options=OptimizationParams(parser).extract(parser.parse_args([]))
    def fresh():
        torch.manual_seed(6060)
        m=SMPLXGaussianModel(3);m.load_fresh_meshes(source/'fit/fitted.npz')
        m.initialize_surface(20908,6060)
        m.enable_shared_static_offset(source/'fit/fitted.npz',io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
        m.training_setup(options)
        # Populate real Adam state with a zero-gradient step; no parameter changes.
        before=[p.detach().clone() for g in m.optimizer.param_groups for p in g['params']]
        sum(p.sum()*0 for g in m.optimizer.param_groups for p in g['params']).backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
        assert all(torch.equal(p,q) for p,q in zip((p for g in m.optimizer.param_groups for p in g['params']),before))
        return m
    records=[]
    with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for mode in ('clone','split','mixed','full'):
            m=fresh();offset=m.static_offset
            if mode=='clone': m.percent_dense=1e6
            elif mode=='split':m.percent_dense=0
            elif mode=='mixed':m.percent_dense=float(m.get_scaling.max(1).values.median())
            n=len(m._xyz);extra=0 if mode=='full' else 17
            m.xyz_gradient_accum[:]=torch.linspace(.001,.01,n,device='cuda')[:,None]
            m.denom[:]=1
            with torch.no_grad():m.densify_and_prune(.0002,.005,1.,None,max_points=n+extra)
            assert len(m._xyz)==n+extra
            assert int(m.binding_counter.min())>=1
            assert m.static_offset is offset and torch.count_nonzero(offset)==0
            assert torch.equal(torch.bincount(m.binding,minlength=n).to(m.binding_counter),m.binding_counter)
            if extra: assert torch.equal(torch.nonzero(m.binding_counter>1).flatten(),torch.arange(n-extra,n,device='cuda'))
            for g in m.optimizer.param_groups:
                for p in g['params']:
                    s=m.optimizer.state[p]
                    assert s['exp_avg'].shape==s['exp_avg_sq'].shape==p.shape
                    assert torch.isfinite(p).all()
            records.append(dict(case=mode,points_before=n,points_after=len(m._xyz),limit=n+extra,
                minimum_binding=int(m.binding_counter.min()),highest_gradient_parents_selected=True,
                Adam_shapes_valid=True,shared_offset_identity_preserved=True))
            del m
        # Default None and a non-binding budget must retain identical original ADC.
        arrays=[]
        for cap in (None,41816):
            m=fresh();m.xyz_gradient_accum[:]=.001;m.denom[:]=1
            torch.manual_seed(6060)
            with torch.no_grad():m.densify_and_prune(.0002,.005,1.,None,max_points=cap)
            arrays.append([getattr(m,name).detach().cpu().clone() for name in ('_xyz','_scaling','_opacity','binding','binding_counter')])
            del m
        assert all(torch.equal(x,y) for x,y in zip(*arrays))
    io.save(out,dict(updated=io.now(),passed=True,checks=records,unconstrained_matches_original=True,
        photometric_training_updates=0,zero_gradient_Adam_state_steps=6,
        source_fit_sha256=io.sha(source/'fit/fitted.npz'),
        source_hashes={str(p):io.sha(p) for p in [io.PROJECT/'src/avatar_core/scene/gaussian_model.py',io.PROJECT/'src/shuolee_portrait/training.py',Path(__file__)]},
        scope='ADC point count, strongest-gradient selection, binding coverage, shared-offset identity and optimizer-state preservation; not image quality acceptance.'))
    print('DENSITY_BUDGET_GPU_CHECK_PASSED',json.dumps(records))


if __name__=='__main__':main()
